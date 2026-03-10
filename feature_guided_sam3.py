#!/usr/bin/env python3
"""SAM3 building segmentation guided by SAM3's own layer-23 features.

Scoring: combined = SAM3_score × pc1_recall
  SAM3_score — language model confidence for "the main building"
  pc1_recall — |mask ∩ hot| / |hot|  where hot = L23 PC1 above image median

  Recall rewards masks that cover the full building hot-zone.
  Precision (fraction of mask pixels that are hot) saturates at 1.0 when any
  mask is fully inside the hot zone — it cannot distinguish a partial wing
  from the whole building. Recall keeps growing until the mask covers all of
  the semantically hot region, selecting the most complete footprint.

Why it works: SAM3 layer-23 PC1 explains 56.8% of token variance — the
dominant semantic axis separates building from background. Multiplying by
the SAM3 score kills tiny fragments (score ≈ 0) and large wrong-region
masks (coverage ≈ 0), rewarding the correct building footprint.

No external models. No hand-tuned thresholds. Both signals from SAM3.

Usage:
  uv run python feature_guided_sam3.py image.png
  uv run python feature_guided_sam3.py "screenshots/*.png" --compare
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from torchvision import transforms

import importlib.resources, types, sys
_pkg = types.ModuleType("pkg_resources")
_pkg.resource_filename = lambda package, path: str(
    importlib.resources.files(package).joinpath(path)
)
sys.modules["pkg_resources"] = _pkg

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


SAM3_TRANSFORM = transforms.Compose([
    transforms.Resize((1008, 1008)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
GRID = 72   # SAM3 ViT-H: 1008 / 14 = 72 patches per side
L23  = 23   # peak semantic layer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", help="Image paths or glob patterns")
    p.add_argument("--prompt", default="the main building")
    p.add_argument("--confidence", type=float, default=0.0)
    p.add_argument("--compare", action="store_true", help="Save side-by-side comparison")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--viz-dir",  default="feature_guided_outputs/viz")
    p.add_argument("--mask-dir", default="feature_guided_outputs/masks")
    p.add_argument("--metadata-out", default="feature_guided_outputs/results.json")
    p.add_argument("--alpha", type=float, default=0.45)
    return p.parse_args()


def extract_pc1_map(model, image: Image.Image, device: torch.device,
                    orig_h: int, orig_w: int) -> np.ndarray:
    """Return (orig_h, orig_w) float32 z-scored L23 PC1 map. Positive = building.

    NOTE: kept for standalone use. In the main inference loop use
    compute_pc1_from_tokens() on tokens already captured by the permanent hook
    registered around run_sam3() to avoid a second vision-backbone forward pass.
    """
    img_t = SAM3_TRANSFORM(image).unsqueeze(0).to(device)
    captured: dict = {}

    def _hook(mod, inp, out):
        if out.ndim == 4 and out.shape[0] == 1:
            captured["tokens"] = out[0].detach().cpu().float()

    h = model.backbone.vision_backbone.trunk.blocks[L23].register_forward_hook(_hook)
    with torch.no_grad():
        model.backbone.vision_backbone(img_t)
    h.remove()

    return compute_pc1_from_tokens(captured["tokens"], orig_h, orig_w)


def compute_pc1_from_tokens(tokens: torch.Tensor, orig_h: int, orig_w: int) -> np.ndarray:
    """PCA on pre-captured L23 tokens → (orig_h, orig_w) float32 z-scored PC1 map.

    Takes tokens already captured by a forward hook during processor.set_image()
    so no additional vision-backbone pass is needed.
    """
    flat = tokens.reshape(-1, 1024).numpy()
    pc1  = PCA(n_components=1, random_state=42).fit_transform(flat).reshape(GRID, GRID)

    if pc1[GRID // 2, GRID // 2] < 0:
        pc1 = -pc1

    pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-8)
    return cv2.resize(pc1.astype(np.float32), (orig_w, orig_h))


def run_sam3(processor: Sam3Processor, image: Image.Image,
             prompt: str, box: Optional[List[float]]) -> Tuple[List[np.ndarray], List[float]]:
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt, state)
    if box:
        state = processor.add_geometric_prompt(box=box, label=True, state=state)
    masks_t, scores_t = state.get("masks"), state.get("scores")
    if masks_t is None or len(masks_t) == 0:
        return [], []
    order = torch.argsort(scores_t, descending=True)
    masks, scores = [], []
    for idx in order:
        m = masks_t[idx]
        if m.ndim == 3:
            m = m.squeeze(0)
        masks.append(m.cpu().numpy().astype(np.uint8))
        scores.append(float(scores_t[idx].item()))
    return masks, scores


def pick_by_combined(masks: List[np.ndarray], sam3_scores: List[float],
                     pc1_map: np.ndarray, orig_h: int, orig_w: int,
                     min_area: float = 0.02, max_area: float = 0.90,
                     ) -> Tuple[int, List[float], List[float]]:
    """Score = SAM3_score × pc1_recall
    pc1_recall = |mask ∩ hot| / |hot|   (hot = PC1 above image median)

    Recall rewards masks that COVER more of the semantically hot region.
    Precision (old formula) saturates at 1.0 for any mask fully inside the
    hot zone, so it cannot distinguish a partial wing from the whole building.
    Recall grows with mask area until the mask covers the full hot zone,
    naturally selecting the most complete building footprint.
    """
    pc1_median  = float(np.median(pc1_map))
    hot         = pc1_map > pc1_median
    total_hot   = float(hot.sum()) + 1e-8
    combined, coverages = [], []
    for i, mask in enumerate(masks):
        m = mask if mask.shape[:2] == (orig_h, orig_w) else cv2.resize(
            mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        area = float(m.mean())
        if area < min_area or area > max_area:
            combined.append(-np.inf)
            coverages.append(float("nan"))
            continue
        recall = float((hot & m.astype(bool)).sum()) / total_hot
        coverages.append(recall)
        combined.append(sam3_scores[i] * recall)
    return int(np.argmax(combined)), combined, coverages


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45,
                 color: Tuple[int, int, int] = (0, 220, 80)) -> np.ndarray:
    if mask.shape[:2] != bgr.shape[:2]:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = bgr.copy().astype(float)
    m   = mask.astype(bool)
    for c, col in enumerate(color):
        out[..., c][m] = out[..., c][m] * (1 - alpha) + col * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours((mask * 255).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 2)
    return out


def _put_label(img_bgr: np.ndarray, lines: List[str]) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    PAD, LINE_H = 8, 24
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(
        [(0, 0), (pil.width, PAD + len(lines) * LINE_H + PAD)], fill=(0, 0, 0, 175))
    pil  = Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(pil)
    y = PAD
    for line in lines:
        draw.text((PAD + 1, y + 1), line, font=font, fill=(0, 0, 0))
        draw.text((PAD,     y),     line, font=font, fill=(255, 255, 255))
        y += LINE_H
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def make_comparison(bgr, sam3_mask, best_mask, sam3_lines, best_lines, alpha):
    specs = [
        (["Original"],                        None,      None),
        (["SAM3 score-pick"] + sam3_lines,    sam3_mask, (0, 140, 255)),
        (["L23 guided pick"]  + best_lines,   best_mask, (0, 220, 80)),
    ]
    panels = []
    for lines, mask, color in specs:
        img = bgr.copy() if mask is None else overlay_mask(bgr, mask, alpha, color)
        h, w = img.shape[:2]
        panels.append(_put_label(cv2.resize(img, (480, int(h * 480 / w))), lines))
    mh  = max(p.shape[0] for p in panels)
    sep = np.full((mh, 3, 3), 50, dtype=np.uint8)
    padded = [np.pad(p, ((0, mh - p.shape[0]), (0, 0), (0, 0))) for p in panels]
    return np.hstack([padded[0], sep, padded[1], sep, padded[2]])


def main() -> None:
    args = parse_args()

    image_paths: List[Path] = []
    for pattern in args.images:
        p = Path(pattern)
        if p.exists():
            image_paths.append(p.resolve())
        else:
            image_paths.extend(m.resolve() for m in sorted(Path().glob(pattern)) if m.is_file())
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise SystemExit("No valid images found.")

    viz_dir  = Path(args.viz_dir);  viz_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.mask_dir); mask_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_out).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    sam3   = build_sam3_image_model()
    sam3.eval().to(device)
    processor = Sam3Processor(sam3)
    processor.set_confidence_threshold(args.confidence)
    print(f"SAM3 ready.  {len(image_paths)} image(s).\n")

    # Register the hook once. processor.set_image() calls backbone.forward_image()
    # which calls vision_backbone.forward() and passes through all trunk blocks,
    # so the hook fires during run_sam3() as a side effect — no second pass needed.
    _l23_captured: dict = {}

    def _l23_hook(mod, inp, out):
        if out.ndim == 4 and out.shape[0] == 1:
            _l23_captured["tokens"] = out[0].detach().cpu().float()

    _hook_handle = sam3.backbone.vision_backbone.trunk.blocks[L23].register_forward_hook(_l23_hook)

    summary = []
    t_total = time.perf_counter()

    for img_idx, image_path in enumerate(image_paths, 1):
        t_img = time.perf_counter()
        print(f"[{img_idx}/{len(image_paths)}] {image_path.name}")
        image  = Image.open(str(image_path)).convert("RGB")
        orig_w, orig_h = image.size
        bgr    = cv2.imread(str(image_path))

        # processor.set_image() triggers _l23_hook as a side effect
        masks, scores = run_sam3(processor, image, args.prompt, None)
        if not masks:
            print(f"  No masks — skipping.  ({time.perf_counter() - t_img:.1f}s)\n")
            continue

        # PCA on already-captured tokens — no second vision-backbone forward pass
        pc1_map = compute_pc1_from_tokens(_l23_captured["tokens"], orig_h, orig_w)
        print(f"  {len(masks)} candidates  score range: {scores[0]:.4f} → {scores[-1]:.4f}")

        best_idx, combined, coverages = pick_by_combined(masks, scores, pc1_map, orig_h, orig_w)
        best_mask = masks[best_idx]
        if best_mask.shape[:2] != (orig_h, orig_w):
            best_mask = cv2.resize(best_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        print(f"  pick: rank {best_idx+1}  score={scores[best_idx]:.4f}  "
              f"area={best_mask.mean():.1%}  pc1_cov={coverages[best_idx]:.3f}  "
              f"combined={combined[best_idx]:.4f}")

        stem = image_path.stem
        cv2.imwrite(str(mask_dir / f"{stem}_mask.png"), best_mask * 255)
        cv2.imwrite(str(viz_dir  / f"{stem}_viz.png"),
                    overlay_mask(bgr, best_mask, args.alpha))

        if args.compare:
            sam3_m = masks[0].copy()
            if sam3_m.shape[:2] != (orig_h, orig_w):
                sam3_m = cv2.resize(sam3_m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(viz_dir / f"{stem}_comparison.png"), make_comparison(
                bgr, sam3_m, best_mask,
                sam3_lines=[f"score={scores[0]:.4f}  area={sam3_m.mean():.1%}",
                             f"pc1_cov={coverages[0]:.3f}"],
                best_lines =[f"score={scores[best_idx]:.4f}  area={best_mask.mean():.1%}",
                              f"pc1_cov={coverages[best_idx]:.3f}  rank {best_idx+1}"],
                alpha=args.alpha,
            ))

        img_elapsed = time.perf_counter() - t_img
        summary.append({
            "image": str(image_path),
            "n_candidates": len(masks),
            "elapsed_s": round(img_elapsed, 2),
            "pick": {
                "rank": best_idx + 1,
                "score": scores[best_idx],
                "pc1_coverage": coverages[best_idx],
                "combined": combined[best_idx],
                "area": float(best_mask.mean()),
            },
        })
        print(f"  elapsed: {img_elapsed:.1f}s\n")

    _hook_handle.remove()

    total_elapsed = time.perf_counter() - t_total
    h, rem = divmod(int(total_elapsed), 3600)
    m, s = divmod(rem, 60)
    elapsed_str = (f"{h}h {m}m {s}s" if h else f"{m}m {s}s" if m else f"{total_elapsed:.1f}s")

    with open(args.metadata_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done.  {len(summary)} image(s) → {args.metadata_out}  |  Elapsed: {elapsed_str}")


if __name__ == "__main__":
    main()
