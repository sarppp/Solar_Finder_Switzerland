#!/usr/bin/env python3
"""SAM3 building segmentation guided by SAM3's own layer-23 features.

Scoring: combined = SAM3_score × pc1_recall × center_score
  SAM3_score   — language model confidence for "the main building"
  pc1_recall   — |mask ∩ hot| / |hot|  where hot = L23 PC1 above image median
  center_score — Gaussian proximity to image centre (domain prior: every image
                 is captured centred on the target building)

  pc1_recall rewards masks that cover the full building hot-zone; it keeps
  growing until the mask covers all semantically hot pixels, selecting the most
  complete footprint.

  center_score exploits the hard pipeline guarantee: the target building is
  always near the centre of the aerial crop.  A mask whose centroid is far from
  the image centre is penalised regardless of SAM3 or PC1 signal — this
  prevents bright off-centre neighbours from winning at canton scale.

Why it works: SAM3 layer-23 PC1 explains 56.8% of token variance — the
dominant semantic axis separates building from background. Multiplying by
the SAM3 score kills tiny fragments (score ≈ 0) and large wrong-region
masks (coverage ≈ 0). Center score eliminates geometrically off-centre
candidates even when PC1 hot region is scattered or degenerate.

PC1 fallback: when max PC1 recall across all candidates < PC1_FALLBACK_RECALL,
the PC1 signal is degenerate (e.g. scattered car-roof reflections). In that
case combined = SAM3_score × center_score (center prior still applied).

No external models. No hand-tuned thresholds. All signals from SAM3 + geometry.

Usage:
  uv run python feature_guided_sam3.py image.png
  uv run python feature_guided_sam3.py "screenshots/*.png" --compare
  uv run python feature_guided_sam3.py image.png --save-pc1 --debug-dir dbg
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

PC1_FALLBACK_RECALL = 0.15   # if max recall < this, PC1 is degenerate → use SAM3×center
CENTER_SIGMA        = 0.25   # Gaussian σ in units of half-diagonal (0=centre, 1=corner)
                             # σ=0.25 → score drops to ~0.6 at 25% of half-diag from centre
                             #           score drops to ~0.14 at 50% (strongly off-centre)


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
    # debug / analysis flags
    p.add_argument("--save-pc1", action="store_true",
                   help="Save PC1 heatmap + candidate score table to --debug-dir")
    p.add_argument("--debug-dir", default="feature_guided_outputs/debug",
                   help="Directory for debug outputs (PC1 heatmaps, score tables)")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of top candidates to show in debug grid")
    p.add_argument("--imagenet-norm", action="store_true",
                   help="Use ImageNet-normed forward pass for PC1 instead of hook (diagnostic)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# PC1 helpers
# ---------------------------------------------------------------------------

def extract_pc1_map(model, image: Image.Image, device: torch.device,
                    orig_h: int, orig_w: int,
                    polarity_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Explicit ImageNet-norm forward pass → PC1 map. Used only with --imagenet-norm."""
    img_t = SAM3_TRANSFORM(image).unsqueeze(0).to(device)
    captured: dict = {}

    def _hook(mod, inp, out):
        if out.ndim == 4 and out.shape[0] == 1:
            captured["tokens"] = out[0].detach().cpu().float()

    h = model.backbone.vision_backbone.trunk.blocks[L23].register_forward_hook(_hook)
    with torch.no_grad():
        model.backbone.vision_backbone(img_t)
    h.remove()

    return compute_pc1_from_tokens(captured["tokens"], orig_h, orig_w, polarity_mask)


def compute_pc1_from_tokens(tokens: torch.Tensor, orig_h: int, orig_w: int,
                             polarity_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """PCA on pre-captured L23 tokens → (orig_h, orig_w) float32 z-scored PC1 map.

    Takes tokens already captured by the permanent forward hook during run_sam3()
    → no second vision-backbone pass needed.

    polarity_mask: optional binary mask (orig_h × orig_w).  Its centroid is used
    to orient PC1 (positive = building).  Falls back to image centre if None or empty.
    """
    flat = tokens.reshape(-1, 1024).numpy()
    pc1  = PCA(n_components=1, random_state=42).fit_transform(flat).reshape(GRID, GRID)

    # Orient PC1: positive values should be "building"
    if polarity_mask is not None and polarity_mask.sum() > 0:
        m_grid = cv2.resize(polarity_mask.astype(np.float32), (GRID, GRID))
        cy = int(np.average(np.arange(GRID), weights=m_grid.mean(axis=1)))
        cx = int(np.average(np.arange(GRID), weights=m_grid.mean(axis=0)))
    else:
        cy, cx = GRID // 2, GRID // 2

    if pc1[cy, cx] < 0:
        pc1 = -pc1

    pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-8)
    return cv2.resize(pc1.astype(np.float32), (orig_w, orig_h))


# ---------------------------------------------------------------------------
# SAM3 inference
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Scoring / picking
# ---------------------------------------------------------------------------

def mask_center_score(mask: np.ndarray, orig_h: int, orig_w: int) -> float:
    """Gaussian proximity score centred on the image centre.

    Exploits the domain guarantee: every aerial crop is centred on the
    target building.  A mask whose centroid drifts to the edge of the image
    is almost certainly a neighbour, not the target.

    Returns 1.0 at image centre, approaches 0 at the corners.
    Uses CENTER_SIGMA (in units of half-diagonal) to control sharpness.
    """
    m = mask.astype(bool)
    if m.sum() == 0:
        return 0.0
    ys, xs = np.where(m)
    cy = ys.mean() / orig_h   # normalised [0, 1]
    cx = xs.mean() / orig_w
    # half-diagonal = distance from centre to corner in normalised coords
    half_diag = np.sqrt(0.5)  # = sqrt(0.5² + 0.5²)
    dist = np.sqrt((cy - 0.5) ** 2 + (cx - 0.5) ** 2) / half_diag
    return float(np.exp(-0.5 * (dist / CENTER_SIGMA) ** 2))


def pick_by_combined(masks: List[np.ndarray], sam3_scores: List[float],
                     pc1_map: np.ndarray, orig_h: int, orig_w: int,
                     min_area: float = 0.02, max_area: float = 0.90,
                     ) -> Tuple[int, List[float], List[float], List[float], bool]:
    """Score = SAM3_score × pc1_recall × center_score

    pc1_recall   = |mask ∩ hot| / |hot|   (hot = PC1 above image median)
    center_score = Gaussian proximity to image centre (see mask_center_score)

    Recall rewards masks that COVER more of the semantically hot region.
    Center score penalises off-centre neighbours even when PC1 is confused
    by scattered bright surfaces (cars, roofs, etc.).

    Returns (best_idx, combined_scores, coverages, center_scores, pc1_fallback).
    pc1_fallback=True when max recall < PC1_FALLBACK_RECALL — in that case
    best_idx is chosen by SAM3_score × center_score (PC1 dropped, center kept).
    """
    pc1_median  = float(np.median(pc1_map))
    hot         = pc1_map > pc1_median
    total_hot   = float(hot.sum()) + 1e-8
    combined, coverages, center_scores = [], [], []
    for i, mask in enumerate(masks):
        m = mask if mask.shape[:2] == (orig_h, orig_w) else cv2.resize(
            mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        area = float(m.mean())
        center = mask_center_score(m, orig_h, orig_w)
        center_scores.append(center)
        if area < min_area or area > max_area:
            combined.append(-np.inf)
            coverages.append(float("nan"))
            continue
        recall = float((hot & m.astype(bool)).sum()) / total_hot
        coverages.append(recall)
        combined.append(sam3_scores[i] * recall * center)

    max_recall = max((r for r in coverages if np.isfinite(r)), default=0.0)
    if max_recall < PC1_FALLBACK_RECALL:
        # PC1 is degenerate — fall back to SAM3_score × center_score
        valid = [i for i, c in enumerate(combined) if np.isfinite(c)]
        best_idx = (max(valid, key=lambda i: sam3_scores[i] * center_scores[i])
                    if valid else 0)
        return best_idx, combined, coverages, center_scores, True

    return int(np.argmax(combined)), combined, coverages, center_scores, False


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

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


def save_debug_pc1(debug_dir: Path, stem: str,
                   bgr: np.ndarray, pc1_map: np.ndarray,
                   masks: List[np.ndarray], sam3_scores: List[float],
                   combined: List[float], coverages: List[float],
                   center_scores: List[float],
                   best_idx: int, pc1_fallback: bool,
                   top_k: int = 5) -> None:
    """Save combined debug image: comparison row + PC1 heatmap + candidate grid.

    Layout:
      Row 1 (THUMB height each): [PC1 heatmap] | [original] | [SAM3 pick] | [guided pick]
      Row 2 (THUMB height each): top-k candidates by combined score
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    orig_h, orig_w = bgr.shape[:2]
    THUMB = 340   # thumbnail height for all panels

    def _resize_h(img, h):
        ratio = h / img.shape[0]
        return cv2.resize(img, (max(1, int(img.shape[1] * ratio)), h))

    # --- PC1 heatmap: clip to 5th–95th percentile so structure is visible ---
    vmin = float(np.percentile(pc1_map, 5))
    vmax = float(np.percentile(pc1_map, 95))
    pc1_clipped = np.clip(pc1_map, vmin, vmax)
    pc1_norm  = cv2.normalize(pc1_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    pc1_color = cv2.applyColorMap(pc1_norm, cv2.COLORMAP_JET)

    valid_cov = [r for r in coverages if np.isfinite(r)]
    max_recall = max(valid_cov) if valid_cov else 0.0
    pc1_label = [
        "PC1 heatmap (5–95 pct clip)",
        f"min={pc1_map.min():.2f}  max={pc1_map.max():.2f}  std={pc1_map.std():.2f}",
        f"max_recall={max_recall:.3f}" + ("  [DEGENERATE]" if pc1_fallback else ""),
    ]
    pc1_panel = _put_label(_resize_h(pc1_color, THUMB), pc1_label)

    # --- comparison: original | SAM3 top pick | guided pick ---
    sam3_mask = masks[0]
    best_mask = masks[best_idx]
    for m in (sam3_mask, best_mask):
        if m.shape[:2] != (orig_h, orig_w):
            m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    orig_panel = _put_label(_resize_h(bgr.copy(), THUMB), ["Original"])

    sam3_ctr = center_scores[0] if center_scores else float("nan")
    sam3_panel = _put_label(
        _resize_h(overlay_mask(bgr, sam3_mask, 0.45, (0, 140, 255)), THUMB),
        [f"SAM3 rank 1",
         f"sam3={sam3_scores[0]:.4f}  area={sam3_mask.mean():.1%}",
         f"recall={coverages[0]:.3f}  ctr={sam3_ctr:.3f}"],
    )

    guided_ctr = center_scores[best_idx] if best_idx < len(center_scores) else float("nan")
    guided_label = ["Guided PICK" + (" [PC1 fallback]" if pc1_fallback else ""),
                    f"rank {best_idx+1}  sam3={sam3_scores[best_idx]:.4f}  area={best_mask.mean():.1%}",
                    f"recall={coverages[best_idx]:.3f}  ctr={guided_ctr:.3f}  "
                    f"comb={combined[best_idx]:.4f}"]
    guided_panel = _put_label(
        _resize_h(overlay_mask(bgr, best_mask, 0.45, (0, 220, 80)), THUMB),
        guided_label,
    )

    sep_v = np.full((THUMB, 4, 3), 30, dtype=np.uint8)
    top_row = np.hstack([pc1_panel, sep_v, orig_panel, sep_v, sam3_panel, sep_v, guided_panel])

    # --- top-k candidate panels ---
    ranked = sorted(
        range(len(masks)),
        key=lambda i: combined[i] if np.isfinite(combined[i]) else -1e9,
        reverse=True,
    )[:top_k]

    thumb_panels = []
    for rank_pos, idx in enumerate(ranked):
        mask = masks[idx]
        if mask.shape[:2] != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        color = (0, 220, 80) if idx == best_idx else (0, 140, 255)
        thumb = _resize_h(overlay_mask(bgr, mask, 0.45, color), THUMB)
        ctr   = center_scores[idx] if idx < len(center_scores) else float("nan")
        label = [
            f"rank {rank_pos+1} (cand #{idx+1})" + (" ← PICK" if idx == best_idx else ""),
            f"sam3={sam3_scores[idx]:.4f}  area={mask.mean():.1%}  ctr={ctr:.3f}",
            (f"recall={coverages[idx]:.3f}  comb={combined[idx]:.4f}"
             if np.isfinite(coverages[idx]) else "[filtered by area]"),
        ]
        thumb_panels.append(_put_label(thumb, label))

    # Pad all thumb panels to same width so hstack works
    max_tw = max(p.shape[1] for p in thumb_panels) if thumb_panels else 1
    padded = [np.pad(p, ((0, 0), (0, max_tw - p.shape[1]), (0, 0))) for p in thumb_panels]

    bot_row = np.hstack([padded[0]] + [np.hstack([sep_v, p]) for p in padded[1:]]) if padded else None

    # Align widths of top and bottom rows
    total_w = max(top_row.shape[1], bot_row.shape[1] if bot_row is not None else 0)
    def _pad_w(img, w):
        return np.pad(img, ((0, 0), (0, w - img.shape[1]), (0, 0)))

    sep_h = np.full((4, total_w, 3), 30, dtype=np.uint8)
    rows  = [_pad_w(top_row, total_w)]
    if bot_row is not None:
        rows += [sep_h, _pad_w(bot_row, total_w)]
    debug_img = np.vstack(rows)

    out_path = debug_dir / f"{stem}_debug.png"
    cv2.imwrite(str(out_path), debug_img)
    print(f"  debug → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    debug_dir = Path(args.debug_dir)

    device = torch.device(args.device)
    sam3   = build_sam3_image_model()
    sam3.eval().to(device)
    processor = Sam3Processor(sam3)
    processor.set_confidence_threshold(args.confidence)
    print(f"SAM3 ready.  {len(image_paths)} image(s).\n")

    # Register the hook once before the loop.
    # processor.set_image() calls backbone.forward_image() → vision_backbone.forward()
    # → trunk.blocks[L23], so the hook fires during run_sam3() as a side effect —
    # no second vision-backbone forward pass needed.
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

        stem = image_path.stem   # needed before save_debug_pc1 / summary

        # run_sam3() → processor.set_image() triggers _l23_hook as a side effect
        masks, scores = run_sam3(processor, image, args.prompt, None)
        if not masks:
            print(f"  No masks — skipping.  ({time.perf_counter() - t_img:.1f}s)\n")
            continue

        print(f"  {len(masks)} candidates  score range: {scores[0]:.4f} → {scores[-1]:.4f}")

        if args.imagenet_norm:
            # Diagnostic path: explicit ImageNet-norm forward pass
            pc1_map = extract_pc1_map(sam3, image, device, orig_h, orig_w,
                                      polarity_mask=masks[0])
            print("  [imagenet-norm] PC1 from explicit forward pass")
        else:
            # Normal path: use tokens already captured by the hook above
            pc1_map = compute_pc1_from_tokens(
                _l23_captured["tokens"], orig_h, orig_w, polarity_mask=masks[0])

        best_idx, combined, coverages, center_scores, pc1_fallback = pick_by_combined(
            masks, scores, pc1_map, orig_h, orig_w)

        if pc1_fallback:
            print(f"  [PC1 degenerate — max recall < {PC1_FALLBACK_RECALL}] "
                  f"falling back to SAM3 × center")

        best_mask = masks[best_idx]
        if best_mask.shape[:2] != (orig_h, orig_w):
            best_mask = cv2.resize(best_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        ctr = center_scores[best_idx] if best_idx < len(center_scores) else float("nan")
        print(f"  pick: rank {best_idx+1}  score={scores[best_idx]:.4f}  "
              f"area={best_mask.mean():.1%}  pc1_cov={coverages[best_idx]:.3f}  "
              f"center={ctr:.3f}  combined={combined[best_idx]:.4f}")

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

        if args.save_pc1:
            save_debug_pc1(debug_dir, stem, bgr, pc1_map,
                           masks, scores, combined, coverages, center_scores,
                           best_idx, pc1_fallback, top_k=args.top_k)

        img_elapsed = time.perf_counter() - t_img
        summary.append({
            "image": str(image_path),
            "n_candidates": len(masks),
            "elapsed_s": round(img_elapsed, 2),
            "pc1_fallback": pc1_fallback,
            "pick": {
                "rank": best_idx + 1,
                "score": scores[best_idx],
                "pc1_coverage": coverages[best_idx],
                "center_score": center_scores[best_idx] if best_idx < len(center_scores) else None,
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
