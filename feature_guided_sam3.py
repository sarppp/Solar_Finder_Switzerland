#!/usr/bin/env python3
"""Feature-guided SAM3 segmentation for aerial building images.

Uses DINOv2 cosine similarity and/or ConvNeXt activation maps to identify
the main building region, then feeds that as a geometric (box) prompt to
SAM3 for much better segmentation than text-only prompting.

Pipeline
--------
1. DINOv2 / ConvNeXt extract features from the image
2. Feature maps are thresholded to find the main building region
3. The bounding box of that region becomes a geometric prompt for SAM3
4. SAM3 segments with both text + geometric guidance

Requirements
------------
- torch, torchvision, timm, sam3, opencv-python, pillow, numpy, scipy, scikit-learn

Example usage
-------------
# Single image, DINOv2-guided
python feature_guided_sam3.py streamlit_site/langnau/outputs2/b1290885_Bleicheweg_11_50m.png

# Multiple images, ConvNeXt-guided
python feature_guided_sam3.py streamlit_site/langnau/outputs2/*.png --guide convnext

# Both guides + text prompt
python feature_guided_sam3.py streamlit_site/langnau/outputs2/*.png --guide both --prompt "the main building"

# Compare text-only vs feature-guided
python feature_guided_sam3.py streamlit_site/langnau/outputs2/b1290885_Bleicheweg_11_50m.png --compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from scipy.ndimage import label, binary_fill_holes
from sklearn.decomposition import PCA
from torchvision import transforms

# Monkey-patch pkg_resources for sam3 compatibility
import importlib.resources
import types
pkg_resources = types.ModuleType("pkg_resources")
pkg_resources.resource_filename = lambda package, path: str(
    importlib.resources.files(package).joinpath(path)
)
import sys
sys.modules["pkg_resources"] = pkg_resources

# now safe to import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feature-guided SAM3 segmentation on aerial building images."
    )
    p.add_argument("images", nargs="+", help="Image paths")
    p.add_argument("--prompt", default="the main building",
                   help="Text prompt for SAM3 (default: 'the main building')")
    p.add_argument("--guide", default="both", choices=["dino", "convnext", "both"],
                   help="Which feature extractor to use for guidance")
    p.add_argument("--compare", action="store_true",
                   help="Also run text-only SAM3 and save a side-by-side comparison image")
    p.add_argument("--threshold-pct", type=int, default=70,
                   help="Percentile threshold for building mask extraction (default: 70)")
    p.add_argument("--box-padding", type=float, default=0.05,
                   help="Fractional padding around the detected building box (default: 0.05)")
    p.add_argument("--confidence", type=float, default=0.3,
                   help="SAM3 confidence threshold (default: 0.3)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--viz-dir", default="feature_guided_outputs/viz")
    p.add_argument("--mask-dir", default="feature_guided_outputs/masks")
    p.add_argument("--metadata-out", default="feature_guided_outputs/results.json")
    p.add_argument("--alpha", type=float, default=0.4, help="Overlay alpha")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------
def load_feature_models(guide: str, device: torch.device):
    """Load DINOv2 and/or ConvNeXt models."""
    models = {}
    if guide in ("dino", "both"):
        print("  Loading DINOv2 (vit_base_patch14_dinov2)...")
        m = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True, num_classes=0, features_only=True,
        )
        m.eval().to(device)
        models["dino"] = m

    if guide in ("convnext", "both"):
        print("  Loading ConvNeXt Base...")
        m = timm.create_model("convnext_base", pretrained=True, features_only=True)
        m.eval().to(device)
        models["convnext"] = m

    return models


DINO_TRANSFORM = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def extract_dino_similarity(model, image: Image.Image, device: torch.device) -> np.ndarray:
    """Return a (H_feat, W_feat) cosine-similarity heatmap from DINOv2.

    Queries multiple patches around the image centre and averages the
    similarity maps so the result is robust to small offsets.
    """
    img_t = DINO_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model(img_t)
        feat = feats[-1]  # (1, C, H, W)

    _, C, H, W = feat.shape
    spatial = feat[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    flat = spatial.reshape(-1, C)
    norm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)

    cy, cx = H // 2, W // 2
    offsets = [(0, 0), (-2, -2), (-2, 2), (2, -2), (2, 2),
               (-4, 0), (4, 0), (0, -4), (0, 4)]
    sim_acc = np.zeros((H, W))
    n = 0
    for dy, dx in offsets:
        qy, qx = cy + dy, cx + dx
        if 0 <= qy < H and 0 <= qx < W:
            q = norm[qy * W + qx: qy * W + qx + 1]
            sim_acc += (norm @ q.T).reshape(H, W)
            n += 1
    sim_acc /= max(n, 1)
    return sim_acc  # values in roughly [0, 1]


def extract_convnext_activation(model, image: Image.Image, device: torch.device) -> np.ndarray:
    """Return a (H_feat, W_feat) average-activation heatmap from ConvNeXt."""
    img_t = CNN_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model(img_t)
        feat = feats[-1]  # (1, C, H, W)

    act = feat[0].cpu().numpy()  # (C, H, W)
    avg_act = act.mean(axis=0)   # (H, W)
    return avg_act


# ---------------------------------------------------------------------------
# Building mask & box extraction
# ---------------------------------------------------------------------------
def heatmap_to_building_box(
    heatmap: np.ndarray,
    orig_w: int,
    orig_h: int,
    threshold_pct: int = 70,
    padding: float = 0.05,
) -> Optional[Tuple[float, float, float, float]]:
    """Threshold a feature heatmap and return a normalised [cx, cy, w, h] box.

    Returns None when no building region is found.
    """
    thresh = np.percentile(heatmap, threshold_pct)
    binary = (heatmap >= thresh).astype(np.uint8)
    binary = binary_fill_holes(binary).astype(np.uint8)

    labeled_arr, n_comp = label(binary)
    if n_comp == 0:
        return None

    sizes = [np.sum(labeled_arr == i) for i in range(1, n_comp + 1)]
    largest = int(np.argmax(sizes)) + 1
    mask = (labeled_arr == largest).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None

    feat_h, feat_w = heatmap.shape
    sx = orig_w / feat_w
    sy = orig_h / feat_h

    x1 = float(xs.min()) * sx
    y1 = float(ys.min()) * sy
    x2 = float(xs.max() + 1) * sx
    y2 = float(ys.max() + 1) * sy

    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0.0, x1 - bw * padding)
    y1 = max(0.0, y1 - bh * padding)
    x2 = min(float(orig_w), x2 + bw * padding)
    y2 = min(float(orig_h), y2 + bh * padding)

    # Normalise to [0, 1]
    cx = ((x1 + x2) / 2) / orig_w
    cy = ((y1 + y2) / 2) / orig_h
    w = (x2 - x1) / orig_w
    h = (y2 - y1) / orig_h

    return (cx, cy, w, h)


# ---------------------------------------------------------------------------
# SAM3 helpers
# ---------------------------------------------------------------------------
def run_sam3_text_only(
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
) -> Tuple[List[np.ndarray], List[float]]:
    """Run SAM3 with text prompt only. Returns (masks, scores)."""
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt, state)

    masks_t = state.get("masks")
    scores_t = state.get("scores")
    if masks_t is None or scores_t is None or len(masks_t) == 0:
        return [], []

    order = torch.argsort(scores_t, descending=True)
    masks_out, scores_out = [], []
    for idx in order:
        m = masks_t[idx]
        if m.ndim == 3:
            m = m.squeeze(0)
        masks_out.append(m.cpu().numpy().astype(np.uint8))
        scores_out.append(float(scores_t[idx].item()))
    return masks_out, scores_out


def run_sam3_guided(
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
    box_norm: Tuple[float, float, float, float],
) -> Tuple[List[np.ndarray], List[float]]:
    """Run SAM3 with text prompt + geometric box prompt."""
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt, state)
    state = processor.add_geometric_prompt(
        box=list(box_norm), label=True, state=state,
    )

    masks_t = state.get("masks")
    scores_t = state.get("scores")
    if masks_t is None or scores_t is None or len(masks_t) == 0:
        return [], []

    order = torch.argsort(scores_t, descending=True)
    masks_out, scores_out = [], []
    for idx in order:
        m = masks_t[idx]
        if m.ndim == 3:
            m = m.squeeze(0)
        masks_out.append(m.cpu().numpy().astype(np.uint8))
        scores_out.append(float(scores_t[idx].item()))
    return masks_out, scores_out


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def overlay_mask(
    bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Green overlay + red contour on a BGR image."""
    if mask.shape[:2] != bgr.shape[:2]:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    binary = mask.astype(np.uint8)
    if binary.max() == 0:
        return bgr.copy()

    overlay = bgr.copy()
    overlay[binary.astype(bool)] = color
    blended = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)

    contours, _ = cv2.findContours(
        binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(blended, contours, -1, (0, 0, 255), thickness=2)
    return blended


def save_comparison(
    image_path: Path,
    results: dict,
    viz_dir: Path,
    alpha: float,
) -> str:
    """Save a side-by-side comparison image of all methods."""
    bgr = cv2.imread(str(image_path))
    n = 1 + len(results)  # original + methods
    fig_w = 400
    canvas = np.zeros((bgr.shape[0], fig_w * n, 3), dtype=np.uint8)

    # Resize helper
    def fit(img):
        h, w = img.shape[:2]
        scale = fig_w / w
        return cv2.resize(img, (fig_w, int(h * scale)))

    col = 0
    orig_resized = fit(bgr)
    canvas[:orig_resized.shape[0], :fig_w] = orig_resized
    cv2.putText(canvas, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    col += 1

    for method_name, data in results.items():
        if not data["masks"]:
            col += 1
            continue
        mask = data["masks"][0]
        score = data["scores"][0]
        vis = overlay_mask(bgr, mask, alpha)
        vis_resized = fit(vis)
        x_off = col * fig_w
        canvas[:vis_resized.shape[0], x_off:x_off + fig_w] = vis_resized
        label_text = f"{method_name} ({score:.2f})"
        cv2.putText(canvas, label_text, (x_off + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        col += 1

    # Trim unused columns
    canvas = canvas[:, :col * fig_w]

    out_path = viz_dir / f"{image_path.stem}_comparison.png"
    cv2.imwrite(str(out_path), canvas)
    return str(out_path)


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
            # Handle absolute paths by extracting directory and pattern
            pattern_path = Path(pattern)
            if pattern_path.is_absolute():
                # For absolute paths, use the parent directory for glob
                parent_dir = pattern_path.parent
                glob_pattern = pattern_path.name
                matches = sorted(parent_dir.glob(glob_pattern))
            else:
                # For relative patterns, use current directory
                matches = sorted(Path().glob(pattern))
            image_paths.extend(m.resolve() for m in matches if m.is_file())
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise SystemExit("No valid images found.")

    viz_dir = Path(args.viz_dir)
    mask_dir = Path(args.mask_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_out).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Load feature models
    print("Loading feature models...")
    feat_models = load_feature_models(args.guide, device)

    # Load SAM3
    print("Loading SAM3...")
    sam3_model = build_sam3_image_model()
    sam3_model.to(device)
    processor = Sam3Processor(sam3_model)
    processor.set_confidence_threshold(args.confidence)
    print("All models loaded!\n")

    summary = {
        "prompt": args.prompt,
        "guide": args.guide,
        "threshold_pct": args.threshold_pct,
        "box_padding": args.box_padding,
        "confidence": args.confidence,
        "results": [],
    }

    for image_path in image_paths:
        print(f"Processing: {image_path}")
        image = Image.open(str(image_path)).convert("RGB")
        orig_w, orig_h = image.size
        img_result: dict = {"image": str(image_path), "methods": {}}

        # --- Feature-guided methods ---
        guided_results: dict = {}

        if "dino" in feat_models:
            print("  Extracting DINOv2 cosine similarity...")
            dino_heatmap = extract_dino_similarity(feat_models["dino"], image, device)
            dino_box = heatmap_to_building_box(
                dino_heatmap, orig_w, orig_h,
                threshold_pct=args.threshold_pct, padding=args.box_padding,
            )
            if dino_box:
                print(f"  DINOv2 box (norm): cx={dino_box[0]:.3f} cy={dino_box[1]:.3f} w={dino_box[2]:.3f} h={dino_box[3]:.3f}")
                print("  Running SAM3 + DINOv2 box...")
                masks, scores = run_sam3_guided(processor, image, args.prompt, dino_box)
                guided_results["dino"] = {"masks": masks, "scores": scores, "box": list(dino_box)}
                if scores:
                    print(f"    Best score: {scores[0]:.3f}, area: {masks[0].sum()/(orig_h*orig_w):.1%}")
            else:
                print("  DINOv2: no building region found")
                guided_results["dino"] = {"masks": [], "scores": [], "box": None}

        if "convnext" in feat_models:
            print("  Extracting ConvNeXt activations...")
            cnn_heatmap = extract_convnext_activation(feat_models["convnext"], image, device)
            cnn_box = heatmap_to_building_box(
                cnn_heatmap, orig_w, orig_h,
                threshold_pct=args.threshold_pct, padding=args.box_padding,
            )
            if cnn_box:
                print(f"  ConvNeXt box (norm): cx={cnn_box[0]:.3f} cy={cnn_box[1]:.3f} w={cnn_box[2]:.3f} h={cnn_box[3]:.3f}")
                print("  Running SAM3 + ConvNeXt box...")
                masks, scores = run_sam3_guided(processor, image, args.prompt, cnn_box)
                guided_results["convnext"] = {"masks": masks, "scores": scores, "box": list(cnn_box)}
                if scores:
                    print(f"    Best score: {scores[0]:.3f}, area: {masks[0].sum()/(orig_h*orig_w):.1%}")
            else:
                print("  ConvNeXt: no building region found")
                guided_results["convnext"] = {"masks": [], "scores": [], "box": None}

        # --- Text-only baseline (always run if --compare) ---
        text_result = {"masks": [], "scores": []}
        if args.compare:
            print(f"  Running SAM3 text-only: '{args.prompt}'...")
            masks, scores = run_sam3_text_only(processor, image, args.prompt)
            text_result = {"masks": masks, "scores": scores}
            if scores:
                print(f"    Best score: {scores[0]:.3f}, area: {masks[0].sum()/(orig_h*orig_w):.1%}")
            else:
                print("    No masks returned")

        # --- Save outputs ---
        bgr = cv2.imread(str(image_path))

        # Find the best guided result
        best_method = None
        best_score = -1.0
        best_mask = None
        for method_name, data in guided_results.items():
            if data["scores"] and data["scores"][0] > best_score:
                best_score = data["scores"][0]
                best_mask = data["masks"][0]
                best_method = method_name

        if best_mask is not None:
            # Save best guided viz
            vis = overlay_mask(bgr, best_mask, args.alpha)
            viz_name = f"{image_path.stem}_guided_{best_method}_score{best_score:.2f}.png"
            cv2.imwrite(str(viz_dir / viz_name), vis)
            print(f"  Saved viz: {viz_dir / viz_name}")

            # Save mask
            mask_name = f"{image_path.stem}_guided_{best_method}_mask.png"
            mask_full = best_mask
            if mask_full.shape[:2] != (orig_h, orig_w):
                mask_full = cv2.resize(mask_full, (orig_w, orig_h),
                                       interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(mask_dir / mask_name), mask_full * 255)
            print(f"  Saved mask: {mask_dir / mask_name}")

        # Save all guided results
        for method_name, data in guided_results.items():
            for i, (mask, score) in enumerate(zip(data["masks"], data["scores"])):
                viz_name = f"{image_path.stem}_{method_name}_rank{i+1}_score{score:.2f}.png"
                vis = overlay_mask(bgr, mask, args.alpha)
                cv2.imwrite(str(viz_dir / viz_name), vis)

        # Save comparison if requested
        if args.compare:
            all_methods = {}
            if text_result["masks"]:
                all_methods["text-only"] = text_result
            for method_name, data in guided_results.items():
                all_methods[method_name] = data
            if all_methods:
                comp_path = save_comparison(image_path, all_methods, viz_dir, args.alpha)
                print(f"  Saved comparison: {comp_path}")

        # Record for JSON
        img_result["methods"] = {
            "text_only": {
                "scores": text_result["scores"],
            },
        }
        for method_name, data in guided_results.items():
            img_result["methods"][method_name] = {
                "scores": data["scores"],
                "box": data.get("box"),
            }
        img_result["best_method"] = best_method
        img_result["best_score"] = best_score
        summary["results"].append(img_result)

        # Print summary for this image
        print(f"  --- Summary for {image_path.name} ---")
        if args.compare and text_result["scores"]:
            print(f"    Text-only:  score={text_result['scores'][0]:.3f}")
        for method_name, data in guided_results.items():
            if data["scores"]:
                print(f"    {method_name:10s}:  score={data['scores'][0]:.3f}")
        if best_method:
            print(f"    BEST: {best_method} (score={best_score:.3f})")
        print()

    # Save JSON summary
    with open(args.metadata_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {args.metadata_out}")


if __name__ == "__main__":
    main()
