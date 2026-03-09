#!/usr/bin/env python3
"""Isolate the building from an aerial screenshot using a SAM3 binary mask.

Takes the original screenshot and the mask produced by feature_guided_sam3.py,
applies the mask (transparent/white/black background), and optionally crops to
the mask bounding box.

Usage:
  python crop_and_clean_image.py --original screenshot.png --mask mask.png
  python crop_and_clean_image.py --original screenshot.png --mask mask.png --crop
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def _ensure_mask_binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def isolate_region(original_bgr: np.ndarray, mask: np.ndarray, background: str) -> np.ndarray:
    mask = _ensure_mask_binary(mask)
    if mask.shape[:2] != original_bgr.shape[:2]:
        mask = cv2.resize(mask, (original_bgr.shape[1], original_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    keep = mask.astype(bool)
    if background == "transparent":
        bgra = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2BGRA)
        alpha = np.zeros(mask.shape[:2], dtype=np.uint8)
        alpha[keep] = 255
        bgra[:, :, 3] = alpha
        return bgra
    bg_color = {"white": (255, 255, 255), "black": (0, 0, 0)}[background]
    out = np.full_like(original_bgr, bg_color)
    out[keep] = original_bgr[keep]
    return out


def process_image(original_path: Path, mask_path: Path, output_dir: Path,
                  crop: bool, padding: int, background: str) -> Path:
    original_bgr = cv2.imread(str(original_path))
    if original_bgr is None:
        raise ValueError(f"Failed to load: {original_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to load mask: {mask_path}")
    mask = _ensure_mask_binary(mask)

    isolated = isolate_region(original_bgr, mask, background=background)

    if crop:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Mask is empty — nothing to crop to")
        h, w = mask.shape[:2]
        x0 = max(0, int(xs.min()) - padding)
        y0 = max(0, int(ys.min()) - padding)
        x1 = min(w - 1, int(xs.max()) + padding)
        y1 = min(h - 1, int(ys.max()) + padding)
        isolated = isolated[y0:y1 + 1, x0:x1 + 1]

    suffix = "_isolated_cropped" if crop else "_isolated"
    output_path = output_dir / f"{original_path.stem}{suffix}_{background}.png"
    cv2.imwrite(str(output_path), isolated)
    return output_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True, help="Original screenshot (no overlay)")
    p.add_argument("--mask",     required=True, help="Binary mask from feature_guided_sam3.py")
    p.add_argument("--output-dir", default="sam3_outputs/cleaned")
    p.add_argument("--crop", action="store_true", help="Crop to mask bounding box")
    p.add_argument("--padding", type=int, default=0, help="Padding in pixels around crop")
    p.add_argument("--background", default="transparent",
                   choices=["transparent", "white", "black"])
    args = p.parse_args()

    original_path = Path(args.original)
    mask_path     = Path(args.mask)
    if not original_path.exists():
        raise FileNotFoundError(original_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out = process_image(original_path, mask_path, output_dir,
                        args.crop, args.padding, args.background)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
