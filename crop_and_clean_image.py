#!/usr/bin/env python3
"""Crop and clean SAM3 processed images to remove colored borders and overlays.

This script takes a SAM3 visualization image (with green/red borders) and its
corresponding mask, then crops the image to the mask boundaries and removes
the colored overlays to produce a clean, normal image.

python crop_and_clean_image.py --original streamlit_site/langnau/outputs2/b2685921_Hinterdorfstrasse_13_50m.png \
    --mask feature_guided_outputs/masks/b2685921_Hinterdorfstrasse_13_50m_guided_convnext_mask.png
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def find_mask_for_image(viz_path: Path, mask_dir: Path) -> Path:
    """Find the corresponding mask file for a visualization."""
    # Extract the base filename without the _sam3_rankX_scoreY.png part
    stem = viz_path.stem
    # Find the position of "_sam3_rank" in the stem
    rank_pos = stem.find("_sam3_rank")
    if rank_pos != -1:
        base_stem = stem[:rank_pos]
    else:
        base_stem = stem
    
    # Look for mask file with similar pattern
    mask_pattern = f"{base_stem}_sam3_rank*_mask.png"
    mask_files = list(mask_dir.glob(mask_pattern))
    
    if not mask_files:
        raise FileNotFoundError(f"No mask found for {viz_path}")
    
    # Return the first (and usually only) match
    return mask_files[0]


def load_original_image(viz_path: Path, original_dir: Path = None) -> np.ndarray:
    """Try to load the original image without SAM3 overlays."""
    
    # Try to find original image in common directories
    if original_dir:
        original_candidates = [
            original_dir / viz_path.name,
            original_dir / f"{viz_path.stem.replace('_sam3_rank1_score0.38', '')}.png",
        ]
        for candidate in original_candidates:
            if candidate.exists():
                return cv2.imread(str(candidate))
    
    # If no original found, we'll work with the viz image and remove overlays
    return cv2.imread(str(viz_path))


def remove_colored_overlays(image: np.ndarray) -> np.ndarray:
    """Remove SAM3 overlays by simply reducing the colored overlay intensity."""
    
    # Work in HSV color space for better color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Detect green overlay areas (SAM3 green is typically hue around 60)
    # But we need to be more conservative to avoid affecting natural green
    green_mask = ((h >= 45) & (h <= 75)) & (s > 50) & (v > 50)
    
    # Detect red contour areas (red can be at hue extremes)
    red_mask = (((h >= 0) & (h <= 10)) | ((h >= 170) & (h <= 180))) & (s > 50) & (v > 50)
    
    # For green areas, drastically reduce saturation to make them less prominent
    s[green_mask] = s[green_mask] * 0.1
    
    # For red areas, also reduce saturation significantly
    s[red_mask] = s[red_mask] * 0.1
    
    # Also reduce the value (brightness) in these areas to make them less visible
    v[green_mask] = v[green_mask] * 0.8
    v[red_mask] = v[red_mask] * 0.8
    
    # Reconstruct the HSV image
    hsv_cleaned = cv2.merge([h, s, v])
    
    # Convert back to BGR
    result = cv2.cvtColor(hsv_cleaned, cv2.COLOR_HSV2BGR)
    
    # Additional pass: directly target very bright green pixels in RGB space
    # This catches any remaining bright green overlays
    b, g, r = cv2.split(result)
    
    # Find pixels where green is much stronger than red and blue
    green_dominant = (g > 150) & (g > r * 1.5) & (g > b * 1.5)
    
    # Reduce green intensity in these areas
    g[green_dominant] = g[green_dominant] * 0.4
    
    # Find pixels where red is dominant (for contours)
    red_dominant = (r > 120) & (r > g * 1.3) & (r > b * 1.3)
    
    # Reduce red intensity in these areas
    r[red_dominant] = r[red_dominant] * 0.4
    
    # Reconstruct RGB image
    result = cv2.merge([b, g, r])
    
    return result


def _ensure_mask_binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def isolate_region(
    original_bgr: np.ndarray,
    mask: np.ndarray,
    background: str,
) -> np.ndarray:
    mask = _ensure_mask_binary(mask)
    if mask.shape[:2] != original_bgr.shape[:2]:
        mask = cv2.resize(mask, (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    keep = mask.astype(bool)
    if background == "transparent":
        bgra = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2BGRA)
        alpha = np.zeros(mask.shape[:2], dtype=np.uint8)
        alpha[keep] = 255
        bgra[:, :, 3] = alpha
        return bgra

    if background == "white":
        bg_color = (255, 255, 255)
    elif background == "black":
        bg_color = (0, 0, 0)
    else:
        raise ValueError("background must be one of: transparent, white, black")

    out = np.empty_like(original_bgr)
    out[:, :] = bg_color
    out[keep] = original_bgr[keep]
    return out


def crop_to_mask_bounds(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop image to the bounding box of the mask."""
    
    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in mask")
    
    # Get the bounding box of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add a small padding (5% of the dimensions)
    padding_x = max(5, int(w * 0.05))
    padding_y = max(5, int(h * 0.05))
    
    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    w = min(image.shape[1] - x, w + 2 * padding_x)
    h = min(image.shape[0] - y, h + 2 * padding_y)
    
    # Crop the image
    cropped = image[y:y+h, x:x+w]
    
    return cropped, (x, y, w, h)


def process_image(
    original_path: Path,
    mask_path: Path,
    output_dir: Path,
    crop: bool,
    padding: int,
    background: str,
) -> Path:
    print(f"Processing original: {original_path}")
    print(f"Using mask: {mask_path}")

    original_bgr = cv2.imread(str(original_path))
    if original_bgr is None:
        raise ValueError(f"Failed to load original image: {original_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to load mask: {mask_path}")
    mask = _ensure_mask_binary(mask)

    isolated = isolate_region(original_bgr, mask, background=background)

    if crop:
        h, w = mask.shape[:2]
        mask_for_crop = mask
        x0, y0 = np.where(mask_for_crop > 0)[1].min(), np.where(mask_for_crop > 0)[0].min()
        x1, y1 = np.where(mask_for_crop > 0)[1].max(), np.where(mask_for_crop > 0)[0].max()
        x0 = max(0, int(x0) - padding)
        y0 = max(0, int(y0) - padding)
        x1 = min(w - 1, int(x1) + padding)
        y1 = min(h - 1, int(y1) + padding)
        isolated = isolated[y0 : y1 + 1, x0 : x1 + 1]

    suffix = "_isolated"
    if crop:
        suffix += "_cropped"
    suffix += f"_{background}"
    output_name = f"{original_path.stem}{suffix}.png"
    output_path = output_dir / output_name

    cv2.imwrite(str(output_path), isolated)
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Isolate SAM3 masked region from original image.")
    parser.add_argument("--original", required=True, help="Path to the original image (no overlay)")
    parser.add_argument("--mask", required=True, help="Path to the SAM3 binary mask image")
    parser.add_argument("--output-dir", default="sam3_outputs/cleaned", help="Directory to save cleaned images")
    parser.add_argument("--crop", action="store_true", help="Crop output to the mask bounding box")
    parser.add_argument("--padding", type=int, default=0, help="Padding (pixels) around bbox when cropping")
    parser.add_argument(
        "--background",
        default="transparent",
        choices=["transparent", "white", "black"],
        help="Background for non-masked area",
    )
    
    args = parser.parse_args()

    original_path = Path(args.original)
    mask_path = Path(args.mask)
    if not original_path.exists():
        raise FileNotFoundError(f"Original image not found: {original_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = process_image(
        original_path=original_path,
        mask_path=mask_path,
        output_dir=output_dir,
        crop=args.crop,
        padding=args.padding,
        background=args.background,
    )
    
    print(f"\n✅ Successfully processed image!")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
