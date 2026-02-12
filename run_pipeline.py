#!/usr/bin/env python3
"""
Unified Solar Panel Detection Pipeline
=======================================
 
Runs the full pipeline in one go:
 
  1. region_building_groups.py  – discover buildings in a region/canton
  2. get_building_screenshot.py – take aerial screenshots
  3. feature_guided_sam3.py     – segment buildings (DINOv2/ConvNeXt + SAM3)
  4. crop_and_clean_image.py    – crop & isolate each building
  5. detect_solar_panels.py     – detect solar panels
 
All intermediate paths are auto-derived from --region / --output-dir so you
only need to specify the search area and the parameters you care about.
 
Examples
--------
 
# Full pipeline for a municipality
python3 run_pipeline.py --region "Langnau im Emmental" --residential-only
 
# Canton-level (large!)
python3 run_pipeline.py --region "Bern" --max-results 500
 
# Skip stages you already ran (resume)
python3 run_pipeline.py --region "Langnau im Emmental" --start-stage 3
 
# Dry-run: just print what would be executed
python3 run_pipeline.py --region "Langnau im Emmental" --dry-run
 
# Override specific stage parameters
python3 run_pipeline.py --region "Payerne" \\
  --min-roof-area 200 \\
  --screenshot-size-m 40 \\
  --detection-models yolo,gemini \\
  --sam-guide dino

python3 run_pipeline.py --region "Langnau im Emmental" \
  --start-stage 1 \
  --stop-stage 5 \
  --limit None \
  --min-roof-area 300.0 \
  --plant-radius 30.0 \
  --filter-mode all \
  --pv-only-plants \
  --neighbor-within-m 200.0 \
  --max-results 2000 \
  --residential-only \
  --residential-gkat-codes 1020,1030,1040 \
  --dedupe-by-esid \
  --dedupe-by-egrid \
  --gwr-tolerance-m 50.0 \
  --tile-size-m 500.0 \
  --min-tile-size-m 125.0 \
  --sleep-s 0.05 \
  --screenshot-size-m 50.0 \
  --screenshot-width 800 \
  --screenshot-height 800 \
  --reuse-screenshot \
  --sam-guide dino \
  --sam-prompt "the main building" \
  --sam-threshold-pct 70 \
  --sam-confidence 0.3 \
  --sam-device cuda \
  --sam-keep-best-only \
  --crop-background transparent \
  --crop-padding 0 \
  --detection-models yolo \
  --yolo-conf 0.25 \
  --gemini-delay-between 2

# Estavayer-le-Lac

python3 run_pipeline.py --region "Payerne" \
  --start-stage 1 \
  --stop-stage 5 \
  --min-roof-area 300.0 \
  --plant-radius 30.0 \
  --filter-mode no_plants \
  --pv-only-plants \
  --neighbor-within-m 200.0 \
  --max-results 2000 \
  --residential-only \
  --residential-gkat-codes 1020,1030,1040 \
  --dedupe-by-esid \
  --dedupe-by-egrid \
  --gwr-tolerance-m 50.0 \
  --tile-size-m 500.0 \
  --min-tile-size-m 125.0 \
  --sleep-s 0.05 \
  --screenshot-size-m 50.0 \
  --screenshot-width 800 \
  --screenshot-height 800 \
  --reuse-screenshot \
  --sam-guide dino \
  --sam-prompt "the main building" \
  --sam-threshold-pct 70 \
  --sam-confidence 0.3 \
  --sam-device cuda \
  --sam-keep-best-only \
  --crop-background transparent \
  --crop-padding 0 \
  --detection-models yolo \
  --yolo-conf 0.25 


python run_pipeline.py \
  --region "Payerne" \
  --start-stage 5 --stop-stage 5 \
  --detection-models gemini \
  --append-model gemini \
  --gemini-delay-between 2.0
"""
 
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
 
 
def _sanitize(name: str) -> str:
    """Turn a region name into a safe directory/file stem."""
    s = re.sub(r"<[^>]+>", "", name)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "region"
 
 
def _merge_detection_results(existing_json: str, new_json: str, model_key: str) -> None:
    """Merge results from *new_json* into *existing_json*, adding only the
    new model's key to each matching result entry (matched by image filename).

    This allows appending e.g. Gemini results to an existing YOLO-only JSON
    without re-running YOLO.
    """
    with open(existing_json, "r", encoding="utf-8") as f:
        existing = json.load(f)
    with open(new_json, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    # Index new results by filename stem
    new_by_stem: dict[str, dict] = {}
    for r in new_data.get("results", []):
        stem = Path(r.get("image_path", "")).stem
        if stem:
            new_by_stem[stem] = r

    merged_count = 0
    for r in existing.get("results", []):
        stem = Path(r.get("image_path", "")).stem
        new_r = new_by_stem.get(stem)
        if new_r and model_key in new_r:
            r[model_key] = new_r[model_key]
            merged_count += 1

    # Update models list
    models = existing.get("models", [])
    if model_key not in models:
        models.append(model_key)
        existing["models"] = models

    with open(existing_json, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"  Merged {merged_count} '{model_key}' results into {existing_json}")


def _flatten_cleaned_pngs(src_dir: str, dst_dir: str) -> int:
    """Re-save transparent PNGs with a white background so YOLO only sees the roof.

    Cleaned images from SAM3 have an alpha channel: the rooftop is opaque and
    everything else is transparent.  However the RGB data behind the transparent
    pixels still contains the original aerial scene (roads, neighbouring
    buildings, etc.).  cv2.imread (used by YOLO) ignores the alpha channel, so
    YOLO ends up detecting solar panels on those neighbouring buildings.

    This function composites each image onto a white background and writes the
    result (3-channel BGR, no alpha) to *dst_dir* with the same filename.
    Returns the number of images processed.
    """
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for src_path in sorted(glob.glob(os.path.join(src_dir, "*.png"))):
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3].astype(np.float32) / 255.0
            bgr = img[:, :, :3].astype(np.float32)
            white = np.full_like(bgr, 255.0)
            flat = (bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None]))
            img = flat.astype(np.uint8)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        cv2.imwrite(dst_path, img)
        count += 1
    return count


def _run(cmd: list[str], label: str, dry_run: bool) -> int:
    """Run a subprocess, streaming output. Returns exit code."""
    cmd_str = " \\\n  ".join(cmd)
    print(f"\n{'='*70}")
    print(f"  STAGE: {label}")
    print(f"{'='*70}")
    print(f"$ {cmd_str}\n")
 
    if dry_run:
        print("  [DRY RUN – skipped]")
        return 0
 
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/app")
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"\n  [{status}] {label}  ({elapsed:.1f}s)")
    return proc.returncode
 
 
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified Solar Panel Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
 
    # ── Global / pipeline control ──────────────────────────────────────
    g = p.add_argument_group("Pipeline control")
    g.add_argument("--region", required=False,
                   help="Region / municipality name to search (e.g. 'Langnau im Emmental')")
    g.add_argument("--canton", required=False,
                   help="Canton name to search (e.g. 'Bern'). Mutually exclusive with --region.")
    g.add_argument("--output-dir", default=None,
                   help="Root output directory (default: /app/streamlit_site/<region_slug>)")
    g.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3, 4, 5],
                   help="Resume from this stage (1-5, default: 1)")
    g.add_argument("--stop-stage", type=int, default=5, choices=[1, 2, 3, 4, 5],
                   help="Stop after this stage (1-5, default: 5)")
    g.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    g.add_argument("--limit", type=int, default=None,
                   help="Limit number of buildings to process in stages 2-5")
 
    # ── Stage 1: region_building_groups ────────────────────────────────
    s1 = p.add_argument_group("Stage 1 – Building discovery (region_building_groups.py)")
    s1.add_argument("--min-roof-area", type=float, default=300.0,
                    help="Minimum roof area in m² (default: 300)")
    s1.add_argument("--plant-radius", type=float, default=30.0,
                    help="Radius to search for nearby power plants (default: 30m)")
    s1.add_argument("--filter-mode", default="all", choices=["all", "no_pv", "pv", "no_plants"],
                    help="Filter buildings by plant presence (default: all)")
    s1.add_argument("--pv-only-plants", action="store_true", default=True,
                    help="Only count photovoltaic plants (default: True)")
    s1.add_argument("--no-pv-only-plants", action="store_true",
                    help="Count all plant types, not just PV")
    s1.add_argument("--neighbor-within-m", type=float, default=200.0,
                    help="Only keep buildings with a neighbor within this distance (default: 200m, 0=disable)")
    s1.add_argument("--max-results", type=int, default=2000,
                    help="Maximum number of results to keep (default: 2000)")
    s1.add_argument("--residential-only", action="store_true",
                    help="Only keep residential buildings (via GWR lookup)")
    s1.add_argument("--residential-gkat-codes", default="1020,1030,1040",
                    help="GKAT codes considered residential (default: 1020,1030,1040)")
    s1.add_argument("--dedupe-by-esid", action="store_true", default=True,
                    help="De-duplicate by ESID (default: True)")
    s1.add_argument("--no-dedupe-by-esid", action="store_true",
                    help="Disable ESID de-duplication")
    s1.add_argument("--dedupe-by-egrid", action="store_true", default=True,
                    help="De-duplicate by EGRID (default: True)")
    s1.add_argument("--no-dedupe-by-egrid", action="store_true",
                    help="Disable EGRID de-duplication")
    s1.add_argument("--gwr-tolerance-m", type=float, default=50.0,
                    help="GWR lookup tolerance (default: 50m)")
    s1.add_argument("--tile-size-m", type=float, default=500.0)
    s1.add_argument("--min-tile-size-m", type=float, default=125.0)
    s1.add_argument("--sleep-s", type=float, default=0.05)
 
    # ── Stage 2: get_building_screenshot ───────────────────────────────
    s2 = p.add_argument_group("Stage 2 – Screenshots (get_building_screenshot.py)")
    s2.add_argument("--screenshot-size-m", type=float, default=50.0,
                    help="Screenshot coverage in meters (default: 50)")
    s2.add_argument("--screenshot-width", type=int, default=800)
    s2.add_argument("--screenshot-height", type=int, default=800)
    s2.add_argument("--reuse-screenshot", action="store_true", default=True,
                    help="Skip screenshots that already exist (default: True)")
    s2.add_argument("--no-reuse-screenshot", action="store_true",
                    help="Re-download all screenshots")
 
    # ── Stage 3: feature_guided_sam3 ───────────────────────────────────
    s3 = p.add_argument_group("Stage 3 – SAM3 segmentation (feature_guided_sam3.py)")
    s3.add_argument("--sam-guide", default="dino", choices=["dino", "convnext", "both"],
                    help="Feature extractor for SAM3 guidance (default: dino)")
    s3.add_argument("--sam-prompt", default="the main building",
                    help="Text prompt for SAM3")
    s3.add_argument("--sam-threshold-pct", type=int, default=70)
    s3.add_argument("--sam-confidence", type=float, default=0.3)
    s3.add_argument("--sam-device", default="cuda", choices=["cuda", "cpu"])
    s3.add_argument("--sam-keep-best-only", action="store_true", default=True,
                    help="After SAM3, keep only the best guided viz/mask and delete ranked extras (default: True)")


    # ── Stage 4: crop_and_clean_image ──────────────────────────────────
    s4 = p.add_argument_group("Stage 4 – Crop & clean (crop_and_clean_image.py)")
    s4.add_argument("--crop-background", default="transparent",
                    choices=["transparent", "white", "black"],
                    help="Background for non-building area (default: transparent)")
    s4.add_argument("--crop-padding", type=int, default=0,
                    help="Padding around crop in pixels (default: 0)")
 
    # ── Stage 5: detect_solar_panels ───────────────────────────────────
    s5 = p.add_argument_group("Stage 5 – Solar panel detection (detect_solar_panels.py)")
    s5.add_argument(
        "--detection-input-dir",
        default=None,
        help="Directory of input images for stage 5 (default: <output-dir>/cleaned)",
    )
    s5.add_argument("--detection-models", default="yolo",
                    help="Detection models: yolo,gemini,openai,ollama or 'all' (default: yolo)")
    s5.add_argument("--yolo-conf", type=float, default=0.25)
    s5.add_argument("--gemini-delay-between", type=float, default=0.0)
    s5.add_argument("--detection-no-viz", action="store_true",
                    help="Skip YOLO visualization output")
    s5.add_argument("--append-model", default=None,
                    help="Append a single model (e.g. 'gemini') to existing detection JSON "
                         "without re-running other models. Merges results into --batch-out.")
 
    return p
 
 
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
 
    # ── Derive paths ───────────────────────────────────────────────────
    if not args.region and not args.canton:
        sys.exit("Provide either --region or --canton")
    if args.region and args.canton:
        sys.exit("Use only one of --region or --canton (not both)")

    region_name = args.region or args.canton
    slug = _sanitize(region_name)
    base_dir = args.output_dir or f"/app/streamlit_site/{slug}"
    base_dir = os.path.abspath(base_dir)
 
    buildings_json   = os.path.join(base_dir, f"{slug}_buildings.json")
    screenshots_dir  = os.path.join(base_dir, "screenshots")
    screenshots_json = os.path.join(base_dir, f"{slug}_screenshots.json")
    sam_viz_dir      = os.path.join(base_dir, "sam3_viz")
    sam_mask_dir     = os.path.join(base_dir, "sam3_masks")
    sam_results_json = os.path.join(base_dir, f"{slug}_sam3_results.json")
    cleaned_dir      = os.path.join(base_dir, "cleaned")
    detection_json   = os.path.join(base_dir, f"{slug}_detections.json")
    detection_viz    = os.path.join(base_dir, "detection_viz")
    label_cache      = os.path.join(base_dir, f"{slug}_labels_cache.json")
    state_file       = f"/tmp/{slug}_pipeline_state.json"

    detection_input_dir = os.path.abspath(args.detection_input_dir) if args.detection_input_dir else cleaned_dir
 
    os.makedirs(base_dir, exist_ok=True)
 
    # Resolve boolean flag overrides
    pv_only = args.pv_only_plants and not args.no_pv_only_plants
    dedupe_esid = args.dedupe_by_esid and not args.no_dedupe_by_esid
    dedupe_egrid = args.dedupe_by_egrid and not args.no_dedupe_by_egrid
    reuse_ss = args.reuse_screenshot and not args.no_reuse_screenshot
 
    print(f"\n{'#'*70}")
    print(f"  Solar Panel Detection Pipeline")
    print(f"  Region:     {region_name}")
    print(f"  Output dir: {base_dir}")
    print(f"  Stages:     {args.start_stage} → {args.stop_stage}")
    if args.limit:
        print(f"  Limit:      {args.limit} buildings")
    if args.dry_run:
        print(f"  MODE:       DRY RUN")
    print(f"{'#'*70}")
 
    # ── Stage 1: Building discovery ────────────────────────────────────
    if args.start_stage <= 1 <= args.stop_stage:
        cmd = [
            sys.executable, "/app/region_building_groups.py",
        ]
        if args.region:
            cmd += ["--region", args.region]
        else:
            cmd += ["--canton", args.canton]
        cmd += [
            "--min-roof-area", str(args.min_roof_area),
            "--plant-radius", str(args.plant_radius),
            "--filter-mode", args.filter_mode,
            "--neighbor-within-m", str(args.neighbor_within_m),
            "--label-bbox-m", "30",
            "--label-cache", label_cache,
            "--tile-size-m", str(args.tile_size_m),
            "--min-tile-size-m", str(args.min_tile_size_m),
            "--restrict-to-region-label",
            "--sleep-s", str(args.sleep_s),
            "--progress-every-pct", "10",
            "--max-results", str(args.max_results),
            "--include-solar-metrics",
            "--include-gwr-attrs",
            "--gwr-tolerance-m", str(args.gwr_tolerance_m),
            "--gwr-match-mode", "point_match_egid",
            "--label-mode", "gwr_prefer",
            "--include-raw-results",
            "--state-file", state_file,
            "--out", buildings_json,
        ]
        if pv_only:
            cmd.append("--pv-only-plants")
        if args.residential_only:
            cmd.append("--residential-only")
        if args.residential_gkat_codes:
            cmd += ["--residential-gkat-codes", args.residential_gkat_codes]
        if dedupe_esid:
            cmd.append("--dedupe-by-esid")
        if dedupe_egrid:
            cmd.append("--dedupe-by-egrid")
 
        rc = _run(cmd, "1/5  Building discovery", args.dry_run)
        if rc != 0 and not args.dry_run:
            sys.exit(rc)
 
    # ── Stage 2: Screenshots ──────────────────────────────────────────
    if args.start_stage <= 2 <= args.stop_stage:
        cmd = [
            sys.executable, "/app/get_building_screenshot.py",
            "--input-json", buildings_json,
            "--screenshots-dir", screenshots_dir,
            "--screenshot-size-m", str(args.screenshot_size_m),
            "--screenshot-width", str(args.screenshot_width),
            "--screenshot-height", str(args.screenshot_height),
        ]
        if reuse_ss:
            cmd.append("--reuse-screenshot")
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
 
        # Capture stdout to save the screenshots JSON
        print(f"\n{'='*70}")
        print(f"  STAGE: 2/5  Screenshots")
        print(f"{'='*70}")
        cmd_str = " \\\n  ".join(cmd)
        print(f"$ {cmd_str}\n")
 
        if not args.dry_run:
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd="/app")
            elapsed = time.time() - t0
            # Print stderr (progress info)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
            # Save stdout (JSON) to file
            if proc.stdout.strip():
                with open(screenshots_json, "w") as f:
                    f.write(proc.stdout)
                print(f"  Screenshots JSON saved to: {screenshots_json}")
            status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            print(f"\n  [{status}] 2/5  Screenshots  ({elapsed:.1f}s)")
            if proc.returncode != 0:
                sys.exit(proc.returncode)
        else:
            print("  [DRY RUN – skipped]")
 
    # ── Stage 3: SAM3 segmentation ────────────────────────────────────
    if args.start_stage <= 3 <= args.stop_stage:
        # Collect screenshot image paths
        screenshot_pattern = os.path.join(screenshots_dir, "*.png")

        cmd = [
            sys.executable, "/app/feature_guided_sam3.py",
            screenshot_pattern,
            "--guide", args.sam_guide,
            "--prompt", args.sam_prompt,
            "--threshold-pct", str(args.sam_threshold_pct),
            "--confidence", str(args.sam_confidence),
            "--device", args.sam_device,
            "--viz-dir", sam_viz_dir,
            "--mask-dir", sam_mask_dir,
            "--metadata-out", sam_results_json,
        ]
 
        rc = _run(cmd, "3/5  SAM3 segmentation", args.dry_run)
        if rc != 0 and not args.dry_run:
            sys.exit(rc)

        # Cleanup SAM outputs to keep best only (remove ranked extras)
        if args.sam_keep_best_only and not args.dry_run:
            removed = 0
            patterns = [
                os.path.join(sam_viz_dir, "*_rank*.png"),
                os.path.join(sam_mask_dir, "*_rank*.png"),
                os.path.join(sam_mask_dir, "*_rank*.json"),
                os.path.join(sam_viz_dir, "*_rank*.json"),
            ]
            for pat in patterns:
                for f in glob.glob(pat):
                    try:
                        os.remove(f)
                        removed += 1
                    except Exception:
                        pass
            if removed:
                print(f"  SAM cleanup: removed {removed} ranked files, keeping best-guided outputs")
 
    # ── Stage 4: Crop & clean ─────────────────────────────────────────
    if args.start_stage <= 4 <= args.stop_stage:
        print(f"\n{'='*70}")
        print(f"  STAGE: 4/5  Crop & clean")
        print(f"{'='*70}")
 
        if not args.dry_run:
            os.makedirs(cleaned_dir, exist_ok=True)
            # Find all mask files and match to originals
            mask_files = sorted(glob.glob(os.path.join(sam_mask_dir, "*_mask.png")))
            if not mask_files:
                print("  WARNING: No mask files found, skipping stage 4")
            else:
                t0 = time.time()
                processed = 0
                failed = 0
                for mask_path in mask_files:
                    mask_name = os.path.basename(mask_path)
                    # Extract original filename: remove _guided_<method>_mask.png
                    # Pattern: <original_stem>_guided_<method>_mask.png
                    m = re.match(r"^(.+?)_guided_\w+_mask\.png$", mask_name)
                    if not m:
                        continue
                    original_stem = m.group(1)
                    original_path = os.path.join(screenshots_dir, f"{original_stem}.png")
                    if not os.path.exists(original_path):
                        print(f"  WARNING: Original not found for mask {mask_name}: {original_path}")
                        failed += 1
                        continue
 
                    cmd = [
                        sys.executable, "/app/crop_and_clean_image.py",
                        "--original", original_path,
                        "--mask", mask_path,
                        "--output-dir", cleaned_dir,
                        "--crop",
                        "--padding", str(args.crop_padding),
                        "--background", args.crop_background,
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True, cwd="/app")
                    if proc.returncode == 0:
                        processed += 1
                    else:
                        failed += 1
                        print(f"  FAILED: {original_stem}: {proc.stderr.strip()[:200]}")
 
                    if args.limit and processed >= args.limit:
                        break
 
                elapsed = time.time() - t0
                print(f"\n  [OK] 4/5  Crop & clean: {processed} processed, {failed} failed  ({elapsed:.1f}s)")
        else:
            print("  [DRY RUN – skipped]")
 
    # ── Stage 5: Solar panel detection ────────────────────────────────
    if args.start_stage <= 5 <= args.stop_stage:
        # Pre-process: flatten alpha channel so YOLO sees only the roof
        # (transparent pixels in cleaned PNGs still contain the original
        # aerial scene which would cause false detections on neighbours).
        flat_dir = os.path.join(base_dir, "cleaned_flat")
        if not args.dry_run:
            print(f"\n{'='*70}")
            print(f"  PRE-STAGE 5: Flatten cleaned PNGs (alpha → white bg)")
            print(f"{'='*70}")
            n_flat = _flatten_cleaned_pngs(detection_input_dir, flat_dir)
            print(f"  Flattened {n_flat} images → {flat_dir}")
        else:
            print(f"  [DRY RUN] Would flatten PNGs from {detection_input_dir} → {flat_dir}")

        cleaned_pattern = os.path.join(flat_dir, "*.png")
        cleaned_files = sorted(glob.glob(cleaned_pattern))
        if args.limit is not None:
            cleaned_files = cleaned_files[: int(args.limit)]

        if not cleaned_files:
            print(f"  WARNING: No input PNGs found for detection; skipping detection: {flat_dir}")
        elif args.append_model:
            # ── Append-model mode: run only the new model, then merge ──
            append_model = args.append_model.strip().lower()
            tmp_json = detection_json + f".{append_model}_tmp.json"
            cmd = [
                sys.executable, "/app/detect_solar_panels.py",
                *cleaned_files,
                "--models", append_model,
                "--batch-out", tmp_json,
                "--no-viz",
            ]
            if append_model == "gemini" and args.gemini_delay_between > 0:
                cmd += ["--gemini-delay-between", str(args.gemini_delay_between)]

            rc = _run(cmd, f"5/5  Append model: {append_model}", args.dry_run)
            if rc != 0 and not args.dry_run:
                sys.exit(rc)

            if not args.dry_run and os.path.exists(detection_json) and os.path.exists(tmp_json):
                _merge_detection_results(detection_json, tmp_json, append_model)
                os.remove(tmp_json)
            elif not args.dry_run and not os.path.exists(detection_json):
                print(f"  WARNING: No existing {detection_json} to merge into; "
                      f"keeping {tmp_json} as-is.")
                os.rename(tmp_json, detection_json)
        else:
            cmd = [
                sys.executable, "/app/detect_solar_panels.py",
                *cleaned_files,
                "--models", args.detection_models,
                "--batch-out", detection_json,
                "--yolo-conf", str(args.yolo_conf),
            ]
            if not args.detection_no_viz:
                cmd += ["--viz-dir", detection_viz]
            else:
                cmd.append("--no-viz")
            if args.gemini_delay_between > 0:
                cmd += ["--gemini-delay-between", str(args.gemini_delay_between)]

            rc = _run(cmd, "5/5  Solar panel detection", args.dry_run)
            if rc != 0 and not args.dry_run:
                sys.exit(rc)
 
    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  Pipeline complete!")
    print(f"{'#'*70}")
    print(f"  Output directory:  {base_dir}")
    print(f"  Buildings JSON:    {buildings_json}")
    print(f"  Screenshots dir:   {screenshots_dir}")
    print(f"  SAM3 masks dir:    {sam_mask_dir}")
    print(f"  Cleaned images:    {cleaned_dir}")
    print(f"  Detection results: {detection_json}")
    print()
 
    # Print quick stats if files exist
    if os.path.exists(buildings_json) and not args.dry_run:
        try:
            with open(buildings_json) as f:
                data = json.load(f)
            n = len(data.get("results", []))
            print(f"  Buildings found:   {n}")
        except Exception:
            pass
 
    if os.path.exists(detection_json) and not args.dry_run:
        try:
            with open(detection_json) as f:
                data = json.load(f)
            results = data.get("results", [])
            n_total = len(results)
            n_solar = sum(
                1 for r in results
                if any(
                    r.get(m, {}).get("has_solar_panel") is True
                    for m in ["yolo", "gemini", "openai", "ollama"]
                    if isinstance(r.get(m), dict)
                )
            )
            print(f"  Images analyzed:   {n_total}")
            print(f"  Solar panels found: {n_solar}")
        except Exception:
            pass
 
 
if __name__ == "__main__":
    main()
 