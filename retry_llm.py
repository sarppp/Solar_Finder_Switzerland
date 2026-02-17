#!/usr/bin/env python3
"""Retry Gemini detections only for failed items and merge the results.

Usage example:
    python retry_gemini.py \
        --detections-json streamlit_site/langnau_im_emmental/langnau_im_emmental_detections.json \
        --chunk-size 10 \
        --delay 3

This script will:
1. Read the detections JSON file.
2. Find entries that are missing a "gemini" field or contain a Gemini error payload.
3. Re-run ``detect_solar_panels.py`` only for those image paths (in chunks to avoid long
   commands), respecting the configured delay between requests.
4. Merge the new Gemini results back into the main detections JSON while keeping other
   model outputs untouched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detections-json",
        required=True,
        help="Path to the detections JSON file to update.",
    )
    parser.add_argument(
        "--detect-script",
        default="/app/detect_solar_panels.py",
        help="Path to detect_solar_panels.py (defaults to /app/detect_solar_panels.py).",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter to use when invoking the detection script.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to wait between Gemini calls (passed to --gemini-delay-between).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25,
        help="Number of images to process per detect_solar_panels invocation.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary retry JSON files (for debugging).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the failing image paths without running detections.",
    )
    return parser.parse_args()


def find_failed_entries(detections_path: Path) -> list[dict]:
    with detections_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    failed: list[dict] = []
    for entry in data.get("results", []):
        gem = entry.get("gemini")
        if not gem:
            failed.append(entry)
            continue
        if isinstance(gem, dict) and "error" in gem:
            failed.append(entry)

    return failed


def chunked(seq: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_detection_chunk(
    chunk_entries: list[dict],
    detect_script: Path,
    python_bin: str,
    delay: float,
    temp_out: Path,
) -> None:
    image_paths = [entry.get("image_path") for entry in chunk_entries]
    missing = [p for p in image_paths if not p or not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Some image paths are missing: " + ", ".join(str(m) for m in missing)
        )

    cmd: list[str] = [
        python_bin,
        str(detect_script),
        *[str(Path(p)) for p in image_paths],
        "--models",
        "gemini",
        "--batch-out",
        str(temp_out),
        "--gemini-delay-between",
        str(delay),
        "--no-viz",
    ]
    print(f"\nRunning Gemini on {len(image_paths)} images → {temp_out}")
    result = subprocess.run(cmd, cwd="/app")
    if result.returncode != 0:
        raise RuntimeError(f"detect_solar_panels failed with exit code {result.returncode}")


def merge_results(main_path: Path, temp_path: Path) -> int:
    if not temp_path.exists():
        raise FileNotFoundError(f"Temp detections file not found: {temp_path}")

    with main_path.open("r", encoding="utf-8") as f:
        main = json.load(f)
    with temp_path.open("r", encoding="utf-8") as f:
        temp = json.load(f)

    temp_by_stem: dict[str, dict] = {}
    for entry in temp.get("results", []):
        stem = Path(entry.get("image_path", "")).stem
        if stem and entry.get("gemini"):
            temp_by_stem[stem] = entry

    updated = 0
    for entry in main.get("results", []):
        stem = Path(entry.get("image_path", "")).stem
        if stem in temp_by_stem:
            entry["gemini"] = temp_by_stem[stem]["gemini"]
            updated += 1

    models = main.get("models") or []
    if "gemini" not in models:
        models.append("gemini")
        main["models"] = models

    with main_path.open("w", encoding="utf-8") as f:
        json.dump(main, f, indent=2, ensure_ascii=False)

    return updated


def main() -> None:
    args = parse_args()
    detections_path = Path(args.detections_json).resolve()
    detect_script = Path(args.detect_script).resolve()

    if not detections_path.exists():
        raise FileNotFoundError(f"Detections JSON not found: {detections_path}")
    if not detect_script.exists():
        raise FileNotFoundError(f"detect_solar_panels.py not found: {detect_script}")

    failed_entries = find_failed_entries(detections_path)
    if not failed_entries:
        print("No missing or failed Gemini entries. Nothing to do.")
        return

    print(f"Found {len(failed_entries)} entries needing Gemini retry.")
    if args.dry_run:
        for entry in failed_entries:
            print(Path(entry["image_path"]).name)
        return

    total_updated = 0
    for idx, chunk in enumerate(chunked(failed_entries, max(1, args.chunk_size)), start=1):
        temp_path = detections_path.with_suffix(
            f".gemini_retry_chunk{idx}.json"
        )
        run_detection_chunk(chunk, detect_script, args.python_bin, args.delay, temp_path)
        updated = merge_results(detections_path, temp_path)
        print(f"Merged chunk {idx}: updated {updated} entries.")
        total_updated += updated
        if not args.keep_temp:
            temp_path.unlink(missing_ok=True)

    print(f"Done. Total entries updated: {total_updated}.")


if __name__ == "__main__":
    main()
