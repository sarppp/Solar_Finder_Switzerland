#!/usr/bin/env python3
"""Retry LLM detections (Gemini/OpenAI/Ollama) for failed items and merge results.

Usage example:
    python retry_llm.py \
        --detections-json streamlit_site/langnau_im_emmental/langnau_im_emmental_detections.json \
        --model gemini \
        --chunk-size 10 \
        --gemini-delay 3

# Gemini with 3 s spacing
python retry_llm.py \
  --detections-json streamlit_site/langnau_im_emmental/langnau_im_emmental_detections.json \
  --model gemini \
  --gemini-delay 3

# OpenAI retries
python retry_llm.py --detections-json ... --model openai

# Ollama retries with custom host/model
python retry_llm.py --detections-json ... --model ollama \
  --ollama-model llava --ollama-host http://localhost:11434
  
This script will:
1. Read the detections JSON file.
2. Find entries that are missing the selected model output or contain an error payload.
3. Re-run ``detect_solar_panels.py`` only for those image paths (in chunks to avoid long
   commands), respecting model-specific options (e.g., Gemini delay between calls).
4. Merge the refreshed model results back into the main detections JSON while keeping other
   model outputs untouched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


SUPPORTED_MODELS = ["gemini", "openai", "ollama"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detections-json",
        required=True,
        help="Path to the detections JSON file to update.",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="gemini",
        help="Which model output to retry (default: gemini).",
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
        "--gemini-delay",
        type=float,
        default=3.0,
        help="Seconds to wait between Gemini calls (passed to --gemini-delay-between).",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Override Ollama model name when retrying ollama outputs.",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Override Ollama host when retrying ollama outputs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25,
        help="Number of images to process per detect_solar_panels invocation.",
    )
    parser.add_argument(
        "--detect-extra-arg",
        action="append",
        default=[],
        help="Additional argument(s) to forward to detect_solar_panels.py (repeatable).",
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


def find_failed_entries(detections_path: Path, model_key: str) -> list[dict]:
    with detections_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    failed: list[dict] = []
    for entry in data.get("results", []):
        payload = entry.get(model_key)
        if not payload:
            failed.append(entry)
            continue
        if isinstance(payload, dict) and "error" in payload:
            failed.append(entry)

    return failed


def chunked(seq: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_detection_chunk(
    chunk_entries: list[dict],
    detect_script: Path,
    python_bin: str,
    model: str,
    gemini_delay: float,
    extra_args: list[str],
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
        model,
        "--batch-out",
        str(temp_out),
        "--no-viz",
    ]

    if model == "gemini":
        cmd += ["--gemini-delay-between", str(gemini_delay)]
    elif model == "ollama":
        # Optional overrides passed only if provided
        if extra_args:
            pass  # extras handled below to keep ordering consistent

    cmd += extra_args

    print(f"\nRunning {model} on {len(image_paths)} images → {temp_out}")
    result = subprocess.run(cmd, cwd="/app")
    if result.returncode != 0:
        raise RuntimeError(f"detect_solar_panels failed with exit code {result.returncode}")


def merge_results(main_path: Path, temp_path: Path, model_key: str) -> int:
    if not temp_path.exists():
        raise FileNotFoundError(f"Temp detections file not found: {temp_path}")

    with main_path.open("r", encoding="utf-8") as f:
        main = json.load(f)
    with temp_path.open("r", encoding="utf-8") as f:
        temp = json.load(f)

    temp_by_stem: dict[str, dict] = {}
    for entry in temp.get("results", []):
        stem = Path(entry.get("image_path", "")).stem
        payload = entry.get(model_key)
        if stem and payload:
            temp_by_stem[stem] = entry

    updated = 0
    for entry in main.get("results", []):
        stem = Path(entry.get("image_path", "")).stem
        if stem in temp_by_stem:
            entry[model_key] = temp_by_stem[stem][model_key]
            updated += 1

    models = main.get("models") or []
    if model_key not in models:
        models.append(model_key)
        main["models"] = models

    with main_path.open("w", encoding="utf-8") as f:
        json.dump(main, f, indent=2, ensure_ascii=False)

    return updated


def main() -> None:
    args = parse_args()
    detections_path = Path(args.detections_json).resolve()
    detect_script = Path(args.detect_script).resolve()
    model_key = args.model.lower()

    if not detections_path.exists():
        raise FileNotFoundError(f"Detections JSON not found: {detections_path}")
    if not detect_script.exists():
        raise FileNotFoundError(f"detect_solar_panels.py not found: {detect_script}")

    failed_entries = find_failed_entries(detections_path, model_key)
    if not failed_entries:
        print("No missing or failed entries for the selected model. Nothing to do.")
        return

    print(f"Found {len(failed_entries)} entries needing {model_key} retry.")
    if args.dry_run:
        for entry in failed_entries:
            print(Path(entry["image_path"]).name)
        return

    extra_args: list[str] = []
    if args.detect_extra_arg:
        extra_args.extend(args.detect_extra_arg)

    if model_key == "ollama":
        if args.ollama_model:
            extra_args += ["--ollama-model", args.ollama_model]
        if args.ollama_host:
            extra_args += ["--ollama-host", args.ollama_host]

    total_updated = 0
    for idx, chunk in enumerate(chunked(failed_entries, max(1, args.chunk_size)), start=1):
        temp_path = detections_path.with_suffix(
            f".{model_key}_retry_chunk{idx}.json"
        )
        run_detection_chunk(
            chunk,
            detect_script,
            args.python_bin,
            model_key,
            args.gemini_delay,
            extra_args,
            temp_path,
        )
        updated = merge_results(detections_path, temp_path, model_key)
        print(f"Merged chunk {idx}: updated {updated} entries.")
        total_updated += updated
        if not args.keep_temp:
            temp_path.unlink(missing_ok=True)

    print(f"Done. Total entries updated: {total_updated}.")


if __name__ == "__main__":
    main()
