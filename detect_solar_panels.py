#!/usr/bin/env python3
"""
Solar Panel Detection using 4 ML models with intelligent retry logic:
1. YOLO (trained for solar panel segmentation) - Local, Fast
2. OpenAI GPT-4o-mini Vision - API, Paid
3. Google Gemini Flash - API, Free tier (with smart rate limiting)
4. Ollama - Local, Free (works with ANY vision model)

Features:
- Exponential backoff with jitter for all API calls
- Automatic retry on rate limits (429 errors)
- Configurable retry strategies per model
- Works with any Ollama vision model (llama3.2-vision, llava, minicpm-v, etc.)

Examples:

A) From JSON with YOLO only
python3 detect_solar_panels.py \
  --input-json screenshots_run.json \
  --limit 10 \
  --models yolo \
  --no-viz

B) Multiple image files directly
python3 detect_solar_panels.py langnau_pv_residential_350/*.png \
  --models yolo \
  --out-dir . \
  --viz-dir yolo_images \
  --batch-out solar_detection_langnau_pv_residential_350_batch.json

C) Single image with all models
python3 detect_solar_panels.py outputs/Hinterdorfstrasse_13_3550_Langnau_im_Emmental_50m.png \
  --models all

D) Use Gemini with aggressive rate limiting (avoid 429 errors)
python3 detect_solar_panels.py outputs/*.png \
  --models gemini \
  --gemini-delay-between 15 \
  --gemini-max-retries 5 \
  --gemini-initial-delay 10

E) Use Ollama with ANY vision model (local, free)
python3 detect_solar_panels.py outputs/*.png \
  --models ollama \
  --ollama-model llava:13b \
  --ollama-host http://localhost:11434

# Try different Ollama models:
--ollama-model llama3.2-vision  # Meta's latest
--ollama-model llava            # Popular 7B model
--ollama-model llava:13b        # Larger variant
--ollama-model minicpm-v        # Efficient Chinese model
--ollama-model bakllava         # Another option

F) Combine multiple models with smart retry
python3 detect_solar_panels.py outputs/*.png \
  --models yolo,ollama,gemini \
  --gemini-delay-between 20 \
  --gemini-max-retries 5

Requirements:
- pip install ollama  # For Ollama support
- Ollama server running with any vision model:
  ollama pull llama3.2-vision  # or llava, minicpm-v, etc.

Retry Logic:
- Exponential backoff: 1s → 2s → 4s → 8s (with random jitter)
- Jitter prevents thundering herd problems
- Automatic detection of rate limit vs connection errors
- Configurable max attempts and delays per model

python3 detect_solar_panels.py streamlit_site/langnau/outputs2/b3597197_B_raustrasse_71l_50m.png --models gemini
"""

import os
import sys
import json
from pathlib import Path
import base64
import argparse
import cv2
import numpy as np
import time
import random
import shutil
import urllib.request
import urllib.error
from functools import wraps
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory for bundled assets (e.g., YOLO weights)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_YOLO_WEIGHTS_NAME = "solar_panel_yolov8s-seg.pt"
DEFAULT_YOLO_WEIGHTS_PATH = BASE_DIR / DEFAULT_YOLO_WEIGHTS_NAME
YOLO_SOLAR_WEIGHTS_URL = "https://huggingface.co/finloop/yolov8s-seg-solar-panels/resolve/main/best.pt"

# ============================================================================
# MODEL CONFIGURATION - Change model names and settings here
# ============================================================================

MODEL_CONFIG = {
    # YOLO Configuration
    "yolo": {
        "model_path": str(DEFAULT_YOLO_WEIGHTS_PATH),
        "default_conf": 0.25,
        "default_imgsz": 640,
    },
    
    # OpenAI Configuration
    "openai": {
        "model_name": "gpt-4o-mini",
        "max_tokens": 300,
    },
    
    # Gemini Configuration
    "gemini": {
        "model_name": "gemini-3-flash-preview",
        "default_max_retries": 3,
        "default_initial_delay": 2.0,
        "default_delay_between": 0.0,
    },
    
    # Ollama Configuration
    "ollama": {
        "default_model": "llama3.2-vision",
        "default_host": "http://localhost:11434",
        "default_max_retries": 3,
        "default_initial_delay": 0.0,
        # Supported models (examples - works with ANY vision model):
        # "llama3.2-vision", "llava", "llava:13b", "llava:34b",
        # "minicpm-v", "bakllava", "cogvlm", "moondream"
    },
}

# ============================================================================

# Model 1: YOLO for solar panel detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not installed - YOLO detection disabled")

# Model 2: OpenAI Vision
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  openai not installed - OpenAI detection disabled")

# Model 3: Gemini
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-genai not installed - Gemini detection disabled")

# Model 4: Ollama (local)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  ollama not installed - Ollama detection disabled")


def retry_with_backoff(max_attempts=3, initial_delay=1.0, max_delay=60.0, exponential_base=2, jitter=True, retry_on_exceptions=None):
    """
    Decorator for retry logic with exponential backoff and jitter.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff (2 = double each time)
        jitter: Add random jitter to prevent thundering herd
        retry_on_exceptions: Tuple of exception types to retry on, or None for all
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    
                    # Check if we should retry this exception
                    if retry_on_exceptions and not isinstance(e, retry_on_exceptions):
                        raise
                    
                    if attempt >= max_attempts:
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (exponential_base ** (attempt - 1)), max_delay)
                    
                    # Add jitter (random 0-100% of delay)
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    error_type = type(e).__name__
                    print(f"   ⚠️  {error_type}: {str(e)[:100]}")
                    print(f"   🔄 Retry {attempt}/{max_attempts} in {delay:.1f}s...")
                    time.sleep(delay)
            
            raise Exception(f"Max retries ({max_attempts}) exceeded")
        return wrapper
    return decorator


def encode_image_base64(image_path):
    """Encode image to base64 for API calls"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def ensure_yolo_weights(model_path):
    """Ensure YOLO weights exist locally; download default weights if missing."""
    path = Path(model_path).expanduser()
    if path.exists():
        return str(path)

    default_path = DEFAULT_YOLO_WEIGHTS_PATH.resolve()
    current_path = path.resolve()

    if current_path != default_path:
        raise FileNotFoundError(
            f"YOLO model weights not found at {path}. Provide a valid --yolo-model path or place the file there."
        )

    print("   ⬇️  Downloading solar-panel YOLOv8 weights (first run)...")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with urllib.request.urlopen(YOLO_SOLAR_WEIGHTS_URL, timeout=90) as resp:
            data = resp.read()
        with open(tmp_path, "wb") as f:
            f.write(data)
        shutil.move(tmp_path, path)
        size_mb = len(data) / (1024 * 1024)
        print(f"   ✅ Downloaded {path.name} ({size_mb:.1f} MB)")
    except Exception as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(
            "Failed to download default YOLO weights. Download manually from "
            f"{YOLO_SOLAR_WEIGHTS_URL} and place it at {path}."
        ) from exc

    return str(path)


def _ensure_parent_dir(path: str | None) -> None:
    if not path:
        return
    parent = Path(str(path)).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def _bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0


def _nms_xyxy(dets, iou_thresh=0.5):
    dets = sorted(dets, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    kept = []
    for d in dets:
        bb = d.get("bbox")
        if not bb or len(bb) != 4:
            continue
        suppress = False
        for k in kept:
            if _bbox_iou_xyxy(bb, k["bbox"]) >= float(iou_thresh):
                suppress = True
                break
        if not suppress:
            kept.append(d)
    return kept


def _tile_boxes(image_path, model, *, conf, imgsz, solar_cls_ids, tile_size, overlap):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]
    ts = int(tile_size)
    ov = int(overlap)
    step = max(1, ts - ov)

    detections = []
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            x1 = min(w, x0 + ts)
            y1 = min(h, y0 + ts)
            x0c = max(0, x1 - ts)
            y0c = max(0, y1 - ts)
            tile = img[y0c:y1, x0c:x1]
            if tile.size == 0:
                continue
            results = model(tile, conf=conf, imgsz=imgsz)
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    bconf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if solar_cls_ids and cls not in solar_cls_ids:
                        continue
                    xyxy = box.xyxy[0].tolist()
                    bx1, by1, bx2, by2 = [float(v) for v in xyxy]
                    bx1 += float(x0c)
                    bx2 += float(x0c)
                    by1 += float(y0c)
                    by2 += float(y0c)
                    class_name = model.names.get(cls, str(cls))
                    detections.append({
                        "class": class_name,
                        "confidence": bconf,
                        "bbox": [bx1, by1, bx2, by2],
                        "class_id": cls,
                    })
    return detections, (w, h)


def _filter_detections(detections, *, image_wh=None, min_area_frac=0.0, center_bias=0.0):
    if not detections:
        return detections
    if image_wh is None:
        return detections

    w, h = image_wh
    img_area = float(w) * float(h)
    cx_img = float(w) / 2.0
    cy_img = float(h) / 2.0
    diag = (float(w) ** 2 + float(h) ** 2) ** 0.5

    out = []
    for d in detections:
        bb = d.get("bbox")
        if not bb or len(bb) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bb]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = bw * bh
        if img_area > 0 and (area / img_area) < float(min_area_frac):
            continue

        if float(center_bias) > 0 and diag > 0:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dist = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5
            norm_dist = dist / diag
            score_mult = max(0.0, 1.0 - float(center_bias) * norm_dist)
            d = dict(d)
            d["confidence"] = float(d.get("confidence", 0.0)) * score_mult
        out.append(d)
    return out


def _filter_by_roi(detections, *, image_wh, roi_xyxy_norm=None, center_window_frac=0.0):
    if not detections:
        return detections
    w, h = image_wh

    roi = None
    if roi_xyxy_norm is not None:
        try:
            x1n, y1n, x2n, y2n = [float(v) for v in roi_xyxy_norm]
            x1 = x1n * float(w)
            x2 = x2n * float(w)
            y1 = y1n * float(h)
            y2 = y2n * float(h)
            roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        except Exception:
            roi = None

    if roi is None and float(center_window_frac) > 0:
        frac = float(center_window_frac)
        frac = max(0.0, min(1.0, frac))
        if frac > 0:
            cx = float(w) / 2.0
            cy = float(h) / 2.0
            hw = (float(w) * frac) / 2.0
            hh = (float(h) * frac) / 2.0
            roi = (cx - hw, cy - hh, cx + hw, cy + hh)

    if roi is None:
        return detections

    rx1, ry1, rx2, ry2 = roi
    kept = []
    for d in detections:
        bb = d.get("bbox")
        if not bb or len(bb) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bb]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            kept.append(d)
    return kept


def detect_yolo(
    image_path,
    model_path=os.path.join(BASE_DIR, "solar_panel_yolov8s-seg.pt"),
    *,
    conf=0.25,
    imgsz=640,
    tile_size=0,
    tile_overlap=0,
    nms_iou=0.5,
    min_area_frac=0.0,
    center_bias=0.0,
    center_window_frac=0.0,
    roi_xyxy_norm=None,
):
    """
    YOLO detection for solar panels (trained model)
    
    Args:
        image_path: Path to image
        model_path: Path to trained YOLO solar panel model
    
    Returns:
        dict with has_solar_panel, confidence, detections
    """
    if not YOLO_AVAILABLE:
        return {"error": "YOLO not available", "has_solar_panel": None, "confidence": 0}

    model_path = ensure_yolo_weights(model_path)

    try:
        model = YOLO(model_path)

        solar_cls_ids = []
        try:
            for k, v in getattr(model, "names", {}).items():
                name = str(v).strip().lower().replace("-", " ")
                if "solar" in name and "panel" in name:
                    solar_cls_ids.append(int(k))
        except Exception:
            solar_cls_ids = []

        if int(tile_size) > 0:
            detections, image_wh = _tile_boxes(
                image_path,
                model,
                conf=float(conf),
                imgsz=int(imgsz),
                solar_cls_ids=solar_cls_ids,
                tile_size=int(tile_size),
                overlap=int(tile_overlap),
            )
            detections = _filter_detections(
                detections,
                image_wh=image_wh,
                min_area_frac=float(min_area_frac),
                center_bias=float(center_bias),
            )
            detections = _filter_by_roi(
                detections,
                image_wh=image_wh,
                roi_xyxy_norm=roi_xyxy_norm,
                center_window_frac=float(center_window_frac),
            )
            detections = _nms_xyxy(detections, iou_thresh=float(nms_iou))
        else:
            results = model(image_path, conf=float(conf), imgsz=int(imgsz))

            img = cv2.imread(image_path)
            image_wh = (int(img.shape[1]), int(img.shape[0])) if img is not None else None

            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    bconf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names.get(cls, str(cls))
                    if solar_cls_ids and cls not in solar_cls_ids:
                        continue
                    detections.append({
                        "class": class_name,
                        "confidence": bconf,
                        "bbox": [float(v) for v in box.xyxy[0].tolist()],
                        "class_id": cls,
                    })

            detections = _filter_detections(
                detections,
                image_wh=image_wh,
                min_area_frac=float(min_area_frac),
                center_bias=float(center_bias),
            )
            if image_wh is not None:
                detections = _filter_by_roi(
                    detections,
                    image_wh=image_wh,
                    roi_xyxy_norm=roi_xyxy_norm,
                    center_window_frac=float(center_window_frac),
                )
            detections = _nms_xyxy(detections, iou_thresh=float(nms_iou))

        max_confidence = 0.0
        for d in detections:
            max_confidence = max(max_confidence, float(d.get("confidence", 0.0)))

        has_panels = len(detections) > 0

        payload = {
            "model": "YOLO",
            "model_path": model_path,
            "has_solar_panel": has_panels,
            "confidence": max_confidence,
            "num_detections": len(detections),
            "detections": detections,
        }

        if not solar_cls_ids:
            payload["warning"] = (
                "Loaded YOLO weights do not contain a 'solar panel' class name. "
                "If this model is a COCO model, it will label objects like 'car'. "
                "Use a YOLO model trained/finetuned for solar panels (ideally a -seg model for masks)."
            )

        return payload
    except Exception as e:
        return {"error": str(e), "has_solar_panel": None, "confidence": 0}


def visualize_yolo_solar_panels(image_path, model_path="solar_panel_yolov8s-seg.pt", output_path=None, conf=0.25, alpha=0.4):
    if not YOLO_AVAILABLE:
        return {"error": "YOLO not available", "output_path": None}

    model_path = ensure_yolo_weights(model_path)

    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Failed to read image: {image_path}", "output_path": None}

    # If caller already computed detections (e.g., tiling + filtering), draw those.
    detections = None
    try:
        detections = getattr(visualize_yolo_solar_panels, "_detections", None)
    except Exception:
        detections = None

    if isinstance(detections, list) and detections:
        green = (0, 255, 0)
        for d in detections:
            bb = d.get("bbox")
            if not bb or len(bb) != 4:
                continue
            x1, y1, x2, y2 = [int(float(v)) for v in bb]
            cv2.rectangle(img, (x1, y1), (x2, y2), green, 2)

        stem = Path(image_path).stem
        if output_path is None:
            output_path = f"annotated_{stem}.png"
        cv2.imwrite(output_path, img)
        return {"output_path": output_path}

    model = YOLO(model_path)
    results = model(image_path, conf=conf)

    solar_cls_ids = []
    for k, v in getattr(model, "names", {}).items():
        name = str(v).strip().lower().replace("-", " ")
        if "solar" in name and "panel" in name:
            solar_cls_ids.append(int(k))

    if not solar_cls_ids:
        stem = Path(image_path).stem
        if output_path is None:
            output_path = f"annotated_{stem}.png"
        cv2.imwrite(output_path, img)
        return {
            "warning": (
                "No 'solar panel' class found in model.names. Saved original image without annotations. "
                "Provide solar-panel-trained weights (preferably segmentation weights) to get green mask/box."
            ),
            "output_path": output_path,
        }

    green = (0, 255, 0)

    for result in results:
        if result.boxes is None:
            continue

        masks = None
        if getattr(result, "masks", None) is not None and getattr(result.masks, "data", None) is not None:
            masks = result.masks.data

        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            if cls not in solar_cls_ids:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cv2.rectangle(img, (x1, y1), (x2, y2), green, 2)

            if masks is not None and i < len(masks):
                mask = masks[i].detach().cpu().numpy()
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask > 0.5
                overlay = img.copy()
                overlay[mask_bool] = green
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    stem = Path(image_path).stem
    if output_path is None:
        output_path = f"annotated_{stem}.png"
    cv2.imwrite(output_path, img)
    return {"output_path": output_path}


def detect_openai(image_path, api_key=None, model_name=None):
    """
    OpenAI Vision detection
    
    Args:
        image_path: Path to image
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        model_name: Model name (default from MODEL_CONFIG)
    """
    if not OPENAI_AVAILABLE:
        return {"error": "OpenAI not available", "has_solar_panel": None, "confidence": 0}
    
    model = model_name or MODEL_CONFIG["openai"]["model_name"]
    max_tokens = MODEL_CONFIG["openai"]["max_tokens"]
    
    try:
        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        # Encode image
        base64_image = encode_image_base64(image_path)
        
        # Call API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this aerial/satellite image of a building. 
                            
Are there solar panels visible on the roof? 

Respond in JSON format:
{
  "has_solar_panel": true/false,
  "confidence": 0.0-1.0,
  "explanation": "brief explanation of what you see"
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens
        )
        
        # Parse response
        text = response.choices[0].message.content
        
        # Try to extract JSON
        try:
            # Find JSON in response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
            else:
                result = {"raw_text": text}
        except:
            result = {"raw_text": text}
        
        return {
            "model": f"OpenAI {model}",
            "has_solar_panel": result.get("has_solar_panel"),
            "confidence": result.get("confidence", 0),
            "explanation": result.get("explanation", result.get("raw_text", "")),
            "raw_response": text
        }
    
    except Exception as e:
        return {"error": str(e), "has_solar_panel": None, "confidence": 0}


def detect_gemini(image_path, api_key=None, max_retries=3, initial_delay=10.0):
    """
    Google Gemini Flash detection with intelligent retry logic
    
    Args:
        image_path: Path to image
        api_key: Google API key (or set GOOGLE_API_KEY env var)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds for exponential backoff
    """
    if not GEMINI_AVAILABLE:
        return {"error": "Gemini not available", "has_solar_panel": None, "confidence": 0}
    
    from PIL import Image
    import re

    model = MODEL_CONFIG["gemini"]["model_name"]
    
    @retry_with_backoff(
        max_attempts=max_retries,
        initial_delay=initial_delay,
        max_delay=120.0,
        exponential_base=2,
        jitter=True
    )
    def _call_gemini():
        client = genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        img = Image.open(image_path)
        
        response = client.models.generate_content(
            model=model,
            contents=[
                """Analyze this aerial/satellite image of a building.

Are there solar panels visible on the roof?

Respond in JSON format:
{
  "has_solar_panel": true/false,
  "confidence": 0.0-1.0,
  "explanation": "brief explanation of what you see"
}""",
                img
            ]
        )
        return response.text
    
    try:
        text = _call_gemini()
        
        # Try to extract JSON
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
            else:
                result = {"raw_text": text}
        except:
            result = {"raw_text": text}
        
        return {
            "model": f"Gemini {model}",
            "has_solar_panel": result.get("has_solar_panel"),
            "confidence": result.get("confidence", 0),
            "explanation": result.get("explanation", result.get("raw_text", "")),
            "raw_response": text
        }
    
    except Exception as e:
        error_str = str(e)
        is_rate_limited = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower()
        
        return {
            "error": error_str,
            "has_solar_panel": None,
            "confidence": 0,
            "rate_limited": is_rate_limited
        }


def detect_ollama(image_path, model="llama3.2-vision", host="http://localhost:11434", max_retries=3, initial_delay=2.0):
    """
    Ollama local model detection - works with ANY vision or multimodal model
    
    Supports: llama3.2-vision, llava, llava:13b, llava:34b, minicpm-v, bakllava, 
              cogvlm, moondream, and any other Ollama model with vision capabilities
    
    Args:
        image_path: Path to image
        model: Any Ollama model name (no restrictions)
        host: Ollama server URL
        max_retries: Maximum retry attempts for connection errors
        initial_delay: Initial delay for exponential backoff
    """
    if not OLLAMA_AVAILABLE:
        return {"error": "Ollama not available", "has_solar_panel": None, "confidence": 0}
    
    from PIL import Image
    import io
    
    prompt = """Analyze this aerial/satellite image of a building roof.

Are there solar panels visible on the roof?

Respond in JSON format:
{
  "has_solar_panel": true/false,
  "confidence": 0.0-1.0,
  "explanation": "brief explanation of what you see"
}"""
    
    @retry_with_backoff(
        max_attempts=max_retries,
        initial_delay=initial_delay,
        max_delay=30.0,
        exponential_base=2,
        jitter=True
    )
    def _call_ollama():
        # Load and encode image
        img = Image.open(image_path)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Call Ollama API - works with any model that supports images
        client = ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_bytes]
            }]
        )
        
        return response.get('message', {}).get('content', '')
    
    try:
        text = _call_ollama()
        
        # Try to extract JSON
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
            else:
                result = {"raw_text": text}
        except:
            result = {"raw_text": text}
        
        return {
            "model": f"Ollama {model}",
            "has_solar_panel": result.get("has_solar_panel"),
            "confidence": result.get("confidence", 0),
            "explanation": result.get("explanation", result.get("raw_text", "")),
            "raw_response": text
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "has_solar_panel": None,
            "confidence": 0,
            "model": f"Ollama {model}"
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", default=None)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", default="all")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--per-image", action="store_true")
    parser.add_argument("--batch-out", default=None)
    parser.add_argument("--viz-dir", default=None)

    parser.add_argument("--yolo-model", default=MODEL_CONFIG["yolo"]["model_path"])
    parser.add_argument("--yolo-conf", type=float, default=MODEL_CONFIG["yolo"]["default_conf"])
    parser.add_argument("--yolo-imgsz", type=int, default=MODEL_CONFIG["yolo"]["default_imgsz"])
    parser.add_argument("--yolo-tile", type=int, default=0)
    parser.add_argument("--yolo-tile-overlap", type=int, default=0)
    parser.add_argument("--yolo-nms-iou", type=float, default=0.5)
    parser.add_argument("--yolo-min-area-frac", type=float, default=0.0)
    parser.add_argument("--yolo-center-bias", type=float, default=0.0)
    parser.add_argument("--yolo-center-window", type=float, default=0.0)
    parser.add_argument("--yolo-roi", default=None)
    
    parser.add_argument("--gemini-max-retries", type=int, default=MODEL_CONFIG["gemini"]["default_max_retries"], help="Max retries for Gemini rate limits")
    parser.add_argument("--gemini-initial-delay", type=float, default=MODEL_CONFIG["gemini"]["default_initial_delay"], help="Initial retry delay for Gemini (seconds)")
    parser.add_argument("--gemini-delay-between", type=float, default=MODEL_CONFIG["gemini"]["default_delay_between"], help="Delay between each Gemini call (seconds)")
    
    parser.add_argument("--ollama-model", default=MODEL_CONFIG["ollama"]["default_model"], help="Ollama model name (e.g., llama3.2-vision, llava)")
    parser.add_argument("--ollama-host", default=MODEL_CONFIG["ollama"]["default_host"], help="Ollama server URL")
    
    args = parser.parse_args()

    yolo_roi = None
    if args.yolo_roi:
        try:
            parts = [p.strip() for p in str(args.yolo_roi).split(",")]
            if len(parts) == 4:
                yolo_roi = [float(p) for p in parts]
        except Exception:
            yolo_roi = None

    if not args.images and not args.input_json:
        raise SystemExit("Provide IMAGE_PATH(s) or --input-json")
    if args.images and args.input_json:
        raise SystemExit("Use either IMAGE_PATH(s) OR --input-json (not both)")

    selected = [m.strip().lower() for m in str(args.models).split(",") if m.strip()]
    if not selected:
        selected = ["all"]
    if "all" in selected:
        selected = ["yolo", "openai", "gemini"]
    
    # Track last Gemini call time for rate limiting
    last_gemini_call = 0.0

    def _iter_paths_from_input_json(p: str):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise SystemExit(f"JSON file does not contain a 'results' list: {p}")
        for r in results:
            if not isinstance(r, dict):
                continue
            ss = r.get("screenshot")
            if not isinstance(ss, dict):
                continue
            sp = ss.get("screenshot")
            if not sp:
                continue
            yield {"input": r, "image_path": str(sp)}

    items = []
    if args.images:
        for p in args.images:
            items.append({"input": None, "image_path": str(p)})
    else:
        for it in _iter_paths_from_input_json(str(args.input_json)):
            items.append(it)

    if args.limit is not None:
        items = items[: int(args.limit)]

    out = {
        "models": selected,
        "num_items": len(items),
        "results": [],
    }

    out_dir: Path | None = Path(str(args.out_dir)).resolve() if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    viz_dir: Path | None = Path(str(args.viz_dir)).resolve() if args.viz_dir else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    for it in items:
        image_path = it["image_path"]
        if not os.path.exists(image_path):
            out["results"].append({
                "image_path": image_path,
                "error": f"Image not found: {image_path}",
                "input": it.get("input"),
            })
            continue

        print(f"\n🔍 Analyzing image: {image_path}")
        results = {"image_path": image_path}
        if it.get("input") is not None:
            results["input"] = it.get("input")

        if "yolo" in selected:
            print("\n1️⃣  YOLO Detection...")
            yolo_result = detect_yolo(
                image_path,
                model_path=str(args.yolo_model),
                conf=float(args.yolo_conf),
                imgsz=int(args.yolo_imgsz),
                tile_size=int(args.yolo_tile),
                tile_overlap=int(args.yolo_tile_overlap),
                nms_iou=float(args.yolo_nms_iou),
                min_area_frac=float(args.yolo_min_area_frac),
                center_bias=float(args.yolo_center_bias),
                center_window_frac=float(args.yolo_center_window),
                roi_xyxy_norm=yolo_roi,
            )
            results["yolo"] = yolo_result
            if yolo_result.get("has_solar_panel") is not None:
                print(f"   Has solar panel: {yolo_result['has_solar_panel']}")
                print(f"   Confidence: {yolo_result.get('confidence', 0):.2f}")
                print(f"   Detections: {yolo_result.get('num_detections', 0)}")
            else:
                print(f"   ❌ {yolo_result.get('error', 'Failed')}")

            if not bool(args.no_viz):
                # Pass filtered detections into the visualizer without changing its signature.
                setattr(visualize_yolo_solar_panels, "_detections", yolo_result.get("detections", []))

                viz_output_path = None
                if viz_dir is not None:
                    viz_output_path = str(viz_dir / f"annotated_{Path(str(image_path)).stem}.png")
                viz_result = visualize_yolo_solar_panels(
                    image_path,
                    model_path=str(args.yolo_model),
                    output_path=viz_output_path,
                    conf=float(args.yolo_conf),
                )
                results["yolo_visualization"] = viz_result
                if viz_result.get("output_path"):
                    print(f"   Visualization saved: {viz_result['output_path']}")
                if viz_result.get("warning"):
                    print(f"   ⚠️  {viz_result['warning']}")

        if "openai" in selected:
            print("\n2️⃣  OpenAI GPT-4o-mini Vision...")
            openai_result = detect_openai(image_path)
            results["openai"] = openai_result
            if openai_result.get("has_solar_panel") is not None:
                print(f"   Has solar panel: {openai_result['has_solar_panel']}")
                print(f"   Confidence: {openai_result.get('confidence', 0):.2f}")
            else:
                print(f"   ❌ {openai_result.get('error', 'Failed')}")

        if "gemini" in selected:
            print("\n3️⃣  Google Gemini Flash...")
            
            # Rate limiting: wait between calls if needed
            if float(args.gemini_delay_between) > 0:
                elapsed = time.time() - last_gemini_call
                if elapsed < float(args.gemini_delay_between):
                    wait_time = float(args.gemini_delay_between) - elapsed
                    print(f"   ⏳ Waiting {wait_time:.1f}s for rate limiting...")
                    time.sleep(wait_time)
            
            gemini_result = detect_gemini(
                image_path,
                max_retries=int(args.gemini_max_retries),
                initial_delay=float(args.gemini_initial_delay)
            )
            last_gemini_call = time.time()
            
            results["gemini"] = gemini_result
            if gemini_result.get("has_solar_panel") is not None:
                print(f"   Has solar panel: {gemini_result['has_solar_panel']}")
                print(f"   Confidence: {gemini_result.get('confidence', 0):.2f}")
            else:
                print(f"   ❌ {gemini_result.get('error', 'Failed')}")

        if "ollama" in selected:
            print("\n4️⃣  Ollama Vision Model...")
            ollama_result = detect_ollama(
                image_path,
                model=str(args.ollama_model),
                host=str(args.ollama_host)
            )
            results["ollama"] = ollama_result
            if ollama_result.get("has_solar_panel") is not None:
                print(f"   Has solar panel: {ollama_result['has_solar_panel']}")
                print(f"   Confidence: {ollama_result.get('confidence', 0):.2f}")
            else:
                print(f"   ❌ {ollama_result.get('error', 'Failed')}")

        out["results"].append(results)

        if bool(args.per_image):
            per_dir = out_dir if out_dir is not None else Path.cwd()
            stem = Path(str(image_path).strip()).stem
            per_path = per_dir / f"solar_detection_{stem}.json"
            per_out = {
                "models": selected,
                "num_items": 1,
                "results": [results],
            }
            _ensure_parent_dir(str(per_path))
            with open(per_path, "w", encoding="utf-8") as f:
                json.dump(per_out, f, indent=2, ensure_ascii=False)

    if args.batch_out:
        out_path = str(args.batch_out)
    elif args.out:
        out_path = str(args.out)
    else:
        stem = "batch" if args.input_json else Path(str(items[0]["image_path"]).strip()).stem
        out_path = f"solar_detection_{stem}.json"

    _ensure_parent_dir(str(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to: {out_path}")


if __name__ == "__main__":
    main()