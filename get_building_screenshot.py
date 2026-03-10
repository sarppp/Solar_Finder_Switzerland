#!/usr/bin/env python3
"""
Get satellite/aerial screenshot of a building from GeoAdmin WMS service


Single address screenshot

python3 get_building_screenshot.py "Hinterdorfstrasse 10 3550 Langnau im Emmental" \
  --screenshots-dir outputs \
  --screenshot-size-m 20 \
  --screenshot-width 800 \
  --screenshot-height 800 \
  --reuse-screenshot

python3 get_building_screenshot.py "Kehrstrasse 9 3550 Langnau im Emmental" \
  --screenshots-dir outputs \
  --screenshot-size-m 20 \
  --screenshot-width 800 \
  --screenshot-height 800 \
  --reuse-screenshot
  

Batch screenshots from your JSON
This uses coordinates.y/x from the JSON, so it does zero address parsing:
python3 get_building_screenshot.py \
  --y 2627021.40 \
  --x 1198617.21 \
  --label "MyBuilding" \
  --screenshots-dir outputs \
  --screenshot-size-m 40 \
  --screenshot-width 800 \
  --screenshot-height 800

python3 get_building_screenshot.py \
  --input-json langnau_pv_clustered.json \
  --screenshots-dir outputs \
  --screenshot-size-m 50 \
  --screenshot-width 800 \
  --screenshot-height 800 \
  --reuse-screenshot

Limit
python3 get_building_screenshot.py \
  --input-json langnau_pv_clustered.json \
  --limit 10 \
  --screenshots-dir outputs \
  --screenshot-size-m 50 \
  --reuse-screenshot
"""

import argparse
import json
import os
import re
import requests
import sys
from PIL import Image
from io import BytesIO

def get_image_metadata(y: float, x: float) -> dict:
    """Query swisstopo for the flight year and resolution of the aerial image at this coordinate."""
    try:
        resp = requests.get(
            "https://api3.geo.admin.ch/rest/services/ech/MapServer/identify",
            params={
                "geometryType": "esriGeometryPoint",
                "geometry": f"{y},{x}",
                "layers": "all:ch.swisstopo.swissimage-product.metadata",
                "mapExtent": f"{y-50},{x-50},{y+50},{x+50}",
                "imageDisplay": "100,100,96",
                "tolerance": "50",
                "sr": "2056",
                "returnGeometry": "false",
            },
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        # Pick the result with the highest flightyear (most current)
        if not results:
            return {}
        best = max(results, key=lambda r: r.get("attributes", {}).get("flightyear", 0))
        attrs = best.get("attributes", {})
        gsd = attrs.get("gsd", "")
        gsd_cm = int(gsd.replace(" cm", "")) if "cm" in str(gsd) else None
        return {
            "flight_year": attrs.get("flightyear"),
            "published_year": attrs.get("bgdi_flugjahr"),
            "resolution_cm": gsd_cm,
        }
    except Exception:
        return {}


def geocode(address):
    """Convert address to LV95 coordinates"""
    resp = requests.get("https://api3.geo.admin.ch/rest/services/ech/SearchServer", 
                        params={"searchText": address, "type": "locations"})
    data = resp.json()
    if not data.get("results"):
        return None
    
    loc = data["results"][0]
    y, x = loc["attrs"]["y"], loc["attrs"]["x"]
    
    # Convert to LV95
    if y < 1000000:
        y += 2000000
        x += 1000000
    
    return {"y": y, "x": x, "label": loc["attrs"].get("label")}

def get_screenshot(y, x, radius_meters=50, width=800, height=800):
    """
    Get aerial screenshot from GeoAdmin WMS
    
    Args:
        y, x: LV95 coordinates
        radius_meters: Radius around point in meters
        width, height: Image dimensions in pixels
    """
    # Calculate bbox
    minY = y - radius_meters
    maxY = y + radius_meters
    minX = x - radius_meters
    maxX = x + radius_meters
    
    # WMS GetMap request
    url = "https://wms.geo.admin.ch/"
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "ch.swisstopo.swissimage",  # Aerial imagery
        "CRS": "EPSG:2056",
        "BBOX": f"{minY},{minX},{maxY},{maxX}",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": "image/png"
    }
    
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    
    return resp.content


def _sanitize_filename(s: str) -> str:
    s2 = re.sub(r"<[^>]+>", "", str(s))
    s2 = s2.strip()
    s2 = re.sub(r"\s+", " ", s2)
    s2 = re.sub(r"[^A-Za-z0-9._ -]+", "_", s2)
    s2 = s2.replace(" ", "_")
    s2 = re.sub(r"_+", "_", s2)
    if not s2:
        return "item"
    return s2[:120]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_screenshot(
    *,
    y: float,
    x: float,
    radius_m: float,
    width: int,
    height: int,
    out_path: str,
    reuse: bool,
) -> dict:
    image_meta = get_image_metadata(y, x)
    if reuse and os.path.exists(out_path):
        return {"screenshot": out_path, "reused": True, "image": image_meta}
    try:
        image_data = get_screenshot(float(y), float(x), float(radius_m), int(width), int(height))
        with open(out_path, "wb") as f:
            f.write(image_data)
        img = Image.open(BytesIO(image_data))
        return {
            "screenshot": out_path,
            "width_px": int(img.size[0]),
            "height_px": int(img.size[1]),
            "coverage_m": float(radius_m) * 2.0,
            "image": image_meta,
        }
    except Exception as e:
        return {"error": str(e), "screenshot": out_path, "image": image_meta}


def _iter_items_from_results_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise SystemExit(f"JSON file does not contain a 'results' list: {path}")
    for r in results:
        if not isinstance(r, dict):
            continue
        bid = r.get("building_id")
        label = r.get("label")
        coords = r.get("coordinates") or {}
        y = coords.get("y")
        x = coords.get("x")
        if y is None or x is None:
            continue
        yield {
            "building_id": bid,
            "label": label,
            "y": float(y),
            "x": float(x),
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("address", nargs="?", default=None)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--y", type=float, default=None)
    parser.add_argument("--x", type=float, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--screenshots-dir", default="outputs")
    parser.add_argument("--screenshot-size-m", type=float, default=50.0)
    parser.add_argument("--screenshot-width", type=int, default=800)
    parser.add_argument("--screenshot-height", type=int, default=800)
    parser.add_argument("--reuse-screenshot", action="store_true")
    args = parser.parse_args()

    has_coords = args.y is not None or args.x is not None
    if has_coords and (args.y is None or args.x is None):
        raise SystemExit("When using coordinates, provide both --y and --x (LV95)")
    if args.input_json and (args.address or has_coords):
        raise SystemExit("Use either --input-json OR (address / coordinates), not both")
    if not args.input_json and not args.address and not has_coords:
        raise SystemExit("Provide an address, coordinates (--y/--x), or --input-json")

    _ensure_dir(str(args.screenshots_dir))
    coverage_m = float(args.screenshot_size_m)
    radius = coverage_m / 2.0

    out = {
        "screenshots_dir": str(args.screenshots_dir),
        "screenshot_size_m": float(args.screenshot_size_m),
        "screenshot_width": int(args.screenshot_width),
        "screenshot_height": int(args.screenshot_height),
        "reuse_screenshot": bool(args.reuse_screenshot),
        "results": [],
    }

    if args.address:
        location = geocode(str(args.address))
        if not location:
            raise SystemExit("Address not found")
        label = location.get("label") or str(args.address)
        stem = _sanitize_filename(label)
        out_path = os.path.join(str(args.screenshots_dir), f"{stem}_{int(coverage_m)}m.png")
        item = {
            "label": label,
            "coordinates": {"y": float(location["y"]), "x": float(location["x"])},
        }
        item["screenshot"] = _save_screenshot(
            y=float(location["y"]),
            x=float(location["x"]),
            radius_m=radius,
            width=int(args.screenshot_width),
            height=int(args.screenshot_height),
            out_path=out_path,
            reuse=bool(args.reuse_screenshot),
        )
        out["results"].append(item)

    if has_coords:
        y = float(args.y)
        x = float(args.x)
        label = str(args.label) if args.label else f"LV95 y={y} x={x}"
        stem = _sanitize_filename(label)
        out_path = os.path.join(str(args.screenshots_dir), f"{stem}_{int(coverage_m)}m.png")
        item = {
            "label": label,
            "coordinates": {"y": y, "x": x},
        }
        item["screenshot"] = _save_screenshot(
            y=y,
            x=x,
            radius_m=radius,
            width=int(args.screenshot_width),
            height=int(args.screenshot_height),
            out_path=out_path,
            reuse=bool(args.reuse_screenshot),
        )
        out["results"].append(item)

    if args.input_json:
        n = 0
        for it in _iter_items_from_results_json(str(args.input_json)):
            if args.limit is not None and n >= int(args.limit):
                break
            bid = it.get("building_id")
            label = it.get("label")
            y = float(it["y"])
            x = float(it["x"])
            stem = _sanitize_filename(f"b{bid}_{label}" if bid is not None else str(label))
            out_path = os.path.join(str(args.screenshots_dir), f"{stem}_{int(coverage_m)}m.png")
            item = {
                "building_id": bid,
                "label": label,
                "coordinates": {"y": y, "x": x},
            }
            item["screenshot"] = _save_screenshot(
                y=y,
                x=x,
                radius_m=radius,
                width=int(args.screenshot_width),
                height=int(args.screenshot_height),
                out_path=out_path,
                reuse=bool(args.reuse_screenshot),
            )
            out["results"].append(item)
            n += 1

    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
