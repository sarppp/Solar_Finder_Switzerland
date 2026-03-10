#!/usr/bin/env python3
"""
Get a satellite screenshot centered on the roof-facet centroid of a building.

Unlike get_building_screenshot.py (which centers on the geocoded address / front
door), this script queries the solar-suitability layer to find the actual roof
facet polygon and centers the crop on its centroid.  This gives a more accurate
crop when the geocoded address point is far from the roof (e.g. tall buildings
or addresses that resolve to the front door).

--size-m controls the total view width/height in metres (same meaning as
--screenshot-size-m in get_building_screenshot.py). When omitted, the view
is auto-sized to the building facet bbox + padding.

The default output is a clean satellite image (no overlay), named to match
the convention of get_building_screenshot.py: {stem}_{size}m.png

Use --with-overlay to additionally save {stem}_overlay.png (solar suitability
polygons composited on top of the satellite).  This flag is intended for
standalone inspection only — the pipeline never passes it.

Usage (single address):
    python3 get_building_wms_overlay.py "Hinterdorfstrasse 13 3550 Langnau im Emmental"
    python3 get_building_wms_overlay.py "Hinterdorfstrasse 13 3550 Langnau im Emmental" --size-m 80
    python3 get_building_wms_overlay.py "Hinterdorfstrasse 13 3550 Langnau im Emmental" --with-overlay

Usage (LV95 coordinates directly, no geocoding):
    python3 get_building_wms_overlay.py --y 2627073.25 --x 1198657.78 --label "Hinterdorf13"

Usage (batch from pipeline buildings JSON):
    python3 get_building_wms_overlay.py --input-json buildings.json --output-dir outputs/screenshots
    python3 get_building_wms_overlay.py --input-json buildings.json --limit 10
"""

import argparse
import json
import os
import re
import sys
import requests
from io import BytesIO
from PIL import Image


GEOADMIN_BASE = "https://api3.geo.admin.ch/rest/services/ech"
WMS_URL = "https://wms.geo.admin.ch/"


def geocode(address: str) -> tuple[float, float]:
    """Return LV95 (y=Easting, x=Northing) for an address string."""
    resp = requests.get(f"{GEOADMIN_BASE}/SearchServer",
                        params={"searchText": address, "type": "locations"},
                        timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise SystemExit(f"Address not found: {address!r}")
    attrs = results[0]["attrs"]
    y = float(attrs["y"])
    x = float(attrs["x"])
    if y < 1_000_000:   # old LV03 offset
        y += 2_000_000
        x += 1_000_000
    return y, x


def get_image_metadata(y: float, x: float) -> dict:
    """Query swisstopo for the flight year and resolution of the aerial image at this coordinate."""
    try:
        resp = requests.get(
            f"{GEOADMIN_BASE}/MapServer/identify",
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


def get_roof_facet(addr_y: float, addr_x: float) -> dict | None:
    """Point-query the solar suitability layer to get the roof facet at a coordinate."""
    resp = requests.get(f"{GEOADMIN_BASE}/MapServer/identify", params={
        "geometryType": "esriGeometryPoint",
        "geometry": f"{addr_y},{addr_x}",
        "layers": "all:ch.bfe.solarenergie-eignung-daecher",
        "mapExtent": f"{addr_y-50},{addr_x-50},{addr_y+50},{addr_x+50}",
        "imageDisplay": "1000,1000,96",
        "tolerance": "10",
        "returnGeometry": "true",
        "sr": "2056",
    }, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return results[0] if results else None


def facet_bbox(facet: dict) -> tuple[float, float, float, float]:
    """Return (min_y, min_x, max_y, max_x) from a facet's polygon rings."""
    all_pts = [p for ring in facet["geometry"]["rings"] for p in ring]
    return (
        min(p[0] for p in all_pts),
        min(p[1] for p in all_pts),
        max(p[0] for p in all_pts),
        max(p[1] for p in all_pts),
    )


def square_crop(min_y, min_x, max_y, max_x, padding: float) -> tuple[float, float, float, float]:
    """Expand bbox by padding, then make it square."""
    side = max((max_y - min_y) + 2 * padding, (max_x - min_x) + 2 * padding)
    cy = (min_y + max_y) / 2
    cx = (min_x + max_x) / 2
    return cy - side / 2, cx - side / 2, cy + side / 2, cx + side / 2


def wms_image(layers: str, bbox: str, width: int, height: int,
              transparent: bool = False) -> Image.Image:
    """Fetch a WMS GetMap image and return as RGBA PIL Image."""
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": layers, "CRS": "EPSG:2056",
        "BBOX": bbox, "WIDTH": str(width), "HEIGHT": str(height),
        "FORMAT": "image/png",
    }
    if transparent:
        params["TRANSPARENT"] = "TRUE"
    resp = requests.get(WMS_URL, params=params, timeout=60)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGBA")


def _sanitize_filename(s: str) -> str:
    """Identical to get_building_screenshot.py so batch stems match."""
    s2 = re.sub(r"<[^>]+>", "", str(s))
    s2 = s2.strip()
    s2 = re.sub(r"\s+", " ", s2)
    s2 = re.sub(r"[^A-Za-z0-9._ -]+", "_", s2)
    s2 = s2.replace(" ", "_")
    s2 = re.sub(r"_+", "_", s2)
    if not s2:
        return "item"
    return s2[:120]


def process_one(y: float, x: float, label: str, output_dir: str,
                size_m: float | None, padding: float, px_size: int,
                with_overlay: bool = False) -> dict:
    """Process a single building by LV95 coordinate. Returns result dict."""
    facet = get_roof_facet(y, x)
    if not facet:
        print(f"  WARNING: no solar facet at y={y:.1f} x={x:.1f}, skipping", file=sys.stderr)
        return {"label": label, "error": "no_facet"}

    attrs = facet["attributes"]
    min_y, min_x, max_y, max_x = facet_bbox(facet)
    facet_cy = (min_y + max_y) / 2
    facet_cx = (min_x + max_x) / 2

    if size_m is not None:
        half = size_m / 2.0
        b0, b1, b2, b3 = facet_cy - half, facet_cx - half, facet_cy + half, facet_cx + half
    else:
        b0, b1, b2, b3 = square_crop(min_y, min_x, max_y, max_x, padding)

    actual_size_m = round(b2 - b0)
    bbox = f"{b0},{b1},{b2},{b3}"
    stem = _sanitize_filename(label)

    # Primary output: clean satellite image, named like get_building_screenshot.py
    sat_path = os.path.join(output_dir, f"{stem}_{actual_size_m}m.png")
    sat = wms_image("ch.swisstopo.swissimage", bbox, px_size, px_size)
    sat.convert("RGB").save(sat_path)

    image_meta = get_image_metadata(y, x)
    print(f"  Saved: {sat_path}  ({actual_size_m}m × {actual_size_m}m  "
          f"bid={attrs.get('building_id')} klasse={attrs.get('klasse')})",
          file=sys.stderr)

    result = {
        "label": label,
        "screenshot": sat_path,
        "building_id": attrs.get("building_id"),
        "klasse": attrs.get("klasse"),
        "roof_area_m2": float(attrs.get("flaeche") or 0),
        "crop_size_m": actual_size_m,
        "coordinates": {"y": y, "x": x},
        "image": image_meta,
    }

    # Optional overlay output (standalone use only)
    if with_overlay:
        overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")
        solar = wms_image("ch.bfe.solarenergie-eignung-daecher", bbox, px_size, px_size, transparent=True)
        Image.alpha_composite(sat, solar).convert("RGB").save(overlay_path)
        print(f"  Overlay: {overlay_path}", file=sys.stderr)
        result["overlay"] = overlay_path

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Satellite screenshot centered on roof-facet centroid (optionally with WMS overlay)."
    )
    # Input modes
    parser.add_argument("address", nargs="?", default=None,
                        help="Street address in Switzerland (optional if using --y/--x or --input-json)")
    parser.add_argument("--y", type=float, default=None, help="LV95 Easting (Swiss Y)")
    parser.add_argument("--x", type=float, default=None, help="LV95 Northing (Swiss X)")
    parser.add_argument("--label", default=None, help="Label for output filename (used with --y/--x)")
    parser.add_argument("--input-json", default=None,
                        help="Batch mode: path to buildings JSON from region_building_groups.py")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max buildings to process in batch mode")
    # Crop / output
    parser.add_argument("--size-m", type=float, default=None,
                        help="Total view size in metres. Overrides auto-sizing.")
    parser.add_argument("--padding", type=float, default=20.0,
                        help="Padding around building bbox when --size-m is not set (default: 20)")
    parser.add_argument("--size", type=int, default=800,
                        help="Output image width/height in pixels (default: 800)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory to save images (default: outputs)")
    # Overlay (standalone only)
    parser.add_argument("--with-overlay", action="store_true",
                        help="Also save an overlay image with solar suitability polygons")
    args = parser.parse_args()

    # Validate input mode
    has_coord = args.y is not None and args.x is not None
    if not args.address and not has_coord and not args.input_json:
        parser.error("Provide an address, --y/--x coordinates, or --input-json")
    if args.y is not None and args.x is None or args.y is None and args.x is not None:
        parser.error("Provide both --y and --x together")

    os.makedirs(args.output_dir, exist_ok=True)

    out = {
        "output_dir": args.output_dir,
        "screenshot_size_m": args.size_m,
        "screenshot_width": args.size,
        "screenshot_height": args.size,
        "results": [],
    }

    # ── Single coordinate mode ─────────────────────────────────────────
    if has_coord:
        label = args.label or f"y{args.y:.0f}_x{args.x:.0f}"
        print(f"Coordinate mode: y={args.y}, x={args.x}  label={label!r}", file=sys.stderr)
        result = process_one(args.y, args.x, label, args.output_dir,
                             args.size_m, args.padding, args.size, args.with_overlay)
        out["results"].append(result)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    # ── Single address mode ────────────────────────────────────────────
    if args.address:
        print(f"Geocoding: {args.address!r}", file=sys.stderr)
        y, x = geocode(args.address)
        print(f"  LV95: y={y:.2f}, x={x:.2f}", file=sys.stderr)
        result = process_one(y, x, args.address, args.output_dir,
                             args.size_m, args.padding, args.size, args.with_overlay)
        out["results"].append(result)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    # ── Batch JSON mode ────────────────────────────────────────────────
    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)
    buildings = data.get("results", [])
    if not isinstance(buildings, list):
        raise SystemExit(f"No 'results' list in {args.input_json}")

    if args.limit:
        buildings = buildings[: args.limit]

    print(f"Batch mode: {len(buildings)} buildings from {args.input_json}", file=sys.stderr)
    for i, b in enumerate(buildings, 1):
        coords = b.get("coordinates") or {}
        y = coords.get("y")
        x = coords.get("x")
        if y is None or x is None:
            continue
        bid = b.get("building_id", i)
        label = b.get("label") or f"b{bid}"
        stem_label = f"b{bid}_{label}" if bid is not None else str(label)
        print(f"[{i}/{len(buildings)}] building_id={bid}  label={str(label)[:40]!r}", file=sys.stderr)
        result = process_one(float(y), float(x), stem_label,
                             args.output_dir, args.size_m, args.padding, args.size,
                             args.with_overlay)
        result["building_id"] = bid
        out["results"].append(result)

    n_ok = sum(1 for r in out["results"] if "error" not in r)
    print(f"\nDone: {n_ok}/{len(out['results'])} screenshots saved", file=sys.stderr)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
