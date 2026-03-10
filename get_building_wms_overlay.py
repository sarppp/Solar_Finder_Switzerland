#!/usr/bin/env python3
"""
Get a satellite screenshot with WMS-rendered solar suitability polygon overlay
for a building at a given address.

The polygon overlay is rendered server-side by the GeoAdmin WMS — no manual
coordinate math. The crop is centered on the building's roof facet centroid
(not the geocoded address point, which is typically the front door).

--size-m controls the total view width/height in metres (same meaning as
--screenshot-size-m in get_building_screenshot.py). When omitted, the view
is auto-sized to the building facet bbox + padding.

Usage:
    python3 get_building_wms_overlay.py "Hinterdorfstrasse 13 3550 Langnau im Emmental"
    python3 get_building_wms_overlay.py "Hinterdorfstrasse 13 3550 Langnau im Emmental" --size-m 80
    python3 get_building_wms_overlay.py "Schützenweg 253 3550 Langnau im Emmental" --size-m 50 --no-overlay
    python3 get_building_wms_overlay.py "Bahnhofstrasse 1 3008 Bern" --size-m 100 --output-dir outputs
"""

import argparse
import os
import re
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


def get_roof_facet(addr_y: float, addr_x: float) -> dict | None:
    """Point-query the solar suitability layer to get the roof facet at the address."""
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


def sanitize(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", str(s)).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return re.sub(r"_+", "_", s)[:80]


def main():
    parser = argparse.ArgumentParser(
        description="Satellite + solar polygon overlay, WMS-rendered (no coord math)."
    )
    parser.add_argument("address", help="Street address in Switzerland")
    parser.add_argument("--size-m", type=float, default=None,
                        help="Total view size in metres (width=height). "
                             "Overrides auto-sizing. E.g. --size-m 80 gives an 80×80m crop.")
    parser.add_argument("--padding", type=float, default=20.0,
                        help="Padding in metres around building bbox when --size-m is not set (default: 20)")
    parser.add_argument("--size", type=int, default=800,
                        help="Output image width/height in pixels (default: 800)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory to save images (default: outputs)")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Also save a clean satellite image without overlay")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Geocode
    print(f"Geocoding: {args.address!r}")
    addr_y, addr_x = geocode(args.address)
    print(f"  LV95: y={addr_y:.2f}, x={addr_x:.2f}")

    # 2. Find roof facet at address → get building centroid
    facet = get_roof_facet(addr_y, addr_x)
    if not facet:
        raise SystemExit("No solar roof facet found at this address.")

    attrs = facet["attributes"]
    bid = attrs.get("building_id")
    klasse = attrs.get("klasse")
    area = attrs.get("flaeche")
    print(f"  building_id={bid}  roof_area={float(area or 0):.1f}m²  suitability_class={klasse}")

    min_y, min_x, max_y, max_x = facet_bbox(facet)
    facet_cy = (min_y + max_y) / 2
    facet_cx = (min_x + max_x) / 2

    if args.size_m is not None:
        # User-specified size centred on the facet centroid
        half = args.size_m / 2.0
        b0, b1, b2, b3 = facet_cy - half, facet_cx - half, facet_cy + half, facet_cx + half
    else:
        # Auto-size: building bbox + padding
        b0, b1, b2, b3 = square_crop(min_y, min_x, max_y, max_x, args.padding)

    bbox = f"{b0},{b1},{b2},{b3}"
    side_m = b2 - b0
    print(f"  Crop: {side_m:.1f}m × {side_m:.1f}m  (centred on facet, not address)")

    W = H = args.size

    # 3. Satellite image
    print("Fetching satellite image...")
    sat = wms_image("ch.swisstopo.swissimage", bbox, W, H)

    # 4. Solar suitability overlay (WMS-rendered, server-side, perfectly aligned)
    print("Fetching solar polygon overlay...")
    solar = wms_image("ch.bfe.solarenergie-eignung-daecher", bbox, W, H, transparent=True)

    # 5. Compose and save
    stem = sanitize(args.address)
    overlay_path = os.path.join(args.output_dir, f"{stem}_overlay.png")
    Image.alpha_composite(sat, solar).convert("RGB").save(overlay_path)
    print(f"Saved: {overlay_path}")

    if args.no_overlay:
        clean_path = os.path.join(args.output_dir, f"{stem}_clean.png")
        sat.convert("RGB").save(clean_path)
        print(f"Saved: {clean_path}")


if __name__ == "__main__":
    main()
