#!/usr/bin/env python3
"""
GeoAdmin API Explorer
=====================
Quick reference + live query tool for all GeoAdmin APIs used in this project.

Run any section to see what the API actually returns for a real address.

Usage:
    uv run python3 geoadmin_explorer.py geocode "Hinterdorfstrasse 13 Langnau im Emmental"
    uv run python3 geoadmin_explorer.py solar "Hinterdorfstrasse 13 Langnau im Emmental"
    uv run python3 geoadmin_explorer.py image-date "Hinterdorfstrasse 13 Langnau im Emmental"
    uv run python3 geoadmin_explorer.py building "Hinterdorfstrasse 13 Langnau im Emmental"
    uv run python3 geoadmin_explorer.py layers swissimage
    uv run python3 geoadmin_explorer.py wms-layers
    uv run python3 geoadmin_explorer.py identify "Hinterdorfstrasse 13 Langnau im Emmental" ch.bfe.solarenergie-eignung-daecher
    uv run python3 geoadmin_explorer.py all "Hinterdorfstrasse 13 Langnau im Emmental"

All results are printed as pretty JSON.
"""

import argparse
import json
import sys
import re
import requests

BASE = "https://api3.geo.admin.ch/rest/services/ech"
WMS  = "https://wms.geo.admin.ch/"

# ── Known layers used in this project ────────────────────────────────────────
KNOWN_LAYERS = {
    # ── Used in the pipeline ──────────────────────────────────────────────────
    "ch.bfe.solarenergie-eignung-daecher": (
        "USED: Solar suitability per roof facet. "
        "Returns polygon, area (flaeche), radiation class (klasse 1-5), azimuth, slope. "
        "Used in get_building_wms_overlay.py (facet centroid) and region_building_groups.py (discovery)."
    ),
    "ch.bfs.gebaeude_wohnungs_register": (
        "USED: Official building register (GWR). "
        "Returns EGID, address, GKAT (building type code), number of floors, year built. "
        "Used in region_building_groups.py to filter residential buildings and de-duplicate."
    ),
    "ch.bfe.elektrizitaetsproduktionsanlagen": (
        "USED: Power plant registry. "
        "Returns solar/wind/hydro plants with coordinates and capacity. "
        "Used in region_building_groups.py to check if a building already has solar (--filter-mode no_pv)."
    ),
    "ch.swisstopo.swissimage": (
        "USED: Satellite/aerial imagery. WMS GetMap only — returns PNG image, not JSON. "
        "Used in get_building_screenshot.py and get_building_wms_overlay.py to fetch the satellite image."
    ),
    "ch.swisstopo.swissimage-product.metadata": (
        "USED: Image tile metadata. "
        "Returns flight year (flightyear), publish year (bgdi_flugjahr), resolution (gsd). "
        "Used in both screenshot scripts to record when the image was taken."
    ),
    # ── Not yet used but relevant ─────────────────────────────────────────────
    "ch.swisstopo.swissbuildings3d_2_0": (
        "NOT USED YET: 3D building geometry. "
        "Returns building height, roof shape, number of floors in 3D. "
        "Would be needed to correct the parallax shift (building lean) in orthophotos."
    ),
    "ch.swisstopo.vec25-gebaeude": (
        "NOT USED YET: 2D building footprint polygons from vec25 map. "
        "Alternative to solar layer for getting building outlines — but less detailed than solar facets."
    ),
}


# ── Core helpers ──────────────────────────────────────────────────────────────

def geocode(address: str) -> dict:
    """
    API: GET /SearchServer?searchText=...&type=locations
    Returns: list of matches with LV95 y/x coordinates, label, bbox
    Used in: get_building_screenshot.py, get_building_wms_overlay.py
    """
    resp = requests.get(f"{BASE}/SearchServer",
                        params={"searchText": address, "type": "locations"},
                        timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise SystemExit(f"Address not found: {address!r}")
    attrs = results[0]["attrs"]
    y, x = float(attrs["y"]), float(attrs["x"])
    if y < 1_000_000:
        y += 2_000_000
        x += 1_000_000
    return {"y": y, "x": x, "label": attrs.get("label"), "raw": results[0]}


def identify(y: float, x: float, layer: str, radius: float = 50,
             return_geometry: bool = False) -> list:
    """
    API: GET /MapServer/identify
    Returns: features at a point for the given layer
    Key params:
      - geometry: "y,x" in LV95 (EPSG:2056)
      - layers: "all:<layer_id>"
      - tolerance: search radius in pixels (~metres at this scale)
      - returnGeometry: true → includes polygon rings
    Used in: get_building_wms_overlay.py (solar facet), region_building_groups.py
    """
    resp = requests.get(f"{BASE}/MapServer/identify", params={
        "geometryType": "esriGeometryPoint",
        "geometry": f"{y},{x}",
        "layers": f"all:{layer}",
        "mapExtent": f"{y-radius},{x-radius},{y+radius},{x+radius}",
        "imageDisplay": "1000,1000,96",
        "tolerance": "10",
        "returnGeometry": str(return_geometry).lower(),
        "sr": "2056",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def wms_getmap_url(y: float, x: float, layer: str = "ch.swisstopo.swissimage",
                   size_m: float = 60, px: int = 800) -> str:
    """
    API: WMS GetMap
    Returns: PNG image (not JSON)
    BBOX format for EPSG:2056: minEasting,minNorthing,maxEasting,maxNorthing
                               (= minY, minX, maxY, maxX in LV95)
    Used in: get_building_screenshot.py, get_building_wms_overlay.py
    """
    half = size_m / 2
    bbox = f"{y-half},{x-half},{y+half},{x+half}"
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": layer, "CRS": "EPSG:2056",
        "BBOX": bbox, "WIDTH": str(px), "HEIGHT": str(px),
        "FORMAT": "image/png",
    }
    req = requests.Request("GET", WMS, params=params).prepare()
    return req.url


def list_layers(keyword: str = "") -> dict:
    """
    API: GET /MapServer/layersConfig
    Returns: ALL available layers with their metadata (name, type, attribution, etc.)
    Useful for discovering layer IDs.
    """
    resp = requests.get(f"{BASE}/MapServer/layersConfig", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if keyword:
        data = {k: v for k, v in data.items() if keyword.lower() in k.lower()
                or keyword.lower() in str(v.get("label", "")).lower()}
    return data


def wms_layers() -> list:
    """
    API: WMS GetCapabilities
    Returns: all layers available in the WMS (for GetMap image requests)
    """
    resp = requests.get(WMS, params={
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetCapabilities"
    }, timeout=30)
    resp.raise_for_status()
    names = re.findall(r"<Name>(ch\.[^<]+)</Name>", resp.text)
    return sorted(set(names))


# ── CLI commands ──────────────────────────────────────────────────────────────

def cmd_geocode(args):
    print(f"\n── GEOCODE ──────────────────────────────────────────────────────")
    print(f"API: {BASE}/SearchServer?searchText=...&type=locations")
    print(f"What: converts address to LV95 coordinates\n")
    result = geocode(args.address)
    print(f"  y (Easting):  {result['y']}")
    print(f"  x (Northing): {result['x']}")
    print(f"  label:        {result['label']}")
    print(f"\nFull first result:")
    print(json.dumps(result["raw"], indent=2, ensure_ascii=False))


def cmd_solar(args):
    print(f"\n── SOLAR ROOF FACETS ────────────────────────────────────────────")
    print(f"Layer: ch.bfe.solarenergie-eignung-daecher")
    print(f"What: solar suitability per roof facet (area, radiation class, polygon)\n")
    loc = geocode(args.address)
    y, x = loc["y"], loc["x"]
    print(f"Coordinates: y={y:.2f}, x={x:.2f}")
    results = identify(y, x, "ch.bfe.solarenergie-eignung-daecher",
                       return_geometry=True)
    print(f"Found {len(results)} facet(s)\n")
    for r in results[:3]:
        attrs = r.get("attributes", {})
        geom = r.get("geometry", {})
        rings = geom.get("rings", [])
        print(f"  building_id: {attrs.get('building_id')}")
        print(f"  klasse:      {attrs.get('klasse')}  (1=best, 5=worst)")
        print(f"  flaeche:     {attrs.get('flaeche')} m²")
        print(f"  ausrichtung: {attrs.get('ausrichtung')} (azimuth degrees)")
        print(f"  neigung:     {attrs.get('neigung')} (slope degrees)")
        print(f"  polygon pts: {sum(len(r) for r in rings)} points in {len(rings)} ring(s)")
        print()
    if len(results) > 3:
        print(f"  ... and {len(results)-3} more facets")
    print("\nFirst full result:")
    if results:
        r0 = dict(results[0])
        if "geometry" in r0:
            r0["geometry"] = f"<{sum(len(r) for r in r0['geometry'].get('rings',[]))} polygon points>"
        print(json.dumps(r0, indent=2, ensure_ascii=False))


def cmd_image_date(args):
    print(f"\n── IMAGE DATE / FLIGHT METADATA ─────────────────────────────────")
    print(f"Layer: ch.swisstopo.swissimage-product.metadata")
    print(f"What: flight year + resolution of the aerial image tile at this point\n")
    loc = geocode(args.address)
    y, x = loc["y"], loc["x"]
    print(f"Coordinates: y={y:.2f}, x={x:.2f}")
    results = identify(y, x, "ch.swisstopo.swissimage-product.metadata", radius=50)
    print(f"Found {len(results)} tile(s) overlapping this point\n")
    for r in sorted(results, key=lambda r: r.get("attributes", {}).get("flightyear", 0)):
        attrs = r.get("attributes", {})
        print(f"  flightyear:    {attrs.get('flightyear')}  ← when photo was taken")
        print(f"  bgdi_flugjahr: {attrs.get('bgdi_flugjahr')}  ← when published")
        print(f"  gsd:           {attrs.get('gsd')}            ← resolution")
        print(f"  kbnum:         {attrs.get('kbnum')}          ← tile ID")
        print()


def cmd_building(args):
    print(f"\n── GWR BUILDING REGISTER ────────────────────────────────────────")
    print(f"Layer: ch.bfs.gebaeude_wohnungs_register")
    print(f"What: official building registry (EGID, address, type, floors, etc.)\n")
    loc = geocode(args.address)
    y, x = loc["y"], loc["x"]
    print(f"Coordinates: y={y:.2f}, x={x:.2f}")
    results = identify(y, x, "ch.bfs.gebaeude_wohnungs_register", radius=30)
    print(f"Found {len(results)} result(s)\n")
    for r in results[:2]:
        attrs = r.get("attributes", {})
        print(json.dumps(attrs, indent=2, ensure_ascii=False))
        print()


def cmd_identify(args):
    print(f"\n── IDENTIFY: {args.layer} ──")
    print(f"API: {BASE}/MapServer/identify")
    print(f"What: any layer, any point\n")
    loc = geocode(args.address)
    y, x = loc["y"], loc["x"]
    print(f"Coordinates: y={y:.2f}, x={x:.2f}")
    results = identify(y, x, args.layer, return_geometry=False)
    print(f"Found {len(results)} result(s)\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


def cmd_layers(args):
    print(f"\n── LAYER CONFIG (keyword: {args.keyword!r}) ──────────────────────")
    print(f"API: {BASE}/MapServer/layersConfig")
    print(f"What: all available layer IDs and their metadata\n")
    data = list_layers(args.keyword)
    print(f"Found {len(data)} matching layer(s):\n")
    for lid, info in list(data.items())[:30]:
        label = info.get("label", {})
        if isinstance(label, dict):
            label = label.get("de") or label.get("en") or ""
        print(f"  {lid}")
        if label:
            print(f"    → {label}")
    if len(data) > 30:
        print(f"\n  ... and {len(data)-30} more. Narrow search with a keyword.")


def cmd_wms_layers(args):
    print(f"\n── WMS LAYERS (GetCapabilities) ─────────────────────────────────")
    print(f"API: {WMS}?SERVICE=WMS&REQUEST=GetCapabilities")
    print(f"What: all layers available for WMS GetMap image requests\n")
    layers = wms_layers()
    kw = getattr(args, "keyword", "")
    if kw:
        layers = [l for l in layers if kw.lower() in l.lower()]
    print(f"Found {len(layers)} layer(s):")
    for l in layers:
        print(f"  {l}")


def cmd_wms_url(args):
    print(f"\n── WMS GetMap URL ───────────────────────────────────────────────")
    print(f"What: builds the URL to fetch a satellite image PNG\n")
    loc = geocode(args.address)
    y, x = loc["y"], loc["x"]
    url = wms_getmap_url(y, x, size_m=float(getattr(args, "size_m", 60)))
    print(f"Paste this in your browser to see the image:\n\n{url}\n")


def cmd_buildings3d(args):
    print(f"\n── 3D BUILDING GEOMETRY ─────────────────────────────────────────")
    print(f"Layer: ch.swisstopo.swissbuildings3d_2_0")
    print(f"NOT available via REST API — it is a bulk download dataset only.")
    print(f"Download from: https://www.swisstopo.admin.ch/en/landscape-model-swissbuildings3d-2-0")
    print(f"\nFor building height via API, use GWR (ch.bfs.gebaeude_wohnungs_register)")
    print(f"which has 'gastw' (number of floors) — multiply by ~3m for rough height estimate.")


def cmd_all(args):
    print(f"\n{'='*70}")
    print(f"  Full API dump for: {args.address}")
    print(f"{'='*70}")
    cmd_geocode(args)
    cmd_solar(args)
    cmd_image_date(args)
    cmd_building(args)
    cmd_wms_url(args)

    print(f"\n── KNOWN LAYERS IN THIS PROJECT ─────────────────────────────────")
    for lid, desc in KNOWN_LAYERS.items():
        print(f"  {lid}")
        print(f"    → {desc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="GeoAdmin API explorer — query any API and see what it returns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def addr(name): a = sub.add_parser(name); a.add_argument("address"); return a

    addr("geocode").description     = "Convert address to LV95 coordinates"
    addr("solar").description       = "Solar roof facets at address"
    addr("image-date").description  = "Flight year + resolution of aerial image"
    addr("building").description    = "GWR building registry data"
    addr("buildings3d").description = "3D building geometry (height, roof shape)"
    addr("all").description         = "Run all queries for an address"
    addr("wms-url").description     = "Print WMS GetMap URL for address"

    p_id = sub.add_parser("identify")
    p_id.add_argument("address")
    p_id.add_argument("layer", help="e.g. ch.bfe.solarenergie-eignung-daecher")

    p_ly = sub.add_parser("layers")
    p_ly.add_argument("keyword", nargs="?", default="",
                      help="Filter by keyword (e.g. swissimage, solar, gebaeude)")

    sub.add_parser("wms-layers").description = "List all WMS GetMap layers"

    args = p.parse_args()
    {
        "geocode":    cmd_geocode,
        "solar":      cmd_solar,
        "image-date": cmd_image_date,
        "building":   cmd_building,
        "buildings3d": cmd_buildings3d,
        "identify":   cmd_identify,
        "layers":     cmd_layers,
        "wms-layers": cmd_wms_layers,
        "wms-url":    cmd_wms_url,
        "all":        cmd_all,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
