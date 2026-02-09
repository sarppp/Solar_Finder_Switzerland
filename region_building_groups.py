"""


python3 "/app/region_building_groups.py" \
  --region "Langnau im Emmental" \
  --min-roof-area 300 \
  --plant-radius 30 \
  --filter-mode all \
  --pv-only-plants \
  --neighbor-within-m 200 \
  --label-bbox-m 30 \
  --label-cache /app/streamlit_site/langnau/langnau_labels_cache.json \
  --tile-size-m 500 \
  --min-tile-size-m 125 \
  --restrict-to-region-label \
  --sleep-s 0.05 \
  --progress-every-pct 10 \
  --max-results 2000 \
  --residential-only \
  --dedupe-by-esid \
  --dedupe-by-egrid \
  --residential-gkat-codes 1020 1030 1040 \
  --include-solar-metrics \
  --include-gwr-attrs \
  --gwr-tolerance-m 50 \
  --gwr-match-mode point_match_egid \
  --label-mode gwr_prefer \
  --include-raw-results \
  --state-file /tmp/langnau_state.json \
  --out "/app/streamlit_site/langnau/langnau_pv_residential_esid_egrid.json"


  --take-screenshots \
  --screenshots-dir /app/streamlit_site/payerne2/outputs \
  --screenshot-size-m 50 \


## SINGLE BUILDING
python3 "/app/region_building_groups.py" \
  --region "Langnau im Emmental" \
  --restrict-to-region-label \
  --min-roof-area 300 \
  --plant-radius 30 \
  --filter-mode no_pv \
  --tile-size-m 500 \
  --min-tile-size-m 125 \
  --sleep-s 0.05 \
  --debug \
  --debug-building-id 2685921 \
  --only-label-debug-building \
  --out single_building_debug.json

"""

#!/usr/bin/env python3
import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

GEOADMIN_BASE = "https://api3.geo.admin.ch/rest/services/ech"


@dataclass
class Plant:
    y: float
    x: float
    sub_category_en: Optional[str] = None
    address: Optional[str] = None
    total_power: Optional[str] = None
    beginning_of_operation: Optional[str] = None
    id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "y": self.y,
            "x": self.x,
            "sub_category_en": self.sub_category_en,
            "address": self.address,
            "total_power": self.total_power,
            "beginning_of_operation": self.beginning_of_operation,
            "id": self.id,
        }


@dataclass
class Facet:
    building_id: int
    facet_id: Optional[str] = None
    area: float = 0.0
    center_y: Optional[float] = None
    center_x: Optional[float] = None
    egid: Optional[str] = None
    slope: Optional[float] = None
    orientation: Optional[float] = None
    mstrahlung: Optional[float] = None
    gstrahlung: Optional[float] = None
    stromertrag: Optional[float] = None
    klasse: Optional[int] = None
    klasse_text: Optional[str] = None


@dataclass
class Building:
    building_id: int
    roof_area_m2: float = 0.0
    center_y: Optional[float] = None
    center_x: Optional[float] = None
    gwr_egid: Optional[str] = None
    avg_slope_deg: Optional[float] = None
    avg_orientation_deg: Optional[float] = None
    avg_mstrahlung: Optional[float] = None
    sum_gstrahlung: Optional[float] = None
    sum_stromertrag: Optional[float] = None
    best_klasse: Optional[int] = None
    best_klasse_text: Optional[str] = None
    cluster_label: Optional[int] = None  # DBSCAN cluster assignment


def _dist_m(y1: float, x1: float, y2: float, x2: float) -> float:
    return math.hypot(y1 - y2, x1 - x2)


class PlantIndex:
    """Spatial index for fast plant lookups using scipy.spatial.cKDTree."""
    
    def __init__(self, plants: list[Plant]):
        self.plants = plants
        self._tree = None
        self._coords = None
        if plants:
            self._coords = [[p.y, p.x] for p in plants]
            try:
                from scipy.spatial import cKDTree
                self._tree = cKDTree(self._coords)
            except ImportError:
                pass  # Fall back to linear search
    
    def query_radius(self, y: float, x: float, radius_m: float) -> list[dict]:
        """Find all plants within radius_m of point (y, x)."""
        if not self.plants:
            return []
        
        if self._tree is not None:
            # Use k-d tree for O(log n) lookup
            indices = self._tree.query_ball_point([y, x], radius_m)
            results = []
            for idx in indices:
                p = self.plants[idx]
                d = _dist_m(y, x, p.y, p.x)
                result = p.to_dict()
                result["distance_m"] = d
                results.append(result)
            results.sort(key=lambda x: x.get("distance_m", 1e18))
            return results
        else:
            # Fall back to linear search
            results = []
            for p in self.plants:
                d = _dist_m(y, x, p.y, p.x)
                if d <= radius_m:
                    result = p.to_dict()
                    result["distance_m"] = d
                    results.append(result)
            results.sort(key=lambda x: x.get("distance_m", 1e18))
            return results


def cluster_buildings(buildings: dict[int, Building], eps_m: float = 200.0, min_samples: int = 3) -> dict[int, Building]:
    """Apply DBSCAN clustering to buildings for urban/rural classification.
    
    Args:
        buildings: Dict of building_id -> Building
        eps_m: Maximum distance between points in a cluster (meters)
        min_samples: Minimum points to form a core cluster
        
    Returns:
        Updated buildings dict with cluster_label field
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        logger.debug("sklearn not available, skipping DBSCAN clustering")
        return buildings
    
    # Extract coordinates for buildings with centers
    coords = []
    building_ids = []
    for bid, b in buildings.items():
        if b.center_y is not None and b.center_x is not None:
            coords.append([b.center_y, b.center_x])
            building_ids.append(bid)
    
    if len(coords) < min_samples:
        return buildings
    
    # Run DBSCAN
    db = DBSCAN(eps=eps_m, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(coords)
    
    # Assign cluster labels to buildings
    # -1 means noise (isolated/rural), 0+ means cluster (urban)
    for bid, label in zip(building_ids, labels):
        buildings[bid].cluster_label = int(label)
    
    # Log cluster statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    logger.debug(f"DBSCAN found {n_clusters} clusters, {n_noise} isolated buildings (rural)")
    
    return buildings

def region_bbox_from_name(place_name: str) -> tuple[float, float, float, float]:
    resp = requests.get(
        f"{GEOADMIN_BASE}/SearchServer",
        params={"searchText": place_name, "type": "locations"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    if not results:
        raise ValueError(f"No locations found for region name: {place_name}")

    best = None
    fallback = None
    for r in results:
        attrs = r.get("attrs", {})
        bbox = attrs.get("geom_st_box2d") or attrs.get("boundingbox")
        if not bbox:
            continue

        if isinstance(bbox, str):
            parts = [p.strip() for p in bbox.replace("BOX(", "").replace(")", "").replace(",", " ").split()]
            if len(parts) >= 4:
                try:
                    min_y, min_x, max_y, max_x = map(float, parts[:4])
                    bbox = (min_y, min_x, max_y, max_x)
                except Exception:
                    bbox = None

        if not bbox or len(bbox) < 4:
            continue

        min_y, min_x, max_y, max_x = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

        if min_y < 1000000:
            min_y += 2000000
            max_y += 2000000
            min_x += 1000000
            max_x += 1000000

        origin = str(attrs.get("origin") or "")
        if best is None and origin == "gg25":
            best = (min_y, min_x, max_y, max_x)
        if fallback is None:
            fallback = (min_y, min_x, max_y, max_x)

    if best is not None:
        return best
    if fallback is not None:
        return fallback
    raise ValueError(f"No bbox available for region: {place_name}")

def identify_envelope(layer_bodid: str, min_y: float, min_x: float, max_y: float, max_x: float) -> dict:
    url = f"{GEOADMIN_BASE}/MapServer/identify"
    params = {
        "geometryType": "esriGeometryEnvelope",
        "geometry": f"{min_y},{min_x},{max_y},{max_x}",
        "layers": f"all:{layer_bodid}",
        "mapExtent": f"{min_y},{min_x},{max_y},{max_x}",
        "imageDisplay": "1000,1000,96",
        "tolerance": "0",
        "returnGeometry": "true",
        "sr": "2056",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()

def gwr_lookup_by_egid(egid: str | int) -> dict | None:
    """Lookup GWR by EGID using SearchServer API (more reliable than coordinate lookup)."""
    if not egid:
        return None
    try:
        resp = requests.get(
            f"{GEOADMIN_BASE}/SearchServer",
            params={
                "searchText": str(egid),
                "type": "locations",
                "sr": "2056",
                "limit": "5",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        # Find result with matching EGID
        for r in results:
            attrs = r.get("attrs", {})
            if str(attrs.get("egid")) == str(egid):
                return attrs
        return None
    except Exception:
        return None


def gwr_lookup(y: float, x: float, tolerance_m: float) -> dict | None:
    url = f"{GEOADMIN_BASE}/MapServer/identify"
    params = {
        "geometry": f"{y},{x}",
        "geometryType": "esriGeometryPoint",
        "layers": "all:ch.bfs.gebaeude_wohnungs_register",
        "mapExtent": f"{y-max(50.0, float(tolerance_m) * 3.0)},{x-max(50.0, float(tolerance_m) * 3.0)},{y+max(50.0, float(tolerance_m) * 3.0)},{x+max(50.0, float(tolerance_m) * 3.0)}",
        "imageDisplay": "1000,1000,96",
        "tolerance": str(float(tolerance_m)),
        "returnGeometry": "true",
        "sr": "2056",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    if not results:
        return None
    first = results[0]
    attrs = first.get("attributes") or {}
    return attrs

def gwr_lookup_all(y: float, x: float, tolerance_m: float) -> list[dict]:
    """Return all GWR results (not just first) for EGID matching."""
    url = f"{GEOADMIN_BASE}/MapServer/identify"
    params = {
        "geometry": f"{y},{x}",
        "geometryType": "esriGeometryPoint",
        "layers": "all:ch.bfs.gebaeude_wohnungs_register",
        "mapExtent": f"{y-max(50.0, float(tolerance_m) * 3.0)},{x-max(50.0, float(tolerance_m) * 3.0)},{y+max(50.0, float(tolerance_m) * 3.0)},{x+max(50.0, float(tolerance_m) * 3.0)}",
        "imageDisplay": "1000,1000,96",
        "tolerance": str(float(tolerance_m)),
        "returnGeometry": "true",
        "sr": "2056",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        return [r.get("attributes", {}) for r in results if r.get("attributes")]
    except Exception:
        return []

def gwr_is_residential(attrs: dict | None, residential_gkat_codes: set[int] | None) -> bool | None:
    if not attrs:
        return None

    # If an allowlist is provided, treat it as a hard constraint: GKAT must be in the allowlist.
    # This prevents non-allowlisted building categories from being kept just because other fields
    # (e.g. GANZWHG) indicate residential.
    if residential_gkat_codes is not None:
        gkat = attrs.get("gkat")
        if gkat is None:
            return None
        try:
            if int(gkat) not in residential_gkat_codes:
                return False
        except Exception:
            return None

    ganzwhg = attrs.get("ganzwhg")
    if ganzwhg is not None:
        try:
            return int(ganzwhg) > 0
        except Exception:
            pass

    gkat = attrs.get("gkat")
    if residential_gkat_codes and gkat is not None:
        try:
            return int(gkat) in residential_gkat_codes
        except Exception:
            pass
    present_count = 0
    for key in ["ganzwhg", "gazzi", "gastw"]:
        if attrs.get(key) is not None:
            present_count += 1
    if present_count >= 2:
        return True
    return None

def get_wms_screenshot(y: float, x: float, radius_m: float, width: int, height: int) -> bytes:
    url = "https://wms.geo.admin.ch/"
    min_y = float(y) - float(radius_m)
    max_y = float(y) + float(radius_m)
    min_x = float(x) - float(radius_m)
    max_x = float(x) + float(radius_m)
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "ch.swisstopo.swissimage",
        "CRS": "EPSG:2056",
        "BBOX": f"{min_y},{min_x},{max_y},{max_x}",
        "WIDTH": str(int(width)),
        "HEIGHT": str(int(height)),
        "FORMAT": "image/png",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.content

def _ensure_parent_dir(path: str | None) -> None:
    if path and (parent := os.path.dirname(str(path))):
        os.makedirs(parent, exist_ok=True)

def _progress_milestones(tiles_processed: int, tiles_total_estimate: int | None, last_pct_printed: int, step_pct: int) -> int:
    if tiles_total_estimate is None or tiles_total_estimate <= 0:
        return last_pct_printed
    pct = int((tiles_processed / tiles_total_estimate) * 100)
    if pct >= last_pct_printed + step_pct:
        pct2 = (pct // step_pct) * step_pct
        return min(100, max(last_pct_printed, pct2))
    return last_pct_printed

def tile_query(
    layer_bodid: str,
    min_y: float,
    min_x: float,
    max_y: float,
    max_x: float,
    tile_size_m: float,
    min_tile_size_m: float,
    max_tiles: int | None,
    sleep_s: float,
    progress_every_pct: int,
    extract_fn: callable,
    progress_label: str = "tiles",
    debug: bool = False,
    state_file: str | None = None,
    resume_key: str = "",
) -> tuple[list[dict], dict]:
    """Generic tiled query with adaptive subdivision and resume support."""
    
    # Try to load existing state
    state = None
    if state_file:
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    full_state = json.load(f)
                    state = full_state.get(resume_key)
                    if state:
                        logger.info(f"Resuming {progress_label} from saved state ({len(state.get('processed_tiles', []))} tiles already done)")
        except Exception as e:
            logger.debug(f"Could not load state: {e}")
    
    tiles_processed = 0
    total_returned = 0
    max_results_in_tile = 0
    tiles_subdivided = 0
    results: list[dict] = []
    processed_tiles: list[tuple] = []
    stack: list[tuple[float, float, float, float]] = []
    
    if state:
        # Restore from state
        results = state.get('results', [])
        processed_tiles = [tuple(t) for t in state.get('processed_tiles', [])]
        tiles_processed = state.get('tiles_processed', 0)
        total_returned = state.get('total_returned', 0)
        max_results_in_tile = state.get('max_results_in_tile', 0)
        tiles_subdivided = state.get('tiles_subdivided', 0)
        stack = [tuple(t) for t in state.get('stack', [])]

    tiles_total_estimate = int(
        max(1, math.ceil((max_y - min_y) / tile_size_m)) * max(1, math.ceil((max_x - min_x) / tile_size_m))
    )
    last_pct_printed = -progress_every_pct
    last_tiles_printed = 0

    # Build initial stack only if we don't have a saved pending stack
    if not stack:
        initial_stack = [(min_y, min_x, max_y, max_x)]
        for tile in initial_stack:
            if tile not in processed_tiles:
                stack.append(tile)
    
    if not stack and processed_tiles:
        logger.info(f"All {progress_label} tiles already processed, using cached results")
        return results, {
            "tiles_processed": tiles_processed,
            "tile_size_m": float(tile_size_m),
            "min_tile_size_m": float(min_tile_size_m),
            "max_tiles": int(max_tiles) if max_tiles is not None else None,
            "total_returned": int(total_returned),
            "results_collected": int(len(results)),
            "max_results_in_tile": int(max_results_in_tile),
            "tiles_subdivided": int(tiles_subdivided),
            "resumed": True,
        }

    def save_state():
        if state_file:
            try:
                full_state = {}
                if os.path.exists(state_file):
                    with open(state_file, 'r') as f:
                        full_state = json.load(f)
                full_state[resume_key] = {
                    'results': results,
                    'processed_tiles': processed_tiles,
                    'stack': stack,
                    'tiles_processed': tiles_processed,
                    'total_returned': total_returned,
                    'max_results_in_tile': max_results_in_tile,
                    'tiles_subdivided': tiles_subdivided,
                }
                _ensure_parent_dir(state_file)
                with open(state_file, 'w') as f:
                    json.dump(full_state, f)
            except Exception as e:
                logger.debug(f"Could not save state: {e}")

    while stack:
        tmin_y, tmin_x, tmax_y, tmax_x = stack.pop()
        tiles_processed += 1
        if max_tiles is not None and tiles_processed > int(max_tiles):
            break

        data = identify_envelope(layer_bodid, tmin_y, tmin_x, tmax_y, tmax_x)
        if sleep_s and sleep_s > 0:
            time.sleep(float(sleep_s))

        res_list = data.get("results", []) or []
        total_returned += len(res_list)
        max_results_in_tile = max(max_results_in_tile, len(res_list))

        width = tmax_x - tmin_x
        height = tmax_y - tmin_y
        if len(res_list) >= 190 and width > float(min_tile_size_m) and height > float(min_tile_size_m):
            tiles_subdivided += 1
            mid_y = (tmin_y + tmax_y) / 2
            mid_x = (tmin_x + tmax_x) / 2
            new_tiles = [
                (tmin_y, tmin_x, mid_y, mid_x),
                (tmin_y, mid_x, mid_y, tmax_x),
                (mid_y, tmin_x, tmax_y, mid_x),
                (mid_y, mid_x, tmax_y, tmax_x),
            ]
            for nt in new_tiles:
                if nt not in processed_tiles:
                    stack.append(nt)
            continue

        for r in res_list:
            extracted = extract_fn(r)
            if extracted is not None:
                results.append(extracted)
        
        # Mark tile as processed and save state
        processed_tiles.append((tmin_y, tmin_x, tmax_y, tmax_x))
        if tiles_processed % 5 == 0:  # Save every 5 tiles
            save_state()

        new_pct = _progress_milestones(tiles_processed, tiles_total_estimate, last_pct_printed, int(progress_every_pct))
        if new_pct != last_pct_printed:
            last_pct_printed = new_pct
            logger.info(f"Progress {progress_label}: ~{last_pct_printed}% (tiles_processed={tiles_processed}, tiles_remaining={len(stack)})")
        if tiles_processed - last_tiles_printed >= 25:
            last_tiles_printed = tiles_processed
            logger.info(f"Progress {progress_label}: tiles_processed={tiles_processed}, tiles_remaining={len(stack)}")

    # Final save
    save_state()

    debug_info = {
        "tiles_processed": int(tiles_processed),
        "tile_size_m": float(tile_size_m),
        "min_tile_size_m": float(min_tile_size_m),
        "max_tiles": int(max_tiles) if max_tiles is not None else None,
        "total_returned": int(total_returned),
        "results_collected": int(len(results)),
        "max_results_in_tile": int(max_results_in_tile),
        "tiles_subdivided": int(tiles_subdivided),
    }

    if debug:
        logger.debug(
            f"DEBUG {progress_label} collection: tiles={tiles_processed}, returned={total_returned}, "
            f"collected={len(results)}, max_in_tile={max_results_in_tile}, subdivided={tiles_subdivided}"
        )

    return results, debug_info

def _extract_plant(r: dict) -> Plant | None:
    geom = r.get("geometry") or {}
    py = geom.get("x")
    px = geom.get("y")
    if py is None or px is None:
        return None
    attrs = r.get("attributes") or {}
    return Plant(
        y=float(py),
        x=float(px),
        sub_category_en=attrs.get("sub_category_en"),
        address=attrs.get("address"),
        total_power=attrs.get("total_power"),
        beginning_of_operation=attrs.get("beginning_of_operation"),
        id=r.get("id"),
    )

def collect_plants_from_bbox(
    min_y: float,
    min_x: float,
    max_y: float,
    max_x: float,
    tile_size_m: float,
    min_tile_size_m: float,
    max_tiles: int | None,
    sleep_s: float,
    progress_every_pct: int,
    debug: bool,
    state_file: str | None = None,
) -> tuple[list[Plant], dict]:
    plants, debug_info = tile_query(
        layer_bodid="ch.bfe.elektrizitaetsproduktionsanlagen",
        min_y=min_y,
        min_x=min_x,
        max_y=max_y,
        max_x=max_x,
        tile_size_m=tile_size_m,
        min_tile_size_m=min_tile_size_m,
        max_tiles=max_tiles,
        sleep_s=sleep_s,
        progress_every_pct=progress_every_pct,
        extract_fn=_extract_plant,
        progress_label="plants",
        debug=debug,
        state_file=state_file,
        resume_key=f"plants_{min_y}_{min_x}",
    )
    # Map debug_info keys for backward compatibility
    mapped_debug = {
        **debug_info,
        "plants_total_returned": debug_info.get("total_returned"),
        "plants_points_collected": debug_info.get("results_collected"),
    }
    return plants, mapped_debug

def _extract_facet(r: dict, include_solar_metrics: bool) -> Facet | None:
    attrs = r.get("attributes") or {}
    b_id = attrs.get("building_id")
    if b_id is None:
        return None
    try:
        b_id_int = int(b_id)
    except Exception:
        return None

    area = attrs.get("flaeche")
    try:
        area_f = float(area) if area is not None else 0.0
    except Exception:
        area_f = 0.0

    bbox = r.get("bbox") or []
    center_y, center_x = None, None
    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            center_y = (float(bbox[0]) + float(bbox[2])) / 2
            center_x = (float(bbox[1]) + float(bbox[3])) / 2
        except Exception:
            pass

    facet = Facet(
        building_id=b_id_int,
        facet_id=r.get("featureId") or r.get("id"),
        area=area_f,
        center_y=center_y,
        center_x=center_x,
        egid=attrs.get("gwr_egid"),
    )

    if include_solar_metrics:
        facet.slope = _to_float(attrs.get("neigung"))
        facet.orientation = _to_float(attrs.get("ausrichtung"))
        facet.mstrahlung = _to_float(attrs.get("mstrahlung"))
        facet.gstrahlung = _to_float(attrs.get("gstrahlung"))
        facet.stromertrag = _to_float(attrs.get("stromertrag"))
        facet.klasse = _to_int(attrs.get("klasse"))
        facet.klasse_text = attrs.get("klasse_text")

    return facet

def _to_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def _to_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None

def _aggregate_buildings(facets: list[Facet], include_solar_metrics: bool) -> dict[int, Building]:
    """Group facets by building_id and compute aggregated metrics."""
    by_building: dict[int, list[Facet]] = defaultdict(list)
    seen_facets: set = set()

    for f in facets:
        fid = f.facet_id
        if fid is not None:
            if fid in seen_facets:
                continue
            seen_facets.add(fid)
        by_building[f.building_id].append(f)

    buildings: dict[int, Building] = {}
    for bid, fs in by_building.items():
        if not fs:
            continue

        # Building center: use average of facet centers
        centers = [(f.center_y, f.center_x) for f in fs if f.center_y is not None and f.center_x is not None]
        if centers:
            by = sum(c[0] for c in centers) / len(centers)
            bx = sum(c[1] for c in centers) / len(centers)
        else:
            by, bx = None, None

        # Roof area: pick facet closest to building center
        if by is not None and bx is not None:
            best_facet = min(fs, key=lambda f: _dist_m(by, bx, f.center_y or by, f.center_x or bx) if f.center_y else float("inf"))
            roof_area = best_facet.area
        else:
            roof_area = max((f.area for f in fs), default=0.0)

        building = Building(
            building_id=bid,
            roof_area_m2=roof_area,
            center_y=by,
            center_x=bx,
        )

        if include_solar_metrics:
            total_area = sum(f.area for f in fs if f.area > 0)
            building.gwr_egid = next((f.egid for f in fs if f.egid), None)

            if total_area > 0:
                weights = [(f, f.area) for f in fs if f.area > 0]
                building.avg_slope_deg = _weighted_mean(weights, "slope")
                building.avg_orientation_deg = _weighted_mean(weights, "orientation")
                building.avg_mstrahlung = _weighted_mean(weights, "mstrahlung")

            building.sum_gstrahlung = sum(f.gstrahlung for f in fs if f.gstrahlung is not None) or None
            building.sum_stromertrag = sum(f.stromertrag for f in fs if f.stromertrag is not None) or None

            best_klasse = max((f.klasse for f in fs if f.klasse is not None), default=None)
            best_klasse_text = next((f.klasse_text for f in fs if f.klasse == best_klasse and f.klasse_text), None)
            building.best_klasse = best_klasse
            building.best_klasse_text = best_klasse_text

        buildings[bid] = building

    return buildings


def _weighted_mean(items: list[tuple[Facet, float]], key: str) -> float | None:
    total_weight = 0.0
    total_val = 0.0
    for item, weight in items:
        v = getattr(item, key, None)
        if v is not None:
            total_val += v * weight
            total_weight += weight
    return total_val / total_weight if total_weight > 0 else None

def collect_buildings_from_region(
    region: str,
    min_roof_area_m2: float,
    tile_size_m: float,
    min_tile_size_m: float,
    max_tiles: int | None,
    include_solar_metrics: bool,
    debug: bool,
    sleep_s: float,
    progress_every_pct: int,
    state_file: str | None = None,
) -> tuple[dict[int, Building], dict]:
    min_y, min_x, max_y, max_x = region_bbox_from_name(region)

    facets, debug_info = tile_query(
        layer_bodid="ch.bfe.solarenergie-eignung-daecher",
        min_y=min_y,
        min_x=min_x,
        max_y=max_y,
        max_x=max_x,
        tile_size_m=tile_size_m,
        min_tile_size_m=min_tile_size_m,
        max_tiles=max_tiles,
        sleep_s=sleep_s,
        progress_every_pct=progress_every_pct,
        extract_fn=lambda r: _extract_facet(r, include_solar_metrics),
        progress_label="roofs",
        debug=debug,
        state_file=state_file,
        resume_key=f"roofs_{region}",
    )

    buildings = _aggregate_buildings(facets, include_solar_metrics)

    facets_with_center = sum(1 for f in facets if f.center_y is not None)
    if debug:
        areas = [b.roof_area_m2 for b in buildings.values()]
        areas.sort(reverse=True)
        logger.debug(
            f"Region bbox LV95: {min_y},{min_x} to {max_y},{max_x} | "
            f"facets returned: {debug_info.get('total_returned')}, unique facets: {len(set(f.facet_id for f in facets if f.facet_id))} | "
            f"buildings: {len(buildings)}, with center: {sum(1 for b in buildings.values() if b.center_y is not None)} | "
            f"top roofs: {[round(a, 2) for a in areas[:10]]}"
        )

    mapped_debug = {
        **debug_info,
        "region_bbox_lv95": {"min_y": min_y, "min_x": min_x, "max_y": max_y, "max_x": max_x},
        "facets_total_returned": debug_info.get("total_returned"),
        "facets_with_bbox": facets_with_center,
        "unique_facets_seen": len(set(f.facet_id for f in facets if f.facet_id)),
        "unique_buildings_before_filter": len(buildings),
        "buildings_with_center": sum(1 for b in buildings.values() if b.center_y is not None),
        "top_roof_area_m2_before_filter": [round(float(a), 2) for a in sorted((b.roof_area_m2 for b in buildings.values()), reverse=True)[:10]],
    }

    # Filter by roof area + center availability
    out: dict[int, Building] = {}
    for bid, b in buildings.items():
        if b.center_y is None or b.center_x is None:
            continue
        if b.roof_area_m2 < min_roof_area_m2:
            continue
        out[bid] = b
    return out, mapped_debug

def reverse_label(y: float, x: float, bbox_m: float = 30.0) -> str | None:
    try:
        resp = requests.get(
            f"{GEOADMIN_BASE}/SearchServer",
            params={
                "bbox": f"{y-bbox_m},{x-bbox_m},{y+bbox_m},{x+bbox_m}",
                "type": "locations",
                "sr": "2056",
                "limit": "20",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        best_address = None
        best_parcel = None
        for r in data.get("results", []) or []:
            attrs = r.get("attrs", {})
            label = attrs.get("label")
            origin = attrs.get("origin")

            if origin == "address" and label:
                best_address = label
                break
            if best_parcel is None and origin == "parcel" and label:
                best_parcel = label

        return best_address or best_parcel
    except Exception:
        return None

def _load_label_cache(path: str | None) -> dict[str, str | None]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): (v if v is None or isinstance(v, str) else str(v)) for k, v in data.items()}
    except Exception:
        return {}
    return {}

def _save_label_cache(path: str | None, cache: dict[str, str | None]) -> None:
    if not path:
        return
    try:
        _ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        return

def _buildings_with_neighbor_within(buildings: dict[int, Building], within_m: float) -> set[int]:
    # Returns building_ids that have at least one other building within within_m.
    # Uses a simple grid (spatial hash) for O(n) average behavior.
    r = float(within_m)
    if r <= 0:
        return set(buildings.keys())

    cell = r
    grid: dict[tuple[int, int], list[tuple[int, float, float]]] = {}

    pts: list[tuple[int, float, float]] = []
    for bid, b in buildings.items():
        y = b.center_y
        x = b.center_x
        if y is None or x is None:
            continue
        pts.append((int(bid), y, x))

    for bid, y, x in pts:
        cy = int(math.floor(y / cell))
        cx = int(math.floor(x / cell))
        grid.setdefault((cy, cx), []).append((bid, y, x))

    keep: set[int] = set()
    for bid, y, x in pts:
        cy = int(math.floor(y / cell))
        cx = int(math.floor(x / cell))
        found = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for obid, oy, ox in grid.get((cy + dy, cx + dx), []):
                    if obid == bid:
                        continue
                    if _dist_m(y, x, oy, ox) <= r:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            keep.add(int(bid))
    return keep

def identify_point(*, layer: str, y: float, x: float, tolerance: float, extent_m: float) -> list[dict]:
    """Query GeoAdmin identify API at a point."""
    url = f"{GEOADMIN_BASE}/MapServer/identify"
    params = {
        "geometry": f"{y},{x}",
        "geometryType": "esriGeometryPoint",
        "tolerance": float(tolerance),
        "sr": 2056,
        "layers": f"all:{layer}",
        "mapExtent": f"{y-extent_m},{x-extent_m},{y+extent_m},{x+extent_m}",
        "imageDisplay": "1000,1000,96",
        "returnGeometry": "true",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    results = data.get("results") or []
    return [r for r in results if isinstance(r, dict)]

def collect_single_building(
    y: float,
    x: float,
    min_roof_area_m2: float,
    include_solar_metrics: bool,
) -> tuple[dict[int, Building], dict]:
    """Collect a single building by coordinate using point query."""
    # Query roof layer at this point
    roof_results = identify_point(
        layer="ch.bfe.solarenergie-eignung-daecher",
        y=y,
        x=x,
        tolerance=5.0,  # Small tolerance to find nearby facets
        extent_m=100.0,
    )
    
    if not roof_results:
        return {}, {"error": "No roof data found at coordinate", "query_y": y, "query_x": x}
    
    # Extract facets
    facets = []
    for r in roof_results:
        facet = _extract_facet(r, include_solar_metrics)
        if facet:
            facets.append(facet)
    
    if not facets:
        return {}, {"error": "No valid facets found", "query_y": y, "query_x": x}
    
    # Aggregate into buildings
    buildings = _aggregate_buildings(facets, include_solar_metrics)
    
    # Filter by roof area
    out: dict[int, Building] = {}
    for bid, b in buildings.items():
        if b.roof_area_m2 < min_roof_area_m2:
            continue
        out[bid] = b
    
    debug_info = {
        "mode": "single_coordinate",
        "query_y": y,
        "query_x": x,
        "facets_found": len(facets),
        "buildings_before_filter": len(buildings),
        "buildings_after_filter": len(out),
    }
    
    return out, debug_info

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=None, help="Region name (e.g., 'Payerne'). Use --coord for single building mode.")
    parser.add_argument("--coord", default=None, help="Single coordinate mode: 'y,x' (e.g., '2561977.054,1185216.497'). Overrides --region.")
    parser.add_argument("--min-roof-area", type=float, default=200.0)
    parser.add_argument("--plant-radius", type=float, default=30.0)
    parser.add_argument("--filter-mode", choices=["all", "no_pv", "pv", "no_plants"], default="all")
    parser.add_argument("--pv-only-plants", action="store_true")
    parser.add_argument("--no-plants-details", action="store_true")
    parser.add_argument("--include-solar-metrics", action="store_true")
    parser.add_argument("--neighbor-within-m", type=float, default=None)
    parser.add_argument("--label-bbox-m", type=float, default=30.0)
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--label-cache", default=None)
    parser.add_argument("--tile-size-m", type=float, default=500.0)
    parser.add_argument("--min-tile-size-m", type=float, default=125.0)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--restrict-to-region-label", action="store_true")
    parser.add_argument("--debug-building-id", type=int, default=None)
    parser.add_argument("--only-label-debug-building", action="store_true")
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--progress-every-pct", type=int, default=10)
    parser.add_argument("--max-results", type=int, default=200)
    parser.add_argument("--take-screenshots", action="store_true")
    parser.add_argument("--screenshots-dir", default=None)
    parser.add_argument("--screenshot-size-m", type=float, default=50.0)
    parser.add_argument("--reuse-screenshot", action="store_true")
    parser.add_argument("--residential-only", action="store_true")
    parser.add_argument(
        "--dedupe-by-egrid",
        action="store_true",
        help="De-duplicate results by GWR EGRID (keep the entry with largest roof area per EGRID).",
    )
    parser.add_argument(
        "--dedupe-by-esid",
        action="store_true",
        help="De-duplicate results by GWR ESID (keep the entry with largest roof area per ESID).",
    )
    parser.add_argument("--gwr-tolerance-m", type=float, default=30.0, help="Tolerance for GWR lookup (default: 30m)")
    parser.add_argument("--gwr-match-mode", choices=["point_first", "point_match_egid", "egid_direct"], default="point_first", help="GWR lookup strategy")
    parser.add_argument("--label-mode", choices=["reverse", "gwr_prefer"], default="reverse", help="Label source: reverse geocoding or GWR")
    parser.add_argument("--include-gwr-attrs", action="store_true")
    parser.add_argument("--residential-gkat-codes", default=None)
    parser.add_argument("--state-file", default=None, help="Path to save/load state for resume capability")
    parser.add_argument("--include-raw-results", action="store_true", help="Include raw API results like req.py for debugging")
    parser.add_argument("--out", default="region_groups.json")
    args = parser.parse_args()

    # Handle single coordinate mode
    coord_y: float | None = None
    coord_x: float | None = None
    single_building_mode = False
    if args.coord:
        # Parse coordinate string (format: "y,x" supporting apostrophes and spaces)
        raw = str(args.coord).strip().replace("'", "").replace(" ", "")
        parts = [p for p in raw.split(",") if p]
        if len(parts) != 2:
            raise SystemExit(f"Invalid --coord format: {args.coord!r}; expected 'y,x'")
        try:
            coord_y = float(parts[0])
            coord_x = float(parts[1])
            single_building_mode = True
            logger.info(f"Single building mode: y={coord_y}, x={coord_x}")
        except ValueError:
            raise SystemExit(f"Invalid coordinate values: {args.coord!r}")
    elif not args.region:
        raise SystemExit("Provide either --region 'RegionName' or --coord 'y,x'")

    if args.skip_labels and args.restrict_to_region_label:
        raise SystemExit("--restrict-to-region-label requires labels; remove it or run without --skip-labels")

    screenshots_dir: str | None
    if args.take_screenshots:
        screenshots_dir = args.screenshots_dir or "outputs"
        os.makedirs(screenshots_dir, exist_ok=True)
    else:
        screenshots_dir = None

    label_cache_path = args.label_cache
    label_cache = _load_label_cache(label_cache_path)

    residential_gkat_codes: set[int] | None = None
    if args.residential_gkat_codes:
        try:
            residential_gkat_codes = {int(v.strip()) for v in args.residential_gkat_codes.split(",") if v.strip()}
        except Exception:
            raise SystemExit("Invalid --residential-gkat-codes; expected comma-separated ints")

    gwr_cache: dict[tuple[float, float, float], dict | None] = {}

    # Collect buildings: either single coordinate or full region
    if single_building_mode and coord_y is not None and coord_x is not None:
        buildings, debug_buildings = collect_single_building(
            y=coord_y,
            x=coord_x,
            min_roof_area_m2=args.min_roof_area,
            include_solar_metrics=args.include_solar_metrics,
        )
        # For plants, use a small bbox around the coordinate
        plant_bbox_radius = max(args.plant_radius * 2, 100.0)
        plants, debug_plants = collect_plants_from_bbox(
            min_y=coord_y - plant_bbox_radius,
            min_x=coord_x - plant_bbox_radius,
            max_y=coord_y + plant_bbox_radius,
            max_x=coord_x + plant_bbox_radius,
            tile_size_m=plant_bbox_radius * 2,
            min_tile_size_m=plant_bbox_radius,
            max_tiles=1,
            sleep_s=0,
            progress_every_pct=100,
            debug=args.debug,
            state_file=None,
        )
    else:
        buildings, debug_buildings = collect_buildings_from_region(
            region=args.region,
            min_roof_area_m2=args.min_roof_area,
            tile_size_m=args.tile_size_m,
            min_tile_size_m=args.min_tile_size_m,
            max_tiles=args.max_tiles,
            include_solar_metrics=args.include_solar_metrics,
            debug=args.debug,
            sleep_s=args.sleep_s,
            progress_every_pct=args.progress_every_pct,
            state_file=args.state_file,
        )

        bbox = (debug_buildings.get("region_bbox_lv95") or {}) if isinstance(debug_buildings, dict) else {}
        plants, debug_plants = collect_plants_from_bbox(
            min_y=float(bbox.get("min_y")),
            min_x=float(bbox.get("min_x")),
            max_y=float(bbox.get("max_y")),
            max_x=float(bbox.get("max_x")),
            tile_size_m=args.tile_size_m,
            min_tile_size_m=args.min_tile_size_m,
            max_tiles=args.max_tiles,
            sleep_s=args.sleep_s,
            progress_every_pct=args.progress_every_pct,
            debug=args.debug,
            state_file=args.state_file,
        )

    if args.pv_only_plants:
        plants = [p for p in plants if "Photovoltaic" in str(p.sub_category_en or "")]

    # Build spatial index for fast plant lookups
    plant_index = PlantIndex(plants)

    # Apply DBSCAN clustering for urban/rural classification
    if len(buildings) > 3:
        buildings = cluster_buildings(buildings, eps_m=args.neighbor_within_m or 200.0, min_samples=3)

    neighbor_keep: set[int] | None = None
    if args.neighbor_within_m is not None and not single_building_mode:
        neighbor_keep = _buildings_with_neighbor_within(buildings, within_m=args.neighbor_within_m)

    items: list[dict] = []
    
    # Use tqdm for progress bar over buildings
    building_iter = tqdm(buildings.values(), desc="Processing buildings", unit="bld", disable=not sys.stdout.isatty())
    
    for b in building_iter:
        if neighbor_keep is not None and b.building_id not in neighbor_keep:
            continue
        if args.only_label_debug_building and args.debug_building_id is not None:
            if b.building_id != args.debug_building_id:
                continue

        if args.debug_building_id is not None and b.building_id == args.debug_building_id:
            logger.debug(f"Found building_id {args.debug_building_id} with roof_area_m2={b.roof_area_m2}")

        y = b.center_y
        x = b.center_x
        if y is None or x is None:
            continue

        nearby_plants = plant_index.query_radius(y, x, radius_m=args.plant_radius)
        has_any = len(nearby_plants) > 0
        if args.pv_only_plants:
            has_pv = has_any
        else:
            has_pv = any("Photovoltaic" in str(p.get("sub_category_en") or "") for p in nearby_plants)

        if args.filter_mode == "no_pv" and has_pv:
            continue
        if args.filter_mode == "pv" and not has_pv:
            continue
        if args.filter_mode == "no_plants" and has_any:
            continue

        gwr_attrs: dict | None = None
        gwr_residential: bool | None = None
        label: str | None = None
        raw_roof_results: list[dict] | None = None
        raw_gwr_results: list[dict] | None = None
        
        if args.residential_only or args.include_gwr_attrs or args.include_raw_results:
            # GWR lookup based on match mode
            if args.gwr_match_mode == "egid_direct" and b.gwr_egid:
                gwr_attrs = gwr_lookup_by_egid(b.gwr_egid)
            elif args.gwr_match_mode == "point_match_egid":
                # Query by point but filter to matching EGID
                key = (round(float(y), 2), round(float(x), 2), args.gwr_tolerance_m)
                if key in gwr_cache:
                    all_results = gwr_cache[key]
                else:
                    all_results = gwr_lookup_all(float(y), float(x), tolerance_m=args.gwr_tolerance_m)
                    gwr_cache[key] = all_results
                # Store raw results if requested
                if args.include_raw_results and all_results:
                    raw_gwr_results = all_results
                # Filter to matching EGID if we have one
                if all_results and b.gwr_egid:
                    for attrs in all_results:
                        if str(attrs.get("egid")) == str(b.gwr_egid):
                            gwr_attrs = attrs
                            break
                elif all_results:
                    gwr_attrs = all_results[0]
            else:  # point_first (default)
                if b.gwr_egid:
                    gwr_attrs = gwr_lookup_by_egid(b.gwr_egid)
                if gwr_attrs is None:
                    key = (round(float(y), 2), round(float(x), 2), args.gwr_tolerance_m)
                    if key in gwr_cache:
                        gwr_attrs = gwr_cache[key]
                    else:
                        try:
                            gwr_attrs = gwr_lookup(float(y), float(x), tolerance_m=args.gwr_tolerance_m)
                        except Exception:
                            gwr_attrs = None
                        gwr_cache[key] = gwr_attrs

            gwr_residential = gwr_is_residential(gwr_attrs, residential_gkat_codes=residential_gkat_codes)
            if args.residential_only and gwr_residential is not True:
                continue

        # Label based on label_mode
        if args.skip_labels:
            label = None
        elif args.label_mode == "gwr_prefer" and gwr_attrs and gwr_attrs.get("label"):
            label = gwr_attrs.get("label")
        else:  # reverse (default)
            cache_key = str(b.building_id)
            cached = label_cache.get(cache_key) if cache_key in label_cache else None
            if cache_key in label_cache and cached is not None:
                label = cached
            else:
                label = reverse_label(y, x, bbox_m=args.label_bbox_m)
                label_cache[cache_key] = label

        if args.restrict_to_region_label:
            in_label = bool(label) and (args.region in str(label))
            in_gwr_muni = False
            if not in_label and gwr_attrs and args.region:
                muni = gwr_attrs.get("ggdename")
                if muni is not None and str(muni) == str(args.region):
                    in_gwr_muni = True

            if not in_label and not in_gwr_muni:
                if args.debug_building_id is not None and b.building_id == args.debug_building_id:
                    logger.debug(
                        f"Filtered by region label: building_id={b.building_id} label={label!r} ggdename={(gwr_attrs or {}).get('ggdename')!r} region={args.region!r}"
                    )
                continue

            if args.debug_building_id is not None and b.building_id == args.debug_building_id:
                logger.debug(
                    f"Passed region label: building_id={b.building_id} label={label!r} ggdename={(gwr_attrs or {}).get('ggdename')!r} region={args.region!r}"
                )

        # Build GWR dict with only non-null fields
        gwr_output = None
        if args.include_gwr_attrs or args.residential_only:
            gwr_output = {"is_residential": gwr_residential}
            if gwr_attrs:
                # Extended list of GWR fields
                for key in ["egid", "egrid", "esid", "gkat", "gklas", "gstat", "ganzwhg", "label", "strname_deinr", 
                           "plz_plz6", "ggdename", "ggdenr", "gdename", "gdenr", "gexpdat"]:
                    val = gwr_attrs.get(key)
                    if val is not None:
                        # Strip HTML from label values
                        if key == "label" and isinstance(val, str):
                            val = val.replace("<b>", "").replace("</b>", "")
                        gwr_output[key] = val

        # Also clean up the main label if it has HTML
        if label and isinstance(label, str):
            label = label.replace("<b>", "").replace("</b>", "")

        screenshot_info: dict | None = None
        item = {
            "building_id": b.building_id,
            "label": label,
            "coordinates": {"y": y, "x": x},
            "roof_area_m2": round(b.roof_area_m2, 2),
            "plant_radius_m": args.plant_radius,
            "has_any_plant": has_any,
            "has_pv_plant": has_pv,
            "plants_within_radius": [] if args.no_plants_details else nearby_plants[:10],
            "avg_slope_deg": round(b.avg_slope_deg, 2) if args.include_solar_metrics and b.avg_slope_deg is not None else None,
            "avg_orientation_deg": round(b.avg_orientation_deg, 2) if args.include_solar_metrics and b.avg_orientation_deg is not None else None,
            "avg_mstrahlung_kwh_m2_year": round(b.avg_mstrahlung, 2) if args.include_solar_metrics and b.avg_mstrahlung is not None else None,
            "sum_gstrahlung": round(b.sum_gstrahlung, 2) if args.include_solar_metrics and b.sum_gstrahlung is not None else None,
            "sum_stromertrag_kwh_year": round(b.sum_stromertrag, 2) if args.include_solar_metrics and b.sum_stromertrag is not None else None,
            "suitability": {"best_klasse": b.best_klasse, "best_klasse_text": b.best_klasse_text} if args.include_solar_metrics else None,
            "cluster_label": b.cluster_label,  # DBSCAN cluster
            "gwr": gwr_output,
            "screenshot": screenshot_info,
        }
        
        # Add raw results if requested
        if args.include_raw_results:
            item["_raw"] = {
                "gwr_results": raw_gwr_results,
            }
        
        items.append(item)

    items.sort(key=lambda r: r.get("roof_area_m2") or 0.0, reverse=True)
    if args.dedupe_by_esid:
        best_by_esid: dict[str, dict] = {}
        deduped: list[dict] = []
        for it in items:
            gwr = it.get("gwr") or {}
            esid = gwr.get("esid")
            if esid is None:
                deduped.append(it)
                continue
            esid_key = str(esid)
            prev = best_by_esid.get(esid_key)
            if prev is None or (it.get("roof_area_m2") or 0.0) > (prev.get("roof_area_m2") or 0.0):
                best_by_esid[esid_key] = it
        deduped.extend(best_by_esid.values())
        items = deduped
        items.sort(key=lambda r: r.get("roof_area_m2") or 0.0, reverse=True)
    if args.dedupe_by_egrid:
        best_by_egrid: dict[str, dict] = {}
        deduped: list[dict] = []
        for it in items:
            gwr = it.get("gwr") or {}
            egrid = gwr.get("egrid")
            if egrid is None:
                deduped.append(it)
                continue
            egrid_key = str(egrid)
            prev = best_by_egrid.get(egrid_key)
            if prev is None or (it.get("roof_area_m2") or 0.0) > (prev.get("roof_area_m2") or 0.0):
                best_by_egrid[egrid_key] = it
        deduped.extend(best_by_egrid.values())
        items = deduped
        items.sort(key=lambda r: r.get("roof_area_m2") or 0.0, reverse=True)
    if args.max_results is not None:
        items = items[:args.max_results]

    _save_label_cache(label_cache_path, label_cache)

    pv_count = sum(1 for r in items if r.get("has_pv_plant"))
    no_pv_count = len(items) - pv_count
    summary = {"region": args.region, "total": len(items), "pv": pv_count, "no_pv": no_pv_count}

    output = {
        "debug": {"buildings": debug_buildings, "plants": debug_plants} if args.debug else None,
        "summary": summary,
        "parameters": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "results": items,
    }

    _ensure_parent_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Region: {args.region} | Total: {len(items)} | PV: {pv_count} | No PV: {no_pv_count} | Saved to: {args.out}")

    # Clean up state file on successful completion
    if args.state_file and os.path.exists(args.state_file):
        try:
            os.remove(args.state_file)
            logger.debug(f"Cleaned up state file: {args.state_file}")
        except Exception:
            pass

    print(json.dumps(output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
