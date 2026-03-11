#!/usr/bin/env python3
"""
Export solar lead list to CSV.

Merges buildings JSON (Stage 1) with detections JSON (Stage 5) into one CSV.
Every field that is available in the pipeline output is included.

Usage:
    uv run python3 export_leads_csv.py \
        --buildings streamlit_site/langnau_run_2025_04/langnau_im_emmental_buildings.json \
        --detections streamlit_site/langnau_run_2025_04/langnau_im_emmental_detections.json \
        --out leads_langnau.csv

    # Stage 1 only (no AI detection yet):
    uv run python3 export_leads_csv.py \\
        --buildings streamlit_site/langnau_run_2025_04/langnau_im_emmental_buildings.json \\
        --out leads_langnau.csv

    # Only HOT + WARM leads:
    uv run python3 export_leads_csv.py ... --hot-only
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# ── GWR lookup tables ─────────────────────────────────────────────────────────

HEATING_LABELS = {
    7410: "Heat pump",
    7411: "Heat pump (air/water)",
    7412: "Heat pump (ground/water)",
    7413: "Heat pump (water/water)",
    7420: "Electric direct",
    7430: "Boiler (general)",
    7431: "Gas boiler",
    7432: "Oil boiler",
    7433: "Wood boiler",
    7436: "Gas condensing boiler",
    7437: "Oil condensing boiler",
    7440: "Stove / room heater",
    7450: "Underfloor heating",
    7460: "District heating",
    7470: "District cooling",
    7480: "Process heat",
    7499: "Other",
}

ENERGY_SOURCE_LABELS = {
    7500: "None",
    7510: "Air",
    7511: "Geothermal probe",
    7512: "Geothermal surface",
    7513: "Water (ground/lake/river)",
    7520: "Gas",
    7530: "Heating oil",
    7540: "Wood (pellets/chips/logs)",
    7550: "Waste heat",
    7560: "Electricity",
    7570: "District heating",
    7580: "Solar thermal",
    7590: "Biogas",
    7598: "Unknown",
    7599: "Other",
}

HOT_WATER_SYSTEM_LABELS = {
    7610: "Electric boiler",
    7620: "Heat pump boiler",
    7630: "Central / collective system",
    7640: "Heat exchanger / district",
    7650: "No hot water / cold water",
    7699: "Other",
}

BUILDING_STATUS_LABELS = {
    1001: "Planned",
    1002: "Construction approved",
    1003: "Under construction",
    1004: "Existing",
    1005: "Not usable",
    1007: "Demolished",
    1008: "Not built",
}

BUILDING_CATEGORY_LABELS = {
    1010: "Residential building (generic)",
    1020: "Detached single-family house",
    1021: "Farm house with residential use",
    1030: "Multi-unit residential building",
    1040: "Apartment block",
    1060: "Farm / agricultural building",
    1080: "Special residential (home, hospital)",
    1110: "Office building",
    1130: "Commercial / trade building",
    1140: "Industrial building",
    1160: "Storage building",
    1220: "School / educational",
    1230: "Cultural / religious",
    1241: "Sports hall",
    1242: "Outdoor sports facility",
    1251: "Hotel / restaurant",
    1261: "Parking structure",
    1264: "Other non-residential",
    1275: "Greenhouse / garden building",
}

BUILDING_CLASS_LABELS = {
    1110: "Detached single-family house",
    1121: "Semi-detached single-family house",
    1122: "Terraced house (end unit)",
    1123: "Terraced house (middle unit)",
    1130: "Multi-family house (2 apartments)",
    1211: "Multi-family house (3+ apartments)",
    1212: "Multi-family house with commercial use",
    1220: "Farm house",
    1230: "Summer house / chalet",
    1231: "Allotment garden house",
    1241: "Building with living quarters (hotel etc.)",
}

BUILDING_PERIOD_LABELS = {
    8011: "Before 1919",
    8012: "1919-1945",
    8013: "1946-1960",
    8014: "1961-1970",
    8015: "1971-1980",
    8016: "1981-1985",
    8017: "1986-1990",
    8018: "1991-1995",
    8019: "1996-2000",
    8020: "2001-2005",
    8021: "2006-2010",
    8022: "2011-2015",
    8023: "2016-2020",
    8025: "2021 or later",
}

SOLAR_CLASS_LABELS = {
    1: "Very good",
    2: "Good",
    3: "Moderate",
    4: "Suitable",
    5: "Poor",
}

KITCHEN_LABELS = {
    3100: "Kitchen",
    3110: "Kitchenette",
    3200: "No kitchen",
}


def _orientation_label(deg) -> str:
    if deg is None:
        return ""
    az = float(deg)
    if az < 0:
        az += 360
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((az + 11.25) / 22.5) % 16]


def _extract_building_id_from_path(image_path: str):
    m = re.search(r"\bb(\d+)_", str(image_path))
    return int(m.group(1)) if m else None



def _lead_score(row: dict) -> int:
    score = 0
    energy = row.get("energy_source_code")
    if energy in (7530, 7520):
        score += 2
    if row.get("has_existing_pv") is False:
        score += 2
    klasse = row.get("solar_class")
    if klasse in (1, 2):
        score += 2
    elif klasse == 3:
        score += 1
    try:
        az = float(row["avg_orientation_deg"])
        if az < 0:
            az += 360
        if 150 <= az <= 210:
            score += 1
    except (TypeError, ValueError, KeyError):
        pass
    if (row.get("roof_area_m2") or 0) > 80:
        score += 1
    try:
        if int(row["year_built"]) < 2000:
            score += 1
    except (TypeError, ValueError, KeyError):
        # Fall back to building period if exact year missing
        period = row.get("building_period_code")
        if period and period <= 8019:   # up to 1996-2000
            score += 1
    return min(score, 10)


def _tier(score: int) -> str:
    if score >= 8:
        return "HOT"
    if score >= 5:
        return "WARM"
    return "COLD"


def load_buildings(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for item in data.get("results", []):
        bid = item.get("building_id")
        if bid is None:
            continue

        raw = item.get("_raw") or {}
        gwr_results = raw.get("gwr_results") or []
        g = gwr_results[0] if gwr_results else {}   # full GWR record

        gwr  = item.get("gwr") or {}                # summarised GWR from pipeline
        coords = item.get("coordinates") or {}
        suit   = item.get("suitability") or {}
        az     = item.get("avg_orientation_deg")

        heating1_code  = g.get("gwaerzh1")
        energy1_code   = g.get("genh1")
        heating2_code  = g.get("gwaerzh2")
        energy2_code   = g.get("genh2")
        hotwater_code  = g.get("gwaerzw1")
        hotwater_e_code = g.get("genw1")

        # Plants summary
        plants = item.get("plants_within_radius") or []
        plant_names = ", ".join(
            p.get("sub_category_en") or p.get("address") or ""
            for p in plants if isinstance(p, dict)
        ) or None

        row = {
            # ── Identification ──────────────────────────────────────────
            "building_id":            bid,
            "egid":                   gwr.get("egid") or g.get("egid"),
            "egrid":                  gwr.get("egrid") or g.get("egrid"),
            "esid":                   gwr.get("esid") or g.get("esid"),
            "parcel_number":          g.get("lparz"),
            "address":                item.get("label"),
            "street_number":          g.get("deinr"),
            "postcode":               g.get("dplz4"),
            "municipality":           g.get("dplzname") or gwr.get("ggdename"),
            "municipality_code":      g.get("ggdenr"),
            "canton":                 g.get("gdekt"),
            "coord_y_lv95":           coords.get("y"),
            "coord_x_lv95":           coords.get("x"),

            # ── Building characteristics (GWR) ──────────────────────────
            "building_status_code":   g.get("gstat"),
            "building_status":        BUILDING_STATUS_LABELS.get(g.get("gstat")),
            "building_category_code": g.get("gkat") or gwr.get("gkat"),
            "building_category":      BUILDING_CATEGORY_LABELS.get(g.get("gkat") or gwr.get("gkat")),
            "building_class_code":    g.get("gklas"),
            "building_class":         BUILDING_CLASS_LABELS.get(g.get("gklas")),
            "year_built":             g.get("gbauj"),
            "building_period_code":   g.get("gbaup"),
            "building_period":        BUILDING_PERIOD_LABELS.get(g.get("gbaup")),
            "floors":                 g.get("gastw"),
            "num_apartments":         g.get("ganzwhg"),
            "footprint_m2":           g.get("garea"),
            "volume_m3":              g.get("gvol"),
            "protected_building":     (None if g.get("gschutzr") is None
                                      else ("No" if g.get("gschutzr") == 0
                                      else f"Yes (code {g.get('gschutzr')})")),

            # ── Primary heating system ───────────────────────────────────
            "heating_system_code":    heating1_code,
            "heating_system":         HEATING_LABELS.get(heating1_code, heating1_code),
            "energy_source_code":     energy1_code,
            "energy_source":          ENERGY_SOURCE_LABELS.get(energy1_code, energy1_code),
            "heating_data_date":      g.get("gwaerdath1"),

            # ── Secondary heating system (if any) ────────────────────────
            "heating2_system_code":   heating2_code,
            "heating2_system":        HEATING_LABELS.get(heating2_code) if heating2_code else None,
            "energy2_source_code":    energy2_code,
            "energy2_source":         ENERGY_SOURCE_LABELS.get(energy2_code) if energy2_code else None,

            # ── Hot water system ─────────────────────────────────────────
            "hot_water_system_code":  hotwater_code,
            "hot_water_system":       HOT_WATER_SYSTEM_LABELS.get(hotwater_code, hotwater_code),
            "hot_water_energy_code":  hotwater_e_code,
            "hot_water_energy":       ENERGY_SOURCE_LABELS.get(hotwater_e_code, hotwater_e_code),

            # ── Apartment / unit details (lists when multiple units in building) ──
            "avg_apartment_area_m2":  round(sum(g["warea"]) / len(g["warea"]), 1)
                                      if isinstance(g.get("warea"), list) and g["warea"] else g.get("warea"),
            "rooms_range":            (f"{min(g['wazim'])}-{max(g['wazim'])}"
                                       if isinstance(g.get("wazim"), list) and len(g["wazim"]) > 1
                                       else (g["wazim"][0] if isinstance(g.get("wazim"), list) and g["wazim"] else g.get("wazim"))),
            "total_rooms":            sum(g["wazim"]) if isinstance(g.get("wazim"), list) else g.get("wazim"),

            # ── Solar potential (Sonnendach.ch) ──────────────────────────
            "roof_area_m2":               item.get("roof_area_m2"),
            "solar_class":                suit.get("best_klasse"),
            "solar_class_label":          SOLAR_CLASS_LABELS.get(suit.get("best_klasse")),
            "avg_slope_deg":              item.get("avg_slope_deg"),
            "avg_orientation_deg":        az,
            "orientation_label":          _orientation_label(az),
            "irradiation_kwh_m2_yr":      item.get("avg_mstrahlung_kwh_m2_year"),
            "total_global_radiation_kwh": item.get("sum_gstrahlung"),
            "estimated_yield_kwh_yr":     item.get("sum_stromertrag_kwh_year"),
            "estimated_kwp":              round(item["sum_stromertrag_kwh_year"] / 1000, 1)
                                          if item.get("sum_stromertrag_kwh_year") else None,

            # ── Existing power plants nearby ─────────────────────────────
            "has_any_plant":          item.get("has_any_plant"),
            "has_existing_pv":        item.get("has_pv_plant"),
            "plant_search_radius_m":  item.get("plant_radius_m"),
            "nearby_plants":          plant_names,
            "_plants":                plants,   # temp, stripped before CSV write

            # ── AI detection (filled by merge_detections) ────────────────
            "yolo_has_solar":         None,
            "yolo_confidence":        None,
            "yolo_num_detections":    None,
            "ollama_has_solar":       None,
            "ollama_confidence":      None,
            "ollama_explanation":     None,
            "gemini_has_solar":       None,
            "gemini_confidence":      None,
            "ai_consensus":           "not_run",

            # ── Lead scoring (filled after all rows loaded) ───────────────
            "lead_score":             None,
            "lead_tier":              None,
        }
        out[bid] = row

    return out


def merge_detections(rows: dict, detections_path: Path) -> None:
    with open(detections_path, encoding="utf-8") as f:
        data = json.load(f)

    for det in data.get("results", []):
        bid = _extract_building_id_from_path(det.get("image_path", ""))
        if bid not in rows:
            continue
        row = rows[bid]

        yolo = det.get("yolo") or {}
        if yolo:
            row["yolo_has_solar"]      = yolo.get("has_solar_panel")
            row["yolo_confidence"]     = yolo.get("confidence")
            row["yolo_num_detections"] = yolo.get("num_detections")

        ollama = det.get("ollama") or {}
        if ollama:
            row["ollama_has_solar"]    = ollama.get("has_solar_panel")
            row["ollama_confidence"]   = ollama.get("confidence")
            explanation = ollama.get("explanation") or ""
            row["ollama_explanation"]  = explanation[:300] if explanation else None

        gemini = det.get("gemini") or {}
        if gemini:
            row["gemini_has_solar"]    = gemini.get("has_solar_panel")
            row["gemini_confidence"]   = gemini.get("confidence")

        votes = [
            row.get(f"{m}_has_solar")
            for m in ("yolo", "ollama", "gemini")
            if row.get(f"{m}_has_solar") is not None
        ]
        if not votes:
            row["ai_consensus"] = "not_run"
        elif all(v is True for v in votes):
            row["ai_consensus"] = "yes"
        elif all(v is False for v in votes):
            row["ai_consensus"] = "no"
        elif any(v is True for v in votes):
            row["ai_consensus"] = "unclear"
        else:
            row["ai_consensus"] = "no"



# How many decimal places each float column should be rounded to.
# Columns not listed are written as-is (integers, strings, booleans).
FLOAT_PRECISION = {
    "coord_y_lv95":           1,
    "coord_x_lv95":           1,
    "avg_slope_deg":          1,
    "avg_orientation_deg":    1,
    "roof_area_m2":           1,
    "footprint_m2":           1,
    "avg_apartment_area_m2":  1,
    "irradiation_kwh_m2_yr":  1,
    "total_global_radiation_kwh": 1,
    "estimated_yield_kwh_yr": 1,
    "estimated_kwp":          1,
    "plant_search_radius_m":  1,
    "installed_kwp_registry": 1,
    "remaining_kwp_potential":1,
    "yolo_confidence":        3,
    "ollama_confidence":      3,
    "gemini_confidence":      3,
}


def _reformat_date(v) -> str:
    """Convert DD.MM.YYYY to YYYY-MM-DD so European Excel doesn't mangle the dots."""
    if not v or not isinstance(v, str):
        return v
    parts = v.strip().split(".")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        dd, mm, yyyy = parts
        return f"{yyyy}-{mm}-{dd}"
    return v


DATE_COLUMNS = {"heating_data_date"}


def _clean_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k in DATE_COLUMNS:
            out[k] = _reformat_date(v)
        elif k in FLOAT_PRECISION and v is not None:
            try:
                out[k] = f"{float(v):.{FLOAT_PRECISION[k]}f}".replace('.', ',')
            except (TypeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out


LEGEND = """\
SOLAR LEAD LIST - COLUMN REFERENCE
===================================

IDENTIFICATION
  building_id               integer   Internal GeoAdmin building identifier
  egid                      integer   Federal Building ID (EGID). Unique across all of Switzerland.
  egrid                     text      Land register parcel ID (EGRID). Used for property lookups.
  esid                      integer   Federal street address entry ID (ESID).
  parcel_number             text      Cadastral parcel number within the municipality.
  address                   text      Full street address.
  street_number             text      House number only.
  postcode                  integer   Swiss postcode (PLZ).
  municipality              text      Municipality name.
  municipality_code         integer   BFS municipality number.
  canton                    text      Canton abbreviation (e.g. BE, ZH, VD).
  coord_y_lv95              decimal   Easting coordinate in Swiss LV95 / EPSG:2056 (metres).
  coord_x_lv95              decimal   Northing coordinate in Swiss LV95 / EPSG:2056 (metres).

BUILDING  (source: Swiss Federal Building Register, GWR)
  building_status_code      integer   GWR status code. 1004 = existing building (normal case).
  building_status           text      Human-readable status (Existing / Under construction / etc.).
  building_category_code    integer   GKAT code. 1020 = detached house, 1040 = apartment block.
  building_category         text      Human-readable building type.
  building_class_code       integer   GKLAS code. More detailed sub-type of the building category.
  building_class            text      Human-readable building class.
  year_built                integer   Exact year of construction (gbauj). Often missing for older buildings.
  building_period_code      integer   GBAUP code. Construction decade when exact year is unknown.
  building_period           text      Human-readable period (e.g. "1971-1980").
  floors                    integer   Number of floors above ground (gastw).
  num_apartments            integer   Number of residential units in the building (ganzwhg).
  footprint_m2              decimal   Ground-floor footprint area in m2 (garea).
  volume_m3                 decimal   Building volume in m3 (gvol). Often not recorded.
  protected_building        text      Heritage protection status. "No" = explicitly not protected.
                                      Blank = not recorded in GWR (also means no known protection).
                                      "Yes (code X)" = protected building — check before any roof work.

HEATING  (source: GWR)
  heating_system_code       integer   GWR code for the primary heating system (gwaerzh1).
  heating_system            text      Human-readable heating system name.
  energy_source_code        integer   GWR code for the primary energy carrier (genh1).
                                      7530 = Heating oil, 7520 = Gas (both are strong sales leads).
                                      7580 = Solar thermal (already partially solar).
  energy_source             text      Human-readable energy source name.
  heating_data_date         text      Date the heating data was last recorded in GWR.
  heating2_system_code      integer   Secondary heating system code (gwaerzh2), if any.
  heating2_system           text      Human-readable secondary heating system.
  energy2_source_code       integer   Secondary energy source code (genh2), if any.
  energy2_source            text      Human-readable secondary energy source.

HOT WATER  (source: GWR)
  hot_water_system_code     integer   GWR code for the hot water system (gwaerzw1).
  hot_water_system          text      Human-readable hot water system.
  hot_water_energy_code     integer   GWR code for the hot water energy source (genw1).
  hot_water_energy          text      Human-readable hot water energy source.
                                      If this runs on oil or electricity, solar thermal is a strong upsell.

APARTMENT DETAILS  (source: GWR unit-level records)
  avg_apartment_area_m2     decimal   Average apartment area across all units in the building (m2).
  rooms_range               text      Room count per apartment. Single value if uniform (e.g. "4"),
                                      range if mixed (e.g. "2-5").
  total_rooms               integer   Sum of all rooms across all apartments in the building.

SOLAR POTENTIAL  (source: Sonnendach.ch / Swiss federal solar atlas)
  roof_area_m2              decimal   Total suitable roof area across all facets (m2).
  solar_class               integer   Best suitability class on this roof. 1 = Very good, 5 = Poor.
  solar_class_label         text      Human-readable class name.
  avg_slope_deg             decimal   Average roof slope in degrees. Optimal range is 20-35 degrees.
  avg_orientation_deg       decimal   Average azimuth. 180 = south (best). 0 = north (worst).
  orientation_label         text      Compass direction (S, SE, SW, N, etc.).
  irradiation_kwh_m2_yr     decimal   Average annual solar irradiation on the roof in kWh/m2/year.
                                      Swiss average is around 1100. Above 1200 is excellent.
  total_global_radiation_kwh decimal  Total annual global radiation summed across all roof facets (kWh).
  estimated_yield_kwh_yr    decimal   Sonnendach estimate of annual electricity production if all
                                      suitable roof surfaces were covered with panels (kWh/year).
                                      This is the theoretical maximum for the building.
  estimated_kwp             decimal   Estimated peak installation size derived from yield / 1000
                                      full-load hours (kWp). Rule of thumb: 1 kWp needs 5-8 m2 of panels.

EXISTING INSTALLATIONS  (source: Swiss power plant registry, ch.bfe.elektrizitaetsproduktionsanlagen)
  has_any_plant             boolean   True if any registered power plant is within the search radius.
  has_existing_pv           boolean   True if a registered photovoltaic plant is within the search radius.
  plant_search_radius_m     decimal   Search radius used to find nearby plants (metres).
  nearby_plants             text      Names or types of plants found nearby, comma-separated.
  installed_kwp_registry    decimal   Total registered installed capacity of nearby PV plants (kWp).
                                      Null if no registered plants found.
  remaining_kwp_potential   decimal   Difference between estimated_kwp and installed_kwp_registry.
                                      Represents the theoretical expansion headroom still available.

AI DETECTION  (source: aerial photo analysis - Stage 5 of the pipeline)
  yolo_has_solar            boolean   Whether the YOLO computer vision model detected solar panels.
  yolo_confidence           decimal   YOLO detection confidence score (0.000 to 1.000).
  yolo_num_detections       integer   Number of separate panel detections YOLO found.
  ollama_has_solar          boolean   Whether the Ollama local vision-language model detected panels.
  ollama_confidence         decimal   Ollama confidence score (0.000 to 1.000).
  ollama_explanation        text      Plain-language explanation from Ollama describing what it saw
                                      on the roof (truncated to 300 characters).
  gemini_has_solar          boolean   Whether Google Gemini detected solar panels.
  gemini_confidence         decimal   Gemini confidence score (0.000 to 1.000).
  ai_consensus              text      Combined verdict across all models that ran.
                                      yes = at least one model detected panels.
                                      no = all models agree no panels present.
                                      unclear = models disagree.
                                      not_run = no AI detection was performed.

LEAD SCORING
  lead_score                integer   Composite score from 0 to 10. Higher is a better prospect.
                                      Scoring breakdown:
                                        +2  primary energy source is oil or gas
                                        +2  no existing PV registered nearby
                                        +2  solar class 1 or 2 (best suitability)
                                        +1  solar class 3
                                        +1  south-facing roof (azimuth 150-210 degrees)
                                        +1  roof area above 80 m2
                                        +1  building built before 2000
  lead_tier                 text      HOT (score 8-10), WARM (score 5-7), COLD (score 0-4).
"""


COLUMNS = [
    # Identification
    "building_id", "egid", "egrid", "esid", "parcel_number",
    "address", "street_number", "postcode", "municipality", "municipality_code", "canton",
    "coord_y_lv95", "coord_x_lv95",
    # Building
    "building_status_code", "building_status",
    "building_category_code", "building_category",
    "building_class_code", "building_class",
    "year_built", "building_period_code", "building_period",
    "floors", "num_apartments", "footprint_m2", "volume_m3", "protected_building",
    # Heating
    "heating_system_code", "heating_system",
    "energy_source_code", "energy_source",
    "heating_data_date",
    "heating2_system_code", "heating2_system",
    "energy2_source_code", "energy2_source",
    # Hot water
    "hot_water_system_code", "hot_water_system",
    "hot_water_energy_code", "hot_water_energy",
    # Apartment details
    "avg_apartment_area_m2", "rooms_range", "total_rooms",
    # Solar potential
    "roof_area_m2", "solar_class", "solar_class_label",
    "avg_slope_deg", "avg_orientation_deg", "orientation_label",
    "irradiation_kwh_m2_yr", "total_global_radiation_kwh",
    "estimated_yield_kwh_yr", "estimated_kwp",
    # Existing plants
    "has_any_plant", "has_existing_pv", "plant_search_radius_m", "nearby_plants",
    # Existing installation capacity (from power plant registry)
    "installed_kwp_registry", "remaining_kwp_potential",
    # AI detection
    "yolo_has_solar", "yolo_confidence", "yolo_num_detections",
    "ollama_has_solar", "ollama_confidence", "ollama_explanation",
    "gemini_has_solar", "gemini_confidence",
    "ai_consensus",
    # Scoring
    "lead_score", "lead_tier",
]


def main():
    p = argparse.ArgumentParser(description="Export solar lead list to CSV")
    p.add_argument("--buildings",  required=True, help="Path to *_buildings.json")
    p.add_argument("--detections", default=None,  help="Path to *_detections.json (optional)")
    p.add_argument("--out",        default="solar_leads.csv", help="Output CSV path")
    p.add_argument("--hot-only",   action="store_true", help="Only export HOT + WARM leads")
    args = p.parse_args()

    buildings_path = Path(args.buildings)
    if not buildings_path.exists():
        sys.exit(f"ERROR: buildings file not found: {buildings_path}")

    print(f"Loading buildings: {buildings_path}")
    rows = load_buildings(buildings_path)
    print(f"  {len(rows)} buildings loaded")

    if args.detections:
        det_path = Path(args.detections)
        if not det_path.exists():
            print(f"  WARNING: detections file not found, skipping: {det_path}")
        else:
            print(f"Loading detections: {det_path}")
            merge_detections(rows, det_path)

    # Fill installed/remaining kWp from plant registry
    for row in rows.values():
        installed_kw = 0.0
        for p in row.pop("_plants", []):
            try:
                installed_kw += float(str(p.get("total_power") or 0).replace(",", ".").replace(" ", ""))
            except ValueError:
                pass
        if installed_kw > 0:
            row["installed_kwp_registry"] = round(installed_kw, 1)
            row["remaining_kwp_potential"] = round(max(0.0, (row.get("estimated_kwp") or 0) - installed_kw), 1)
        else:
            row["installed_kwp_registry"] = None
            row["remaining_kwp_potential"] = None

    for row in rows.values():
        row["lead_score"] = _lead_score(row)
        row["lead_tier"]  = _tier(row["lead_score"])

    all_rows = sorted(rows.values(), key=lambda r: r["lead_score"], reverse=True)
    if args.hot_only:
        all_rows = [r for r in all_rows if r["lead_tier"] in ("HOT", "WARM")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig writes a BOM so Excel auto-detects UTF-8 encoding.
    # Semicolon delimiter is standard for European (Swiss/German/French) Excel.
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", delimiter=";")
        writer.writeheader()
        writer.writerows(_clean_row(r) for r in all_rows)

    legend_path = out_path.with_name(out_path.stem + "_legend.txt")
    with open(legend_path, "w", encoding="utf-8") as f:
        f.write(LEGEND)

    hot      = sum(1 for r in all_rows if r["lead_tier"] == "HOT")
    warm     = sum(1 for r in all_rows if r["lead_tier"] == "WARM")
    cold     = sum(1 for r in all_rows if r["lead_tier"] == "COLD")
    oil_gas  = sum(1 for r in all_rows if r.get("energy_source_code") in (7530, 7520))
    detected = sum(1 for r in all_rows if r.get("ai_consensus") in ("yes", "unclear"))

    print(f"\nExported {len(all_rows)} rows ({len(COLUMNS)} columns) to {out_path}")
    print(f"Legend written to {legend_path}")
    print(f"  HOT: {hot}  WARM: {warm}  COLD: {cold}")
    print(f"  Oil/gas heated buildings: {oil_gas}")
    print(f"  AI detected possible solar: {detected}")


if __name__ == "__main__":
    main()
