"""
Export buildings and detections to Excel.

Usage:
python streamlit_site/export_to_excel.py \
    --buildings streamlit_site/payerne/payerne_buildings.json \
    --detections streamlit_site/payerne/payerne_detections.json \
    --out streamlit_site/payerne/payerne_export.xlsx
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _flatten(obj: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    if out is None:
        out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            _flatten(v, key, out)
        return out

    if isinstance(obj, list):
        out[prefix] = json.dumps(obj, ensure_ascii=False)
        return out

    out[prefix] = obj
    return out


def _to_french_bool(v: Any) -> Any:
    if v is True:
        return "oui"
    if v is False:
        return "non"
    return v


def _convert_bools_to_french(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df2 = df.copy()
    for col in df2.columns:
        s = df2[col]
        if pd.api.types.is_bool_dtype(s):
            df2[col] = s.map(_to_french_bool)
            continue

        non_null = s.dropna()
        if not non_null.empty and non_null.map(lambda x: isinstance(x, bool)).all():
            df2[col] = s.map(_to_french_bool)

    return df2


def _extract_building_id_from_image_path(image_path: Any) -> int | None:
    if not image_path:
        return None
    m = re.search(r"\bb(\d+)_", str(image_path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _stringify_nested(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df2 = df.copy()
    for col in df2.columns:
        s = df2[col]
        if not pd.api.types.is_object_dtype(s):
            continue

        def _coerce(v: Any) -> Any:
            if isinstance(v, (dict, list)):
                return json.dumps(v, ensure_ascii=False)
            return v

        df2[col] = s.map(_coerce)
    return df2


def _load_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ ERROR reading {path}: {e}")
        sys.exit(1)


def build_buildings_rows(buildings_json: dict[str, Any]) -> list[dict[str, Any]]:
    results = buildings_json.get("results")
    if not isinstance(results, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        d = dict(item)
        d.pop("_raw", None)
        rows.append(_flatten(d))
    return rows


def _json_normalize_results(data: Any) -> pd.DataFrame:
    if not isinstance(data, dict):
        return pd.DataFrame()
    results = data.get("results")
    if not isinstance(results, list) or not results:
        return pd.DataFrame()
    return pd.json_normalize(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a minimal Excel sheet: building_id, label, and model has_solar_panel flags."
    )
    parser.add_argument(
        "--buildings",
        default="streamlit_site/payerne/payerne_buildings.json",
        help="Path to buildings JSON",
    )
    parser.add_argument(
        "--detections",
        default="streamlit_site/payerne/payerne_detections.json",
        help="Path to detections JSON",
    )
    parser.add_argument(
        "--out",
        default="streamlit_site/payerne/payerne_export.xlsx",
        help="Output .xlsx path",
    )
    args = parser.parse_args()

    # Determine paths
    repo_root = Path(__file__).resolve().parent
    buildings_path = (repo_root / args.buildings).resolve()
    detections_path = (repo_root / args.detections).resolve()
    out_path = (repo_root / args.out).resolve()

    # --- DEBUGGING OUTPUT ---
    print("\n--- Resolving Paths ---")
    print(f"Script Location: {Path(__file__).resolve()}")
    print(f"Repository Root: {repo_root}")
    print(f"Target Buildings JSON: {buildings_path}")
    print(f"Target Detections JSON: {detections_path}")
    print(f"Target Excel Output: {out_path}")
    print("-----------------------\n")

    # Strict checks so it doesn't fail silently
    if not buildings_path.exists():
        print(f"❌ ERROR: Cannot find buildings JSON at {buildings_path}")
        sys.exit(1)
        
    if not detections_path.exists():
        print(f"❌ ERROR: Cannot find detections JSON at {detections_path}")
        sys.exit(1)

    print("✅ Found JSON files. Loading data...")
    buildings_json = _load_json(buildings_path)
    detections_json = _load_json(detections_path)

    buildings_dict = buildings_json if isinstance(buildings_json, dict) else {}
    detections_dict = detections_json if isinstance(detections_json, dict) else {}

    # Minimal buildings columns
    buildings_rows = []
    for it in (buildings_dict.get("results") or []):
        if not isinstance(it, dict):
            continue
        row = {
            "building_id": it.get("building_id"),
            "label": it.get("label"),
            "has_any_plant": it.get("has_any_plant"),
            "roof_area_m2": it.get("roof_area_m2"),
            "avg_mstrahlung_kwh_m2_year": it.get("avg_mstrahlung_kwh_m2_year"),
            "sum_gstrahlung": it.get("sum_gstrahlung"),
            "sum_stromertrag_kwh_year": it.get("sum_stromertrag_kwh_year"),
        }
        buildings_rows.append(row)

    print(f"🔍 Extracted {len(buildings_rows)} buildings.")
    buildings_df = pd.DataFrame(buildings_rows)

    if buildings_df.empty:
        print("❌ ERROR: No building data found in the JSON file. Aborting.")
        sys.exit(1)

    # Flatten detections and extract model has_solar_panel
    detections_df = _json_normalize_results(detections_dict)
    print(f"🔍 Extracted {len(detections_df)} detections.")
    
    if not detections_df.empty and "image_path" in detections_df.columns:
        detections_df = detections_df.copy()
        detections_df["building_id"] = detections_df["image_path"].map(_extract_building_id_from_image_path)

    # Pick only the model has_solar_panel columns
    model_cols = {}
    for model in ["yolo", "gemini", "openai", "ollama"]:
        col = f"{model}.has_solar_panel"
        if col in detections_df.columns:
            model_cols[model] = detections_df[["building_id", col]].rename(columns={col: f"{model}_has_solar_panel"})

    # Merge model columns into buildings
    combined_df = buildings_df
    for model, df_model in model_cols.items():
        combined_df = combined_df.merge(
            df_model,
            on="building_id",
            how="left",
        )

    # Ensure the expected columns exist (add missing ones with empty)
    for model in ["yolo", "gemini", "openai", "ollama"]:
        col_name = f"{model}_has_solar_panel"
        if col_name not in combined_df.columns:
            combined_df[col_name] = None

    # Reorder columns
    ordered_cols = [
        "building_id",
        "label",
        "has_any_plant",
        "roof_area_m2",
        "avg_mstrahlung_kwh_m2_year",
        "sum_gstrahlung",
        "sum_stromertrag_kwh_year"
    ]
    ordered_cols.extend([f"{m}_has_solar_panel" for m in ["yolo", "gemini", "openai", "ollama"]])
    combined_df = combined_df[[c for c in ordered_cols if c in combined_df.columns]]

    combined_df = _convert_bools_to_french(combined_df)

    # Rename columns to French
    french_columns = {
        "building_id": "id_batiment",
        "label": "libelle",
        "has_any_plant": "installation_existante",
        "roof_area_m2": "surface_toit_m2",
        "avg_mstrahlung_kwh_m2_year": "irradiation_moy_kwh_m2_an",
        "sum_gstrahlung": "somme_irradiation_globale",
        "sum_stromertrag_kwh_year": "somme_production_elec_kwh_an",
        "yolo_has_solar_panel": "yolo_panneau_solaire",
        "gemini_has_solar_panel": "gemini_panneau_solaire",
        "openai_has_solar_panel": "openai_panneau_solaire",
        "ollama_has_solar_panel": "ollama_panneau_solaire"
    }
    combined_df.rename(columns=french_columns, inplace=True)

    print("💾 Attempting to write Excel file...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            combined_df.to_excel(writer, sheet_name="Export", index=False)
        print(f"✅ SUCCESS: Exported {len(combined_df)} rows to {out_path}")
    except ModuleNotFoundError:
        print("❌ ERROR: The 'openpyxl' library is missing. Please run: pip install openpyxl")
        sys.exit(1)
    except PermissionError:
        print(f"❌ ERROR: Permission denied. Make sure the file {out_path} is not currently open in Excel.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR writing Excel file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()