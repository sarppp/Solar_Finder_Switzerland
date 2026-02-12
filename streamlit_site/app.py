import json
import re
from pathlib import Path
from urllib.parse import urlencode

import requests
import streamlit as st

# streamlit run app.py --server.port 8501 --server.address 0.0.0.0

PREVIEW_LIMIT = 100
PREVIEW_JSON = "streamlit_site/payerne/payerne_buildings.json"
SCREENSHOT_DIRS = ["streamlit_site/payerne/screenshots"]
DETECTIONS_BATCH_JSON = "streamlit_site/payerne/payerne_detections.json"
YOLO_VIZ_DIRS = ["streamlit_site/payerne/detection_viz"]
# Optional per-image detection JSON directories (fallback); keep empty by default.
DETECTIONS_DIRS = []

BASE_MAP = {
    "lang": "fr",
    "topic": "energie",
    "bgLayer": "ch.swisstopo.swissimage",
    "catalogNodes": "2419,2420,2427,2480,2429,2431,2434,2436,2767,2441,3206",
    "layers": "ch.swisstopo.amtliches-strassenverzeichnis,ch.bfe.solarenergie-eignung-daecher",
    "layers_opacity": "0.85,0.65",
    "zoom": "12",
}

PLAIN_MAP = {
    "lang": "fr",
    "topic": "energie",
    "bgLayer": "ch.swisstopo.swissimage",
    "zoom": "12",
}


def to_lv03(y: float, x: float) -> tuple[float, float]:
    """Convert LV95 coordinates to LV03 if needed.

    Convention in this codebase:
    - y ~= easting
    - x ~= northing
    """
    y2 = float(y)
    x2 = float(x)
    if y2 > 1_000_000:
        y2 -= 2_000_000
    if x2 > 1_000_000:
        x2 -= 1_000_000
    return y2, x2


def slugify_label(label: str) -> str:
    s = re.sub(r"<[^>]+>", " ", str(label or ""))
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s


@st.cache_data(show_spinner=False)
def find_screenshot_path(label: str, building_id: int | None = None) -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    dirs = [repo_root / str(d) for d in SCREENSHOT_DIRS]

    slug = slugify_label(label)

    for d in dirs:
        if not d.exists():
            continue
        if building_id is not None:
            bid_candidates = sorted(d.glob(f"b{int(building_id)}_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            if bid_candidates:
                return str(bid_candidates[0])

        candidates = sorted(d.glob(f"{slug}_*m.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return str(candidates[0])

    # Fallback: substring match.
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
            if slug and slug in p.name:
                return str(p)
    return None


def resolve_yolo_viz_path(viz: str | None) -> str | None:
    if not viz:
        return None

    p = Path(str(viz))
    if p.is_absolute() and p.exists():
        return str(p)

    repo_root = Path(__file__).resolve().parents[1]
    cand = repo_root / str(viz)
    if cand.exists():
        return str(cand)

    name = Path(str(viz)).name
    for d in YOLO_VIZ_DIRS:
        p2 = repo_root / str(d) / name
        if p2.exists():
            return str(p2)

    return None


@st.cache_data(show_spinner=False)
def load_detection_for_image(image_path: str) -> dict | None:
    repo_root = Path(__file__).resolve().parents[1]
    img_name = Path(image_path).name
    stem = Path(img_name).stem

    batch_path = repo_root / str(DETECTIONS_BATCH_JSON)
    if batch_path.exists():
        try:
            with open(batch_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results = data.get("results") if isinstance(data, dict) else None
            if isinstance(results, list):
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    rp = r.get("image_path")
                    if rp:
                        detection_stem = Path(str(rp)).stem
                        # Check if screenshot stem is a prefix of detection stem
                        if detection_stem.startswith(stem):
                            return {"results": [r], "num_items": 1, "models": data.get("models")}
        except Exception:
            pass

    for d in DETECTIONS_DIRS or []:
        p = repo_root / str(d) / f"solar_detection_{stem}.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def build_map_url(y: float, x: float) -> str:
    params = dict(BASE_MAP)
    y_lv03, x_lv03 = to_lv03(y, x)
    # GeoAdmin URL convention: X=northing, Y=easting.
    params["X"] = str(x_lv03)
    params["Y"] = str(y_lv03)
    return "https://map.geo.admin.ch/?" + urlencode(params)


def build_embed_url(y: float, x: float) -> str:
    params = dict(BASE_MAP)
    y_lv03, x_lv03 = to_lv03(y, x)
    params["X"] = str(x_lv03)
    params["Y"] = str(y_lv03)
    # New map viewer (2024+): embedded view is enabled via URL parameter embed=true
    # and keeps the full UI (incl. search box).
    params["embed"] = "true"
    return "https://map.geo.admin.ch/?" + urlencode(params)


def build_plain_embed_url(y: float, x: float) -> str:
    params = dict(PLAIN_MAP)
    y_lv03, x_lv03 = to_lv03(y, x)
    params["X"] = str(x_lv03)
    params["Y"] = str(y_lv03)
    params["embed"] = "true"
    return "https://map.geo.admin.ch/?" + urlencode(params)


def sanitize_item_for_display(item: dict) -> dict:
    if not isinstance(item, dict):
        return {}
    d = dict(item)
    d.pop("screenshot", None)
    return d


def add_crosshair(url: str, enabled: bool) -> str:
    if not enabled:
        return url
    # GeoAdmin supports a built-in center marker via the crosshair parameter.
    # Common values seen in the wild: crosshair=true or crosshair=cross.
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}crosshair=cross"


@st.cache_data(show_spinner=False)
def geocode(query: str) -> tuple[float, float] | None:
    resp = requests.get(
        "https://api3.geo.admin.ch/rest/services/ech/SearchServer",
        params={"searchText": query, "type": "locations"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    if not results:
        return None
    attrs = results[0].get("attrs") or {}
    y = attrs.get("y")
    x = attrs.get("x")
    if y is None or x is None:
        return None
    return float(y), float(x)


@st.cache_data(show_spinner=False)
def load_preview_items() -> list[dict]:
    repo_root = Path(__file__).resolve().parents[1]
    p = repo_root / str(PREVIEW_JSON)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results") or []
    if isinstance(results, list):
        return results[:PREVIEW_LIMIT]
    return []


def main() -> None:
    st.set_page_config(page_title="Solar Map", layout="wide")

    # Safety: if a stale "global PREVIEW_*" line existed from an earlier edit, ensure it is not present.

    if "y" not in st.session_state or "x" not in st.session_state:
        # Prefer centering on the first preview item (valid LV95 -> LV03 conversion).
        items = load_preview_items()
        if items:
            coords = (items[0].get("coordinates") or {}) if isinstance(items[0], dict) else {}
            y0 = coords.get("y")
            x0 = coords.get("x")
            if y0 is not None and x0 is not None:
                st.session_state.y = float(y0)
                st.session_state.x = float(x0)
            else:
                # Fallback: a known-valid LV03 center (easting ~600k, northing ~200k)
                st.session_state.y = 626973.0
                st.session_state.x = 198648.0
        else:
            st.session_state.y = 626973.0
            st.session_state.x = 198648.0
    if "selected_building" not in st.session_state:
        st.session_state.selected_building = None

    st.title("Solar Map")

    with st.sidebar:
        st.header("Debug")
        show_iframe = st.checkbox("Show embedded map (iframe)", value=True)
        show_plain_iframe = st.checkbox("Show plain map at bottom", value=False)
        test_geoadmin = st.checkbox("Test GeoAdmin connectivity", value=False)
        show_preview = st.checkbox("Show preview panel", value=True)

    selected = st.session_state.selected_building
    has_selected = isinstance(selected, dict) and bool(selected)

    url = add_crosshair(build_map_url(float(st.session_state.y), float(st.session_state.x)), enabled=has_selected)
    embed_url = add_crosshair(build_embed_url(float(st.session_state.y), float(st.session_state.x)), enabled=has_selected)
    plain_embed_url = add_crosshair(build_plain_embed_url(float(st.session_state.y), float(st.session_state.x)), enabled=has_selected)

    if test_geoadmin:
        st.sidebar.write("Testing...")
        try:
            r1 = requests.get(
                "https://api3.geo.admin.ch/rest/services/ech/SearchServer",
                params={"searchText": "Bern", "type": "locations"},
                timeout=10,
            )
            st.sidebar.write(f"SearchServer status: {r1.status_code}")
        except Exception as e:
            st.sidebar.error(f"SearchServer error: {e}")

        try:
            r2 = requests.get(embed_url, timeout=10)
            st.sidebar.write(f"embed.html status: {r2.status_code}")
        except Exception as e:
            st.sidebar.error(f"embed.html error: {e}")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(f"[Open in new tab]({url})")
        if show_iframe:
            st.components.v1.iframe(embed_url, height=560, scrolling=True)
            st.caption(
                "If the map area is blank/black/keeps loading, open it in a new tab (some environments block iframes)."
            )
        else:
            st.info("Iframe disabled. Use the 'Open in new tab' link above.")
        
        if show_plain_iframe:
            st.divider()
            st.markdown("**Map (without overlay)**")
            st.components.v1.iframe(plain_embed_url, height=560, scrolling=True)

        # Selected building details
        if isinstance(selected, dict) and selected:
            st.divider()
            st.subheader("Selected building")

            sel_label = selected.get("label") or ""
            coords = selected.get("coordinates") or {}
            sel_y = coords.get("y")
            sel_x = coords.get("x")
            st.markdown(sel_label, unsafe_allow_html=True)
            st.caption(f"building_id: {selected.get('building_id')} | roof_area_m2: {selected.get('roof_area_m2')}")
            st.caption(f"LV95: y={sel_y} x={sel_x}")

            bid = selected.get("building_id")
            img_path = find_screenshot_path(sel_label, building_id=bid if isinstance(bid, int) else None)
            det = load_detection_for_image(img_path) if img_path else None

            cimg, cdet = st.columns([1, 1], gap="large")
            with cimg:
                st.subheader("Screenshot")
                if img_path and Path(img_path).exists():
                    st.image(img_path, use_container_width=True)
                else:
                    st.info("No screenshot found in /app/outputs for this label.")

            with cdet:
                st.subheader("Solar detection")
                if det is None:
                    st.info("No solar_detection_*.json found for this screenshot.")
                else:
                    item0 = (det.get("results") or [{}])[0] if isinstance(det.get("results"), list) else {}

                    with st.expander("YOLO", expanded=True):
                        st.json(item0.get("yolo") or {})
                        viz = (item0.get("yolo_visualization") or {}).get("output_path")
                        viz_path = resolve_yolo_viz_path(viz)
                        if viz_path and Path(viz_path).exists():
                            st.image(str(viz_path), use_container_width=True)

                    with st.expander("OpenAI", expanded=False):
                        st.json(item0.get("openai") or {})

                    with st.expander("Gemini", expanded=False):
                        st.json(item0.get("gemini") or {})

                    with st.expander("Ollama", expanded=False):
                        st.json(item0.get("ollama") or {})

    with right:
        if show_preview:
            st.subheader("Preview")
            items = load_preview_items()
            if not items:
                st.info(f"Preview file not found or empty: {PREVIEW_JSON}")
            else:
                def _opt_label(it: dict) -> str:
                    bid = it.get("building_id")
                    lab = it.get("label") or ""
                    lab_txt = re.sub(r"<[^>]+>", " ", str(lab)).strip()
                    return f"{bid} | {lab_txt}" if bid is not None else lab_txt

                options = list(range(len(items)))
                default_idx = 0
                cur = st.session_state.get("selected_building")
                if isinstance(cur, dict) and cur in items:
                    default_idx = items.index(cur)

                if "preview_selected_idx" not in st.session_state:
                    st.session_state.preview_selected_idx = int(default_idx)

                choice = st.selectbox(
                    "Select building",
                    options=options,
                    format_func=lambda i: _opt_label(items[int(i)]),
                    index=int(st.session_state.preview_selected_idx),
                    key="_preview_selectbox",
                )

                chosen = items[int(choice)] if 0 <= int(choice) < len(items) else None
                if isinstance(chosen, dict):
                    with st.expander("Details"):
                        st.json(sanitize_item_for_display(chosen))

                    coords = chosen.get("coordinates") or {}
                    y = coords.get("y")
                    x = coords.get("x")
                    if y is not None and x is not None:
                        st.session_state.y = float(y)
                        st.session_state.x = float(x)

                    new_idx = int(choice)
                    if int(st.session_state.preview_selected_idx) != new_idx or st.session_state.selected_building != chosen:
                        st.session_state.preview_selected_idx = new_idx
                        st.session_state.selected_building = chosen
                        st.rerun()
        else:
            st.info("Preview panel hidden (enable it in the sidebar).")


if __name__ == "__main__":
    main()
