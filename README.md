# Solar Roof Toolkit

A set of scripts for collecting building candidates, grabbing aerial screenshots, segmenting roofs with SAM3 + feature guidance, detecting solar panels with multiple models, and post-processing the masks.

## Setup

1) Python deps are in `requirements.txt` (includes `sam3 @ git+https://github.com/facebookresearch/sam3.git`).
2) Recommended: create a virtualenv and install via `pip install -r requirements.txt`.
3) Optional model/assets:
   - YOLO weights: `/app/solar_panel_yolov8s-seg.pt` (used by `detect_solar_panels.py`).
   - SAM3 + DINOv2/ConvNeXt are downloaded on first run of `feature_guided_sam3.py`.
4) API keys (put in `.env`):
   - `OPENAI_API_KEY` for OpenAI vision.
   - `GOOGLE_API_KEY` for Gemini.
   - `OPEN_ROUTER_API_KEY` for OpenRouter.

## Scripts

### 1) `region_building_groups.py`
Collect roof facets + nearby plants from Swiss GeoAdmin for a region or a single coordinate. Supports resume, filtering, DBSCAN clustering, and optional solar metrics.

Examples:
```bash
# Region sweep with PV/residential filtering
python region_building_groups.py \
  --region "Bern" \
  --min-roof-area 300 \
  --plant-radius 30 \
  --filter-mode all \
  --pv-only-plants \
  --neighbor-within-m 200 \
  --label-cache /app/streamlit_site/langnau/langnau_labels_cache.json \
  --tile-size-m 500 --min-tile-size-m 125 \
  --restrict-to-region-label \
  --max-results 2000 \
  --residential-only \
  --dedupe-by-esid --dedupe-by-egrid \
  --residential-gkat-codes 1020 1030 1040 \
  --include-solar-metrics --include-gwr-attrs \
  --gwr-match-mode point_match_egid \
  --label-mode gwr_prefer \
  --include-raw-results \
  --state-file /tmp/bern_state.json \
  --out /app/streamlit_site/bern/bern_pv_residential_esid_egrid.json

# Single building debug
python region_building_groups.py \
  --coord "2561977.054,1185216.497" \
  --min-roof-area 300 \
  --plant-radius 30 \
  --filter-mode no_pv \
  --debug --debug-building-id 2685921 \
  --only-label-debug-building \
  --out single_building_debug.json
```

Notes:
- Outputs JSON with `results[]`, summary stats, and optional raw API payloads.
- Uses GeoAdmin WMS/MapServer; set `--sleep-s` if you hit rate limits.
- For region mode, bbox is derived from `--region`; `--state-file` enables resume.

### 2) `get_building_screenshot.py`
Grab GeoAdmin aerial tiles for an address, coordinates, or a results JSON.

Examples:
```bash
# Address
python get_building_screenshot.py "Dorfbergstrasse 2 3000 Bern" \
  --screenshots-dir outputs --screenshot-size-m 20 --screenshot-width 800 --screenshot-height 800 --reuse-screenshot

# Coordinates
python get_building_screenshot.py \
  --y 2627021.40 --x 1198617.21 --label "MyBuilding" \
  --screenshots-dir outputs --screenshot-size-m 40 --screenshot-width 800 --screenshot-height 800

# From region results JSON (uses coordinates.y/x)
python get_building_screenshot.py \
  --input-json /app/streamlit_site/bern/bern_pv_residential_esid_egrid.json \
  --screenshots-dir outputs --screenshot-size-m 50 --screenshot-width 800 --screenshot-height 800 --reuse-screenshot --limit 10
```

Outputs JSON with per-item coords and screenshot paths.

### 3) `streamlit_site/app.py`
Streamlit UI to browse results, screenshots, and analyses.

Typical run:
```bash
cd streamlit_site
streamlit run app.py
```

Notes:
- Expects cached labels/results under `streamlit_site/bern/` as referenced in the examples above.
- Update paths in the app if you add new regions.

### 4) `feature_guided_sam3.py`
Feature-guided SAM3 segmentation using DINOv2 and/or ConvNeXt heatmaps as geometric prompts. Can optionally compare against text-only SAM3.

Examples:
```bash
# DINO-guided on one image
python feature_guided_sam3.py streamlit_site/bern/outputs2/b2188992_Dorfbergstrasse_2_50m.png

# ConvNeXt-guided on a batch
python feature_guided_sam3.py streamlit_site/bern/outputs2/*.png --guide convnext

# Both guides + text-only comparison
python feature_guided_sam3.py streamlit_site/bern/outputs2/*.png --guide both --compare --prompt "the main building"
```

Outputs:
- Vizzes in `feature_guided_outputs/viz` (overlays + comparisons if `--compare`).
- Masks in `feature_guided_outputs/masks`.
- JSON summary at `feature_guided_outputs/results.json`.

### 5) `detect_solar_panels.py`
Multi-model solar panel detection with retry logic.
- Models: YOLO (local), OpenAI vision, Gemini, Ollama (local vision).
- Supports tiling, ROI, per-image and batch JSON outputs, and visualizations.

Examples:
```bash
# YOLO only from JSON screenshots
python detect_solar_panels.py --input-json screenshots_run.json --limit 10 --models yolo --no-viz

# Single image with all models
python detect_solar_panels.py outputs/b2188992_Dorfbergstrasse_2_50m.png --models all

# Gemini with slower calls to avoid 429s and tire the API 
python detect_solar_panels.py outputs/*.png --models gemini --gemini-delay-between 20 --gemini-max-retries 5
```

Notes:
- Set `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `OPEN_ROUTER_API_KEY` for API models.
- YOLO uses `/app/solar_panel_yolov8s-seg.pt`; adjust via `--yolo-model`.
- Ollama needs a running server; choose any vision-capable model via `--ollama-model`.

### 6) `crop_and_clean_image.py`
Clean SAM3 visualization overlays and optionally crop to the mask bounds.

Example:
```bash
python crop_and_clean_image.py \
  --original streamlit_site/bern/outputs2/b2188992_Dorfbergstrasse_2_50m.png \
  --mask feature_guided_outputs/masks/b2188992_Dorfbergstrasse_2_50m_guided_convnext_mask.png \
  --output-dir sam3_outputs/cleaned --crop --padding 10 --background transparent
```

Outputs cleaned PNGs (transparent/white/black background) in `sam3_outputs/cleaned` by default.

## Tips
- Large downloads (SAM3, DINOv2/ConvNeXt) happen on first run; ensure GPU if using `--device cuda`.
- GeoAdmin calls can rate limit; use `--sleep-s` and `--progress-every-pct` to throttle.
- Keep heavy outputs in ignored dirs (`sam3_outputs/`, `feature_guided_outputs/`, etc.).
