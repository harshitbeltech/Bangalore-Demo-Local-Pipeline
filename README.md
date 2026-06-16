# Bangalore Demo — Live Multi-Stream Violation + LPR Pipeline

Live, in-memory pipeline that ingests RTSP (or replays video files) for many cameras,
detects + tracks vehicles, runs traffic-violation models, runs LPR **only on violators**,
saves two evidence images per violation to Google Drive (with a local fallback), deletes
everything else, and appends a row per violation to a Google Sheet + local CSV.

See `docs/ARCHITECTURE.md` for the full design.

## Quick start

```bash
source /home/cv-gpu-2/harshit_workspace/.venv/bin/activate
pip install -r requirements.txt          # google-api-python-client, gspread, PyYAML, ...

# 1) Make sure the Triton server is running (serves yolo_infer, feature_extractor,
#    number_yolo, ocr_model_vit, seatbelt_infer, helmet_infer, helmet_classifier,
#    phone_infer, hsrp_infer, side_view_infer, uncovered_vehicle).

# 2) Validate config + Triton reachability:
python run.py --check

# 3) Run (uses config/cameras.yaml — cam01 replays a goa_feed video by default):
python run.py
```

Stop with Ctrl-C; still-tracked vehicles are finalized on shutdown.

## Configuration

- `config/cameras.yaml` — one entry per camera. `kind: file` replays a local video
  (set `loop: true`); `kind: rtsp` connects to a live URL. Toggle `enabled`. Add up to 22.
- `config/settings.yaml` — Triton URL, target FPS, batch sizes, track thresholds,
  Drive folder / Sheet IDs, output dir.

## Google Drive / Sheets

Without credentials the pipeline runs in **local fallback**: evidence → `output/evidence/`,
rows → `output/violations.csv`. To go live, follow `credentials/README.md` (service account),
then set `evidence_folder_id` and `sheet_id` in `config/settings.yaml`. No code change needed.

## Output columns

`Timestamp | Number plate | Seatbelt | Helmet | Triple rider | Phone user | Uncovered |
HSRP | Side view | Evidence image (violation) | Evidence image (plate)`

Both evidence images are **full frames** (the plate image is not cropped, so it keeps scene context).

A row is written **only when a vehicle has ≥1 vision violation**.

## Notes / limits (single RTX 3090)

- GPU memory is fine (~6.4 / 24 GB with all models). Detection batches across cameras.
- Per-vehicle violation models are the bottleneck; one 3090 is marginal at full 22-cam
  density. Tune on 1–4 cameras first (just edit `cameras.yaml`).
- `VideoSource` uses OpenCV/FFMPEG **CPU** decode; 22 live H.264 streams will saturate the
  8-core CPU. NVDEC/GStreamer decode is a future optimization.
- HSRP and "Fake number plate" are enrichment columns computed only after a vision violation
  triggers LPR (they need the plate). They cannot, by design, trigger a row on their own.
