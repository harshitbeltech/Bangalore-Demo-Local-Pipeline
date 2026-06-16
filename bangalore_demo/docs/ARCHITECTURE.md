# Bangalore Demo — Live Multi-Stream Violation + LPR Pipeline

**Status:** design approved 2026-06-03. Target: live RTSP (or file-replay) for up to 22 cameras
on a single RTX 3090, tuned first on 1–4 streams.

## 1. Why a new pipeline

`vit_lpr_testing/run_pipeline_new.py` is **offline/batch/disk-backed/single-threaded**: it extracts
*every* frame of one video to disk, then runs each stage over all frames serially. That cannot serve
22 concurrent live streams, and it would fill the (already 98%-full) disk in minutes.

`bangalore_demo/` inverts this: **frames live in RAM**, stages run concurrently across cameras, and
**only two JPEGs ever touch disk per violation** (the violation evidence frame + the plate crop),
which are deleted immediately after they are uploaded to Google Drive.

## 2. Data flow

```
[Cam i ingest thread]  RTSP/file -> decode -> drop stale -> sample to TARGET_FPS (5)
        | (cam_id, frame, ts) into bounded queue (drop-oldest backpressure)
        v
[ODN detector]   batch frames across cameras -> yolo_infer -> detections
        v
[Tracker]        per-camera DeepSort (Kalman + feature_extractor) -> stable track ids
        |        each active track keeps a small in-RAM ring buffer of {frame_id, bbox, image}
        v   (track exits: miss_count > MAX_MISS)
[Violation stage]  per vehicle: seatbelt | helmet+triple+side_view | phone | hsrp | uncovered
        v
   any real violation? --no--> drop the track's frames from RAM, nothing saved
        | yes
        v
[LPR]   number_yolo -> ocr_model_vit -> OCR_post   (ONLY on violators)
        v
[Evidence]  pick best violation frame + best plate/HSRP crop -> encode 2 JPEGs to tmp
        v
[Drive]  upload both -> shareable links -> DELETE the 2 local files
        v
[Sheet]  append one row -> local CSV (source of truth) + Google Sheet
```

### Key decisions
- **Bounded queues, drop-oldest.** A slow stage sheds frames (logged) instead of growing RAM or
  falling behind real time. We never guarantee every frame is processed — we guarantee real-time.
- **Evaluate-once-per-vehicle.** Violations + LPR run when a track exits (confident + gone), not on
  every frame. Sampling picks up to 16 evenly-spaced frames from the track's lifetime.
- **In-RAM ring buffer per track**, freed the instant the track exits (saved or not).
- **Worker pools, not max_workers=1.** Violation + LPR run concurrently across vehicles.
- **Reuse, don't reinvent.** Calls the exact proven Triton pre/post-processing from
  `cv_module_utils_vit_lpr` (LPR), `violation_model_utils/modules` (violations), `DeepSort_Tracker`
  (tracking), `Side_view_mirror_detection`, `uncovered_covered_vehicle_detection`. The only change
  from the batch pipeline is the frame source: in-RAM instead of `cv2.imread(local_img_path)`.

## 3. Sheet schema (one row per violating vehicle)

| # | Column | Source |
|---|--------|--------|
| 1 | Timestamp | frame time `YYYY-MM-DD HH:MM:SS` |
| 2 | Number plate | `ocr_model_vit` (LPR) |
| 3 | Seatbelt | `seatbelt_infer` -> `no` / `driver` / `driver+passenger` |
| 4 | Helmet | `helmet_infer` + `helmet_classifier` |
| 5 | Triple rider | helmet decider `overloading-two-wheeler` |
| 6 | Phone user | `phone_infer` |
| 7 | Uncovered | `uncovered_vehicle` |
| 8 | HSRP | `hsrp_infer` |
| 9 | Side view | `side_view_infer` |
| 10 | Evidence image (violation) | Drive link (full frame) |
| 11 | Evidence image (number plate) | Drive link (full frame, not cropped) |

A row is written **only when a vehicle has >= 1 real vision violation** (cols 3-9). No-violation
vehicles never hit the sheet, never upload, never persist.

Per-vehicle model routing (same as batch pipeline):
- `FOUR_WHEELERS = {car, truck, auto, bus, jcb, vehicle}` -> seatbelt
- `TWO_WHEELERS  = {bike, bicycle, man}` -> helmet + triple + side_view
- `truck` -> uncovered
- phone + hsrp -> always

## 4. Google Drive / Sheets setup (service account)

1. Create a GCP project + service account; enable **Drive API** and **Sheets API**; download its JSON key.
2. Save it at `bangalore_demo/credentials/service_account.json` (gitignored).
3. Create a Drive folder for evidence, share it with the service-account email (Editor); put the
   **folder ID** in `config/settings.yaml`.
4. Create a Google Sheet, share it with the same email (Editor); put the **sheet ID** in settings.
   (Or leave blank and the pipeline auto-creates one on first run and prints the link.)

**Local fallback:** until the key exists, the pipeline runs fully end-to-end but writes evidence to
`bangalore_demo/output/evidence/` and rows to `bangalore_demo/output/violations.csv`. Dropping the
key in flips it to live Drive/Sheets via config — no code change.

## 5. Module layout

```
bangalore_demo/
  config/settings.yaml      triton url, fps, drive folder/sheet id, batch sizes, thresholds
  config/cameras.yaml       per-camera: id, source (rtsp url or file path), enabled
  credentials/              service_account.json (gitignored)
  core/_env.py              sys.path + numpy shim + shared constants (import FIRST)
  core/config.py            yaml loaders
  core/source.py            VideoSource (rtsp/file) + ingest thread, drop-oldest sampling
  core/detector.py          ODN batched inference (yolo_infer) via ODN_Utility
  core/tracker.py           per-camera DeepSort wrapper
  core/lpr.py               per-vehicle LPR (in-memory)
  core/violations.py        per-vehicle violation models (in-memory) + row decision
  core/drive.py             Drive + Sheets client, local fallback
  core/evidence.py          encode 2 JPEGs, upload, delete locals
  core/sheet.py             CSV + Sheet append (column schema)
  pipeline.py               orchestrator: threads, queues, worker pools
  run.py                    entrypoint
```

## 6. Capacity notes (single RTX 3090)
- GPU memory with all 11 models loaded: ~6.4 / 24 GB — not a constraint.
- Detection (ODN+DeepSort) at 22x5fps: ~0.7 of a serialized GPU-second — fine with batching.
- Per-vehicle violation models are the squeeze; one 3090 is marginal at full 22-cam vehicle density.
  Tune on 1–4 cams first; scale camera count via `cameras.yaml`.
- CPU H.264 decode of 22 streams will saturate the 8-core CPU — NVDEC is a later optimization
  (current `VideoSource` uses OpenCV/FFMPEG CPU decode).
