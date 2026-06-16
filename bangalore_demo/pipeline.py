"""Orchestrator: ingest → batched ODN → per-camera DeepSort → violations → LPR → evidence → sheet.

Threading model
---------------
- One CameraIngest thread per camera → its shard's bounded, drop-oldest frame queue.
- `detect_shards` detect/track threads (>=1). Each owns a disjoint subset of cameras:
  it pulls a batch from its own queue, runs ONE yolo_infer call, then routes each
  frame to that camera's DeepSort tracker (tracker state is single-threaded per shard).
  Sharding parallelises the per-frame tracking work that caps single-thread throughput.
- A ThreadPoolExecutor of `violation_workers`: each finalized vehicle is evaluated
  off-thread — phase-A violations, then LPR + HSRP only on violators, then evidence + sheet row.
- A monitor thread prints a heartbeat.
"""
import core._env  # noqa: F401

import asyncio
import contextlib
import logging
import queue as queue_mod
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from core import evidence as evidence_mod
from core import violations as violations_mod
from core.detector import OdnDetector
from core.drive import DriveSheets
from core.sheet import SheetWriter
from core.source import CameraIngest
from core.tracker import CameraTracker

logger = logging.getLogger("bangalore.pipeline")


class _Shard:
    """One detect/track unit: its own queue, ODN detector, tracker subset and thread."""
    def __init__(self, shard_id, cameras, grpc_url, qmax, tracker_kwargs):
        self.id = shard_id
        self.cameras = cameras            # list of camera dicts assigned to this shard
        self.cam_ids = {c["id"] for c in cameras}
        self.frame_q = queue_mod.Queue(maxsize=qmax)
        self.detector = OdnDetector(grpc_url)
        self.trackers = {
            c["id"]: CameraTracker(c["id"], grpc_url, **tracker_kwargs)
            for c in cameras
        }
        self.thread = None


class Pipeline:
    def __init__(self, settings: dict, cameras: list):
        self.settings = settings
        self.cameras = cameras
        self.grpc_url = settings["triton"]["grpc_url"]
        p = settings["pipeline"]
        self.target_fps = p["target_fps"]
        self.odn_batch = p["odn_batch_size"]
        self.odn_wait = p["odn_batch_wait_ms"] / 1000.0
        self.max_miss = p["max_miss"]
        self.max_life = p.get("sample_frames", 16) * 3
        self.eval_after = p.get("eval_after_frames", 6)
        self.stage_log = p.get("stage_logging", False)
        self.ingest_backend = p.get("ingest_backend", "cv2")
        self.detect_shards = max(1, int(p.get("detect_shards", 1)))

        self.stop_event = threading.Event()

        # Partition cameras across shards (round-robin) so each shard's queue is fed
        # only by its own cameras. Per-shard queue depth keeps the no-loss buffer per
        # camera roughly constant regardless of shard count.
        n_shards = min(self.detect_shards, len(cameras)) or 1
        groups = [[] for _ in range(n_shards)]
        for i, c in enumerate(cameras):
            groups[i % n_shards].append(c)
        tracker_kwargs = dict(max_miss=self.max_miss, max_life=self.max_life,
                              eval_after=self.eval_after, stage_log=self.stage_log)
        self.shards = [
            # per-shard queue sized to THIS shard's cameras, so sharding doesn't
            # multiply the worst-case frame-buffer RAM.
            _Shard(i, g, self.grpc_url,
                   max(p["ingest_queue_max"] * len(g), p["ingest_queue_max"]),
                   tracker_kwargs)
            for i, g in enumerate(groups) if g
        ]
        self.cam_to_shard = {c["id"]: s for s in self.shards for c in s.cameras}

        self.drive = DriveSheets(settings)
        self.sheet = SheetWriter(settings, self.drive)
        self.vpool = ThreadPoolExecutor(max_workers=p["violation_workers"],
                                        thread_name_prefix="viol")
        self.jpeg_quality = settings["output"]["jpeg_quality"]

        # Global GPU lock: serializes Triton inference so detection never overlaps a
        # vehicle's violation/LPR burst. Disabled by default now (serialize_gpu=false):
        # in-process concurrency to Triton is safe (every evaluate_vehicle already fires
        # 4 models at once; each caller uses a distinct gRPC client), and the lock left
        # the GPU idle (~5%) while frames dropped. With it off, detection, the shards and
        # eval workers all use the GPU concurrently.
        self.serialize_gpu = p.get("serialize_gpu", True)
        self.gpu_lock = threading.Lock()

        # counters
        self.ingests = []
        self._frames = 0
        self._vehicles = 0
        self._rows = 0
        self._gpu_faults = 0
        self._lock = threading.Lock()

    def _gpu(self):
        return self.gpu_lock if self.serialize_gpu else contextlib.nullcontext()

    # ── vehicle evaluation (worker) ──────────────────────────────────────────
    def _handle_vehicle(self, vdata: dict):
        try:
            with self._gpu():  # hold the GPU while this vehicle's models + LPR run
                result = violations_mod.evaluate_vehicle(vdata, self.grpc_url, self.settings)
            if not result:
                return
            try:
                viol_link, plate_link = evidence_mod.save_evidence(
                    result, self.drive, self.jpeg_quality)
            except Exception as e:
                # Never lose the row to a Drive failure — CSV is the source of truth.
                logger.warning(f"evidence upload failed (row still recorded): {e}")
                viol_link, plate_link = "", ""
            self.sheet.write(result, viol_link, plate_link)
            with self._lock:
                self._rows += 1
        except Exception as e:
            logger.warning(f"vehicle eval error: {e}")
        finally:
            vdata.clear()  # free in-RAM frames immediately

    # ── batched detect + track loop (one per shard) ──────────────────────────
    def _collect_batch(self, shard):
        batch = []
        try:
            batch.append(shard.frame_q.get(timeout=0.5))
        except queue_mod.Empty:
            return batch
        deadline = time.time() + self.odn_wait
        while len(batch) < self.odn_batch and time.time() < deadline:
            try:
                batch.append(shard.frame_q.get_nowait())
            except queue_mod.Empty:
                break
        return batch

    def _detect_track_loop(self, shard):
        # One shared event loop, created + owned by THIS thread, drives this shard's
        # DeepSort trackers. grpc.aio binds to it, so trackers must be initialised here.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for tr in shard.trackers.values():
            tr.init_in_thread(loop)

        while not self.stop_event.is_set():
            batch = self._collect_batch(shard)
            if not batch:
                continue
            frames = [f for (_, _, f) in batch]
            finalized_all = []
            try:
                with self._gpu():  # detection + tracking hold the GPU together
                    dets_list = shard.detector.detect_batch(frames)
                    for (cam_id, frame_id, frame), dets in zip(batch, dets_list):
                        if self.stage_log:
                            classes = ",".join(sorted({d.get("class_name", "?") for d in dets})) or "-"
                            logger.info("[%s f%s] ODN: %d detections (%s)",
                                        cam_id, frame_id, len(dets), classes)
                        finalized = shard.trackers[cam_id].update(frame_id, frame, dets)
                        for vid, vdata in finalized:
                            vdata["cam_id"] = cam_id
                            finalized_all.append(vdata)
                with self._lock:
                    self._frames += len(batch)
                self._gpu_faults = 0
            except Exception as e:
                msg = str(e)
                if "illegal instruction" in msg or "CUDA" in msg or "CuTensor" in msg:
                    self._gpu_faults += 1
                    logger.error(f"GPU fault #{self._gpu_faults}: {e}")
                    if self._gpu_faults >= 5:
                        logger.error("Persistent GPU fault — Triton CUDA context is corrupted. "
                                     "Stopping pipeline; restart Triton before re-running.")
                        self.stop_event.set()
                    time.sleep(0.5)
                else:
                    logger.warning(f"detect/track error (shard {shard.id}): {e}")
                continue
            for vdata in finalized_all:
                with self._lock:
                    self._vehicles += 1
                self.vpool.submit(self._handle_vehicle, vdata)

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            time.sleep(10.0)
            dropped = sum(t.dropped for t in self.ingests)
            qsize = sum(s.frame_q.qsize() for s in self.shards)
            with self._lock:
                logger.info(f"[hb] frames={self._frames} vehicles={self._vehicles} "
                            f"rows={self._rows} queue={qsize} dropped={dropped}")

    # ── warmup ────────────────────────────────────────────────────────────────
    def _warmup(self):
        """Pre-load every Triton model with dummy inputs BEFORE ingest starts.

        First-time inference triggers ~20s of model loading (TensorRT deserialize /
        CUDA init) which would stall detection and overflow the queue (dropped frames).
        Model loading is GPU-global, so one dummy pass front-loads it -> no startup loss.
        """
        import numpy as np
        from core import lpr as lpr_mod
        t0 = time.time()
        logger.info("Warming up models (pre-loading Triton; cameras not open yet)...")
        n = max(2, self.settings["pipeline"].get("sample_frames", 4))
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)

        def life():
            return [{"frame_id": str(i), "image": dummy, "bbox": [[100, 100], [500, 500]]}
                    for i in range(n)]

        detector = self.shards[0].detector
        try:
            with self._gpu():
                detector.detect_batch([dummy])                     # yolo_infer
        except Exception as e:
            logger.warning(f"warmup detect: {e}")
        for cls in ("car", "bike", "truck"):                       # seatbelt/phone/helmet/side_view/uncovered
            try:
                with self._gpu():
                    violations_mod.evaluate_vehicle(
                        {"cam_id": "warmup", "vehicle_id": "warmup",
                         "class_name": cls, "life_time": life()},
                        self.grpc_url, self.settings)
            except Exception as e:
                logger.warning(f"warmup {cls}: {e}")
        try:                                                        # number_yolo + ocr_model_vit
            with self._gpu():
                lpr_mod.run_lpr_vehicle(
                    {"vehicle_id": "warmup", "class_name": "car", "life_time": life()},
                    self.grpc_url)
        except Exception as e:
            logger.warning(f"warmup lpr: {e}")
        logger.info(f"Warmup done in {time.time() - t0:.1f}s — starting cameras.")

    # ── lifecycle ────────────────────────────────────────────────────────────
    def run(self):
        self._warmup()
        ingests = [
            CameraIngest(c, self.cam_to_shard[c["id"]].frame_q, self.target_fps,
                         self.stop_event, backend=self.ingest_backend)
            for c in self.cameras
        ]
        self.ingests = ingests
        for t in ingests:
            t.start()
        for s in self.shards:
            s.thread = threading.Thread(target=self._detect_track_loop, args=(s,),
                                        name=f"detect-track-{s.id}", daemon=True)
            s.thread.start()
        mon = threading.Thread(target=self._monitor_loop, name="monitor", daemon=True)
        mon.start()
        logger.info(f"Pipeline running on {len(self.cameras)} camera(s) across "
                    f"{len(self.shards)} detect shard(s). Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Shutdown requested…")
        finally:
            self.shutdown()

    def shutdown(self):
        self.stop_event.set()
        for s in self.shards:
            if s.thread:
                s.thread.join(timeout=5.0)
        # finalize any vehicles still being tracked
        for s in self.shards:
            for cam_id, tr in s.trackers.items():
                for vid, vdata in tr.flush():
                    vdata["cam_id"] = cam_id
                    self.vpool.submit(self._handle_vehicle, vdata)
        self.vpool.shutdown(wait=True)
        with self._lock:
            logger.info(f"Done. frames={self._frames} vehicles={self._vehicles} rows={self._rows}")
        logger.info(f"CSV: {self.sheet.csv_path}")
