"""Per-camera DeepSort tracking + vehicle lifecycle.

Wraps DeepSort (DeepSort_Tracker/Tracking) with a private asyncio loop so it can
run inside a worker thread, and manages the in-RAM ring buffer of frames per track.
A vehicle is "finalized" (ready for violation/LPR evaluation) once it has been
unseen for > max_miss frames.
"""
import core._env  # noqa: F401

import asyncio
import logging
import time

from Tracking import DeepSort

logger = logging.getLogger("bangalore.tracker")
# DeepSort internals are very chatty (per-frame "Matches"/feature logs); keep them quiet.
_ds_logger = logging.getLogger("bangalore.deepsort")
_ds_logger.setLevel(logging.ERROR)

DS_CLASS_NAMES = "/home/cv-gpu-2/harshit_workspace/DeepSort_Tracker/class_names.txt"
DS_TRACK_ONLY = "/home/cv-gpu-2/harshit_workspace/DeepSort_Tracker/track_only.json"


def _make_settings(grpc_url: str):
    class EnsembleSettings:
        input_name = "input"
        output_name = "output"
        triton_inference_server_url = grpc_url
        classes_path = DS_CLASS_NAMES
        track_only = DS_TRACK_ONLY
        input_type = "FP32"
        model_name = "feature_extractor"
        client_timeout = 30
        batch_size = 1
    return EnsembleSettings()


class CameraTracker:
    """One per camera. Holds independent DeepSort state + per-track frame buffers."""

    def __init__(self, cam_id: str, grpc_url: str, max_miss: int = 30,
                 max_life: int = 48, eval_after: int = 6, stage_log: bool = False):
        self.cam_id = cam_id
        self.grpc_url = grpc_url
        self.max_miss = max_miss
        self.max_life = max_life  # cap frames retained per track (RAM bound)
        self.eval_after = eval_after  # evaluate a track once it has this many frames
        self.stage_log = stage_log
        self._loop = None
        self.ds = None
        # vid -> {"life_time": [...], "miss_count": int, "class_name": str, "evaluated": bool}
        self.active = {}

    def init_in_thread(self, loop):
        """Create the DeepSort instance bound to `loop`, FROM the thread that owns it.

        grpc.aio binds its channel to the running loop, so the loop must be created,
        set as current, and used for track() all in the SAME thread. Otherwise the
        feature_extractor Future ends up "attached to a different loop" and returns
        0 features (tracking silently breaks).
        """
        self._loop = loop
        self.ds = DeepSort(_make_settings(self.grpc_url), logger=_ds_logger,
                           track_time_threshold=99999)

    def _track(self, img, dets):
        input_data = {"frame": img, "detections": dets, "ingested_time": time.time()}
        try:
            return self._loop.run_until_complete(self.ds.track(input_data))
        except Exception as e:
            logger.warning(f"[{self.cam_id}] DeepSort error: {e}")
            return {}

    def update(self, frame_id, img, dets) -> list:
        """Feed one frame. Returns list of (vehicle_id, vehicle_data) READY to evaluate.

        A vehicle is emitted for evaluation exactly once — as soon as it has
        `eval_after` tracked frames (so we don't wait for it to leave the frame),
        or on exit if it never reached that count. Emitted tracks stay alive for
        continued tracking but are not re-evaluated.
        """
        out = self._track(img, dets)
        seen = set()
        new_ids = []
        for det in out.get("sorted_detections", []):
            vid = det["vehicle_id"]
            seen.add(vid)
            entry = {
                "frame_id": str(frame_id),
                "image": img,
                "bbox": det["bbox_location"],
                "confidence": float(det.get("confidence", 0.0)),
                "captured_time": time.time(),
                "ingested_time": time.time(),
            }
            if vid not in self.active:
                new_ids.append(vid)
            v = self.active.setdefault(
                vid, {"life_time": [], "miss_count": 0, "evaluated": False,
                      "class_name": det.get("class_name", "vehicle")})
            v["life_time"].append(entry)
            # ring buffer: keep an evenly-decimated window when over capacity
            if len(v["life_time"]) > self.max_life:
                v["life_time"] = v["life_time"][::2]
            v["miss_count"] = 0

        ready = []
        # (a) confirmed mid-life: enough frames and not yet evaluated -> snapshot now
        for vid, v in self.active.items():
            if not v["evaluated"] and len(v["life_time"]) >= self.eval_after:
                v["evaluated"] = True
                snap = {"vehicle_id": vid, "class_name": v["class_name"],
                        "life_time": list(v["life_time"])}  # shallow copy; track stays alive
                ready.append((vid, snap))

        # (b) exit: bump miss; pop when gone. Evaluate on exit only if never evaluated.
        exited = 0
        for vid in list(self.active):
            if vid not in seen:
                self.active[vid]["miss_count"] += 1
                if self.active[vid]["miss_count"] > self.max_miss:
                    vdata = self.active.pop(vid)
                    exited += 1
                    if not vdata["evaluated"]:
                        vdata["vehicle_id"] = vid
                        ready.append((vid, vdata))

        if self.stage_log:
            logger.info("[%s f%s] DeepSort: dets_in=%d tracked=%d new=%d active=%d "
                        "ready_eval=%d exited=%d",
                        self.cam_id, frame_id, len(dets), len(seen), len(new_ids),
                        len(self.active), len(ready), exited)
        return ready

    def flush(self) -> list:
        """Finalize still-active tracks not yet evaluated (call on shutdown)."""
        out = []
        for vid, vdata in self.active.items():
            if not vdata.get("evaluated"):
                vdata["vehicle_id"] = vid
                out.append((vid, vdata))
        self.active.clear()
        return out
