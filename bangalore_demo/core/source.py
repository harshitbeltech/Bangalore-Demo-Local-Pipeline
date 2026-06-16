"""Video sources + per-camera ingest threads.

VideoSource abstracts an RTSP url or a local file path. The ingest thread decodes,
drops stale frames, samples down to target_fps, and pushes (cam_id, frame_id, frame)
into a bounded queue with drop-oldest backpressure so a slow consumer can never grow
RAM or fall behind real time.
"""
import logging
import queue as queue_mod
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger("bangalore.source")


class VideoSource:
    """Opens an RTSP stream or a local file; reconnects/loops as needed."""

    def __init__(self, cam: dict):
        self.cam_id = cam["id"]
        self.kind = cam.get("kind", "file")
        self.source = cam["source"]
        self.loop = cam.get("loop", self.kind == "file")
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if self.kind == "rtsp":
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        ok = self.cap.isOpened()
        if not ok:
            logger.warning(f"[{self.cam_id}] could not open {self.source}")
        return ok

    def native_fps(self) -> float:
        if not self.cap:
            return 0.0
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        return fps if fps > 0 else 25.0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            if self.kind == "file" and self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
        return (ok, frame)

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None


class FfmpegNvdecSource:
    """Decode an RTSP stream on the GPU's NVDEC engine via an ffmpeg subprocess.

    Software HEVC decode of a 1440p stream costs ~0.5 CPU core PER camera; on an
    8-core box that starves detection. NVDEC moves decode to dedicated GPU silicon
    (separate from the CUDA cores Triton uses), cutting per-stream CPU ~4x. The
    `fps` filter samples to target_fps INSIDE ffmpeg, so we never transfer the 80%
    of frames the old path decoded-then-discarded.

    Emits BGR np.ndarray frames already sampled to target_fps (stride handled here,
    so the ingest thread pushes every frame this returns).
    """

    def __init__(self, cam: dict, target_fps: int):
        self.cam_id = cam["id"]
        self.source = cam["source"]
        self.kind = cam.get("kind", "rtsp")
        self.loop = cam.get("loop", self.kind == "file")
        self.target_fps = max(1, int(target_fps))
        self.proc = None
        self.w = self.h = 0
        self.frame_bytes = 0

    # frames are already sampled inside ffmpeg → ingest must not re-stride
    samples_internally = True

    def _probe(self):
        """Return (width, height, codec_name) or (0,0,None) on failure."""
        if not shutil.which("ffprobe"):
            return 0, 0, None
        cmd = ["ffprobe", "-hide_banner", "-loglevel", "error"]
        if self.kind == "rtsp":
            cmd += ["-rtsp_transport", "tcp"]
        cmd += ["-select_streams", "v:0", "-show_entries",
                "stream=width,height,codec_name", "-of",
                "default=noprint_wrappers=1:nokey=0", self.source]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        except Exception as e:
            logger.warning(f"[{self.cam_id}] ffprobe failed: {e}")
            return 0, 0, None
        info = {}
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
        try:
            return int(info["width"]), int(info["height"]), info.get("codec_name")
        except (KeyError, ValueError):
            return 0, 0, None

    def open(self) -> bool:
        if not shutil.which("ffmpeg"):
            logger.warning(f"[{self.cam_id}] ffmpeg not found; NVDEC unavailable")
            return False
        self.w, self.h, codec = self._probe()
        if self.w <= 0 or self.h <= 0:
            logger.warning(f"[{self.cam_id}] could not probe {self.source}")
            return False
        self.frame_bytes = self.w * self.h * 3
        # pick the matching CUVID decoder; fall back to software decode-on-GPU-hwaccel
        cuvid = {"hevc": "hevc_cuvid", "h264": "h264_cuvid"}.get(codec)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
               "-fflags", "nobuffer", "-flags", "low_delay"]
        if self.kind == "rtsp":
            cmd += ["-rtsp_transport", "tcp"]
        cmd += ["-hwaccel", "cuda"]
        if cuvid:
            cmd += ["-c:v", cuvid]
        if self.kind == "file" and self.loop:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-i", self.source,
                "-an", "-vf", f"fps={self.target_fps}",
                "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=self.frame_bytes)
        except Exception as e:
            logger.warning(f"[{self.cam_id}] ffmpeg spawn failed: {e}")
            return False
        logger.info(f"[{self.cam_id}] open (nvdec {codec} {self.w}x{self.h}) "
                    f"-> fps={self.target_fps}")
        return True

    def _read_exact(self, n: int):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                return None  # EOF / process died
            buf += chunk
        return buf

    def read(self):
        if self.proc is None or self.proc.stdout is None:
            return (False, None)
        raw = self._read_exact(self.frame_bytes)
        if raw is None:
            return (False, None)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w, 3)
        return (True, frame)

    def native_fps(self) -> float:
        return float(self.target_fps)

    def release(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


class CameraIngest(threading.Thread):
    """Per-camera decode thread → bounded drop-oldest queue of sampled frames."""

    def __init__(self, cam: dict, out_queue: queue_mod.Queue, target_fps: int,
                 stop_event: threading.Event, backend: str = "cv2"):
        super().__init__(daemon=True, name=f"ingest-{cam['id']}")
        self.cam = cam
        self.cam_id = cam["id"]
        self.out = out_queue
        self.target_fps = max(1, int(target_fps))
        self.stop_event = stop_event
        self.backend = backend
        self.frame_id = 0
        self.dropped = 0

    def _make_source(self):
        if self.backend == "ffmpeg_nvdec":
            return FfmpegNvdecSource(self.cam, self.target_fps)
        return VideoSource(self.cam)

    def _push(self, frame):
        self.frame_id += 1
        item = (self.cam_id, self.frame_id, frame)
        try:
            self.out.put_nowait(item)
        except queue_mod.Full:
            # drop-oldest: discard the stalest frame, enqueue the fresh one
            try:
                self.out.get_nowait()
                self.dropped += 1
            except queue_mod.Empty:
                pass
            try:
                self.out.put_nowait(item)
            except queue_mod.Full:
                self.dropped += 1

    def run(self):
        src = self._make_source()
        # NVDEC source samples to target_fps inside ffmpeg → push every frame (stride 1)
        internal = getattr(src, "samples_internally", False)
        while not self.stop_event.is_set():
            if not src.open():
                time.sleep(2.0)
                continue
            if internal:
                stride = 1
            else:
                native = src.native_fps()
                stride = max(1, int(round(native / self.target_fps)))
                logger.info(f"[{self.cam_id}] open ({src.kind}) native_fps={native:.1f} "
                            f"sample_stride={stride}")
            raw_idx = 0
            while not self.stop_event.is_set():
                ok, frame = src.read()
                if not ok or frame is None:
                    logger.warning(f"[{self.cam_id}] stream ended/failed; reconnecting")
                    break
                raw_idx += 1
                if raw_idx % stride != 0:
                    continue
                self._push(frame)
            src.release()
            if not self.stop_event.is_set():
                time.sleep(1.0)
        logger.info(f"[{self.cam_id}] ingest stopped (dropped={self.dropped})")
