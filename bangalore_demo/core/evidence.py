"""Encode the two evidence images, upload them, and never keep them on disk.

Given a VehicleResult, produce a violation-evidence JPEG (full frame) and a plate
JPEG (also the FULL frame — not cropped — so the plate keeps its scene context).
Upload both via DriveSheets, which returns shareable links; in Drive mode nothing
touches local disk, in fallback mode the bytes are written under output/evidence/.
"""
import logging

import cv2
import numpy as np

logger = logging.getLogger("bangalore.evidence")


def _encode(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else b""


def save_evidence(result: dict, drive, quality: int = 90):
    """Upload violation + plate images (both full frames). Returns (violation_link, plate_link)."""
    cam = result["cam_id"]
    ts = result["timestamp"].replace(":", "").replace(" ", "_").replace("-", "")
    plate = (result.get("plate") or "noplate").replace(" ", "")
    base = f"{cam}_{ts}_{plate}"

    viol_img = result.get("violation_image")
    plate_img = result.get("plate_image")  # full frame from the best plate-read frame (no crop)

    viol_link, plate_link = "", ""
    if viol_img is not None:
        b = _encode(viol_img, quality)
        if b:
            viol_link = drive.upload_jpeg(b, f"{base}_violation.jpg")
    if plate_img is not None:
        b = _encode(plate_img, quality)
        if b:
            plate_link = drive.upload_jpeg(b, f"{base}_plate.jpg")
    return viol_link, plate_link
