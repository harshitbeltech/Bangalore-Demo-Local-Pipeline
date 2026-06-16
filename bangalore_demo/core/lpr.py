"""Per-vehicle LPR (in-memory port of run_pipeline_new.py run_lpr()).

Flow:  preprocess_violation -> number_yolo (FP16) -> postprocess_lpr
       -> ocr_model_vit -> OCR_post.ocr_post
Only invoked on vehicles that already have a vision violation.
"""
import core._env as _env  # noqa: F401

import logging
import threading
import numpy as np
import tritonclient.grpc as grpcclient

logger = logging.getLogger("bangalore.lpr")

YOLO_INPUT_SIZE = (640, 640)
YOLO_MAX_BATCH = 16
YOLO_CONF_THRES = 0.25
YOLO_IOU_THRES = 0.45

# VIT-LPR package (importlib alias to avoid colliding with ODN cv_module_utils)
_pkg, _post = _env.load_new_lpr_module()
preprocess_violation = _pkg.preprocess_violation
postprocess_lpr = _pkg.postprocess_lpr
OCR_post = _post.OCR_post

# Thread-local Triton client + OCR_post instance (one per worker thread).
_local = threading.local()


def _resources(grpc_url: str):
    if not hasattr(_local, "client"):
        _local.client = grpcclient.InferenceServerClient(url=grpc_url)
        _local.ocr_post = OCR_post(logger=logger)
    return _local.client, _local.ocr_post


def _run_number_yolo(client, batch):
    """number_yolo over batch (T,3,H,W) in <=16 chunks; FP16 in/out. Returns (T,5,N)."""
    n = batch.shape[0]
    parts, i = [], 0
    while i < n:
        size = min(YOLO_MAX_BATCH, n - i)
        while True:
            chunk = np.ascontiguousarray(batch[i:i + size], dtype=np.float16)
            try:
                inp = grpcclient.InferInput("images", chunk.shape, "FP16")
                inp.set_data_from_numpy(chunk)
                res = client.infer(model_name="number_yolo", inputs=[inp])
                parts.append(res.as_numpy("output0").astype(np.float32))
                i += size
                break
            except Exception as e:
                if size > 1:
                    size //= 2
                    continue
                if not parts:
                    raise
                logger.warning(f"number_yolo batch=1 failed at {i}/{n}: {e}")
                parts.append(np.zeros((1,) + parts[-1].shape[1:], dtype=np.float32))
                i += 1
                break
    return np.concatenate(parts, axis=0)


def run_lpr_vehicle(vehicle_data: dict, grpc_url: str):
    """Returns ocr_post_out dict (with result.text, evidence, hsrp) or None.

    vehicle_data["life_time"] entries must each carry an in-memory "image" (BGR),
    a "bbox" [[x1,y1],[x2,y2]] and a "frame_id".
    """
    client, ocr_post_inst = _resources(grpc_url)
    vehicle_id = vehicle_data.get("vehicle_id", "?")

    list_images = []
    for entry in vehicle_data.get("life_time", []):
        img = entry.get("image")
        if img is None:
            continue
        item = dict(entry)
        item["image"] = img
        list_images.append(item)
    if not list_images:
        return None

    batch, metas = preprocess_violation(list_images, size=YOLO_INPUT_SIZE, log=logger)
    for item in list_images:
        item.pop("image", None)
    if not metas:
        return None

    try:
        yolo_output = _run_number_yolo(client, batch)
    except Exception as e:
        logger.warning(f"[LPR] {vehicle_id} number_yolo failed: {e}")
        return None

    entity_dict = {
        "_id": vehicle_id,
        "class_name": vehicle_data.get("class_name", "car"),
        "life_time": [m.get("meta", {}) for m in metas],
    }

    ocr_input, crop_frame_idx, plate_bbox_per_slot = postprocess_lpr(
        yolo_output, metas,
        conf_thres=YOLO_CONF_THRES, iou_thres=YOLO_IOU_THRES,
        parseq_size=(128, 32), max_slots=20, log=logger,
    )

    ocr_input = np.ascontiguousarray(ocr_input, dtype=np.float32)
    inp = grpcclient.InferInput("images", ocr_input.shape, "FP32")
    inp.set_data_from_numpy(ocr_input)
    ocr_res = client.infer(model_name="ocr_model_vit", inputs=[inp])
    ocr_out_raw = ocr_res.as_numpy("logits")

    ocr_post_out = ocr_post_inst.ocr_post(
        ocr_out_raw, crop_frame_idx, plate_bbox_per_slot, entity_dict
    )
    return ocr_post_out
