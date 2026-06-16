"""Per-vehicle violation evaluation (in-memory port of run_pipeline_new.py STAGE 5).

Phase A (no LPR needed): seatbelt | helmet+triple+side_view | phone | uncovered.
If any Phase-A violation fires, run LPR (plate) then HSRP + fake-plate enrichment.
Returns a VehicleResult consumed by evidence.py / sheet.py.
"""
import core._env as _env  # noqa: F401

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

from core import lpr as lpr_mod

logger = logging.getLogger("bangalore.violations")

SEATBELT_BATCH = 16
MAX_BATCH_V10 = 16

# Per-thread seatbelt instance. The decider holds per-call state, so a single shared
# instance is NOT safe once violation_workers > 1 (two vehicles' evals would race and
# could cross-contaminate results). Thread-local gives each worker its own instance.
_seatbelt_local = threading.local()


def _client(grpc_url):
    return grpcclient.InferenceServerClient(url=grpc_url)


def _sample_frames(list_images, max_frames):
    if len(list_images) <= max_frames:
        return list_images
    idx = np.linspace(0, len(list_images) - 1, max_frames, dtype=int)
    return [list_images[i] for i in idx]


def _build_list_images(life_time):
    """In-memory: entries already hold the BGR image (no disk read)."""
    out = []
    for e in life_time:
        img = e.get("image")
        if img is not None:
            out.append({"frame_id": e["frame_id"], "bbox": e["bbox"], "image": img})
    return out


def _get_seatbelt_instance():
    _env.install_numpy_core_shim()  # needed to unpickle the seatbelt regressor (numpy>=2)
    if not hasattr(_seatbelt_local, "instance"):
        from modules.deciders.seatbelt import seatbeltDecisionCheck
        _seatbelt_local.instance = seatbeltDecisionCheck(img_side=352)
    return _seatbelt_local.instance


# ── Phase-A model runners (faithful ports) ───────────────────────────────────

def _run_seatbelt(list_images, grpc_url):
    from modules.common import preprocess_violation
    from modules.yolo_post import yolo_v10_frame_detections
    sampled = _sample_frames(list_images, SEATBELT_BATCH)
    client = _client(grpc_url)
    sb = _get_seatbelt_instance()
    batch, crops = preprocess_violation(sampled, size=(352, 352))
    inp = grpcclient.InferInput("images", batch.shape, "FP32")
    inp.set_data_from_numpy(batch.astype(np.float32))
    raw = client.infer(model_name="seatbelt_infer", inputs=[inp]).as_numpy("output0")
    dets = yolo_v10_frame_detections(raw, ["wind", "sb-person", "person-nsb", "per-idk"],
                                     img_side=352, crops=crops)
    decisions = sb.decide_seatbelt(dets_by_frame=dets, crops=crops)
    drv = next((r for r in decisions if r.get("type") == "no-seatbelt-driver"), {})
    psg = next((r for r in decisions if r.get("type") == "no-seatbelt-passenger"), {})
    return {
        "driver_decision": drv.get("review_decision", ""),
        "driver_frame": drv.get("frame_id"),
        "passenger_decision": psg.get("review_decision", ""),
        "passenger_frame": psg.get("frame_id"),
    }


def _run_helmet(list_images, grpc_url):
    from modules.common import preprocess_violation
    from modules.yolo_post import yolo_v10_frame_detections
    from modules.deciders.helmet import HelmetDecider
    sampled = _sample_frames(list_images, MAX_BATCH_V10)
    client = _client(grpc_url)
    batch, crops = preprocess_violation(sampled, size=(352, 352))
    inp = grpcclient.InferInput("images", batch.shape, "FP32")
    inp.set_data_from_numpy(batch.astype(np.float32))
    raw = client.infer(model_name="helmet_infer", inputs=[inp]).as_numpy("output0")
    dets = yolo_v10_frame_detections(raw, ["helmet", "head", "half-helmet"],
                                     img_side=352, crops=crops)
    decider = HelmetDecider()
    clf_batch, state = decider.prepare_classifier_batch(dets, crops)
    clf_logits = None
    if clf_batch.shape[0] > 0:
        clf_inp = grpcclient.InferInput("input", clf_batch.shape, "FP32")
        clf_inp.set_data_from_numpy(clf_batch.astype(np.float32))
        clf_logits = client.infer(
            model_name="helmet_classifier", inputs=[clf_inp],
            outputs=[grpcclient.InferRequestedOutput("logits")],
        ).as_numpy("logits")
    decisions = decider.decide_helmet(state, clf_logits)
    rider = next((r for r in decisions if r.get("type") == "no-helmet-rider"), {})
    pillion = next((r for r in decisions if r.get("type") == "no-helmet-pillion"), {})
    triple = next((r for r in decisions if r.get("type") == "overloading-two-wheeler"), {})
    return {
        "rider_decision": rider.get("review_decision", ""),
        "rider_frame": rider.get("frame_id"),
        "pillion_decision": pillion.get("review_decision", ""),
        "pillion_frame": pillion.get("frame_id"),
        "triple": bool(triple.get("result", False)),
        "triple_frame": triple.get("frame_id"),
    }


def _run_phone(list_images, grpc_url):
    from modules.common import preprocess_violation
    from modules.yolo_post import yolo_frame_detections
    from modules.deciders.phone import decide_phone
    sampled = _sample_frames(list_images, MAX_BATCH_V10)
    client = _client(grpc_url)
    batch, crops = preprocess_violation(sampled, size=(384, 384))
    inp = grpcclient.InferInput("images", batch.shape, "FP32")
    inp.set_data_from_numpy(batch.astype(np.float32))
    raw = client.infer(model_name="phone_infer", inputs=[inp]).as_numpy("output0")
    dets = yolo_frame_detections(raw, ["phone", "wind"], img_side=384)
    decisions = decide_phone(dets, crops)
    entry = decisions[0] if decisions else {}
    return {"phone": bool(entry.get("result", False)), "frame": entry.get("frame_id")}


def _run_side_view(list_images, grpc_url):
    from modules_2.inference import load_and_preprocess, postprocess_output, decide_violation
    sampled = _sample_frames(list_images, MAX_BATCH_V10)
    client = _client(grpc_url)
    frame_results = []
    for data in sampled:
        tensor = load_and_preprocess(data, size=(384, 384))
        inp = grpcclient.InferInput("images", tensor.shape, "FP32")
        inp.set_data_from_numpy(tensor.astype(np.float32))
        logits = client.infer(model_name="side_view_infer", inputs=[inp],
                              outputs=[grpcclient.InferRequestedOutput("output0")]).as_numpy("output0")
        res = postprocess_output(logits)
        res["frame_id"] = data.get("frame_id")
        frame_results.append(res)
    final = decide_violation(frame_results)
    return {"side_view": bool(final.get("violation", False)), "frame": final.get("frame_id")}


def _run_uncovered(list_images, grpc_url):
    uv_pre = _env.load_uv_module("preprocessing").preprocess_violation
    uv_post = _env.load_uv_module("postprocessing").postprocess_uncovered
    sampled = _sample_frames(list_images, MAX_BATCH_V10)
    client = _client(grpc_url)
    batch, meta = uv_pre(sampled, size=(224, 224), max_batch_size=16)
    inp = grpcclient.InferInput("input", batch.shape, "FP32")
    inp.set_data_from_numpy(batch.astype(np.float32))
    logits = client.infer(model_name="uncovered_vehicle", inputs=[inp],
                         outputs=[grpcclient.InferRequestedOutput("logits")]).as_numpy("logits")
    res = uv_post(logits, meta)
    return {"uncovered": bool(res.get("violation", False)), "frame": res.get("frame_id")}


def _hsrp_preprocess(image, hsrp_bbox):
    # Ported from HSRP_client/modules/inference.py (the model itself is unchanged):
    # crop the plate bbox, then keep only the LEFT 35% (hologram region) — that is how
    # the HSRP classifier was trained — then RGB / resize 224 / normalize / CHW.
    x1, y1 = int(hsrp_bbox[0][0]), int(hsrp_bbox[0][1])
    x2, y2 = int(hsrp_bbox[1][0]), int(hsrp_bbox[1][1])
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    crop = crop[:, :max(1, int(crop.shape[1] * 0.35)), :]   # left hologram region
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return np.transpose(crop, (2, 0, 1))[np.newaxis]


def _run_hsrp(ocr_post_out, frame_lookup, grpc_url):
    hsrp_bbox = ocr_post_out.get("hsrp")
    evidence = ocr_post_out.get("evidence", {}) or {}
    frame_id = evidence.get("frame_id")
    if not hsrp_bbox or frame_id is None:
        return None
    img = frame_lookup.get(str(frame_id))
    if img is None:
        return None
    tensor = _hsrp_preprocess(img, hsrp_bbox)
    if tensor is None:
        return None
    client = _client(grpc_url)
    inp = grpcclient.InferInput("input", tensor.shape, "FP32")
    inp.set_data_from_numpy(tensor)
    logit = float(client.infer(model_name="hsrp_infer", inputs=[inp]).as_numpy("output").flatten()[0])
    # HSRP_client postprocess: sigmoid, then prob < 0.5 => "no-hsrp" (violation).
    prob = 1.0 / (1.0 + np.exp(-logit))
    return {"hsrp_violation": bool(prob < 0.5)}


# ── Orchestration ────────────────────────────────────────────────────────────

def evaluate_vehicle(vehicle_data: dict, grpc_url: str, settings: dict) -> Optional[dict]:
    """Run phase-A violations; if any fire, run LPR + HSRP. Returns a row dict or None."""
    _env.install_numpy_core_shim()  # safe now (all C-ext imports done); needed for joblib models
    cam_id = vehicle_data.get("cam_id", "?")
    vid = vehicle_data.get("vehicle_id", "?")
    class_name = vehicle_data.get("class_name", "vehicle")
    life_time = vehicle_data.get("life_time", [])
    stage_log = settings.get("pipeline", {}).get("stage_logging", False)
    t0 = time.time()
    if not life_time:
        return None
    if stage_log:
        logger.info("[%s v%s] VIOLATION: start class=%s frames=%d", cam_id, vid, class_name, len(life_time))

    frame_lookup = {e["frame_id"]: e["image"] for e in life_time if e.get("image") is not None}
    list_images = _build_list_images(life_time)
    if not list_images:
        return None

    do_seatbelt = class_name in _env.FOUR_WHEELERS
    do_helmet = class_name in _env.TWO_WHEELERS
    do_uncovered = class_name == "truck"

    fields = {k: "no" for k in
              ("seatbelt", "helmet", "triple_rider", "phone", "uncovered",
               "hsrp", "side_view")}
    evidence_frame_id = None  # frame id of the triggering violation

    futures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        if do_seatbelt:
            futures["seatbelt"] = pool.submit(_run_seatbelt, list_images, grpc_url)
        if do_helmet:
            futures["helmet"] = pool.submit(_run_helmet, list_images, grpc_url)
            futures["side_view"] = pool.submit(_run_side_view, list_images, grpc_url)
        if do_uncovered:
            futures["uncovered"] = pool.submit(_run_uncovered, list_images, grpc_url)
        futures["phone"] = pool.submit(_run_phone, list_images, grpc_url)

        for key, fut in futures.items():
            try:
                res = fut.result()
            except Exception as e:
                logger.warning(f"[{cam_id}] {key} error: {e}")
                continue
            if key == "seatbelt":
                drv = res["driver_decision"].lower() in ("accept", "review")
                psg = res["passenger_decision"].lower() in ("accept", "review")
                parts = (["driver"] if drv else []) + (["passenger"] if psg else [])
                if parts:
                    fields["seatbelt"] = "+".join(parts)
                    evidence_frame_id = evidence_frame_id or res.get("driver_frame") or res.get("passenger_frame")
            elif key == "helmet":
                rider = res["rider_decision"].lower() in ("accept", "review")
                pillion = res["pillion_decision"].lower() in ("accept", "review")
                if rider or pillion:
                    fields["helmet"] = "yes"
                    evidence_frame_id = evidence_frame_id or res.get("rider_frame") or res.get("pillion_frame")
                if res.get("triple"):
                    fields["triple_rider"] = "yes"
                    evidence_frame_id = evidence_frame_id or res.get("triple_frame")
            elif key == "side_view" and res.get("side_view"):
                fields["side_view"] = "yes"
                evidence_frame_id = evidence_frame_id or res.get("frame")
            elif key == "phone" and res.get("phone"):
                fields["phone"] = "yes"
                evidence_frame_id = evidence_frame_id or res.get("frame")
            elif key == "uncovered" and res.get("uncovered"):
                fields["uncovered"] = "yes"
                evidence_frame_id = evidence_frame_id or res.get("frame")

    has_violation = any(fields[k] != "no" for k in
                        ("seatbelt", "helmet", "triple_rider", "phone", "uncovered", "side_view"))
    if stage_log:
        fired = ",".join(k for k in fields if fields[k] != "no") or "none"
        logger.info("[%s v%s] VIOLATION: phase-A done in %.2fs -> %s",
                    cam_id, vid, time.time() - t0, fired)
    if not has_violation:
        return None  # no row, no LPR, no upload — frames will be freed by caller

    # ── Violator: run LPR, then HSRP + fake-plate enrichment ──
    plate = ""
    plate_evidence_id = None
    plate_bbox = None
    try:
        ocr_post_out = lpr_mod.run_lpr_vehicle(vehicle_data, grpc_url)
    except Exception as e:
        logger.warning(f"[{cam_id}] LPR error: {e}")
        ocr_post_out = None

    if ocr_post_out:
        plate = (ocr_post_out.get("result", {}) or {}).get("text", "") or ""
        ev = ocr_post_out.get("evidence", {}) or {}
        plate_evidence_id = ev.get("frame_id")
        plate_bbox = ocr_post_out.get("hsrp")
        try:
            hres = _run_hsrp(ocr_post_out, frame_lookup, grpc_url)
            if hres and hres.get("hsrp_violation"):
                fields["hsrp"] = "yes"
        except Exception as e:
            logger.warning(f"[{cam_id}] HSRP error: {e}")

    # Only record a row when we have BOTH a violation and a readable plate.
    # A violation with no recognised plate is dropped (no row, no upload).
    plate = plate.strip()
    if stage_log:
        logger.info("[%s v%s] LPR: plate=%r", cam_id, vid, plate)
    if not plate:
        logger.info("[%s v%s] VIOLATION: has violation but no readable plate — skipping row",
                    cam_id, vid)
        return None
    if stage_log:
        logger.info("[%s v%s] VIOLATION: ROW ready plate=%s total=%.2fs",
                    cam_id, vid, plate, time.time() - t0)

    # Resolve evidence images from the in-RAM frame lookup.
    viol_img = frame_lookup.get(str(evidence_frame_id)) if evidence_frame_id is not None else None
    if viol_img is None:
        viol_img = life_time[-1]["image"]
    plate_img = frame_lookup.get(str(plate_evidence_id)) if plate_evidence_id is not None else viol_img

    return {
        "cam_id": cam_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "plate": plate,
        "class_name": class_name,
        "fields": fields,
        "violation_image": viol_img,
        "plate_image": plate_img,
        "plate_bbox": plate_bbox,
    }
