"""ODN object detection (Triton model `yolo_infer`), batched across cameras.

Mirrors run_pipeline_new.py run_odn() pre/post-processing but accepts a batch of
in-memory frames from multiple cameras in a single yolo_infer call.
"""
import core._env  # noqa: F401  (sys.path + shim must load first)

import logging
import numpy as np
import tritonclient.grpc as grpcclient
from cv_module_utils import ODN_Utility

logger = logging.getLogger("bangalore.detector")

ODN_NAMES = "/home/cv-gpu-2/violation_modules/object_detection_network/workspace/obj.names"
ODN_INPUT_SIZE = (512, 320)   # (W, H), as in run_pipeline_new.py


class OdnDetector:
    def __init__(self, grpc_url: str):
        self.util = ODN_Utility(logger=logger, names_path=ODN_NAMES)
        self.client = grpcclient.InferenceServerClient(url=grpc_url)

    def detect_batch(self, frames: list) -> list:
        """frames: list of BGR np.ndarray. Returns list (aligned) of detection lists.

        Each detection: {"class_name", "confidence", "bbox_location": [[x1,y1],[x2,y2]]}.
        """
        if not frames:
            return []
        pre, sizes = [], []
        for img in frames:
            h, w = img.shape[:2]
            try:
                pre.append(self.util._preprocess_yolo(img, input_size=ODN_INPUT_SIZE))
                sizes.append((h, w))
            except Exception as e:
                logger.warning(f"ODN preprocess failed: {e}")
                pre.append(None)
                sizes.append((h, w))

        valid_idx = [i for i, p in enumerate(pre) if p is not None]
        results = [[] for _ in frames]
        if not valid_idx:
            return results

        batch = np.concatenate([pre[i] for i in valid_idx], axis=0).astype(np.float32)
        inp = grpcclient.InferInput("images", batch.shape, "FP32")
        inp.set_data_from_numpy(batch)
        out = self.client.infer(model_name="yolo_infer", inputs=[inp]).as_numpy("output0")

        for bi, i in enumerate(valid_idx):
            preds = out[bi] if out.ndim == 3 else out
            h, w = sizes[i]
            try:
                res = self.util.get_odn_result(
                    None, preds, input_size=ODN_INPUT_SIZE, orig_size=(h, w)
                )
                results[i] = res.get("detections", [])
            except Exception as e:
                logger.warning(f"ODN postprocess failed: {e}")
                results[i] = []
        return results
