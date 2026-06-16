"""Environment setup — MUST be imported before any model/util import.

Replicates the sys.path + numpy compat shim from
vit_lpr_testing/run_pipeline_new.py so the reused modules
(cv_module_utils_vit_lpr, violation_model_utils, DeepSort_Tracker,
Side_view_mirror_detection, no_parking, uncovered_*) import cleanly.

Import this module first in every entrypoint:  `import core._env  # noqa`
"""
import os
import sys

# ── RTSP transport MUST be set before cv2/FFMPEG is imported ──────────────────
# Force TCP (UDP corrupts HEVC over the public internet) and a 5s socket timeout
# so a dead camera errors out and the ingest thread reconnects instead of hanging.
# Format: '|'-separated key;value pairs consumed by OpenCV's FFMPEG backend.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000",
)
# Cap FFMPEG's own log spam (libavcodec emits per-frame HEVC slice/RPS errors on
# any packet loss). 8 = AV_LOG_FATAL. Our own ingest logger still reports
# reconnects, so we lose only the noise, not real stream-failure signal.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

# ── sys.path setup MUST happen before any module-level reuse imports ──────────
# This mirrors run_pipeline_new.py EXACTLY (order + priority matters): the
# cv_pipeline venv site-packages must sit at the front so the reused C-extensions
# (numpy/scipy/torch) resolve to the ABI they were built against — otherwise
# `from cv_module_utils import ...` segfaults with a structseq ABI error.
for _p in (
    "/home/cv-gpu-2/violation_modules/object_detection_network/cv_module_utils",
    "/home/cv-gpu-2/violation_modules/LicensePlateRecognition/cv_module_utils_1",
    "/home/cv-gpu-2/harshit_workspace/violation_model_utils",
    "/home/cv-gpu-2/harshit_workspace/DeepSort_Tracker",
    "/home/cv-gpu-2/cv_pipeline/Side_view_mirror_detection",
    "/home/cv-gpu-2/harshit_workspace/no_parking",
    "/home/cv-gpu-2/cv_pipeline/.venv/lib/python3.8/site-packages",
):
    sys.path.insert(0, _p)

# uncovered_covered_vehicle_detection is loaded via importlib inside its model
# function to avoid colliding with violation_model_utils/modules (also 'modules').
_UV_MODULES_DIR = "/home/cv-gpu-2/harshit_workspace/uncovered_covered_vehicle_detection/modules"

# torch / torchvision (appended so the venv paths above take priority)
sys.path.append("/home/cv-gpu-2/.local/lib/python3.8/site-packages")

# ── numpy compat shim (LAZY) ─────────────────────────────────────────────────
# joblib models saved with numpy>=2.0 reference numpy._core, absent in numpy<2.
# It is required only to UNPICKLE those models (e.g. the seatbelt regressor).
# It must NOT be installed eagerly: aliasing numpy.core -> numpy._core while
# `cv_module_utils` is being imported re-registers numpy's structseq types and
# segfaults the interpreter ("structseq.c:398: bad argument"). So we install it
# lazily at runtime, after all C-extension imports are done, via
# install_numpy_core_shim() (called once before the violation deciders load).
_shim_installed = False


def install_numpy_core_shim():
    """Idempotently make numpy._core importable for unpickling numpy>=2 models."""
    global _shim_installed
    if _shim_installed:
        return
    import numpy as np
    if not hasattr(np, "_core"):
        import numpy.core as _np_core
        sys.modules["numpy._core"] = _np_core
        for _sub in ("multiarray", "numeric", "umath", "fromnumeric", "_methods",
                     "_dtype", "_internal", "records", "function_base", "shape_base"):
            _src = f"numpy.core.{_sub}"
            _dst = f"numpy._core.{_sub}"
            if _src in sys.modules and _dst not in sys.modules:
                sys.modules[_dst] = sys.modules[_src]
            elif _dst not in sys.modules:
                try:
                    import importlib
                    sys.modules[_dst] = importlib.import_module(_src)
                except ImportError:
                    pass
    _shim_installed = True

# ── Shared constants (mirror run_pipeline_new.py) ─────────────────────────────
FOUR_WHEELERS = {"car", "truck", "auto", "bus", "jcb", "vehicle"}
TWO_WHEELERS = {"bike", "bicycle", "man"}

# New VIT-LPR package, loaded via importlib alias (avoids colliding with the
# already-imported ODN `cv_module_utils`).
LPR_VIT_PKG_ROOT = "/home/cv-gpu-2/harshit_workspace/cv_module_utils_vit_lpr/cv_module_utils"
LPR_VIT_ALIAS = "lpr_vit_pkg"


def load_new_lpr_module():
    """Return (pkg, post_subpkg) for the VIT-LPR cv_module_utils package."""
    import importlib.util
    if LPR_VIT_ALIAS in sys.modules:
        return sys.modules[LPR_VIT_ALIAS], sys.modules[f"{LPR_VIT_ALIAS}.post"]

    pkg_spec = importlib.util.spec_from_file_location(
        LPR_VIT_ALIAS,
        f"{LPR_VIT_PKG_ROOT}/__init__.py",
        submodule_search_locations=[LPR_VIT_PKG_ROOT],
    )
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules[LPR_VIT_ALIAS] = pkg
    pkg_spec.loader.exec_module(pkg)

    post_spec = importlib.util.spec_from_file_location(
        f"{LPR_VIT_ALIAS}.post",
        f"{LPR_VIT_PKG_ROOT}/post/__init__.py",
        submodule_search_locations=[f"{LPR_VIT_PKG_ROOT}/post"],
    )
    post = importlib.util.module_from_spec(post_spec)
    sys.modules[f"{LPR_VIT_ALIAS}.post"] = post
    post_spec.loader.exec_module(post)
    return pkg, post


def load_uv_module(name: str):
    """Load an uncovered-vehicle module by file path (avoids 'modules' collision)."""
    import importlib.util
    key = f"_uv_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, f"{_UV_MODULES_DIR}/{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[key] = mod
    return sys.modules[key]
