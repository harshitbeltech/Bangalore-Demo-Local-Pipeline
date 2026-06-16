#!/usr/bin/env python3
"""Entrypoint for the Bangalore live violation + LPR demo.

    python run.py                       # uses config/settings.yaml + config/cameras.yaml
    python run.py --cameras config/cameras.yaml --settings config/settings.yaml
    python run.py --check               # validate config + Triton reachability, then exit

Run inside the shared venv:
    source /home/cv-gpu-2/harshit_workspace/.venv/bin/activate
"""
import core._env  # noqa: F401  (must be first)

import argparse
import logging
import sys

import requests

from core.config import load_settings, load_cameras


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def check_triton(settings) -> bool:
    url = settings["triton"]["http_health"]
    try:
        r = requests.get(url, timeout=3)
        return r.status_code == 200
    except Exception as e:
        logging.error(f"Triton not reachable at {url}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Bangalore live violation + LPR demo")
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--cameras", default="config/cameras.yaml")
    ap.add_argument("--check", action="store_true",
                    help="validate config + Triton reachability, then exit")
    ap.add_argument("--only", default="",
                    help="comma-separated camera ids to run (e.g. cam01,cam04); subset of enabled")
    ap.add_argument("--shard", default="",
                    help="run a disjoint slice for multi-instance, as I/N (e.g. 1/4, 2/4 ...)")
    args = ap.parse_args()

    settings = load_settings(args.settings)
    setup_logging(settings["logging"]["level"])
    log = logging.getLogger("bangalore")

    cameras = load_cameras(args.cameras)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        cameras = [c for c in cameras if c["id"] in wanted]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        cameras = [c for k, c in enumerate(cameras) if k % n == (i - 1)]
        log.info(f"Shard {i}/{n} selected.")
    if not cameras:
        log.error("No cameras left after --only/--shard filtering. Exiting.")
        sys.exit(1)
    log.info(f"Loaded {len(cameras)} enabled camera(s): {[c['id'] for c in cameras]}")

    triton_ok = check_triton(settings)
    if triton_ok:
        log.info("Triton is ready.")
    else:
        log.warning("Triton health check FAILED. Start the Triton server before running "
                    "(models: yolo_infer, feature_extractor, number_yolo, ocr_model_vit, "
                    "seatbelt_infer, helmet_infer, helmet_classifier, phone_infer, hsrp_infer, "
                    "side_view_infer, uncovered_vehicle).")

    if args.check:
        log.info("Config OK. Triton %s.", "ready" if triton_ok else "NOT ready")
        sys.exit(0 if triton_ok else 1)

    if not triton_ok:
        log.error("Refusing to start without Triton. Re-run with --check after starting it.")
        sys.exit(1)

    # Import here so --check works even if heavy deps are missing.
    from pipeline import Pipeline
    Pipeline(settings, cameras).run()


if __name__ == "__main__":
    main()
