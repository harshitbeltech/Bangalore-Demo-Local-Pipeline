#!/usr/bin/env python3
"""Live end-to-end check of the Drive + Sheets path (no Triton needed).

Uploads a tiny test image to the configured evidence folder and appends one test row
to the sheet, using the pipeline's real DriveSheets/SheetWriter code. Prints the URLs.

    source /home/cv-gpu-2/harshit_workspace/.venv/bin/activate
    python verify_drive.py
"""
import core._env  # noqa: F401

import io
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from core.config import load_settings
from core.drive import DriveSheets
from core.sheet import HEADER


def tiny_jpeg() -> bytes:
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (200, 30, 30)).save(buf, "JPEG")
        return buf.getvalue()
    except Exception:
        # Minimal valid 8x8 JPEG.
        return bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
            "070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c"
            "2837292c30313434341f27393d38323c2e333432ffc0000b08000800080101110"
            "0ffc4001f0000010501010101010100000000000000000102030405060708090a0bff"
            "c400b5100002010303020403050504040000017d01020300041105122131410613516"
            "107227114328191a1082342b1c11552d1f02433627282090a161718191a2526272829"
            "2a3435363738393a434445464748494a535455565758595a636465666768696a73747"
            "5767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4"
            "b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1"
            "f2f3f4f5f6f7f8f9faffda0008010100003f00f7fa28a28a28a28a28a28a2803ffd9")


def main():
    settings = load_settings("config/settings.yaml")
    drive = DriveSheets(settings)
    if drive.local_only:
        sys.exit("\n❌ Still in LOCAL FALLBACK — no valid Google credentials.\n"
                 "Run `python authorize.py` first (and check the log lines above).")

    url = drive.upload_jpeg(tiny_jpeg(), "_verify_drive_test.jpg")
    print(f"\n✅ Uploaded image: {url}")

    drive.ensure_header(HEADER)
    drive.append_row(["_VERIFY_", "TEST123", "Yes", "", "", "", "", "", "", url, url])
    sheet_url = f"https://docs.google.com/spreadsheets/d/{drive.sheet_id}"
    print(f"✅ Appended test row to sheet: {sheet_url}")
    print("\nOpen both links to confirm, then delete the test image/row.")


if __name__ == "__main__":
    main()
