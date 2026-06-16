"""Violations sheet writer: local CSV (source of truth) + Google Sheet mirror.

Column order matches the approved schema. One row per violating vehicle.
"""
import csv
import logging
import os
import threading

logger = logging.getLogger("bangalore.sheet")

HEADER = [
    "Timestamp", "Number plate", "Seatbelt", "Helmet", "Triple rider",
    "Phone user", "Uncovered", "HSRP", "Side view",
    "Evidence image for violation", "Evidence image for number plate",
]


class SheetWriter:
    def __init__(self, settings: dict, drive):
        self.drive = drive
        self.csv_path = os.path.join(settings["output"]["dir"], "violations.csv")
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_csv()
        self.drive.ensure_header(HEADER)

    def _init_csv(self):
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(HEADER)

    def write(self, result: dict, viol_link: str, plate_link: str):
        f = result["fields"]
        row = [
            result["timestamp"],
            result.get("plate", ""),
            f["seatbelt"], f["helmet"], f["triple_rider"], f["phone"],
            f["uncovered"], f["hsrp"], f["side_view"],
            viol_link, plate_link,
        ]
        with self._lock:
            with open(self.csv_path, "a", newline="") as fh:
                csv.writer(fh).writerow(row)
            self.drive.append_row(row)
        logger.info(f"[{result['cam_id']}] ROW plate={result.get('plate','')!r} "
                    f"seatbelt={f['seatbelt']} helmet={f['helmet']} triple={f['triple_rider']} "
                    f"phone={f['phone']} uncovered={f['uncovered']} hsrp={f['hsrp']} "
                    f"side_view={f['side_view']}")
