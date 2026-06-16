"""YAML config loaders for settings + cameras."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(_resolve(path)) as f:
        cfg = yaml.safe_load(f)
    # Normalise output dir to absolute
    cfg["output"]["dir"] = str(_resolve(cfg["output"]["dir"]))
    # Resolve credential paths relative to repo root
    for key in ("service_account", "oauth_client", "oauth_token"):
        p = cfg["drive"].get(key, "")
        cfg["drive"][key] = str(_resolve(p)) if p else ""
    return cfg


def load_cameras(path: str = "config/cameras.yaml") -> list:
    with open(_resolve(path)) as f:
        cfg = yaml.safe_load(f)
    cams = [c for c in cfg.get("cameras", []) if c.get("enabled", False)]
    if not cams:
        raise ValueError(f"No enabled cameras in {path}")
    return cams
