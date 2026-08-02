"""See, Point, Fly evaluation for the CARLA-Air cooperative tasks."""

import json
from pathlib import Path


def load_config() -> dict:
    """Find and parse ``config.json``.  Raises if file is missing or malformed."""
    here = Path(__file__).resolve().parent
    for directory in (here, here.parent):
        candidate = directory / "config.json"
        if candidate.is_file():
            with open(candidate, encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError("config.json not found in spf_eval/ or its parent")
