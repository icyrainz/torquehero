"""Settings of the app loop, persisted in `config_dir()/app.json` (not in Config).

`ffb_first_run_done` stays false until the player raises the FFB gain once with
ffb_up; until then the game starts at FIRST_RUN_GAIN (SPEC 5 rule 12).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .config import config_dir

SETTINGS_FILE = "app.json"
CHARTS_DIR = "charts"


@dataclass
class AppSettings:
    ffb_first_run_done: bool = False
    difficulty: str = "normal"      # last difficulty picked in the menu
    charts_dir: str | None = None   # folder of charts for the menu; None = config_dir()/charts

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppSettings:
        """Defaults when the file is missing or unreadable; unknown keys are ignored."""
        path = Path(path) if path else config_dir() / SETTINGS_FILE
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError):
            return cls()
        if not isinstance(d, dict):
            return cls()
        s = cls()
        for f in fields(cls):
            v = d.get(f.name)
            if f.name == "ffb_first_run_done" and isinstance(v, bool):
                s.ffb_first_run_done = v
            elif f.name != "ffb_first_run_done" and isinstance(v, str):
                setattr(s, f.name, v)
        return s

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else config_dir() / SETTINGS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path

    def charts_path(self) -> Path:
        return Path(self.charts_dir) if self.charts_dir else config_dir() / CHARTS_DIR
