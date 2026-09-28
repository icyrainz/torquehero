"""Tunables and their persistence.

Every judgement threshold lives here (SPEC section 3), plus FFB and audio
defaults (section 5). Times are seconds, lane units span -1..1, angles are
degrees of wheel rotation, gains are 0..1.

Only shared game tunables live here. Other modules (bindings, companion, ...)
persist their own settings in their own file under `config_dir()`, not in Config.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)

APP_NAME = "torquehero"
CONFIG_FILE = "config.json"


@dataclass
class Config:
    # --- wheel ---
    wheel_range_deg: float = 1080.0  # lock-to-lock as set in the wheel driver
    play_range_deg: float = 90.0    # +/- degrees of rotation that map to lane -1..1
    audio_offset: float = 0.0       # seconds; positive = notes judged later
    spin_mode: str = "auto"         # "net", "abs", or "auto" (net with a bound wheel, abs on keyboard/mouse)

    # --- timing windows (seconds, each side of the note time) ---
    perfect_window: float = 0.05
    good_window: float = 0.12

    # --- position tolerance (lane units) ---
    perfect_pos: float = 0.08
    good_pos: float = 0.20
    road_tolerance: float = 0.20    # on-road check that opens the melody filter

    # --- analog thresholds (0..1) ---
    press_threshold: float = 0.5    # brake/clutch/handbrake count as pressed past this
    expr_perfect: float = 0.15      # mean absolute error, expr and fader notes
    expr_good: float = 0.30

    # --- spin (degrees of rotation within dur) ---
    spin_perfect_deg: float = 360.0
    spin_good_deg: float = 240.0
    spin_tolerance_deg: float = 15.0  # perfect as soon as the rotation reaches spin_perfect_deg minus this

    # --- riser (fraction of dur held, then release timing uses the windows above) ---
    riser_perfect_hold: float = 0.7
    riser_good_hold: float = 0.5

    # --- playability (SPEC 9), checked by chart.playability() ---
    limb_travel: float = 0.4        # seconds a hand or foot needs to move between two controls
    max_lane_rate: dict[str, float] = field(  # gate to gate steering speed, lane units per second
        default_factory=lambda: {"easy": 1.0, "normal": 1.5, "hard": 2.5})

    # --- scoring ---
    score_perfect: int = 100
    score_good: int = 50
    combo_step: int = 10            # multiplier rises by 1 every combo_step hits
    max_multiplier: int = 4

    # --- snapshot ---
    upcoming_horizon: float = 8.0   # seconds of special notes listed in Snapshot.upcoming
    popup_life: float = 0.8         # seconds a judgement popup stays in Snapshot.popups

    # --- force feedback (SPEC section 5) ---
    ffb_gain: float = 0.5           # global gain on top of every effect, never above 1.0
    echo_max_deg_s: float = 180.0   # echo centre speed limit, degrees of wheel per second

    # --- audio ---
    master_volume: float = 0.8

    # --- render ---
    lookahead: float = 2.5          # seconds of road visible ahead of the hit line; Snapshot.notes span

    def __post_init__(self) -> None:
        self.ffb_gain = min(1.0, max(0.0, float(self.ffb_gain)))
        self.spin_tolerance_deg = max(0.0, min(self.spin_perfect_deg - self.spin_good_deg,
                                               float(self.spin_tolerance_deg)))

    def lane_to_deg(self, x: float) -> float:
        return x * self.play_range_deg

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        """Build from a dict, ignoring unknown keys so old files keep loading.
        A partial `max_lane_rate` is merged over the defaults."""
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        if isinstance(kw.get("max_lane_rate"), dict):
            kw["max_lane_rate"] = {**cls().max_lane_rate, **kw["max_lane_rate"]}
        return cls(**kw)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load from a .json or .toml file; defaults when the file is missing. Key by key
        (`load_settings`): a bad key falls back to its default with one warning, and
        `settings_fallback(cfg)` is then true."""
        path = Path(path) if path else config_dir() / CONFIG_FILE
        if path.suffix == ".toml":
            return cls.from_dict(tomllib.loads(path.read_text())) if path.exists() else cls()
        return load_settings(cls, path, {"max_lane_rate": _lane_rates, "ffb_gain": _gain}, build=cls.from_dict)

    def save(self, path: str | Path | None = None) -> Path:
        """Save as JSON (TOML is read-only: the stdlib cannot write it). A file that fell
        back on load is first copied to `<name>.bad`."""
        path = Path(path) if path else config_dir() / CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        backup_bad(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


def _gain(v) -> bool:
    """A gain outside 0..1 (15 meant as 15%) is a bad value, not a clamp to full gain."""
    return valid_like(0.5, v) and 0.0 <= v <= 1.0


def finite(v) -> bool:
    """A finite number; a huge JSON integer (OverflowError) is not."""
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


def _lane_rates(v) -> bool:
    return isinstance(v, dict) and all(isinstance(k, str) and valid_like(1.0, x) for k, x in v.items())


def valid_like(ref, v) -> bool:
    """`v` has the type of the default `ref`: a finite number for a number, true/false for
    a flag, text for text, an object for an object. A None default accepts anything."""
    if isinstance(ref, bool):
        return isinstance(v, bool)
    if isinstance(ref, int | float):
        return isinstance(v, int | float) and not isinstance(v, bool) and finite(v)
    if isinstance(ref, str | dict | list):
        return isinstance(v, type(ref))
    return True


_fell_back: set[Path] = set()   # settings files that fell back on load; backed up before the first save


def settings_fallback(obj) -> bool:
    """True when `obj` came from a settings file that fell back (a bad file or a bad key)."""
    return bool(getattr(obj, "_fallback", False))


def load_settings(cls, path: Path, checks: dict | None = None, build=None):
    """A settings dataclass from the JSON object in `path`, key by key: unknown keys are
    ignored, a key that fails its check (`checks[name]`, else `valid_like` the default)
    keeps its default. Missing file: the defaults. Any fallback logs one warning, marks
    the result (`settings_fallback`) and backs the file up before the first save."""
    checks, build = checks or {}, build or (lambda kw: cls(**kw))
    defaults = cls()
    try:
        d = read_json_object(path)
    except (OSError, ValueError) as e:
        return _fallback(cls(), path, f"{path} is not usable ({e}): using the defaults")
    if d is None:
        return defaults
    kw, bad = {}, []
    for f in fields(cls):
        if f.name in d:
            v = d[f.name]
            if checks[f.name](v) if f.name in checks else valid_like(getattr(defaults, f.name), v):
                kw[f.name] = v
            else:
                bad.append(f"{f.name}={v!r}")
    try:
        obj = build(kw)
    except (TypeError, ValueError) as e:
        return _fallback(cls(), path, f"{path} is not usable ({e}): using the defaults")
    if bad:
        return _fallback(obj, path, f"{path}: ignored {', '.join(bad)}: using the defaults for them")
    return obj


def _fallback(obj, path: Path, msg: str):
    log.warning(msg)
    obj._fallback = True
    _fell_back.add(path.resolve())
    return obj


def backup_bad(path: Path) -> None:
    """Before the first save over a file that fell back on load, copy it to `<name>.bad`."""
    key = path.resolve()
    if key in _fell_back:
        if path.exists():
            shutil.copyfile(path, path.with_name(path.name + ".bad"))   # OSError: the marker stays
        _fell_back.discard(key)


def read_json_object(path: Path) -> dict | None:
    """The JSON object in `path`; None when the file is missing. ValueError when it is not an object."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if not isinstance(d, dict):
        raise ValueError("not a JSON object")
    return d


def config_dir() -> Path:
    """Platform config dir. `TORQUEHERO_CONFIG_DIR` overrides it."""
    if env := os.environ.get("TORQUEHERO_CONFIG_DIR"):
        return Path(env)
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME
