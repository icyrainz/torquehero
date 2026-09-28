"""Chart format v1 (SPEC 8.1) and the audio manifest (SPEC 8.2).

Times are seconds from song start. Lane positions are -1..1 (left..right).
Paths in the audio manifest are relative to the chart file.
"""
from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter

FORMAT = 1

# Each note kind belongs to exactly one layer.
KIND_LAYER = {
    "gate": "melody", "spin": "melody", "kick": "kick", "hat": "hat", "expr": "expr",
    "stab": "pads", "tom": "fills", "riser": "riser", "fader": "faders",
}
NOTE_KINDS = tuple(KIND_LAYER)
AUDIO_MODES = ("trigger", "gate", "filter", "level", "levers", "riser")
# Audio modes each layer may use (SPEC 8.9).
ALLOWED_MODES = {
    "melody": ("filter", "gate"), "expr": ("level",), "faders": ("levers",),
    "kick": ("trigger", "gate"), "hat": ("trigger", "gate"), "pads": ("trigger", "gate"),
    "fills": ("trigger", "gate"), "riser": ("riser",),
}
SPIN_DIRS = ("cw", "ccw")  # cw = steer_deg rising (to the right)
LAYER_MODES = ("you", "auto")

# Optional note fields and their default values; a default is omitted on save.
_NOTE_DEFAULTS = {
    "x": None, "listen": False, "blind": False, "dur": None, "vel": None, "open": False,
    "curve": None, "gate": None, "sample": None, "side": None, "lever": None, "dir": None,
}
_REQUIRED = {
    "gate": ("x",), "spin": ("dur",), "kick": (), "hat": (), "expr": ("dur", "curve"),
    "stab": ("gate",), "tom": ("side",), "riser": ("dur",), "fader": ("dur", "lever", "curve"),
}
# Every field a kind may carry besides t, kind, layer.
_FIELDS = {
    "gate": {"x", "listen", "blind"}, "spin": {"dur", "dir"}, "kick": {"vel"}, "hat": {"open"},
    "expr": {"dur", "curve"}, "stab": {"gate", "sample"}, "tom": {"side"}, "riser": {"dur"},
    "fader": {"dur", "lever", "curve"},
}
TEMPLATE_FIELDS = ("gate", "side")
ECHO_PEAK_FACTOR = 1.5  # peak speed / mean speed of one smoothstep segment of phrase_x

DEFAULT_LAYER_AUDIO = {
    "melody": {"stem": "lead", "mode": "filter"},
    "expr": {"stem": "pad", "mode": "level"},
    "faders": {"stem": "bass", "mode": "levers"},
    "kick": {"oneshot": "kick", "mode": "trigger"},
    "hat": {"oneshot": "hat", "mode": "trigger"},
    "pads": {"oneshot": "stab{gate}", "mode": "trigger"},
    "fills": {"oneshot": "tom_{side}", "mode": "trigger"},
    "riser": {"oneshot": "riser", "mode": "riser"},
}


class ChartError(ValueError):
    """Invalid chart. `errors` lists every problem found, each naming its location."""

    def __init__(self, errors: list[str], source: str = "chart"):
        self.errors = errors
        super().__init__(f"{source}: {len(errors)} error(s)\n  " + "\n  ".join(errors))


def smoothstep(u: float) -> float:
    return 0.0 if u <= 0 else 1.0 if u >= 1 else u * u * (3 - 2 * u)


def curve_at(curve: list[list[float]], dt: float) -> float:
    """Value of a `[[dt, v], ...]` curve at `dt` seconds into the note. Linear
    between points, held flat before the first and after the last."""
    if not curve:
        return 0.0
    if dt <= curve[0][0]:
        return curve[0][1]
    for (a, va), (b, vb) in zip(curve, curve[1:], strict=False):
        if dt <= b:
            return va + (vb - va) * ((dt - a) / (b - a) if b > a else 1.0)
    return curve[-1][1]


def phrase_x(points: list[tuple[float, float]], t: float) -> float:
    """Echo demonstration path through listen gates `[(t, x), ...]` (sorted): a
    smoothstep from each gate's x to the next, reaching every x at its time. Held
    at the first x before it and the last x after. Peak speed of a segment is
    ECHO_PEAK_FACTOR times its mean speed."""
    if not points:
        return 0.0
    if t <= points[0][0]:
        return points[0][1]
    for (ta, xa), (tb, xb) in zip(points, points[1:], strict=False):
        if t <= tb:
            return xa + (xb - xa) * smoothstep((t - ta) / (tb - ta) if tb > ta else 1.0)
    return points[-1][1]


def road_at(road: list[RoadKey], t: float) -> float:
    """Centreline through keyframes `road` at `t`: hold `a.x` for the first 55% of each
    keyframe interval, then smoothstep to `b.x`. 0 before the first keyframe, last x after."""
    if not road or t < road[0].t:
        return 0.0
    i = bisect_right([k.t for k in road], t) - 1
    a = road[i]
    if i + 1 >= len(road):
        return a.x
    b = road[i + 1]
    u = (t - a.t) / (b.t - a.t)
    return a.x if u < 0.55 else a.x + (b.x - a.x) * smoothstep((u - 0.55) / 0.45)


def fill_template(tmpl: str, gate: int = 0, side: str = "") -> str:
    """Fill a one-shot template: `{gate}` 1..6, `{side}` "l"/"r"."""
    return tmpl.format(gate=gate, side=side.lower())


def template_fields(tmpl: str) -> set[str]:
    return {f for _, f, _, _ in Formatter().parse(tmpl) if f is not None}


@dataclass
class Note:
    t: float
    kind: str
    layer: str
    x: float | None = None           # gate: lane position
    listen: bool = False             # gate: echo demonstration, not judged
    blind: bool = False              # gate: echo playback, road hidden
    dur: float | None = None         # spin, expr, riser, fader: seconds
    vel: float | None = None         # kick: 0..1 (auto-layer velocity)
    open: bool = False               # hat: open hat
    curve: list[list[float]] | None = None  # expr, fader: [[dt, v], ...], v 0..1
    gate: int | None = None          # stab: shifter gate 1..6
    sample: str | None = None        # stab: one-shot name overriding the layer template
    side: str | None = None          # tom: "L" or "R"
    lever: int | None = None         # fader: 0 or 1
    dir: str | None = None           # spin: "cw" or "ccw"; None accepts either

    @property
    def end(self) -> float:
        return self.t + (self.dur or 0.0)

    def value_at(self, now: float) -> float:
        """Target curve value (expr, fader) at song time `now`."""
        return curve_at(self.curve or [], now - self.t)

    def to_dict(self) -> dict:
        d = {"t": self.t, "kind": self.kind, "layer": self.layer}
        for k, default in _NOTE_DEFAULTS.items():
            v = getattr(self, k)
            if v != default:
                d[k] = v
        return d


@dataclass
class Section:
    t: float
    name: str
    weight: float  # 0..1, drives FFB section weight


@dataclass
class BeatMark:
    t: float
    s: float  # strength 0..1


@dataclass
class RoadKey:
    t: float
    x: float  # lane units


@dataclass
class LayerAudio:
    mode: str
    stem: str | None = None
    oneshot: str | None = None  # may be a template: "stab{gate}", "tom_{side}"

    def to_dict(self) -> dict:
        d = {"mode": self.mode}
        if self.stem is not None:
            d["stem"] = self.stem
        if self.oneshot is not None:
            d["oneshot"] = self.oneshot
        return d


@dataclass
class AudioManifest:
    sr: int = 48000
    backing: list[str] = field(default_factory=list)
    stems: dict[str, str] = field(default_factory=dict)
    oneshots: dict[str, str] = field(default_factory=dict)
    layers: dict[str, LayerAudio] = field(
        default_factory=lambda: {k: LayerAudio(**v) for k, v in DEFAULT_LAYER_AUDIO.items()})

    def oneshot_for(self, layer: str, note: Note | None = None, *, gate: int = 0, side: str = "") -> str | None:
        """Resolved one-shot name for a layer, filling `{gate}` and `{side}` (lower case)."""
        la = self.layers.get(layer)
        if la is None or la.oneshot is None:
            return None
        if note is not None and note.sample:
            return note.sample
        gate = note.gate if note is not None and note.gate else gate
        side = note.side if note is not None and note.side else side
        return fill_template(la.oneshot, gate, side)

    def mode(self, layer: str) -> str | None:
        la = self.layers.get(layer)
        return la.mode if la else None

    @classmethod
    def from_dict(cls, d: dict) -> AudioManifest:
        layers = {k: LayerAudio(**v) for k, v in d["layers"].items()} if "layers" in d else None
        m = cls(sr=d.get("sr", 48000), backing=list(d.get("backing", [])),
                stems=dict(d.get("stems", {})), oneshots=dict(d.get("oneshots", {})))
        if layers is not None:
            m.layers = layers
        return m

    def to_dict(self) -> dict:
        return {"sr": self.sr, "backing": self.backing, "stems": self.stems, "oneshots": self.oneshots,
                "layers": {k: v.to_dict() for k, v in self.layers.items()}}


@dataclass
class Chart:
    title: str
    bpm: float
    length: float
    artist: str = ""
    sections: list[Section] = field(default_factory=list)
    beats: list[BeatMark] = field(default_factory=list)
    road: list[RoadKey] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    audio: AudioManifest = field(default_factory=AudioManifest)
    base_dir: Path = field(default_factory=Path.cwd, repr=False, compare=False)
    default_layers: dict[str, str] = field(default_factory=dict)  # "defaults.layers": layer -> "you" | "auto"

    # --- geometry ---

    def road_x(self, t: float) -> float:
        """Melody centreline at `t` (see `road_at`)."""
        return road_at(self.road, t)

    def blind_zones(self, good_window: float = 0.12) -> list[tuple[float, float]]:
        """(start, end) of every blind zone. A blind run (consecutive blind gates) opens one
        beat before its first gate, never before the previous listen run closes, and closes
        `good_window` after its last gate. Its zone runs from the last road keyframe at or
        before the open to the first keyframe at or after the close (the song end if none)."""
        lead = 60.0 / self.bpm
        runs: list[tuple[str, list[float]]] = []
        for n in self.notes:
            if n.kind != "gate":
                continue
            k = "listen" if n.listen else "blind" if n.blind else ""
            if runs and runs[-1][0] == k:
                runs[-1][1].append(n.t)
            else:
                runs.append((k, [n.t]))
        keys = [k.t for k in self.road]
        zones = []
        listen_close = -math.inf
        for k, ts in runs:
            if k == "listen":
                listen_close = ts[-1] + good_window
            elif k == "blind":
                opens, closes = max(ts[0] - lead, listen_close), ts[-1] + good_window
                i, j = bisect_right(keys, opens) - 1, bisect_left(keys, closes)
                zones.append((keys[i] if i >= 0 else opens, keys[j] if j < len(keys) else max(closes, self.length)))
        return zones

    def drawn_road(self, good_window: float = 0.12) -> list[RoadKey]:
        """Road keyframes as drawn: every keyframe strictly inside a blind zone pinned to x = 0,
        so the road blends to the centre line and back by the `road_at` rule."""
        zones = self.blind_zones(good_window)
        return [RoadKey(k.t, 0.0 if any(a < k.t < b for a, b in zones) else k.x) for k in self.road]

    def drawn_road_x(self, t: float, good_window: float = 0.12) -> float:
        """The road as shown at `t` (SPEC 10, one drawn road): blind runs removed."""
        return road_at(self.drawn_road(good_window), t)

    def section_at(self, t: float) -> Section | None:
        cur = None
        for s in self.sections:
            if s.t > t:
                break
            cur = s
        return cur

    def path(self, rel: str) -> Path:
        """Absolute path of a file named in the manifest."""
        return (self.base_dir / rel).resolve()

    # --- I/O ---

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | Path | None = None, *, validate: bool = True,
                  play_range_deg: float = 90.0, echo_max_deg_s: float = 180.0) -> Chart:
        if validate:
            errors = validate_dict(d, play_range_deg, echo_max_deg_s)
            if errors:
                raise ChartError(errors, str(base_dir or "chart"))
        return cls(
            title=d["title"], artist=d.get("artist", ""), bpm=float(d["bpm"]), length=float(d["length"]),
            sections=[Section(float(s["t"]), s["name"], float(s["weight"])) for s in d.get("sections", [])],
            beats=[BeatMark(float(b["t"]), float(b.get("s", 1.0))) for b in d.get("beats", [])],
            road=[RoadKey(float(k["t"]), float(k["x"])) for k in d.get("road", [])],
            notes=[Note(**n) for n in d.get("notes", [])],
            audio=AudioManifest.from_dict(d.get("audio", {})),
            base_dir=Path(base_dir) if base_dir else Path.cwd(),
            default_layers=dict(d.get("defaults", {}).get("layers", {})),
        )

    def to_dict(self) -> dict:
        d = {
            "format": FORMAT, "title": self.title, "artist": self.artist, "bpm": self.bpm, "length": self.length,
            "sections": [{"t": s.t, "name": s.name, "weight": s.weight} for s in self.sections],
            "beats": [{"t": b.t, "s": b.s} for b in self.beats],
            "road": [{"t": k.t, "x": k.x} for k in self.road],
            "notes": [n.to_dict() for n in self.notes],
            "audio": self.audio.to_dict(),
        }
        if self.default_layers:
            d["defaults"] = {"layers": dict(self.default_layers)}
        return d

    def controls_used(self) -> set[str]:
        """Performance controls (SPEC 8.3 names) that this chart's notes need."""
        used = set()
        for n in self.notes:
            match n.kind:
                case "gate" | "spin":
                    used.add("steer")
                case "kick":
                    used.add("brake")
                case "hat":
                    used.add("clutch")
                case "expr":
                    used.add("throttle")
                case "stab":
                    used.add(f"gate{n.gate}")
                case "tom":
                    used.add("paddle_l" if n.side == "L" else "paddle_r")
                case "riser":
                    used.add("handbrake")
                case "fader":
                    used.add(f"lever{n.lever}")
        return used

    def validate(self, play_range_deg: float = 90.0, echo_max_deg_s: float = 180.0) -> None:
        """Raise ChartError listing every problem. Pass Config.play_range_deg and
        Config.echo_max_deg_s to check the echo limit against the player's settings."""
        errors = validate_dict(self.to_dict(), play_range_deg, echo_max_deg_s)
        if errors:
            raise ChartError(errors, self.title or "chart")

    @classmethod
    def load(cls, path: str | Path, play_range_deg: float = 90.0, echo_max_deg_s: float = 180.0) -> Chart:
        path = Path(path)
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ChartError([f"not valid JSON: {e}"], str(path)) from e
        errors = validate_dict(d, play_range_deg, echo_max_deg_s)
        if errors:
            raise ChartError(errors, str(path))
        return cls.from_dict(d, path.parent.resolve(), validate=False)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n")
        self.base_dir = path.parent.resolve()


# --- validation ---

def _num(v) -> bool:
    """A finite number that fits a float (JSON integers can be arbitrarily large)."""
    if not isinstance(v, int | float) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(float(v))
    except OverflowError:
        return False


def _check_curve(curve, dur, where: str, errors: list[str]) -> None:
    if not isinstance(curve, list) or not curve:
        errors.append(f"{where}: 'curve' must be a non-empty list of [dt, v]")
        return
    prev = -1.0
    for j, p in enumerate(curve):
        if not (isinstance(p, list | tuple) and len(p) == 2 and _num(p[0]) and _num(p[1])):
            errors.append(f"{where}: curve[{j}] must be [dt, v] with finite numbers")
            return
        dt, v = p
        if dt < prev:
            errors.append(f"{where}: curve[{j}] dt {dt} is before the previous point")
        if _num(dur) and not 0 <= dt <= dur + 1e-9:
            errors.append(f"{where}: curve[{j}] dt {dt} outside 0..dur ({dur})")
        if not 0 <= v <= 1:
            errors.append(f"{where}: curve[{j}] v {v} outside 0..1")
        prev = dt


def _check_sorted(items: list, name: str, errors: list[str]) -> None:
    ts = [x.get("t") for x in items if isinstance(x, dict) and _num(x.get("t"))]
    for i in range(1, len(ts)):
        if ts[i] < ts[i - 1]:
            errors.append(f"{name}[{i}]: t {ts[i]} is before {name}[{i - 1}] t {ts[i - 1]} (must be sorted by t)")
            return


def validate_dict(d: dict, play_range_deg: float = 90.0, echo_max_deg_s: float = 180.0) -> list[str]:
    """Every problem in a chart dict, as readable strings. Empty list means valid.
    Never raises: a note with any error is left out of the later cross-checks."""
    errors: list[str] = []
    if not isinstance(d, dict):
        return ["chart must be a JSON object"]
    if d.get("format") != FORMAT:
        errors.append(f"format: expected {FORMAT}, got {d.get('format')!r}")
    if not isinstance(d.get("title"), str):
        errors.append("title: required string")
    if "artist" in d and not isinstance(d["artist"], str):
        errors.append("artist: must be a string")
    for k in ("bpm", "length"):
        if not _num(d.get(k)) or d[k] <= 0:
            errors.append(f"{k}: required positive finite number")
    length = d.get("length") if _num(d.get("length")) else None

    lists = {}
    for k in ("sections", "beats", "road", "notes"):
        v = d.get(k, [])
        if not isinstance(v, list):
            errors.append(f"{k}: must be a list")
            v = []
        lists[k] = v
        _check_sorted(v, k, errors)

    for i, s in enumerate(lists["sections"]):
        w = f"sections[{i}]"
        if not isinstance(s, dict) or not _num(s.get("t")) or not isinstance(s.get("name"), str):
            errors.append(f"{w}: needs finite 't' and string 'name'")
            continue
        if not _num(s.get("weight")) or not 0 <= s["weight"] <= 1:
            errors.append(f"{w}: 'weight' must be 0..1")
    for i, b in enumerate(lists["beats"]):
        if not isinstance(b, dict) or not _num(b.get("t")):
            errors.append(f"beats[{i}]: needs finite 't'")
        elif "s" in b and (not _num(b["s"]) or not 0 <= b["s"] <= 1):
            errors.append(f"beats[{i}]: 's' must be 0..1")
    for i, k in enumerate(lists["road"]):
        if not isinstance(k, dict) or not _num(k.get("t")) or not _num(k.get("x")):
            errors.append(f"road[{i}]: needs finite 't' and 'x'")
        elif not -1 <= k["x"] <= 1:
            errors.append(f"road[{i}]: x {k['x']} outside -1..1")

    valid_notes: list[tuple[int, dict]] = []
    for i, n in enumerate(lists["notes"]):
        if not isinstance(n, dict):
            errors.append(f"notes[{i}]: must be an object")
            continue
        kind = n.get("kind")
        w = f"notes[{i}] ({kind} @ {n.get('t')})"
        n_errors = len(errors)
        if not _num(n.get("t")) or n["t"] < 0:
            errors.append(f"{w}: 't' must be a finite number >= 0")
            continue
        if not isinstance(kind, str) or kind not in KIND_LAYER:
            errors.append(f"{w}: unknown kind {kind!r}, expected one of {', '.join(NOTE_KINDS)}")
            continue
        if n.get("layer") != KIND_LAYER[kind]:
            errors.append(f"{w}: layer must be {KIND_LAYER[kind]!r}, got {n.get('layer')!r}")
        extra = set(n) - {"t", "kind", "layer"}
        unknown = extra - set(_NOTE_DEFAULTS)
        if unknown:
            errors.append(f"{w}: unknown field(s) {', '.join(sorted(unknown))}")
        foreign = extra - unknown - _FIELDS[kind]
        if foreign:
            errors.append(f"{w}: field(s) {', '.join(sorted(foreign))} do not belong to kind {kind!r}")
        missing = [f for f in _REQUIRED[kind] if f not in n]
        if missing:
            errors.append(f"{w}: missing field(s) {', '.join(missing)}")
            continue
        for b in ("listen", "blind", "open"):
            if b in n and not isinstance(n[b], bool):
                errors.append(f"{w}: '{b}' must be true or false")
        if "dur" in n:
            if not _num(n["dur"]) or n["dur"] <= 0:
                errors.append(f"{w}: 'dur' must be a positive finite number")
            elif length is not None and n["t"] + n["dur"] > length + 1e-9:
                errors.append(f"{w}: ends at {n['t'] + n['dur']}, after the song length {length}")
        if length is not None and n["t"] > length:
            errors.append(f"{w}: t is after the song length {length}")
        if kind == "gate":
            if not _num(n["x"]) or not -1 <= n["x"] <= 1:
                errors.append(f"{w}: x must be -1..1")
            if n.get("listen") is True and n.get("blind") is True:
                errors.append(f"{w}: 'listen' and 'blind' are exclusive")
        elif kind == "kick" and "vel" in n and (not _num(n["vel"]) or not 0 <= n["vel"] <= 1):
            errors.append(f"{w}: vel must be 0..1")
        elif kind in ("expr", "fader"):
            _check_curve(n["curve"], n.get("dur"), w, errors)
            if kind == "fader" and n["lever"] not in (0, 1):
                errors.append(f"{w}: lever must be 0 or 1")
        elif kind == "stab":
            if not isinstance(n["gate"], int) or isinstance(n["gate"], bool) or not 1 <= n["gate"] <= 6:
                errors.append(f"{w}: gate must be an integer 1..6")
            if "sample" in n and (not isinstance(n["sample"], str) or not n["sample"]):
                errors.append(f"{w}: sample must be a non-empty string")
        elif kind == "tom" and n["side"] not in ("L", "R"):
            errors.append(f"{w}: side must be 'L' or 'R'")
        elif kind == "spin" and "dir" in n and n["dir"] not in SPIN_DIRS:
            errors.append(f"{w}: dir must be 'cw' or 'ccw'")
        if len(errors) == n_errors:
            valid_notes.append((i, n))

    errors += _check_echo_rate(valid_notes, play_range_deg, echo_max_deg_s)
    if "defaults" in d:
        errors += _validate_defaults(d["defaults"])

    audio = d.get("audio")
    if audio is None:
        errors.append("audio: required manifest (SPEC 8.2)")
    else:
        errors += _validate_audio(audio, [n for _, n in valid_notes])
    return errors


def _check_echo_rate(notes: list[tuple[int, dict]], play_range_deg: float, max_deg_s: float) -> list[str]:
    """Peak speed of phrase_x between consecutive listen gates (no other gate between)."""
    errors = []
    prev: dict | None = None
    for i, n in notes:
        if n["kind"] != "gate" or not _num(n.get("x")):
            continue
        if n.get("listen") is True:
            if prev is not None and n["t"] == prev["t"] and n["x"] != prev["x"]:
                errors.append(f"notes[{i}] (listen gate @ {n['t']}): same time as the previous listen gate "
                              f"but a different x; the base cannot be in two places")
            elif prev is not None and n["t"] > prev["t"]:
                peak = ECHO_PEAK_FACTOR * abs(n["x"] - prev["x"]) * play_range_deg / (n["t"] - prev["t"])
                if peak > max_deg_s + 1e-6:
                    errors.append(f"notes[{i}] (listen gate @ {n['t']}): echo path peaks at {peak:.0f} deg/s, "
                                  f"over the limit of {max_deg_s:.0f} deg/s")
            prev = n
        else:
            prev = None
    return errors


def _validate_defaults(defaults) -> list[str]:
    if not isinstance(defaults, dict):
        return ["defaults: must be an object"]
    errors = [f"defaults: unknown field {k!r}" for k in defaults if k != "layers"]
    layers = defaults.get("layers", {})
    if not isinstance(layers, dict):
        return errors + ["defaults.layers: must map layer names to 'you' or 'auto'"]
    for k, v in layers.items():
        if k not in DEFAULT_LAYER_AUDIO:
            errors.append(f"defaults.layers: unknown layer {k!r}")
        elif v not in LAYER_MODES:
            errors.append(f"defaults.layers.{k}: must be 'you' or 'auto', got {v!r}")
    return errors


def _validate_audio(a, notes: list[dict]) -> list[str]:
    errors: list[str] = []
    if not isinstance(a, dict):
        return ["audio: must be an object"]
    if not isinstance(a.get("sr"), int) or isinstance(a.get("sr"), bool) or a["sr"] <= 0:
        errors.append("audio.sr: required positive integer")
    if not isinstance(a.get("backing", []), list) or not all(isinstance(p, str) for p in a.get("backing", [])):
        errors.append("audio.backing: must be a list of paths")
    for k in ("stems", "oneshots"):
        v = a.get(k, {})
        if not isinstance(v, dict) or not all(isinstance(p, str) for p in v.values()):
            errors.append(f"audio.{k}: must map names to paths")
    stems = a.get("stems") if isinstance(a.get("stems"), dict) else {}
    oneshots = a.get("oneshots") if isinstance(a.get("oneshots"), dict) else {}
    layers = a.get("layers", {})
    if not isinstance(layers, dict):
        return errors + ["audio.layers: must be an object"]

    dud_layers = [k for k in ("pads", "fills") if isinstance(layers.get(k), dict)
                  and layers[k].get("mode") in ("trigger", "gate") and layers[k].get("oneshot") is not None]
    by_layer: dict[str, list[dict]] = {}
    for n in notes:
        by_layer.setdefault(KIND_LAYER[n["kind"]], []).append(n)
    for layer in by_layer:
        if layer not in layers:
            errors.append(f"audio.layers.{layer}: required, the chart has {layer} notes")

    for name, la in layers.items():
        w = f"audio.layers.{name}"
        if name not in DEFAULT_LAYER_AUDIO:
            errors.append(f"{w}: unknown layer, expected one of {', '.join(DEFAULT_LAYER_AUDIO)}")
        if not isinstance(la, dict):
            errors.append(f"{w}: must be an object")
            continue
        unknown = set(la) - {"mode", "stem", "oneshot"}
        if unknown:
            errors.append(f"{w}: unknown field(s) {', '.join(sorted(unknown))}")
        mode = la.get("mode")
        if mode not in AUDIO_MODES:
            errors.append(f"{w}: mode must be one of {', '.join(AUDIO_MODES)}, got {mode!r}")
            continue
        if name in ALLOWED_MODES and mode not in ALLOWED_MODES[name]:
            errors.append(f"{w}: mode {mode!r} is not allowed for {name}, expected {' or '.join(ALLOWED_MODES[name])}")
        if mode in ("gate", "filter", "level", "levers") and "stem" not in la:
            errors.append(f"{w}: mode {mode!r} needs 'stem'")
        if mode in ("trigger", "riser") and "oneshot" not in la:
            errors.append(f"{w}: mode {mode!r} needs 'oneshot'")
        if "stem" in la and not isinstance(la["stem"], str):
            errors.append(f"{w}: stem must be a string")
        elif "stem" in la and la["stem"] not in stems:
            errors.append(f"{w}: stem {la['stem']!r} is not in audio.stems")
        tmpl = la.get("oneshot")
        if tmpl is None:
            continue
        if not isinstance(tmpl, str):
            errors.append(f"{w}: oneshot must be a string")
            continue
        try:
            bad = template_fields(tmpl) - set(TEMPLATE_FIELDS)
        except ValueError as e:
            errors.append(f"{w}: oneshot template {tmpl!r} is malformed: {e}")
            continue
        if bad:
            errors.append(f"{w}: oneshot template {tmpl!r} uses {', '.join(sorted(bad))}; "
                          f"only {{gate}} and {{side}} are allowed")
            continue
        try:
            names = _emittable(name, mode, tmpl, by_layer.get(name, []))
        except (ValueError, KeyError, IndexError, AttributeError, TypeError) as e:
            errors.append(f"{w}: oneshot template {tmpl!r} cannot be filled: {e}")
            continue
        for needed in sorted(names):
            if needed not in oneshots:
                errors.append(f"audio.oneshots: missing {needed!r}, which the {name} layer can play")
    if dud_layers and "dud" not in oneshots:
        errors.append(f"audio.oneshots: missing 'dud', which the {' and '.join(dud_layers)} layer can play")
    return errors


def _emittable(layer: str, mode: str, tmpl: str, notes: list[dict]) -> set[str]:
    """One-shot names the game can emit for a layer entry, notes or not (see state.Sound).
    Stabs are the exception: only the gates and samples the notes use. "dud" is
    checked by the caller."""
    if mode not in ("trigger", "gate", "riser"):
        return set()
    names = set()
    for n in notes:
        if isinstance(n.get("sample"), str) and n["sample"]:
            names.add(n["sample"])
        elif layer == "pads":
            names.add(fill_template(tmpl, n["gate"]))
    if layer == "fills":
        names |= {fill_template(tmpl, 0, "l"), fill_template(tmpl, 0, "r")}
    elif layer != "pads":
        names.add(fill_template(tmpl))
    if mode == "riser":
        names.add("impact")
    return names


# --- playability (SPEC 9) ---

FREE_HAND_JUMP = 0.4  # lane units a gate may move while no hand is firmly on the wheel (rule 5)
_EPS = 1e-3  # 1 ms slack on time margins, so a margin of exactly one beat passes


def playability(chart_or_dict, difficulty: str = "normal", cfg=None) -> list[str]:
    """Every violation of SPEC 9 rules 1 to 11, as readable strings naming the rule,
    note indices and times. Empty list means playable. Never raises; an invalid
    chart dict gives one message. `cfg` is a Config (defaults when None).
    `validate_dict` does not call this; the CLI and the generators do."""
    from .config import Config

    try:
        cfg = cfg or Config()
        if isinstance(chart_or_dict, Chart):
            chart = chart_or_dict
        else:
            errors = validate_dict(chart_or_dict, cfg.play_range_deg, cfg.echo_max_deg_s)
            if errors:
                return [f"chart is not valid ({len(errors)} error(s)); playability not checked"]
            chart = Chart.from_dict(chart_or_dict, validate=False)
        if difficulty not in cfg.max_lane_rate:
            return [f"unknown difficulty {difficulty!r}, expected one of {', '.join(cfg.max_lane_rate)}"]
        return _playability(chart, cfg.max_lane_rate[difficulty], cfg.limb_travel, cfg.expr_perfect)
    except Exception as e:  # noqa: BLE001 - a report, never a crash
        return [f"playability check failed: {type(e).__name__}: {e}"]


def _at(i: int, n: Note) -> str:
    return f"notes[{i}] ({n.kind} @ {n.t:.3f})"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 - _EPS and b0 < a1 - _EPS


def _playability(chart: Chart, lane_rate: float, travel: float, jump_tol: float) -> list[str]:
    out: list[str] = []
    notes = list(enumerate(chart.notes))
    by = {k: [(i, n) for i, n in notes if n.kind == k] for k in NOTE_KINDS}
    gates = by["gate"]
    beat = 60.0 / chart.bpm

    # Rules 1-3, hands. Spins and toms hold both hands on the wheel, padded by the travel time.
    wheel = [(i, n, n.t - travel, n.end + travel) for i, n in by["spin"] + by["tom"]]
    for i, n in by["stab"]:
        for j, r in by["riser"]:
            if r.t - travel + _EPS < n.t < r.end + travel - _EPS:
                out.append(f"rule 3: {_at(i, n)} is within {travel} s of the riser {_at(j, r)} (right hand)")
    for j, w, lo, hi in wheel:
        for i, n in by["stab"]:
            if lo + _EPS < n.t < hi - _EPS:
                out.append(f"rule 3: {_at(i, n)} is within {travel} s of {_at(j, w)} (both hands on the wheel)")
        for i, n in by["riser"] + by["fader"]:
            if _overlap(n.t, n.end, lo, hi):
                out.append(f"rule 3: {_at(i, n)} overlaps {_at(j, w)} or its {travel} s margin "
                           f"(both hands on the wheel)")

    # Rule 4: one hand for both levers. It can hold both, so a hand-over at the same instant is fine.
    faders = sorted(by["fader"], key=lambda p: p[1].t)
    for a, (i, n) in enumerate(faders):
        for j, m in faders[a + 1:]:
            if m.lever != n.lever and _overlap(n.t, n.end, m.t, m.end):
                out.append(f"rule 4: {_at(i, n)} (lever {n.lever}) overlaps {_at(j, m)} (lever {m.lever})")
    for lever in (0, 1):
        run = [(i, n) for i, n in faders if n.lever == lever]
        for (i, n), (j, m) in zip(run, run[1:], strict=False):
            if m.t - n.end >= travel - _EPS:
                continue  # time to move the lever
            end, start = curve_at(n.curve or [], n.dur or 0.0), curve_at(m.curve or [], 0.0)
            if abs(start - end) > jump_tol:
                out.append(f"rule 4: lever {lever} target jumps from {end:.2f} at the end of {_at(i, n)} "
                           f"to {start:.2f} at {_at(j, m)}")

    # Rule 5: a fader with a right-hand note leaves no hand firmly on the wheel.
    right = [(i, n, n.t - travel, n.end + travel) for i, n in by["stab"] + by["riser"]]
    for i, f in by["fader"]:
        for j, r, lo, hi in right:
            if not _overlap(f.t, f.end, lo, hi):
                continue
            span = (max(f.t, lo), min(f.end, hi))
            for k, (gi, g) in enumerate(gates):
                if k and span[0] <= g.t <= span[1] and abs((g.x or 0.0) - (gates[k - 1][1].x or 0.0)) > FREE_HAND_JUMP:
                    out.append(f"rule 5: {_at(gi, g)} moves more than {FREE_HAND_JUMP} lane units while "
                               f"{_at(i, f)} and {_at(j, r)} hold both hands off the wheel")

    # Rules 6-7, feet. Near an expr note the right foot is on the throttle, so the
    # left foot plays kicks as well as hats.
    exprs = [(n.t - travel, n.end + travel) for _, n in by["expr"]]
    left = sorted(by["hat"] + [(i, n) for i, n in by["kick"] if any(lo < n.t < hi for lo, hi in exprs)],
                  key=lambda p: p[1].t)
    for (i, n), (j, m) in zip(left, left[1:], strict=False):
        if n.kind != m.kind and m.t - n.t < travel - _EPS:
            out.append(f"rule 7: {_at(i, n)} and {_at(j, m)} are {m.t - n.t:.3f} s apart for the left foot "
                       f"(an expr note holds the right foot on the throttle); need {travel} s")

    # Rule 8: steering speed, gate to gate (listen gates are played by the base).
    played = [(i, n) for i, n in gates if not n.listen]
    for (i, n), (j, m) in zip(played, played[1:], strict=False):
        dx, dt = abs((m.x or 0.0) - (n.x or 0.0)), m.t - n.t
        if dx > _EPS and (dt <= 0 or dx / dt > lane_rate + 1e-6):
            rate = f"{dx / dt:.2f}" if dt > 0 else "an instant"
            out.append(f"rule 8: {_at(i, n)} to {_at(j, m)} needs {rate} lane units/s, over {lane_rate}")

    # Rule 9: the road starts at x 0 on t 0 and never steps.
    road = chart.road
    if not road or road[0].t != 0 or road[0].x != 0:
        out.append("rule 9: road needs a keyframe at t 0 with x 0")
    for k in range(1, len(road)):
        if road[k].t == road[k - 1].t and road[k].x != road[k - 1].x:
            out.append(f"rule 9: road steps at t {road[k].t:.3f} (road[{k - 1}] and road[{k}])")

    # Rule 10: spins carry a direction and alternate; the offset stays within one turn.
    offset = 0
    for i, n in by["spin"]:
        if n.dir is None:
            out.append(f"rule 10: {_at(i, n)} has no dir")
            continue
        offset += 1 if n.dir == "cw" else -1
        if abs(offset) > 1:
            out.append(f"rule 10: {_at(i, n)} turns {n.dir} again; the offset would pass one turn")
            offset = max(-1, min(1, offset))
    if offset:
        out.append("rule 10: spins end one turn off centre; the last spin needs a partner in the other direction")

    # Rule 11: no gate within one beat of a spin.
    for i, s in by["spin"]:
        for j, g in gates:
            if s.t - beat + _EPS < g.t < s.end + beat - _EPS:
                out.append(f"rule 11: {_at(j, g)} is within one beat ({beat:.3f} s) of {_at(i, s)}")
    return out
