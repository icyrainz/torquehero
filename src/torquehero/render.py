"""raylib scene drawn from a `Snapshot` and the chart (SPEC 2, 3, 8.4, 8.6, 9, 10).

A sunset road that bends with the melody; notes flow toward the hit line. The
look follows `docs/design/browser-prototype.html` (proj, drawSky, drawRoad, drawNotes, drawPlayer).

Layout: the road and every judged note stay inside the centre region, 16:9 of
the window height (the middle 1920 px of a 5760x1080 surround). Scenery,
shoulders and roadside posts use the rest. A wider window widens the world; the
focal length comes from the centre width, so nothing stretches.

Blind echo phase (SPEC 10): nothing reveals the path. The road is drawn straight
and dim at lane 0, blind gates are hidden, and everything anchored to the road
(shoulder notes, riser band, fader lanes, lever markers, spin ring, posts,
popups) is anchored to that straight centre line. `SceneGeo` owns every x.

Use from the app:

    with Window(span=(5760, 1080, 0, 0)) as win:   # or fullscreen=True, hidden=True
        view = win.own(Renderer(chart, cfg, win.settings))       # closed with the window
        keys = RaylibKeys()                        # KeySource for input.py
        while not win.should_close():
            win.frame(lambda: view.draw(snapshot, win.layout))

Maths (Layout, RoadPath, SceneGeo, parse_span, sky_state, ...) is pure and has
no raylib import. pyray is imported inside functions only. Smoke test:
`python -m torquehero.render --hidden --frames 300 --screenshot out.png`;
timing: `python -m torquehero.render --bench`.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .chart import Chart
from .config import Config, backup_bad, config_dir, load_settings
from .state import (
    GATES,
    LAYERS,
    InputState,
    LayerStatus,
    NoteView,
    Popup,
    RiserStatus,
    Snapshot,
    spin_running,
)

log = logging.getLogger("torquehero.render")

# --- palette (demo COL) ---

Color = tuple[int, int, int, int]


def hexc(s: str, a: float = 1.0) -> Color:
    s = s.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), round(255 * a)


def fade(c: Color, a: float) -> Color:
    return c[0], c[1], c[2], max(0, min(255, round(c[3] * a)))


def lerp_color(a: Color, b: Color, u: float) -> Color:
    u = min(1.0, max(0.0, u))
    return tuple(round(x + (y - x) * u) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


COL = {
    "bg": hexc("#0c0a1c"), "bg2": hexc("#15122b"), "line": hexc("#2b2647"),
    "ink": hexc("#f2efe8"), "dim": hexc("#a9a3c2"),
    "sun": hexc("#ff7a45"), "cyan": hexc("#34e5ff"), "magenta": hexc("#ff3fa4"), "amber": hexc("#ffc844"),
    "red": hexc("#ff4d4d"), "green": hexc("#5cff8a"), "violet": hexc("#b48cff"), "mint": hexc("#6ef2d2"),
    "sky": hexc("#8ad8ff"), "post": hexc("#3a2f66"),
    "ground_top": hexc("#2a1440"), "ground_bot": hexc("#0a0716"),
    "sky_mid_lo": hexc("#33184a"), "sky_mid_hi": hexc("#4a1f5e"),
    "horizon_lo": hexc("#b8503a"), "horizon_flash": hexc("#ffb070"),
}
SHOULDER_COL = (hexc("#1a1531"), hexc("#1d1838"))   # even, odd beat
PAVED_COL = (hexc("#221d42"), hexc("#252047"))
PAVED_BAR = hexc("#2e2858")
WHITE = (255, 255, 255, 255)
# Layer display name and colour (demo LAYERS, plus faders).
LAYER_STYLE = {
    "melody": ("MELODY", COL["cyan"]), "kick": ("KICK", COL["red"]), "hat": ("HATS", COL["amber"]),
    "expr": ("SWELL", COL["green"]), "pads": ("STABS", COL["magenta"]), "fills": ("FILLS", COL["sun"]),
    "riser": ("RISER", COL["violet"]), "faders": ("FADERS", COL["mint"]),
}
LEVER_COL = (COL["mint"], COL["sky"])
RESULT_COL = {"perfect": COL["cyan"], "good": COL["green"], "miss": COL["red"]}
POPUP_COL = {"PERFECT": COL["cyan"], "GOOD": COL["green"], "MISS": COL["red"]}

# --- geometry: lane units across, seconds along (demo constants) ---

SPEED = 6.0             # depth units per second of song time
Z0 = 1.6                # depth of the hit line
BEHIND = 0.35           # seconds of road drawn past the hit line (clipped at the screen bottom)
ROAD_HALF = 0.95        # paved road half width
SHOULDER_HALF = 1.7     # shoulder outer edge
SIDE_LANE = 1.35        # kick/hat left of the road, stab/tom right
RISER_LANE = 1.55       # riser band, right shoulder
FADER_BASE, FADER_SWING = 1.8, 0.35  # fader lanes outside the shoulders: base + swing * target
POST_ROWS = (2.6, 5.0, 9.0, 15.0)    # roadside post rows, lane units each side of the road
CENTRE_ASPECT = 16 / 9
FOCAL_FRAC = 0.25       # focal length / centre width
HORIZON_FRAC = 0.42     # horizon height / window height
HIT_FRAC = 0.87         # hit line height / window height
AUTO_ALPHA = 0.35       # auto layers are drawn dimmed
PIN_FADE = 0.15         # seconds a pending note fades while pinned on the hit line
SAMPLE_DT = 0.03        # road sampling step, seconds
LONG_DT = 0.05          # riser/fader ribbon sampling step, seconds
MAX_POINTS = 2048       # vertex buffer size
LEFT_LAYERS = ("kick", "hat", "expr")  # popups of these layers sit left of the road (game.OFF_LANE)


@dataclass(frozen=True)
class Layout:
    """Window geometry, computed once per size. Screen x grows right, y grows down."""

    w: int
    h: int
    cw: float = field(init=False)        # centre region width: 16:9 of the height, or the whole window
    cx: float = field(init=False)
    cx0: float = field(init=False)
    focal: float = field(init=False)
    horizon: float = field(init=False)
    cam_h: float = field(init=False)     # camera height that puts the hit line at HIT_FRAC
    hit_y: float = field(init=False)
    near_z: float = field(init=False)    # depth at the bottom edge
    min_trel: float = field(init=False)  # earliest time offset worth drawing
    unit: float = field(init=False)      # HUD unit (demo: 2.6% of the stage height)
    s: float = field(init=False)         # pixel scale relative to a 1080 px tall centre region

    def __post_init__(self) -> None:
        w, h = self.w, self.h
        cw = min(float(w), round(h * CENTRE_ASPECT))
        focal = FOCAL_FRAC * cw
        horizon = HORIZON_FRAC * h
        cam_h = (HIT_FRAC - HORIZON_FRAC) * h * Z0 / focal
        near_z = focal * cam_h / (h - horizon)
        stage = min(h, cw / CENTRE_ASPECT)
        for k, v in (("cw", cw), ("cx", w / 2), ("cx0", (w - cw) / 2), ("focal", focal), ("horizon", horizon),
                     ("cam_h", cam_h), ("hit_y", HIT_FRAC * h), ("near_z", near_z),
                     ("min_trel", max(-BEHIND, (near_z - Z0) / SPEED)), ("unit", stage * 0.026),
                     ("s", stage / 1080)):
            object.__setattr__(self, k, v)

    def scale(self, trel):
        """Pixels per lane unit at time offset `trel` (scalar or array)."""
        return self.focal / (Z0 + trel * SPEED)

    def proj(self, x, trel):
        """Lane position `x` at `trel` seconds ahead of now -> (sx, sy). Works on arrays."""
        z = Z0 + trel * SPEED
        return self.cx + self.focal * x / z, self.horizon + self.focal * self.cam_h / z


class RoadPath:
    """Vectorised `Chart.road_x`: hold a.x for 55% of each keyframe interval, then
    smoothstep to b.x; 0 before the first keyframe, last x after the last."""

    def __init__(self, chart: Chart):
        self.t = np.array([k.t for k in chart.road], dtype=float)
        self.x = np.array([k.x for k in chart.road], dtype=float)

    @classmethod
    def from_keys(cls, keys: list[tuple[float, float]]) -> RoadPath:
        road = cls.__new__(cls)
        road.t = np.array([t for t, _ in keys], dtype=float)
        road.x = np.array([x for _, x in keys], dtype=float)
        return road

    def at(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if len(self.t) == 0:
            return np.zeros_like(t)
        i = np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1)
        j = np.minimum(i + 1, len(self.t) - 1)
        span = self.t[j] - self.t[i]
        u = np.where(span > 0, (t - self.t[i]) / np.where(span > 0, span, 1.0), 0.0)
        v = np.clip((u - 0.55) / 0.45, 0.0, 1.0)
        x = self.x[i] + (self.x[j] - self.x[i]) * v * v * (3 - 2 * v)
        return np.where(t < self.t[0], 0.0, x)


def parse_span(s: str) -> tuple[int, int, int, int]:
    """'5760x1080+0+0' -> (w, h, x, y). Offsets may be negative ('1920x1080-1920+0')."""
    m = re.fullmatch(r"\s*(\d+)x(\d+)\+?(-?\d+)\+?(-?\d+)\s*", s)
    if not m or not re.fullmatch(r"\s*\d+x\d+[+-]-?\d+[+-]-?\d+\s*", s):
        raise ValueError(f"span must look like WxH+X+Y, got {s!r}")
    w, h, x, y = (int(g) for g in m.groups())
    if w <= 0 or h <= 0:
        raise ValueError(f"span size must be positive, got {s!r}")
    return w, h, x, y


def span_on_monitors(span: tuple[int, int, int, int], monitors: list[tuple[int, int, int, int]]) -> bool:
    """Whether `span` (w, h, x, y) lies on the monitors (x, y, w, h), checked on a grid
    of points (enough for side-by-side monitors)."""
    w, h, x, y = span
    for sx in np.linspace(x + 0.5, x + w - 0.5, 25):
        for sy in (y + 0.5, y + h / 2, y + h - 0.5):
            if not any(mx <= sx < mx + mw and my <= sy < my + mh for mx, my, mw, mh in monitors):
                return False
    return True


def alive_fraction(snap: Snapshot) -> float:
    """Fraction of layers alive (auto layers count as alive): the mix the world shows."""
    if not snap.layers:
        return 1.0
    return sum(1 for v in snap.layers.values() if v.alive or v.mode == "auto") / len(snap.layers)


def beat_pulse(beat_phase: float, beat_strength: float) -> float:
    """1 on a strong beat, decaying to 0 before the next one."""
    return beat_strength * (1.0 - min(1.0, max(0.0, beat_phase))) ** 3


@dataclass(frozen=True)
class SkyState:
    top: Color
    mid: Color
    horizon: Color
    sun_r: float        # sun radius / window height
    star_alpha: float   # 0..1


def sky_state(glow: float, beat_phase: float, beat_strength: float) -> SkyState:
    """How the sky reacts to the mix: `glow` is the alive fraction, the beat pulses it."""
    p = beat_pulse(beat_phase, beat_strength)
    horizon = lerp_color(COL["horizon_lo"], COL["sun"], glow)
    return SkyState(
        top=COL["bg"],
        mid=lerp_color(COL["sky_mid_lo"], COL["sky_mid_hi"], glow),
        horizon=lerp_color(horizon, COL["horizon_flash"], 0.3 * p * glow),
        sun_r=(0.15 + 0.05 * glow) * (1 + 0.04 * p),
        star_alpha=min(1.0, 0.25 + 0.4 * glow + 0.15 * p),
    )


def steer_lane(snap: Snapshot) -> float:
    """Player position in lane units: `Snapshot.steer_lane` (turn offset removed; the
    game always fills it), else `input.steer`. Nothing is smoothed across frames, so on a
    `steer_offset_changed` or `steer_lane_jump` frame the marker is placed, never animated."""
    v = getattr(snap, "steer_lane", None)
    return snap.input.steer if v is None else float(v)


def turn_back_dir(snap: Snapshot, chart: Chart) -> int:
    """Direction of the "turn back" hint: +1 clockwise, -1 counter-clockwise, toward the
    turn offset; 0 when no hint shows. `steer_unwind` is also true mid-spin (the wheel
    is half way round), so the hint waits until no spin is running."""
    if not getattr(snap, "steer_unwind", False):
        return 0
    if spin_running(snap, chart):
        return 0
    offset = float(getattr(snap, "steer_offset_deg", 0.0) or 0.0)
    return 1 if offset > snap.input.steer_deg else -1


def turn_offset(snap: Snapshot) -> int:
    """Whole turns of `Snapshot.steer_offset_deg` (SPEC 9 rule 10); 0 when absent."""
    return round(float(getattr(snap, "steer_offset_deg", 0.0) or 0.0) / 360.0)


def marker_on_road(snap: Snapshot, geo: SceneGeo) -> bool:
    """Whether the player marker shows its on-road colour: `Snapshot.on_road`, which is
    true in every blind zone (SPEC 10). A game without that rule (charts without
    `drawn_road`) gets the zone forced here, so sweeping the wheel reveals nothing."""
    return snap.on_road or snap.phase == "attract" or (not geo.chart_drawn and geo.in_zone(snap.now))


def spin_dir(note) -> int:
    """+1 clockwise, -1 counter-clockwise, 0 when the chart gives no direction."""
    return {"cw": 1, "ccw": -1}.get(getattr(note, "dir", None) or "", 0)


def blind_runs(chart: Chart, good_window: float) -> list[tuple[float, float]]:
    """(open, close) of each run of consecutive blind gates, as the game opens the
    echo repeat: one beat before the first gate until one good window after the last.
    A run never opens before the previous listen run closes (SPEC 10), so a listen
    gate less than a beat before it keeps the road from road_x."""
    lead, runs = 60.0 / chart.bpm, []
    run: list[float] = []
    kind, listen_close = "", -math.inf
    for n in (n for n in chart.notes if n.kind == "gate"):
        k = "listen" if n.listen else "blind" if n.blind else ""
        if k != kind and run:
            if kind == "blind":
                runs.append((max(run[0] - lead, listen_close), run[-1] + good_window))
            elif kind == "listen":
                listen_close = run[-1] + good_window
            run = []
        kind = k
        run.append(n.t)
    if run and kind == "blind":
        runs.append((max(run[0] - lead, listen_close), run[-1] + good_window))
    return runs


def blind_zones(chart: Chart, runs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Fallback for `Chart.blind_zones`: from the last road keyframe at or before a run
    opens to the first keyframe at or after it closes (the song end if none)."""
    keys = [k.t for k in chart.road]
    zones = []
    for opens, closes in runs:
        i, j = bisect_right(keys, opens) - 1, bisect_left(keys, closes)
        zones.append((keys[i] if i >= 0 else opens, keys[j] if j < len(keys) else max(closes, chart.length)))
    return zones


BLIND_RAMP = 0.4   # seconds over which the road dims into and out of a blind run


class SceneGeo:
    """Every lane x the renderer draws, from the chart and a Snapshot. Pure; cached
    per note and per beat at init.

    Blind echo (SPEC 10, one drawn road): every road keyframe strictly inside a blind
    zone is pinned to lane 0, so the road blends to the centre line and back by the
    road's own rule and no drawn position, at any sample time, depends on the blind
    gates, whatever the current phase. Blind gates are never drawn."""

    def __init__(self, chart: Chart, lookahead: float, good_window: float = 0.12, use_chart_road: bool = True):
        self.chart, self.look = chart, lookahead
        self.runs = blind_runs(chart, good_window)
        # The drawn road (SPEC 10): keyframes strictly inside a blind zone pinned to lane 0.
        # Built once per chart from the chart's own keyframes when it has them (the same
        # good_window the game uses), else by the same rule here.
        self.chart_drawn = callable(getattr(chart, "drawn_road", None)) and use_chart_road
        if self.chart_drawn:
            self.zones = list(chart.blind_zones(good_window))
            keys = [(k.t, k.x) for k in chart.drawn_road(good_window)]
        else:
            self.zones = blind_zones(chart, self.runs)
            keys = [(k.t, 0.0 if any(a < k.t < b for a, b in self.zones) else k.x) for k in chart.road]
        self.road = RoadPath.from_keys(keys)
        notes = chart.notes
        self.note_road = self.road.at([n.t for n in notes]).tolist() if notes else []
        self.note_end_road = self.road.at([n.end for n in notes]).tolist() if notes else []
        # expr/fader curves as arrays for np.interp (same as chart.curve_at: linear, flat ends)
        self.curves = {i: (np.array([p[0] for p in n.curve]) + n.t, np.array([p[1] for p in n.curve]))
                       for i, n in enumerate(notes) if n.curve}
        if chart.beats:
            self.beat_t = np.array([b.t for b in chart.beats])
            self.beat_s = np.array([b.s for b in chart.beats])
        else:
            step = 60.0 / chart.bpm
            self.beat_t = np.arange(0.0, chart.length + step, step)
            self.beat_s = np.where(np.arange(len(self.beat_t)) % 4 == 0, 1.0, 0.5)
        self.beat_list = self.beat_t.tolist()
        self.beat_strong = (self.beat_s >= 0.99).tolist()
        self.beat_road = self.road.at(self.beat_t).tolist() if len(self.beat_t) else []
        self.beat_dim = self.blind_weight(self.beat_t).tolist() if len(self.beat_t) else []

    def in_blind(self, t: float) -> bool:
        return any(o <= t <= c for o, c in self.runs)

    def in_zone(self, t: float) -> bool:
        """Inside a blind zone: from the last road keyframe before a run opens to the
        first keyframe after it closes."""
        return any(a <= t <= b for a, b in self.zones)

    def blind_weight(self, t) -> np.ndarray:
        """1 inside a blind run, smoothstep to 0 over BLIND_RAMP either side."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        w = np.zeros_like(t)
        for o, c in self.runs:
            d = np.maximum(np.maximum(o - t, t - c), 0.0)
            u = np.clip(1 - d / BLIND_RAMP, 0.0, 1.0)
            w = np.maximum(w, u * u * (3 - 2 * u))
        return w

    def road_now(self, snap: Snapshot) -> float:
        return float(self.road.at(snap.now)[0])

    def road_samples(self, snap: Snapshot, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """Time offsets from t0 to t1 including every beat boundary, and the drawn road x there."""
        now = snap.now
        lo, hi = bisect_left(self.beat_list, now + t0), bisect_right(self.beat_list, now + t1)
        trel = np.unique(np.concatenate([np.arange(t0, t1, SAMPLE_DT), self.beat_t[lo:hi] - now, [t1]]))
        return trel, self.road.at(now + trel)

    def beat_x(self, i: int) -> float:
        return self.beat_road[i]

    def value(self, i: int, t):
        ts, vs = self.curves[i]
        return np.interp(t, ts, vs)

    def fader_off(self, i: int, t):
        side = -1.0 if self.chart.notes[i].lever == 0 else 1.0
        return side * (FADER_BASE + FADER_SWING * self.value(i, t))

    def note_x(self, i: int) -> float | None:
        """Anchor lane x of note `i` at its own time; None for expr (drawn in the HUD)
        and for blind gates (never drawn)."""
        n = self.chart.notes[i]
        base = self.note_road[i]
        match n.kind:
            case "gate":
                return None if n.blind else n.x or 0.0
            case "kick" | "hat":
                return base - SIDE_LANE
            case "stab" | "tom":
                return base + SIDE_LANE
            case "riser":
                return base + RISER_LANE
            case "fader":
                return base + float(self.fader_off(i, n.t))
            case "spin":
                return base
        return None

    def spin_x(self, i: int, now: float) -> tuple[float, float]:
        """Spin ring (x, trel): it waits on the hit line while the spin runs."""
        n = self.chart.notes[i]
        tt = min(max(now, n.t), n.end)
        return float(self.road.at(tt)[0]), tt - now

    def ribbon(self, i: int, now: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Riser band or fader lane of note `i`: (trel samples, lane x), or None if not in view."""
        n = self.chart.notes[i]
        t_a, t_b = max(0.0, n.t - now), min(self.look, n.end - now)
        if t_b <= t_a:
            return None
        ts = np.linspace(t_a, t_b, max(2, int((t_b - t_a) / LONG_DT) + 1))
        off = RISER_LANE if n.kind == "riser" else self.fader_off(i, now + ts)
        return ts, self.road.at(now + ts) + off

    def riser_end_x(self, i: int) -> float:
        return self.note_end_road[i] + RISER_LANE

    def lever_x(self, lever: int, value: float, snap: Snapshot) -> float:
        side = -1.0 if lever == 0 else 1.0
        return self.road_now(snap) + side * (FADER_BASE + FADER_SWING * min(1.0, max(0.0, value)))

    def popup_x(self, pop: Popup) -> float:
        """The game places popups from the judged road (and melody popups at the gate's
        x). A popup born in a blind zone sits on the drawn road by layer for its whole
        life, even a late miss finalized after the run closes."""
        if not self.in_zone(pop.t0):
            return pop.x
        base = float(self.road.at(pop.t0)[0])
        if pop.layer == "melody":
            return base
        return base - SIDE_LANE if pop.layer in LEFT_LAYERS else base + SIDE_LANE

    def positions(self, snap: Snapshot) -> list[tuple[str, float]]:
        """Every lane x this snapshot puts on screen, labelled. The renderer draws with
        the same methods; tests compare these across charts with different blind gates."""
        now = snap.now
        out: list[tuple[str, float]] = [("hit", self.road_now(snap))]
        _, xs = self.road_samples(snap, -BEHIND, self.look)
        out += [("road", float(x)) for x in xs]
        lo, hi = bisect_left(self.beat_list, now - BEHIND), bisect_right(self.beat_list, now + self.look)
        out += [("beat", self.beat_x(i)) for i in range(lo, hi)]
        for v in snap.notes:
            n = self.chart.notes[v.index]
            if n.kind == "spin":
                out.append(("spin", self.spin_x(v.index, now)[0]))
            elif n.kind in ("riser", "fader"):
                rb = self.ribbon(v.index, now)
                if rb is not None:
                    out += [(n.kind, float(x)) for x in rb[1]]
                if n.kind == "riser":
                    out.append(("drop", self.riser_end_x(v.index)))
                elif n.t <= now <= n.end:
                    val = snap.input.lever0 if n.lever == 0 else snap.input.lever1
                    out.append(("lever", self.lever_x(n.lever or 0, val, snap)))
            elif (x := self.note_x(v.index)) is not None:
                out.append((n.kind, x))
        out += [("popup", self.popup_x(p)) for p in snap.popups]
        return out


def snapshot_from_dict(d: dict) -> Snapshot:
    """Inverse of `Snapshot.to_dict()` / `to_json()` (null floats stay None)."""
    d = dict(d)
    inp = dict(d.pop("input", {}))
    for k in ("down", "pressed", "released", "system", "bound", "fallback"):
        inp[k] = frozenset(inp.get(k, ()))
    known = {f.name for f in fields(InputState)}
    top = {f.name for f in fields(Snapshot)} - {"layers", "riser", "notes", "popups"}
    snap = Snapshot(**{k: v for k, v in d.items() if k in top})
    snap.input = InputState(**{k: v for k, v in inp.items() if k in known and v is not None})
    snap.layers = {k: LayerStatus(**v) for k, v in d.get("layers", {}).items()}
    snap.riser = RiserStatus(**d.get("riser", {}))
    snap.notes = [NoteView(**v) for v in d.get("notes", [])]
    snap.popups = [Popup(**v) for v in d.get("popups", [])]
    return snap


# --- settings (own file under config_dir, not Config) ---

SETTINGS_FILE = "render.json"


@dataclass
class RenderSettings:
    fps: int = 144              # frame cap when vsync is off
    vsync: bool = True
    msaa: bool = True
    font: str | None = None     # TTF path; None picks a system font

    @classmethod
    def load(cls, path: str | Path | None = None) -> RenderSettings:
        """Defaults when the file is missing; key by key, a bad key keeps its default (`load_settings`)."""
        path = Path(path) if path else config_dir() / SETTINGS_FILE
        return load_settings(cls, path, {
            "fps": lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 < v <= 1000,
            "font": lambda v: v is None or isinstance(v, str)})

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else config_dir() / SETTINGS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        backup_bad(path)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path


FONT_CANDIDATES = (
    "C:/Windows/Fonts/bahnschrift.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf",
)
HUD_SYMBOLS = "°·×—−…"  # every non-ASCII symbol the HUD draws; arrows are drawn as shapes
FONT_CODEPOINTS = tuple(sorted({*range(32, 127), *range(0xA0, 0x180), *map(ord, HUD_SYMBOLS)}))
FONT_SIZE = 96


def displayable(s: str, charset) -> str:
    """`s` with every character outside the font atlas replaced by '?'."""
    return "".join(c if ord(c) in charset else "?" for c in s)


# --- raylib side ---

def monitors(rl) -> list[tuple[int, int, int, int]]:
    out = []
    for i in range(rl.get_monitor_count()):
        pos = rl.get_monitor_position(i)
        out.append((int(pos.x), int(pos.y), rl.get_monitor_width(i), rl.get_monitor_height(i)))
    return out


VSYNC_PROBE_FRAMES = 120
MINIMISED_SLEEP = 0.05


def needs_frame_cap(frame_times: list[float], refresh_hz: float) -> bool:
    """True when frames come faster than 60% of the refresh period: vsync is ignored."""
    if not frame_times or refresh_hz <= 0:
        return False
    return sum(frame_times) / len(frame_times) < 0.6 / refresh_hz


def export_png(rl, img, path: str | Path) -> None:
    """Encode through raylib into memory, write with Python (non-ASCII paths on Windows)."""
    import raylib

    n = rl.ffi.new("int *")
    data = raylib.ExportImageToMemory(img, b".png", n)
    if data == rl.ffi.NULL or n[0] <= 0:
        raise RuntimeError(f"could not encode {path}")
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(rl.ffi.buffer(data, n[0])))
    finally:
        raylib.MemFree(data)


class Window:
    """The game window, a context manager. `span` (w, h, x, y) makes a borderless
    window at that position, for a triple-monitor surround; `fullscreen` takes the
    current monitor; `hidden` renders off screen (smoke tests). `frame(draw,
    capture=path)` saves the finished frame in every mode. `close()` is idempotent
    and also closes everything registered with `own()`; a failure while opening
    closes the window before raising. ESC does not close the window."""

    def __init__(self, size: tuple[int, int] = (1920, 1080), span: tuple[int, int, int, int] | None = None,
                 fullscreen: bool = False, hidden: bool = False, title: str = "Torque Hero",
                 settings: RenderSettings | None = None):
        if span and fullscreen:
            raise ValueError("--span and --fullscreen exclude each other: a span is already borderless")
        import pyray as rl

        self.rl, self.hidden, self.rt, self._open = rl, hidden, None, False
        self._owned: list = []
        self._layout: Layout | None = None
        st = self.settings = settings or RenderSettings()
        self._vsync_check = not hidden and st.vsync   # see _pace()
        self._frame_times: list[float] = []
        self._last_frame: float | None = None
        rl.set_trace_log_level(rl.LOG_WARNING)
        flags = rl.FLAG_WINDOW_HIDDEN if hidden else (
            (rl.FLAG_VSYNC_HINT if st.vsync else 0) | (rl.FLAG_MSAA_4X_HINT if st.msaa else 0)
            | (rl.FLAG_WINDOW_UNDECORATED if span else 0)
            # macOS Retina: without it the viewport covers a quarter of the framebuffer.
            # Not on Windows, where it would scale a span past the monitors at 125%.
            | (rl.FLAG_WINDOW_HIGHDPI if sys.platform == "darwin" else 0))
        rl.set_config_flags(flags)
        w, h = (span[0], span[1]) if span else (0, 0) if fullscreen and not hidden else size
        # raylib centres the first window on the monitor and crashes on macOS when it
        # is larger than the monitor, so open small and resize.
        rl.init_window(0 if w == 0 else 320, 0 if h == 0 else 180, title)
        if not rl.is_window_ready():
            raise RuntimeError("could not open a window: no display found (on macOS the screen must be awake)")
        self._open = True
        try:
            if not hidden and w:
                rl.set_window_size(w, h)
            if span and not hidden:
                rl.set_window_position(span[2], span[3])
            if fullscreen and not hidden:
                rl.toggle_borderless_windowed()
            rl.set_exit_key(rl.KEY_NULL)          # ESC is pause (input.py), never close
            rl.set_target_fps(0 if hidden or st.vsync else st.fps)  # vsync paces; a second cap stutters
            rl.rl_disable_backface_culling()
            self.w, self.h = (w, h) if hidden else (rl.get_screen_width(), rl.get_screen_height())
            if hidden:
                self.rt = rl.load_render_texture(self.w, self.h)
            self._log_start(span, (w, h))
        except BaseException:
            self.close()
            raise

    def _log_start(self, span, requested: tuple[int, int]) -> None:
        rl = self.rl
        mons = monitors(rl)
        render = (rl.get_render_width(), rl.get_render_height())
        log.info("window: span %s, requested %sx%s, screen %sx%s, render %sx%s%s", span, *requested,
                 self.w, self.h, *render, " (hidden, drawn off screen)" if self.hidden else "")
        for i, m in enumerate(mons):
            log.info("monitor %d: at %+d%+d, %dx%d", i, *m)
        if self.hidden:
            return
        if requested[0] and render != requested:
            log.warning("framebuffer %sx%s differs from the requested %sx%s: check display scaling "
                        "(100%%, or override high-DPI scaling for python.exe)", *render, *requested)
        if span and mons and not span_on_monitors(span, mons):
            log.warning("span %s does not lie on the monitors %s", span, mons)

    def __enter__(self) -> Window:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def own(self, obj):
        """Register an object with `close()` to be closed with the window; returns it."""
        self._owned.append(obj)
        return obj

    @property
    def layout(self) -> Layout:
        if self._layout is None or (self._layout.w, self._layout.h) != (self.w, self.h):
            self._layout = Layout(self.w, self.h)
        return self._layout

    def should_close(self) -> bool:
        return self.rl.window_should_close()

    def focused(self) -> bool:
        return self.rl.is_window_focused()

    def frame(self, draw, capture: str | Path | None = None) -> bool:
        """Run `draw()` for one frame; save the finished frame to `capture` if given.
        Returns False when the frame was skipped (window minimised or 0x0)."""
        rl = self.rl
        if self.rt is not None:
            rl.begin_texture_mode(self.rt)
            draw()
            rl.end_texture_mode()
            if capture:
                self._save(rl.load_image_from_texture(self.rt.texture), capture, flip=True)
            rl.begin_drawing()
            rl.end_drawing()
            return True
        if not rl.is_window_fullscreen():
            self.w, self.h = rl.get_screen_width(), rl.get_screen_height()
        if self.w <= 0 or self.h <= 0 or rl.is_window_minimized():
            rl.begin_drawing()
            rl.end_drawing()
            time.sleep(MINIMISED_SLEEP)           # no busy loop while minimised
            self._last_frame = None
            return False
        rl.begin_drawing()
        draw()
        if capture:
            rl.rl_draw_render_batch_active()      # flush the batch: the back buffer now holds the frame
            self._save(self._read_screen(), capture)
        rl.end_drawing()
        self._pace()
        return True

    def _read_screen(self):
        """The back buffer at framebuffer size (load_image_from_screen scales by the
        content scale, which loses pixels under Windows display scaling)."""
        rl = self.rl
        w, h = rl.get_render_width(), rl.get_render_height()
        pixels = rl.rl_read_screen_pixels(w, h)   # RGBA, already flipped to top-down
        return rl.ffi.new("Image *", {"data": pixels, "width": w, "height": h, "mipmaps": 1,
                                      "format": rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8})[0]

    def _pace(self) -> None:
        """Vsync is only a hint: a driver setting (NVIDIA global vsync off, G-Sync) can
        ignore it and the loop then runs unbounded. Measure the first frames; if they
        come much faster than the refresh rate, apply the configured cap."""
        if not self._vsync_check:
            return
        now = time.perf_counter()
        if self._last_frame is not None:
            self._frame_times.append(now - self._last_frame)
        self._last_frame = now
        if len(self._frame_times) >= VSYNC_PROBE_FRAMES:
            self._vsync_check = False
            rl = self.rl
            hz = rl.get_monitor_refresh_rate(rl.get_current_monitor()) or 60
            if needs_frame_cap(self._frame_times, hz):
                rl.set_target_fps(self.settings.fps)
                log.warning("vsync is not pacing frames (mean %.2f ms at %d Hz): capping at %d fps",
                            1000 * sum(self._frame_times) / len(self._frame_times), hz, self.settings.fps)

    def _save(self, img, path: str | Path, flip: bool = False) -> None:
        rl = self.rl
        try:
            if flip:
                rl.image_flip_vertical(img)
            rl.image_format(img, rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)
            # blending leaves the texture's alpha below 255; the window ignores it, a PNG does not
            px = np.frombuffer(rl.ffi.buffer(img.data, img.width * img.height * 4), np.uint8)
            px[3::4] = 255
            export_png(rl, img, path)
        finally:
            rl.unload_image(img)

    def close(self) -> None:
        while self._owned:
            obj = self._owned.pop()
            try:
                obj.close()
            except Exception:
                log.exception("closing %s", type(obj).__name__)
        if self.rt is not None:
            self.rl.unload_render_texture(self.rt)
            self.rt = None
        if self._open:
            self._open = False
            self.rl.close_window()


class RaylibKeys:
    """`state.KeySource` over raylib. Key names are raylib names without `KEY_`
    ("W", "SPACE", "LEFT_SHIFT"); "MOUSE_LEFT", "MOUSE_RIGHT", "MOUSE_MIDDLE"
    name mouse buttons. Unknown names read as not down."""

    def __init__(self):
        import pyray as rl

        self.rl = rl
        self._codes: dict[str, tuple[bool, int] | None] = {}

    def _code(self, key: str) -> tuple[bool, int] | None:
        if key not in self._codes:
            rl, code = self.rl, None
            if key.startswith("MOUSE_"):
                v = getattr(rl, f"MOUSE_BUTTON_{key[6:]}", None)
                code = (True, v) if v is not None else None
            else:
                v = getattr(rl, f"KEY_{key}", None)
                code = (False, v) if v is not None else None
            self._codes[key] = code
        return self._codes[key]

    def is_down(self, key: str) -> bool:
        c = self._code(key)
        if c is None:
            return False
        return self.rl.is_mouse_button_down(c[1]) if c[0] else self.rl.is_key_down(c[1])

    def pressed(self, key: str) -> bool:
        c = self._code(key)
        if c is None:
            return False
        return self.rl.is_mouse_button_pressed(c[1]) if c[0] else self.rl.is_key_pressed(c[1])

    def mouse_x_norm(self) -> float:
        w = self.rl.get_screen_width()
        return min(1.0, max(0.0, self.rl.get_mouse_x() / w)) if w > 0 else 0.5


class Painter:
    """Thin drawing helpers over pyray: fonts, text alignment, vertex buffers."""

    def __init__(self, settings: RenderSettings | None = None):
        import pyray as rl
        import raylib

        self.rl, self.raw = rl, raylib
        self.owned = False
        self.font, self.charset = self._load_font((settings or RenderSettings()).font)
        self._safe: dict[str, str] = {}
        self.buf = rl.ffi.new("Vector2[]", MAX_POINTS)
        self.xy = np.frombuffer(rl.ffi.buffer(self.buf), dtype=np.float32).reshape(-1, 2)

    def _load_font(self, path: str | None):
        rl = self.rl
        for p in ((path,) if path else ()) + FONT_CANDIDATES:
            try:
                data = Path(p).read_bytes()      # read in Python: non-ASCII paths on Windows
            except OSError:
                continue
            cps = rl.ffi.new("int[]", list(FONT_CODEPOINTS))
            buf = rl.ffi.from_buffer(data)
            font = self.raw.LoadFontFromMemory(Path(p).suffix.lower().encode() or b".ttf",
                                               rl.ffi.cast("unsigned char *", buf), len(data), FONT_SIZE,
                                               cps, len(FONT_CODEPOINTS))
            if font.glyphCount <= 0:
                continue
            rl.gen_texture_mipmaps(rl.ffi.addressof(font, "texture"))
            rl.set_texture_filter(font.texture, rl.TEXTURE_FILTER_TRILINEAR)
            charset = frozenset(font.glyphs[i].value for i in range(font.glyphCount))
            log.info("font %s: %d of %d glyphs", p, len(charset), len(FONT_CODEPOINTS))
            missing = [c for c in HUD_SYMBOLS if ord(c) not in charset]
            if missing:
                log.warning("font %s lacks HUD symbols %s; they draw as '?'", p, "".join(missing))
            self.owned = True
            return font, charset
        log.warning("no TTF font found; using the raylib default font")
        return rl.get_font_default(), frozenset(range(32, 127))

    def close(self) -> None:
        if self.owned:
            self.rl.unload_font(self.font)
            self.owned = False

    def safe(self, s: str) -> str:
        out = self._safe.get(s)
        if out is None:
            if len(self._safe) > 4096:
                self._safe.clear()
            out = self._safe[s] = displayable(s, self.charset)
        return out

    def text_width(self, s: str, size: float) -> float:
        return self.rl.measure_text_ex(self.font, self.safe(s), size, size * 0.02).x

    def text(self, s: str, x: float, y: float, size: float, col: Color, align: str = "left") -> None:
        """Draw `s` with its top at `y`; `align` left, center or right of `x`."""
        s = self.safe(s)
        if align != "left":
            w = self.rl.measure_text_ex(self.font, s, size, size * 0.02).x
            x -= w / 2 if align == "center" else w
        self.rl.draw_text_ex(self.font, s, (x, y), size, size * 0.02, col)

    def rect(self, x: float, y: float, w: float, h: float, col: Color) -> None:
        self.rl.draw_rectangle_rec((x, y, w, h), col)

    def strip(self, n: int, col: Color) -> None:
        """Triangle strip over the first `n` points of `xy`."""
        if n >= 3:
            self.raw.DrawTriangleStrip(self.buf, n, col)

    def polyline(self, n: int, thick: float, col: Color) -> None:
        if n >= 2:
            self.raw.DrawSplineLinear(self.buf, n, thick, col)

    def ribbon(self, lx: np.ndarray, ly: np.ndarray, rx: np.ndarray, ry: np.ndarray, col: Color) -> None:
        """Strip between a left and a right edge (arrays of equal length)."""
        n = min(len(lx), MAX_POINTS // 2)
        xy = self.xy
        xy[0:2 * n:2, 0], xy[0:2 * n:2, 1] = lx[:n], ly[:n]
        xy[1:2 * n:2, 0], xy[1:2 * n:2, 1] = rx[:n], ry[:n]
        self.strip(2 * n, col)

    def line(self, x: np.ndarray, y: np.ndarray, thick: float, col: Color) -> None:
        n = min(len(x), MAX_POINTS)
        self.xy[:n, 0], self.xy[:n, 1] = x[:n], y[:n]
        self.polyline(n, thick, col)

    def arrow_head(self, cx: float, cy: float, r: float, deg: float, direction: int, size: float,
                   col: Color) -> None:
        """Arrow head on a circle of radius `r` at angle `deg`, pointing along the
        rotation (+1 clockwise on screen, -1 counter-clockwise)."""
        ang = math.radians(deg)
        tx, ty = cx + r * math.cos(ang), cy + r * math.sin(ang)
        tan = ang + direction * math.pi / 2
        tip = (tx + size * 1.4 * math.cos(tan), ty + size * 1.4 * math.sin(tan))
        nx, ny = math.cos(ang) * size, math.sin(ang) * size
        self.rl.draw_triangle(tip, (tx + nx, ty + ny), (tx - nx, ty - ny), col)

    def turn_arrow(self, x: float, y: float, r: float, direction: int, thick: float, col: Color) -> None:
        """A small circular arrow glyph (the fonts lack U+21BB and U+21BA)."""
        a0, a1 = (-60.0, 210.0) if direction > 0 else (-30.0, 240.0)
        self.rl.draw_ring((x, y), r - thick / 2, r + thick / 2, a0, a1, 24, col)
        self.arrow_head(x, y, r, a1 if direction > 0 else a0, direction, thick * 1.6, col)


def _rgba_texture(rl, arr: np.ndarray):
    """Upload an (h, w, 4) uint8 array as a bilinear texture."""
    h, w = arr.shape[:2]
    img = rl.gen_image_color(w, h, (0, 0, 0, 0))
    rl.ffi.memmove(img.data, np.ascontiguousarray(arr).tobytes(), arr.nbytes)
    tex = rl.load_texture_from_image(img)
    rl.unload_image(img)
    rl.set_texture_filter(tex, rl.TEXTURE_FILTER_BILINEAR)
    return tex


class Scenery:
    """Pre-rendered textures for one window size: stars, sun, two mountain ranges.
    Built with numpy once per size, so a frame draws a few quads."""

    MARGIN = 64  # px of parallax travel each side, at s = 1

    def __init__(self, rl, lay: Layout):
        self.rl, self.lay = rl, lay
        s = lay.s
        self.margin = int(self.MARGIN * s) + 2
        w, hy = lay.w + 2 * self.margin, int(lay.horizon)

        # stars: deterministic scatter over the upper 60% of the sky
        n = int(70 * w / max(1.0, lay.cw))
        i = np.arange(n)
        sx = ((i * 733) % 997) / 997 * w
        sy = ((i * 271) % 613) / 613 * hy * 0.6
        stars = np.zeros((max(1, hy), w, 4), np.uint8)
        size = max(1, round(1.5 * s))
        for dy in range(size):
            for dx in range(size):
                stars[np.clip(sy.astype(int) + dy, 0, hy - 1), np.clip(sx.astype(int) + dx, 0, w - 1)] = 255
        self.stars = _rgba_texture(rl, stars)

        # sun: vertical gradient disc with stripes cut out (drawn scaled per frame)
        r = int(lay.h * 0.21)
        yy, xx = np.mgrid[0:2 * r, 0:2 * r] + 0.5
        d = np.hypot(xx - r, yy - r)
        alpha = np.clip(r - d, 0, 1)
        u = (yy / (2 * r))[..., None]
        top, bot = np.array(hexc("#ffe36e")[:3]), np.array(hexc("#ff3f6a")[:3])
        rgb = top + (bot - top) * u
        stripe = np.zeros_like(d, bool)
        # demo: sy = hy - R*0.25, stripes at sy + R*(0.15 + 0.15 i), 2 + 1.5 i px tall
        for k in range(6):
            y0 = r + r * (0.15 + 0.15 * k)
            stripe |= (yy >= y0) & (yy < y0 + (2 + 1.5 * k) * r / (0.2 * 840))
        alpha = np.where(stripe, 0.0, alpha)
        sun = np.dstack([rgb, alpha[..., None] * 255]).astype(np.uint8)
        self.sun, self.sun_r = _rgba_texture(rl, sun), r

        # mountains: demo profile, px frequencies scaled with the window
        self.ranges = []
        ranges = ((0.14, 0.011, 1.3, hexc("#1b1233"), 22), (0.09, 0.019, 4.1, hexc("#241a44"), 40))
        for amp, freq, off, col, par in ranges:
            f = freq / s
            x = np.arange(w) - self.margin
            hgt = lay.h * amp * (0.55 + 0.45 * np.sin(x * f + off) * np.sin(x * f * 0.37 + off * 2)
                                 + 0.2 * np.sin(x * f * 3.1))
            hgt = np.maximum(0, hgt)
            mh = int(lay.h * amp * 1.8) + 2
            rows = np.arange(mh)[:, None] + 0.5
            a = np.clip(rows - (mh - hgt[None, :]), 0, 1)
            img = np.zeros((mh, w, 4), np.uint8)
            img[..., :3] = col[:3]
            img[..., 3] = (a * 255).astype(np.uint8)
            self.ranges.append((_rgba_texture(rl, img), mh, par * s))

    def close(self) -> None:
        for tex in (self.stars, self.sun, *(t for t, _, _ in self.ranges)):
            self.rl.unload_texture(tex)


class Renderer:
    """Draws the world and the HUD for one Snapshot. Holds no note state: per-note
    view state comes from `Snapshot.notes`, paired with `chart.notes[index]`."""

    def __init__(self, chart: Chart, cfg: Config | None = None, settings: RenderSettings | None = None):
        import pyray as rl

        from . import hud

        self.rl, self.hud = rl, hud
        self.chart, self.cfg = chart, cfg or Config()
        self.geo = SceneGeo(chart, self.cfg.lookahead, self.cfg.good_window)
        self._scenery: Scenery | None = None
        self.paint = Painter(settings)

    def close(self) -> None:
        if self._scenery:
            self._scenery.close()
            self._scenery = None
        self.paint.close()

    def scenery(self, lay: Layout) -> Scenery:
        if self._scenery is None or self._scenery.lay != lay:
            if self._scenery:
                self._scenery.close()
            self._scenery = Scenery(self.rl, lay)
        return self._scenery

    # --- frame ---

    def draw(self, snap: Snapshot, lay: Layout | None = None) -> None:
        rl = self.rl
        lay = lay or Layout(rl.get_screen_width(), rl.get_screen_height())
        rl.clear_background(COL["bg"])
        self.draw_sky(snap, lay, alive_fraction(snap))
        self.draw_ground(snap, lay)
        self.draw_road(snap, lay)
        if snap.phase != "attract":
            self.draw_notes(snap, lay)
        self.draw_player(snap, lay)
        if snap.phase != "attract":
            self.draw_popups(snap, lay)
        self.hud.draw(self, snap, lay)

    def draw_sky(self, snap: Snapshot, lay: Layout, glow: float) -> None:
        rl, sc = self.rl, self.scenery(lay)
        sky = sky_state(glow, snap.beat_phase, snap.beat_strength)
        hy, s, steer = lay.horizon, lay.s, steer_lane(snap)
        mid = int(hy * 0.55)
        rl.draw_rectangle_gradient_v(0, 0, lay.w, mid, sky.top, sky.mid)
        rl.draw_rectangle_gradient_v(0, mid, lay.w, int(hy) - mid + 1, sky.mid, sky.horizon)
        rl.draw_texture(sc.stars, int(-sc.margin - steer * 6 * s), 0, fade(COL["ink"], sky.star_alpha))
        r = lay.h * sky.sun_r
        sx, sy = lay.cx + lay.cw * 0.13 - steer * 10 * s, hy - r * 0.25
        src = (0, 0, sc.sun_r * 2, sc.sun_r * 2)
        rl.draw_texture_pro(sc.sun, src, (sx - r, sy - r, 2 * r, 2 * r), (0, 0), 0, WHITE)
        for tex, mh, par in sc.ranges:
            rl.draw_texture(tex, int(-sc.margin + steer * par), int(hy + 1 - mh), WHITE)

    def draw_ground(self, snap: Snapshot, lay: Layout) -> None:
        """Ground gradient, a synthwave grid across the whole width, roadside posts."""
        rl, p, s, geo = self.rl, self.paint, lay.s, self.geo
        hy = int(lay.horizon)
        rl.draw_rectangle_gradient_v(0, hy, lay.w, lay.h - hy, COL["ground_top"], COL["ground_bot"])
        now, t0, t1 = snap.now, lay.min_trel, self.cfg.lookahead
        grid = fade(COL["magenta"], 0.10 + 0.10 * beat_pulse(snap.beat_phase, snap.beat_strength))
        thin = max(1.0, s)
        reach = (lay.w / 2) / lay.scale(t0) + 2
        for k in range(-int(reach / 2) - 1, int(reach / 2) + 2):
            ax, ay = lay.proj(2.0 * k, t1 * 4)
            bx, by = lay.proj(2.0 * k, t0)
            rl.draw_line_ex((ax, ay), (bx, by), thin, grid)
        beats, strong = geo.beat_list, geo.beat_strong
        lo, hi = bisect_left(beats, now + t0), bisect_right(beats, now + t1 * 2)
        weak_grid = fade(grid, 0.8)
        for i in range(lo, hi):
            _, y = lay.proj(0.0, beats[i] - now)
            rl.draw_line_ex((0, y), (lay.w, y), thin, grid if strong[i] else weak_grid)
        # roadside posts every beat, rows reaching out onto the side screens
        for i in range(lo, hi):
            tr = beats[i] - now
            if tr > t1:
                break
            base = geo.beat_x(i)
            k = lay.scale(tr)
            a = min(1.0, (t1 - tr) / 0.6)
            post, bulb = fade(COL["post"], a), (fade(COL["cyan"], a), fade(COL["magenta"], a))
            glow = (fade(COL["cyan"], 0.18 * a), fade(COL["magenta"], 0.18 * a))
            for row, off in enumerate(POST_ROWS):
                if row % 2 and not strong[i]:
                    continue
                ph = (0.55 + 0.25 * row) * k
                for side in (0, 1):
                    bx, by = lay.proj(base + (2 * side - 1) * off, tr)
                    if not -0.2 * lay.w < bx < 1.2 * lay.w:
                        continue
                    p.rect(bx - 0.02 * k, by - ph, max(1.0, 0.04 * k), ph, post)
                    rl.draw_circle(int(bx), int(by - ph), max(1.5, 0.05 * k), bulb[side])
                    rl.draw_circle(int(bx), int(by - ph), max(3.0, 0.14 * k), glow[side])

    def draw_road(self, snap: Snapshot, lay: Layout) -> None:
        """Road strips per beat, dimmed through blind runs; stripes, edges and the
        centre line fade out there (blind_weight per beat segment)."""
        p, rl, s, geo = self.paint, self.rl, lay.s, self.geo
        now, t0, t1 = snap.now, lay.min_trel, self.cfg.lookahead
        trel, xs = geo.road_samples(snap, t0, t1)
        z = Z0 + trel * SPEED
        sy = lay.horizon + lay.focal * lay.cam_h / z
        k = lay.focal / z
        cx = lay.cx + k * xs
        sl, sr, pl, pr = cx - k * SHOULDER_HALF, cx + k * SHOULDER_HALF, cx - k * ROAD_HALF, cx + k * ROAD_HALF
        beats, strong = geo.beat_list, geo.beat_strong
        lo, hi = bisect_left(beats, now + t0), bisect_right(beats, now + t1)
        cuts = [0, *np.searchsorted(trel, geo.beat_t[lo:hi] - now - 1e-9).tolist(), len(trel) - 1]
        edge, centre = fade(COL["magenta"], 0.6), fade(COL["cyan"], 0.35)
        dims = geo.blind_weight(now + trel[[(cuts[j] + cuts[j + 1]) // 2 for j in range(len(cuts) - 1)]])
        for j in range(len(cuts) - 1):
            a, b = cuts[j], cuts[j + 1] + 1
            if b - a < 2:
                continue
            bi = lo - 1 + j
            odd = bi % 2
            bar = 0 <= bi < len(strong) and strong[bi]
            dim = float(dims[j])
            alpha = 1.0 - 0.65 * dim
            p.ribbon(sl[a:b], sy[a:b], sr[a:b], sy[a:b], fade(SHOULDER_COL[odd], alpha))
            p.ribbon(pl[a:b], sy[a:b], pr[a:b], sy[a:b], fade(PAVED_BAR if bar else PAVED_COL[odd], alpha))
            if dim < 1.0:
                p.line(pl[a:b], sy[a:b], 2.5 * s, fade(edge, 1 - dim))
                p.line(pr[a:b], sy[a:b], 2.5 * s, fade(edge, 1 - dim))
                p.line(cx[a:b], sy[a:b], 1.2 * s, fade(centre, 1 - dim))
        strong_line, weak_line = fade(COL["cyan"], 0.35), fade(COL["cyan"], 0.12)
        for i in range(lo, hi):
            if geo.beat_dim[i] >= 1.0:
                continue
            tr = beats[i] - now
            x = geo.beat_x(i)
            ax, ay = lay.proj(x - SHOULDER_HALF, tr)
            bx, _ = lay.proj(x + SHOULDER_HALF, tr)
            rl.draw_line_ex((ax, ay), (bx, ay), max(1.0, s * (1.5 if strong[i] else 1)),
                            fade(strong_line if strong[i] else weak_line, 1 - geo.beat_dim[i]))

    def draw_notes(self, snap: Snapshot, lay: Layout) -> None:
        geo, rl = self.geo, self.rl
        xc = geo.road_now(snap)
        ax, ay = lay.proj(xc - SHOULDER_HALF, 0.0)
        bx, _ = lay.proj(xc + SHOULDER_HALF, 0.0)
        rl.draw_line_ex((ax, ay), (bx, ay), 3 * lay.s, COL["ink"])
        notes, now, look = self.chart.notes, snap.now, self.cfg.lookahead
        for v in reversed(snap.notes):
            n = notes[v.index]
            st = snap.layers.get(n.layer)
            a = AUTO_ALPHA if st is not None and st.mode == "auto" else 1.0
            if n.kind in ("riser", "fader"):
                self._long_note(n, v, snap, lay, a)
                continue
            if n.kind == "spin":
                self._spin(n, v, snap, lay, a)
                continue
            trel = n.t - now
            x = geo.note_x(v.index)
            if v.done or x is None or trel > look:
                continue
            if trel < 0:
                a *= max(0.0, 1 + trel / PIN_FADE)
                trel = 0.0
            if a > 0:
                self._tap_note(n, x, trel, lay, a)

    def _tap_note(self, n, x: float, trel: float, lay: Layout, a: float) -> None:
        rl, p = self.rl, self.paint
        k = lay.scale(trel)
        sx, sy = lay.proj(x, trel)
        match n.kind:
            case "gate":
                w, h = k * 0.5, max(2.0, k * 0.04)
                col = COL["amber"] if n.listen else COL["cyan"]
                p.rect(sx - w / 2, sy - h * 2, w, h * 4, fade(col, a * (0.5 if n.listen else 0.25)))
                p.rect(sx - w / 2, sy - h / 2, w, h, fade(col, a))
                p.rect(sx - w / 2, sy - h * 2, h, h * 4, fade(col, a))
                p.rect(sx + w / 2 - h, sy - h * 2, h, h * 4, fade(col, a))
            case "kick":
                w, h = k * 0.26 * (0.7 + 0.3 * (n.vel if n.vel is not None else 1.0)), k * 0.07
                p.rect(sx - w / 2, sy - h / 2, w, h, fade(COL["red"], a))
            case "hat":
                w, h = k * 0.12, max(1.5, k * 0.025)
                if n.open:
                    rl.draw_rectangle_lines_ex((sx - w / 2, sy - h * 1.5, w, h * 3), max(1.0, h * 0.6),
                                               fade(COL["amber"], a))
                else:
                    p.rect(sx - w / 2, sy - h / 2, w, h, fade(COL["amber"], a))
            case "stab":
                r = k * 0.1
                rl.draw_poly((sx, sy), 6, r, 30, fade(COL["magenta"], a))
                size = max(6.0, r * 1.3)
                p.text(str(n.gate), sx, sy - size * 0.52, size, fade(COL["bg"], a), "center")
            case "tom":
                r, d = k * 0.08, -1 if n.side == "L" else 1
                rl.draw_triangle((sx + d * r, sy), (sx - d * r * 0.6, sy - r), (sx - d * r * 0.6, sy + r),
                                 fade(COL["sun"], a))

    def _spin(self, n, v: NoteView, snap: Snapshot, lay: Layout, a: float) -> None:
        """Ring with arrow heads in the direction of rotation; progress fills that way."""
        now = snap.now
        if n.t - now > self.cfg.lookahead or (v.done and now > n.end + 0.3):
            return
        rl, p = self.rl, self.paint
        x, trel = self.geo.spin_x(v.index, now)
        sx, sy = lay.proj(x, trel)
        r = lay.scale(trel) * 0.22
        d = spin_dir(n)
        ink = fade(COL["ink"], a)
        prog = 1.0 if v.done and v.result in ("perfect", "auto") else (v.progress or 0.0)
        ring = max(2.0, r * 0.12)
        rl.draw_ring((sx, sy), r - ring / 2, r + ring / 2, 0, 360, 48, ink)
        if prog > 0:
            col = RESULT_COL.get(v.result or "", COL["cyan"])
            lo_a, hi_a = sorted((-90.0, -90.0 + 360 * prog * (d or 1)))
            rl.draw_ring((sx, sy), r - ring, r + ring, lo_a, hi_a, 48, fade(col, a))
        if d:
            for deg in (-90.0, 30.0, 150.0):
                p.arrow_head(sx, sy, r, deg, d, ring * 1.6, ink)
        size = max(8.0, r * 0.5)
        y = sy - r * 1.35 - size
        if d:
            w = p.text_width("SPIN", size)
            p.text("SPIN", sx - size * 0.35, y, size, ink, "center")
            p.turn_arrow(sx - size * 0.35 + w / 2 + size * 0.45, y + size * 0.5, size * 0.3, d,
                         max(1.5, size * 0.1), ink)
        else:
            p.text("SPIN", sx, y, size, ink, "center")

    def _long_note(self, n, v: NoteView, snap: Snapshot, lay: Layout, a: float) -> None:
        """Riser band (HOLD at the start, DROP at the end) and fader lanes."""
        now, look, p, rl, geo = snap.now, self.cfg.lookahead, self.paint, self.rl, self.geo
        if n.t - now > look or n.end - now < -0.3:
            return
        riser = n.kind == "riser"
        if riser:
            col, half = fade(COL["violet"], 0.3 if v.done else 1.0), 0.045
        else:
            col, half = LEVER_COL[1 if n.lever else 0], 0.03
        rb = geo.ribbon(v.index, now)
        if rb is not None:
            ts, x = rb
            z = Z0 + ts * SPEED
            k = lay.focal / z
            sy = lay.horizon + lay.focal * lay.cam_h / z
            p.ribbon(lay.cx + k * (x - half), sy, lay.cx + k * (x + half), sy, fade(col, a))
        if riser:
            ink = fade(COL["ink"], a)
            if n.end - now <= look:
                tr = max(0.0, n.end - now)
                sx, sy = lay.proj(geo.riser_end_x(v.index), tr)
                k = lay.scale(tr)
                size = max(7.0, k * 0.08)
                rl.draw_circle(int(sx), int(sy), k * 0.1, fade(COL["violet"], a))
                p.text("DROP", sx, sy - k * 0.14 - size, size, ink, "center")
            if 0 <= n.t - now <= look:
                tr = n.t - now
                sx, sy = lay.proj(geo.note_x(v.index) or 0.0, tr)
                k = lay.scale(tr)
                size = max(7.0, k * 0.08)
                p.text("HOLD", sx, sy - k * 0.14 - size, size, ink, "center")
        elif n.t <= now <= n.end:
            val = snap.input.lever0 if n.lever == 0 else snap.input.lever1
            sx, sy = lay.proj(geo.lever_x(n.lever or 0, val, snap), 0.0)
            rl.draw_circle(int(sx), int(sy), max(3.0, lay.scale(0) * 0.05), fade(COL["ink"], a))

    def draw_player(self, snap: Snapshot, lay: Layout) -> None:
        rl = self.rl
        ffb = snap.ffb if isinstance(snap.ffb, dict) else {}
        torque = float(ffb.get("torque") or 0.0)
        lane = steer_lane(snap)
        sx, sy = lay.proj(lane, 0.0)
        sy += torque * 2 * lay.s
        s = lay.scale(0.0) * 0.085
        on = marker_on_road(snap, self.geo)
        rl.draw_ellipse(int(sx), int(sy), s * 2, s, fade(COL["cyan"], 0.22) if on else fade(COL["red"], 0.18))
        self._diamond(sx, sy, s, lane * 0.3, COL["ink"] if on else COL["red"])
        if snap.echo == "listen" and snap.echo_target is not None:
            gx, gy = lay.proj(snap.echo_target, 0.0)
            self._diamond(gx, gy, s, 0.0, fade(COL["amber"], 0.8))

    def _diamond(self, x: float, y: float, s: float, rot: float, col: Color) -> None:
        c, si = math.cos(rot), math.sin(rot)
        pts = [(0, -s * 1.1), (s * 0.9, 0), (0, s * 0.7), (-s * 0.9, 0)]
        (ax, ay), (bx, by), (cx, cy), (dx, dy) = ((x + px * c - py * si, y + px * si + py * c) for px, py in pts)
        self.rl.draw_triangle((ax, ay), (bx, by), (dx, dy), col)
        self.rl.draw_triangle((bx, by), (cx, cy), (dx, dy), col)

    def draw_popups(self, snap: Snapshot, lay: Layout) -> None:
        life = self.cfg.popup_life
        size = lay.unit * 1.75
        for pop in snap.popups:
            age = snap.now - pop.t0
            if not 0 <= age <= life:
                continue
            sx, sy = lay.proj(self.geo.popup_x(pop), 0.0)
            col = fade(POPUP_COL.get(pop.text, COL["ink"]), 1 - age / life)
            self.paint.text(pop.text, sx, sy - lay.h * 0.06 - age * lay.h * 0.12 - size, size, col, "center")


# --- harness: a chart played by a script, on a fake clock ---

SAMPLE_CHART = Path(__file__).with_name("preview_chart.json")   # shipped with the package
FIXTURE_SNAPSHOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "snapshot.json"


def scripted_input(chart: Chart, t: float, dt: float, prev: frozenset[str], play_range_deg: float = 90.0
                   ) -> InputState:
    """A near-perfect player on a bound wheel: on the road, pedals and gates on their
    notes, levers on their curves, the handbrake held through risers, spins turned their
    way (the wheel stays a turn round after a spin, as on the rig). The first stab is
    played on the wrong gate so a layer goes dark, and after the first spin the wheel
    strays half a turn back for a moment so the "turn back" state shows."""
    road = chart.road_x(t)
    down: set[str] = set()
    throttle = lever0 = lever1 = 0.0
    bad_stab = next((n.t for n in chart.notes if n.kind == "stab"), None)
    first_spin = next((n for n in chart.notes if n.kind == "spin"), None)
    turns = 0.0
    for n in chart.notes:
        if n.t - dt > t:
            break
        near = abs(t - n.t) <= dt / 2 + 1e-9
        match n.kind:
            case "kick" if near:
                down.add("brake")
            case "hat" if near:
                down.add("clutch")
            case "stab" if near:
                wrong = 3 if n.gate == 2 else 2
                down.add(f"gate{wrong}" if n.t == bad_stab else f"gate{n.gate}")
            case "tom" if near:
                down.add("paddle_l" if n.side == "L" else "paddle_r")
            case "riser" if n.t <= t < n.end:
                down.add("handbrake")
            case "spin" if t >= n.t:
                turns += (spin_dir(n) or 1) * min(1.0, (t - n.t) / (0.9 * (n.dur or 1.0)))
            case "expr" if n.t <= t <= n.end:
                throttle = n.value_at(t) + 0.05 * math.sin(7 * t)
            case "fader" if n.t <= t <= n.end:
                if n.lever == 0:
                    lever0 = n.value_at(t)
                else:
                    lever1 = n.value_at(t) + 0.04
    lane = max(-1.0, min(1.0, road + 0.06 * math.sin(3.1 * t)))
    steer_deg = lane * play_range_deg + turns * 360.0
    if first_spin is not None and first_spin.dur:
        u = (t - first_spin.end - 0.1) / 0.45
        if 0 < u < 1:
            steer_deg -= (spin_dir(first_spin) or 1) * 250.0 * math.sin(math.pi * u)
    analog = {"brake": 1.0 if "brake" in down else 0.0, "clutch": 1.0 if "clutch" in down else 0.0,
              "handbrake": 1.0 if "handbrake" in down else 0.0}
    down_f = frozenset(down)
    pressed = down_f - prev
    return InputState(
        steer=max(-1.0, min(1.0, steer_deg / play_range_deg)), steer_deg=steer_deg,
        throttle=max(0.0, min(1.0, throttle)), lever0=lever0, lever1=max(0.0, min(1.0, lever1)),
        down=down_f, pressed=pressed, released=prev - down_f,
        bound=frozenset(("steer", "brake", "clutch", "throttle", "handbrake", "lever0", "lever1", "paddle_l",
                         "paddle_r", *GATES)),
        velocity={c: 1.0 for c in pressed}, **analog,
    )


def harness_phase(k: int, frames: int, t: float, length: float) -> str:
    """Phase shown on harness frame `k`: attract first, one paused and one
    calibrate frame a third of the way in, countdown before 0, results after the end."""
    third = frames // 3
    if k == 0:
        return "attract"
    if k == third:
        return "paused"
    if k == third + 1:
        return "calibrate"
    if t < 0:
        return "countdown"
    if t > length:
        return "results"
    return "play"


def harness_snapshots(chart: Chart, cfg: Config, frames: int, t_start: float, t_end: float,
                      sample: Snapshot | None = None):
    """Yield (k, t, Snapshot): the real Game on `chart` with the scripted player,
    hats on auto so a dimmed layer shows."""
    from .game import Game

    modes = {k: "you" for k in LAYERS}
    modes["hat"] = "auto"
    game = Game(chart, cfg, modes)
    prev: frozenset[str] = frozenset()
    log_lines = list(sample.ffb.get("log", [])) if sample else ["ECHO spring centre -> 0 deg", "WEIGHT 30%"]
    dt = (t_end - t_start) / max(1, frames - 1)
    for k in range(frames):
        t = t_start + k * dt
        if t >= 0:
            inp = scripted_input(chart, t, dt, prev, cfg.play_range_deg)
            prev = inp.down
            game.update(t, dt, inp)
        snap = game.snapshot()
        snap.now = t if t < 0 else snap.now
        snap.phase = harness_phase(k, frames, t, chart.length)  # type: ignore[assignment]
        if k == 1 and sample is not None:
            snap = sample
        snap.ffb = {"torque": 0.6 * math.sin(t * 2.3) * (0.3 + snap.weight), "log": log_lines}
        yield k, t, snap


def run_harness(args) -> int:
    chart = Chart.load(args.chart)
    cfg = Config()
    sample = snapshot_from_dict(json.loads(Path(args.snapshot).read_text())) if args.snapshot else None
    if sample is not None and chart.title != "Fixture":
        sample = None  # the sample snapshot belongs to the fixture chart
    frames = max(3, args.frames)
    t_start, t_end = -2.0, chart.length + 1.0
    dt = (t_end - t_start) / (frames - 1)
    shot_at = args.at if args.at is not None else chart.length * 0.63
    shot_k = min(range(frames), key=lambda k: abs(t_start + k * dt - shot_at))
    span = parse_span(args.span) if args.span else None
    size = parse_span(args.size + "+0+0")[:2] if args.size else (1920, 1080)
    if span and args.fullscreen:
        raise ValueError("--span and --fullscreen exclude each other: a span is already borderless")
    times: list[float] = []
    with Window(size=size, span=span, fullscreen=args.fullscreen, hidden=args.hidden,
                settings=RenderSettings.load()) as win:
        view = win.own(Renderer(chart, cfg))
        for k, _t, snap in harness_snapshots(chart, cfg, frames, t_start, t_end, sample):
            if k == shot_k and args.phase:
                snap.phase = args.phase

            def draw(s=snap):
                t0 = time.perf_counter()
                view.draw(s, win.layout)
                times.append(time.perf_counter() - t0)

            win.frame(draw, args.screenshot if k == shot_k else None)
            if not args.hidden and win.should_close():
                break
        size_used = (win.w, win.h)
    if args.screenshot:
        print(f"screenshot: {args.screenshot} (song time {t_start + shot_k * dt:.2f} s)")
    if args.bench and times:
        ms = np.array(times) * 1000
        rest = ms[1:] if len(ms) > 1 else ms  # frame 0 builds the scenery textures
        print(f"draw {len(ms)} frames at {size_used[0]}x{size_used[1]}: mean {rest.mean():.2f} ms, "
              f"p99 {np.percentile(rest, 99):.2f} ms, max {rest.max():.2f} ms (first frame {ms[0]:.1f} ms)")
    return 0


def _parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--chart", default=str(SAMPLE_CHART), help="chart to animate (default: the packaged sample)")
    p.add_argument("--snapshot", default=str(FIXTURE_SNAPSHOT) if FIXTURE_SNAPSHOT.exists() else "",
                   help="sample snapshot drawn on frame 1 with the fixture chart ('' to skip)")
    p.add_argument("--frames", type=int, default=300, help="frames to render across the song")
    p.add_argument("--screenshot", help="save one frame as an image here (with or without --hidden)")
    p.add_argument("--at", type=float, help="song time of the saved frame (default 63%% of the song)")
    p.add_argument("--phase", choices=("attract", "countdown", "play", "paused", "calibrate", "results"),
                   help="phase to show on the saved frame")
    p.add_argument("--span", help="borderless window WxH+X+Y, e.g. 5760x1080+0+0 for triples")
    p.add_argument("--size", help="window size WxH (default 1920x1080)")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--hidden", action="store_true", help="render off screen")
    p.add_argument("--bench", action="store_true",
                   help="time the draw call; default 2000 frames at 5760x1080, hidden")
    return p


def _run(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.bench:
        if not (args.span or args.size or args.fullscreen):
            args.span, args.hidden = "5760x1080+0+0", True
        if args.frames == 300:
            args.frames = 2000
    try:
        return run_harness(args)
    except (RuntimeError, ValueError) as e:
        print(f"torquehero.render: {e}", file=sys.stderr)
        return 2


def add_cli(sub) -> None:
    p = sub.add_parser("preview", help="animate a chart with a scripted player (renderer check)")
    _parser(p).set_defaults(func=_run)


def main(argv: list[str] | None = None) -> int:
    return _run(_parser(argparse.ArgumentParser(prog="python -m torquehero.render")).parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
