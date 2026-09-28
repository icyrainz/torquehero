"""Shared data passed between input, game, audio, FFB, render and companion.

Units everywhere: times are seconds from song start, lane positions are -1..1
(left..right across the play range), analog controls are 0..1, wheel angles are
degrees. This module imports nothing heavy; every other module may import it.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol, runtime_checkable

# --- layers (SPEC 3) ---

LAYERS = ("melody", "kick", "hat", "expr", "pads", "fills", "riser", "faders")
LayerMode = Literal["you", "auto"]

# Performance controls each layer needs bound to be played by the player.
LAYER_CONTROLS: dict[str, tuple[str, ...]] = {
    "melody": ("steer",),
    "kick": ("brake",),
    "hat": ("clutch",),
    "expr": ("throttle",),
    "pads": ("gate1", "gate2", "gate3", "gate4", "gate5", "gate6"),
    "fills": ("paddle_l", "paddle_r"),
    "riser": ("handbrake",),
    "faders": ("lever0", "lever1"),
}

# --- difficulty (SPEC 10) ---

DIFFICULTIES = ("easy", "normal", "hard")
# Layers each difficulty gives the player; the song plays the rest (auto). Easy also drops spins and echo.
DIFFICULTY_LAYERS: dict[str, tuple[str, ...]] = {
    "easy": ("melody", "kick", "riser"),
    "normal": ("melody", "kick", "riser", "hat", "pads", "fills"),
    "hard": LAYERS,
}

# --- control names (SPEC 8.3) ---

GATES = ("gate1", "gate2", "gate3", "gate4", "gate5", "gate6")
UNIPOLAR = ("brake", "clutch", "throttle", "handbrake", "lever0", "lever1")  # 0..1, axis or button
BIPOLAR = ("steer",)                                                         # lane units -1..1
BUTTONS = (*GATES, "paddle_l", "paddle_r")                                   # button, or axis past 0.5
PERFORMANCE_CONTROLS = (*BIPOLAR, *UNIPOLAR, *BUTTONS)
SYSTEM_CONTROLS = (
    "menu_up", "menu_down", "menu_ok", "menu_back", "pause",
    "trim_minus", "trim_plus", "vol_up", "vol_down", "ffb_up", "ffb_down",
)
CONTROLS = (*PERFORMANCE_CONTROLS, *SYSTEM_CONTROLS)


def default_layer_modes(bound: set[str] | frozenset[str],
                        fallback: set[str] | frozenset[str] = frozenset(),
                        needed: set[str] | frozenset[str] | None = None,
                        defaults: dict[str, str] | None = None) -> dict[str, LayerMode]:
    """SPEC default layer modes. A layer is `you` only when every control it needs is
    served (in `bound` or `fallback`); otherwise `auto`, so a half-bound layer is
    never left unplayable. `needed` is `Chart.controls_used()`; without it, a layer
    needs all its controls. A layer the chart does not use is `you` if any of its
    controls is served (free play). `melody` is always `you`. `defaults` is
    `Chart.default_layers`, the starting point: a layer it sets to "auto" is auto;
    a layer it sets to "you" is still auto when its controls are not served."""
    defaults = defaults or {}
    served = set(bound) | set(fallback)
    modes: dict[str, LayerMode] = {}
    for layer, ctrls in LAYER_CONTROLS.items():
        need = [c for c in ctrls if needed is None or c in needed]
        ok = all(c in served for c in need) if need else any(c in served for c in ctrls)
        mode: LayerMode = "you" if layer == "melody" or ok else "auto"
        modes[layer] = "auto" if defaults.get(layer) == "auto" else mode
    return modes


@dataclass
class InputState:
    """One frame of input, built by `input.py`, read by the game and the views.

    Analog values: `steer` is lane units -1..1 (clamped at the play range),
    `steer_deg` is raw wheel rotation in degrees (unclamped, 0 = centre, positive
    = right). The unipolar controls are 0..1 with 0 at rest.

    Threshold: `Config.press_threshold` (0.5) is the one threshold for every
    control. `down` holds every performance control past it now: buttons, gates,
    paddles, and also held pedals, handbrake and levers. `pressed` holds those that
    crossed it upwards this frame, `released` those that crossed it downwards.
    `steer` never appears in these sets. `system` holds system controls pressed
    this frame.

    Velocity: `velocity` maps each control in `pressed` to its hit strength 0..1.
    `input.py` owns the mapping (from the rise rate of the axis; a key or button
    gives 1.0). The value on the crossing frame is always near the threshold, so
    it is not a velocity. The game reads `velocity.get(ctrl, 1.0)`.

    Sources: `bound` holds controls bound to a real device (joystick, wheel, pedals,
    button box). `fallback` holds controls served by keyboard/mouse. A control is
    in at most one of them. `spin_mode` "auto" is `net` only when `steer` is in `bound`.
    """

    steer: float = 0.0
    steer_deg: float = 0.0
    brake: float = 0.0
    clutch: float = 0.0
    throttle: float = 0.0
    handbrake: float = 0.0
    lever0: float = 0.0
    lever1: float = 0.0
    down: frozenset[str] = frozenset()
    pressed: frozenset[str] = frozenset()
    released: frozenset[str] = frozenset()
    system: frozenset[str] = frozenset()
    bound: frozenset[str] = frozenset()
    fallback: frozenset[str] = frozenset()
    velocity: dict[str, float] = field(default_factory=dict)

    def value(self, name: str) -> float:
        """Value of any performance control: analog as is, buttons 1.0 when in `down`."""
        if name in BUTTONS:
            return 1.0 if name in self.down else 0.0
        return float(getattr(self, name))

    @property
    def gate(self) -> int:
        """Shifter gate engaged now, 1..6, or 0 in neutral."""
        for i, g in enumerate(GATES, 1):
            if g in self.down:
                return i
        return 0

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("down", "pressed", "released", "system", "bound", "fallback"):
            d[k] = sorted(d[k])
        return d


@runtime_checkable
class KeySource(Protocol):
    """Keyboard and mouse as seen by `input.py`. The renderer supplies the raylib
    implementation; tests use a fake. Key names are raylib names without the
    prefix, e.g. "W", "SPACE", "LEFT_SHIFT", "ONE"."""

    def is_down(self, key: str) -> bool: ...
    def pressed(self, key: str) -> bool: ...          # went down this frame
    def mouse_x_norm(self) -> float: ...              # 0..1 across the window


# --- game events (returned by Game.update) ---

Result = Literal["perfect", "good", "miss"]


@dataclass(frozen=True)
class Judgement:
    """A note was judged at song time `t`. `note_index` indexes `chart.notes`,
    `note_t` is that note's chart time. `error` is seconds for timing notes, lane
    units for gates, degrees short of `spin_perfect_deg` for spins, mean error for
    expr/fader, and release offset in seconds for risers. `vel` is the hit velocity
    (kick), 0..1."""

    t: float
    layer: str
    note_kind: str
    note_index: int
    note_t: float
    result: Result
    error: float = 0.0
    vel: float = 1.0
    kind: str = field(default="judgement", init=False)


@dataclass(frozen=True)
class Sound:
    """Play a one-shot, or start/stop a loop, at song time `t`.

    `name` is a key of `audio.oneshots`: the layer's template filled with
    `{gate}` (1..6) or `{side}` ("l"/"r"), a note's `sample`, "dud" (wrong gate or
    side), or "impact" (riser release). Chart validation guarantees that every
    name the game can emit exists, for every layer with a `trigger`, `gate` (with
    oneshot) or `riser` entry in audio.layers, notes or not: kick, hat, tom_l and
    tom_r, the riser loop and impact, dud when pads or fills have an entry, every
    stab gate and sample the notes use. The one exception: free-play stabs on
    gates the chart never uses. The audio engine skips unknown names.

    `cause`: "hit" (judged hit, a pull during a riser note, or the release that
    judged a riser as a hit), "miss" (wrong gate or side, or the release that
    judged a riser as a miss), "free" (input outside any window), "auto" (auto
    layer at its note time). `note_index`/`note_t` name the chart note, None for
    free play. `vel` is `InputState.velocity` for player presses.

    Timing contract: the audio engine schedules auto-layer sounds itself from the
    chart, sample-accurately, and IGNORES Sound events with cause="auto". The game
    still emits them for views and FFB. All other causes are played on receipt."""

    t: float
    layer: str
    name: str
    action: Literal["play", "start", "stop"] = "play"
    vel: float = 1.0
    cause: Literal["hit", "free", "auto", "miss"] = "hit"
    note_index: int | None = None
    note_t: float | None = None
    kind: str = field(default="sound", init=False)


@dataclass(frozen=True)
class FfbCue:
    """A one-off force-feedback cue. `name`: "kick" (perfect melody hit, `dir`
    -1 left / +1 right), "rumble" (miss), "riser_start", "riser_release"."""

    t: float
    name: Literal["kick", "rumble", "riser_start", "riser_release"]
    dir: float = 0.0
    kind: str = field(default="ffb", init=False)


@dataclass(frozen=True)
class Beat:
    """A chart beat was crossed. `strength` 0..1."""

    t: float
    strength: float
    kind: str = field(default="beat", init=False)


@dataclass(frozen=True)
class SectionChange:
    """A chart section began. `weight` 0..1 drives FFB section weight."""

    t: float
    name: str
    weight: float
    kind: str = field(default="section", init=False)


GameEvent = Judgement | Sound | FfbCue | Beat | SectionChange


# --- snapshot (SPEC 8.4) ---

Phase = Literal["attract", "countdown", "play", "paused", "calibrate", "results"]
EchoState = Literal["none", "listen", "repeat"]


@dataclass
class LayerStatus:
    mode: LayerMode = "you"
    alive: bool = True       # false after a miss until the next hit; auto layers are always alive
    hit: int = 0
    total: int = 0


@dataclass
class RiserStatus:
    """The one source of truth for the FFB riser wind-up (FfbCue riser_start and
    riser_release are one-off markers only). `active` and `held` stay valid from
    the note's t until the riser is judged, which can be up to drop + good_window;
    for an auto riser, from t to the drop."""

    active: bool = False     # a riser note has started and is not judged yet
    progress: float = 0.0    # 0..1 through the riser note, 1 after the drop
    held: bool = False       # handbrake pulled now (or auto layer)
    held_frac: float = 0.0   # fraction of dur held so far


@dataclass
class Popup:
    text: str                # "PERFECT", "GOOD", "MISS"
    layer: str
    x: float                 # lane units, where the renderer draws it
    t0: float                # song time it appeared


@dataclass
class NoteView:
    """View state of one chart note near `now`; the renderer pairs it with
    `chart.notes[index]`. `progress` 0..1: spin = rotation done / spin_perfect_deg,
    riser/expr/fader = time through the note; None for other kinds."""

    index: int
    done: bool = False
    result: str | None = None   # "perfect", "good", "miss", "auto", or None while pending
    progress: float | None = None


def spin_in_progress(note, done: bool, now: float) -> bool:
    """A spin counts as in progress from its start time until it is finalized
    (game, render and FFB all use this rule)."""
    return note.kind == "spin" and not done and note.t <= now


def spin_running(snap: Snapshot, chart) -> bool:
    """True while any spin note in `snap.notes` is in progress."""
    notes = chart.notes
    return any(0 <= v.index < len(notes) and spin_in_progress(notes[v.index], v.done, snap.now)
               for v in snap.notes)


@dataclass
class SongInfo:
    """Static song facts for the companion timeline, from `Game.song_info()`."""

    title: str
    artist: str
    bpm: float
    length: float
    sections: list[dict]             # [{t, name, weight}, ...]
    layer_modes: dict[str, str]      # layer -> "you" | "auto"
    layers_used: list[str]           # layers with at least one note, in LAYERS order

    def to_dict(self) -> dict:
        return asdict(self)


def _finite(v):
    """Replace non-finite floats with None, recursively."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_finite(x) for x in v]
    return v


@dataclass
class Snapshot:
    """Everything a view needs to draw one frame. Built by `Game.snapshot()`;
    `app.py` sets `phase` and fills `ffb` from the FFB engine."""

    phase: Phase = "attract"
    now: float = 0.0                          # song seconds (judge clock, audio offset applied)
    length: float = 0.0
    score: int = 0
    combo: int = 0
    max_combo: int = 0
    multiplier: int = 1
    accuracy: float = 0.0                     # (perfect + 0.5 good) / judged, 0..1
    counts: dict[str, int] = field(default_factory=lambda: {"perfect": 0, "good": 0, "miss": 0})
    layers: dict[str, LayerStatus] = field(default_factory=lambda: {k: LayerStatus() for k in LAYERS})
    input: InputState = field(default_factory=InputState)
    section: str = ""
    weight: float = 0.0                       # 0..1
    beat_phase: float = 0.0                   # 0 on a beat, rising to 1 at the next
    beat_strength: float = 0.0                # strength of the last beat crossed
    road_x: float = 0.0                       # drawn road at `now` (Chart.drawn_road_x), lane units
    on_road: bool = True                      # |steer_lane - road_x| <= tolerance, or listen, blind zone, spin
    echo: EchoState = "none"
    # Goal (lane units) of the base during listen. The FFB engine rate-limits its
    # spring centre toward it, including the step when a listen region opens.
    echo_target: float | None = None
    riser: RiserStatus = field(default_factory=RiserStatus)
    expr_target: float | None = None          # 0..1 during an expr note
    fader_targets: list[float | None] = field(default_factory=lambda: [None, None])
    upcoming: list[dict] = field(default_factory=list)  # special notes in the next 8 s: {t, kind, layer, ..., index}
    notes: list[NoteView] = field(default_factory=list)  # every note from now - 0.5 s to now + Config.lookahead
    popups: list[Popup] = field(default_factory=list)
    ffb: dict = field(default_factory=lambda: {"torque": 0.0, "log": []})  # torque -1..1, log newest first
    # Turn offset (SPEC 9 rule 10): 0, +360 or -360 degrees. The FFB engine adds it to every centre.
    steer_offset_deg: float = 0.0
    steer_lane: float = 0.0                   # steer minus the offset; road_x during a spin (not judged)
    steer_unwind: bool = False                # |steer_deg - offset| > 180, never in a spin: turn back, FFB springs off
    steer_offset_changed: bool = False        # the offset changed this frame (steer_lane may jump)
    steer_lane_jump: bool = False             # steer_lane may jump: offset changed, or a spin started or ended

    def to_dict(self) -> dict:
        d = asdict(self)
        d["input"] = self.input.to_dict()
        return d

    def to_json(self) -> str:
        """Compact JSON. Non-finite floats become null; never emits NaN or Infinity."""
        return json.dumps(_finite(self.to_dict()), separators=(",", ":"), allow_nan=False)
