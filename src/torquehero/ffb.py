"""Force feedback (SPEC 5): device backends and the engine that turns game events
and the Snapshot into effects.

The engine owns every safety rule; a backend only talks to the device. Engine
units: magnitudes 0..1 (constant levels -1..1, sign = direction), seconds, wheel
degrees. Backends take milliseconds and spring centres as -1..1 of the full
wheel axis (lock to lock). Effect slots are named; each has one effect type.

Safety pipeline, every frame:
1. Each effect: strength (settings) then its cap -> pre-gain magnitude.
2. Budget (rule 6): one-shots together <= ONESHOT_BUDGET, all active effects
   together <= BUDGET. Over budget, ramped effects are scaled down first, then
   one-shots get less.
3. Ramp (rule 2): pre-gain values and the effective gain each slew at a quarter
   of the limit, so their product never moves more than RAMP_LIMIT in RAMP_WINDOW.
   Kick and beat pulse step, within STEP_CAP. Only stop_all is immediate.
4. Gain (rule 1): output = pre-gain value * effective gain (0..1).
Step sizes use the engine's own clock, never more than the caller's dt. That
clock must be wall time (time.perf_counter), never the song clock.

Centres follow the turn offset (SPEC 9 rule 10): every spring centre adds
Snapshot.steer_offset_deg; while Snapshot.steer_unwind is true only the damper runs,
and during an active spin (`update(..., spinning=True)`) the weight spring lets go.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import logging
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from .config import Config, _fell_back, backup_bad, config_dir, valid_like
from .state import Beat, FfbCue, GameEvent, InputState, RiserStatus, SectionChange, Snapshot, spin_running

log = logging.getLogger(__name__)

EFFECT_TYPES = ("sine", "constant", "spring", "damper")

# --- safety limits (SPEC 5) ---

RAMP_LIMIT = 0.5        # rule 2: no change larger than this ...
RAMP_WINDOW = 0.010     # ... inside this many seconds
STEP_CAP = 0.6          # rule 2: the kick and the beat pulse may step, capped here
ECHO_MAX_DEG_S = 180.0  # rule 3, whatever Config says
BUDGET = 1.0            # rule 6: sum of pre-gain magnitudes of all active effects
ONESHOT_BUDGET = STEP_CAP  # rule 6: sum of active one-shots
# Output is sample-and-hold, so a 10 ms window can hold the step that covers the
# previous frame plus steps covering 10 ms. Effect and gain each take a quarter:
# a step moves the output by at most 2 * RAMP_LIMIT / 4, the rate is RAMP_LIMIT / RAMP_WINDOW / 2.
MAX_STEP = RAMP_LIMIT / 4
SLEW_RATE = RAMP_LIMIT / RAMP_WINDOW / 4
MAX_DT = 0.05           # a frame hitch counts as at most this
DEADMAN_MS = 200        # every ramped effect ends by itself unless refreshed; a hung loop goes quiet
REFRESH_S = DEADMAN_MS / 2000   # re-run a ramped effect when half its length is left
GAP_LOG_S = 0.15        # a longer gap between updates is logged; each slot's own age decides what ended
DEADMAN_S = DEADMAN_MS / 1000   # the device plays a ramped slot at most this long after its last run
SPRING_DEADMAN_MS = 32  # software springs and brake: a stalled frame holds a constant force, so they end
                        # sooner; 32 ms at echo cap 0.3 keeps a hands-off 0.04 kg m2 / 25 Nm wheel
                        # under 360 deg/s (40 ms gave 424 deg/s on it). Ended at 24 ms: at 60 Hz a
                        # frame that long already trips the rate floor.
ENDED_FRACTION = 0.75   # a ramped slot not re-run for this fraction of its device length is treated as ended
ENDED_S = DEADMAN_S * ENDED_FRACTION   # (150 ms for DEADMAN_MS);
                        # the margin covers call order and device timer jitter
EXPIRY_MARGIN_S = 0.02  # a one-shot holds its budget one frame past its length
ECHO_LEAD_DEG = 45.0    # the echo centre never leads the wheel by more than this
MIN_WHEEL_RANGE_DEG = 180.0
RETRY_S, MAX_FAILS = 1.0, 5
DROP_ORDER = ("pattern", "rumble", "pulse", "kick", "riser", "echo", "weight_spring", "brake", "weight_damper")

# --- effect shapes ---

BEAT_STALE = 0.2        # beats crossed longer ago than this (mid-song start) do not pulse
PULSE_MS, PULSE_PERIOD_MS = 80, 40
KICK_MS, KICK_LEVEL = 60, 0.8
RUMBLE_MS, RUMBLE_PERIOD_MS, RUMBLE_MAG, RUMBLE_EDGE_MS = 250, 125, 0.4, 20
# Riser (SPEC 10): a ~12 Hz buzz built from whole device sine cycles. Each cycle is one device
# sine of exactly one period (RISER_CYCLE_MS, an integer so length = period), fired from the frame
# loop once the previous cycle has ended (re-running a playing effect would cut it short). The
# device plays each cycle whole whatever the frame times, stalls or budget scaling; a stall at most
# drops cycles. Amplitude and budget share are fixed when a cycle is fired; the amplitude grows with
# the wind-up and fades in and out over 2 cycles.
# Each riser cycle starts at phase 90 (a cosine). A whole cycle has zero net impulse, but from
# phase 0 its velocity (a/w)(1 - cos wt) never changes sign and moves a free wheel a T^2 / (2 pi)
# one way each cycle; from phase 90 the velocity is (a/w) sin wt and the wheel is back where it
# started at the end of every cycle. (Pulse and rumble stay at phase 0: SPEC decision.)
# The riser buzz is experimental and OFF by default (FfbSettings strength.riser = 0).
# Measured (tests/test_ffb.py riser_run: hands-off free wheel, gain 1.0, no weight spring, no base
# damping, full 3 s riser released at 8 phases of a cycle): worst 8.1 deg displacement on 0.04 kg m2
# at 25 Nm with random 7-19 ms frames; 2.7 deg with equal 144 Hz frames; 3.7 deg with beat pulses
# forcing budget scaling. Phase-0 cycles gave 92.5 deg under the same conditions.
RISER_CYCLE_MS = 83     # period = length: one whole cycle (12.05 Hz)
RISER_HZ = 1000 / RISER_CYCLE_MS
RISER_PHASE_DEG = 90.0  # each riser cycle starts as a cosine: no net displacement per cycle
RISER_GAP_S = 0.003     # the next cycle waits this long past the previous end (call latency)
RISER_ONESHOT_GAP_S = 0.010   # no riser cycle within this long of a pulse or rumble
RISER_RISE = 0.8        # amplitude per second while held
RISER_FADE_S = 2 / RISER_HZ   # the buzz fades in and out over 2 cycles: start, release, retry
# first-run test patterns
PATTERN_SPRING_DEG = 45.0   # centre pattern: full pull this far from centre (echo stiffness: stable hands off)

# software springs (SPEC 10): a device condition spring on a 1080 deg axis reaches full force only
# half a turn off centre, so the engine computes a constant level from the angle error each frame
ECHO_FULL_DEG = 45.0        # echo: full level at this error (the lead limit)
WEIGHT_FULL_DEG = 90.0      # section weight: full level this far from the offset
# Stability (tests/test_ffb.py, simulated hands-off wheel, one frame of latency, 60 and 144 Hz):
# no growing oscillation and overshoot < 10 deg at every gain for inertia >= 0.04 kg m2 up to
# 25 Nm and >= 0.05 kg m2 up to 30 Nm. A lighter rim on a stronger base can oscillate at gain 1.0
# and 60 Hz; run the app at 120 Hz or more.
K_V = 0.001                 # velocity term, level per deg/s of wheel speed at gain 1 (divided by sqrt(gain))
VEL_MAX = 720.0             # wheel speed clamp, deg/s; the speed is the median of the last 3 samples
# rate floor: frame rate = 1 / worst frame time over RATE_WINDOW_S. Below RATE_FULL_HZ, K_V scales by
# r = rate / RATE_FULL_HZ and spring levels by r squared; below RATE_OFF_HZ the springs are off
# (damper kept) until the rate is back above RATE_ON_HZ
RATE_WINDOW_S, RATE_FULL_HZ, RATE_OFF_HZ, RATE_ON_HZ = 0.5, 100.0, 50.0, 55.0
# stall brake (SPEC 10): after a gap, a spring dead-man end or the rate floor switching the springs
# off, level = -K_BRAKE x wheel speed (cap 0.3) until the springs are back or the wheel is slow.
# It only opposes motion. The brake is a velocity loop with one frame of delay; its gain per frame is
# K x 57.3 x Tmax / J x frame time, stable below about 0.52. K scales with BRAKE_FRAME_S / frame time
# (never above K_BRAKE), so the gain per frame is the same at every rate: 0.25 for 0.04 kg m2 at
# 25 Nm, 0.20 for 0.05 kg m2, under half the limit for every wheel in the tests.
K_BRAKE = 0.001             # level per deg/s at BRAKE_FRAME_S or shorter frames
BRAKE_STOP_DEG_S = 30.0     # disarmed when all 3 speed samples are below this
BRAKE_FRAME_S = 1 / 144
ECHO_RECOVER_DEG_S = 30.0   # after an echo restart the centre returns to the phrase this slowly
PATTERN_KICK_MS = 150       # kick pattern: long enough to read the direction
PATTERN_ECHO_DEG_S = 90.0   # echo pattern: centre speed
PATTERN_READY_DEG = 5.0     # echo pattern: starts once the wheel is this close to centre


@dataclass(frozen=True)
class Slot:
    type: str
    cap: float
    length_ms: int = 0      # one-shots: played by the device for this long; 0 = ramped
    deadman_ms: int = DEADMAN_MS   # ramped: device length of each (re)run; ended at ENDED_FRACTION of it

    @property
    def deadman_s(self) -> float:
        return self.deadman_ms / 1000

    @property
    def refresh_s(self) -> float:
        """Re-run when half the length is left; short (sustained) slots every frame."""
        return 0.0 if self.deadman_ms < DEADMAN_MS else self.deadman_s / 2


# Rule (SPEC 10): every sustained constant force is short and capped. A stalled frame holds it for
# at most its device length, so a free wheel gains at most cap x Tmax x length / J:
# 0.3 x 25 Nm x 0.032 s / 0.04 kg m2 = 6 rad/s = 344 deg/s, under 360 deg/s.
SUSTAINED_CAP = 0.3     # for each sustained slot AND for their sum (a stall holds all of them)

SLOTS: dict[str, Slot] = {
    "kick": Slot("constant", STEP_CAP, KICK_MS),       # one-shots keep their fixed lengths:
    "rumble": Slot("sine", RAMP_LIMIT, RUMBLE_MS),     # a stall does not hold them
    "pulse": Slot("sine", STEP_CAP, PULSE_MS),
    "riser": Slot("sine", SUSTAINED_CAP, RISER_CYCLE_MS),                            # whole buzz cycles
    "pattern": Slot("constant", SUSTAINED_CAP, deadman_ms=SPRING_DEADMAN_MS),        # centre test pattern
    "weight_spring": Slot("constant", SUSTAINED_CAP, deadman_ms=SPRING_DEADMAN_MS),  # software spring
    "weight_damper": Slot("damper", 0.5),              # a condition effect: it only opposes motion
    "echo": Slot("constant", SUSTAINED_CAP, deadman_ms=SPRING_DEADMAN_MS),           # software spring
    "brake": Slot("constant", SUSTAINED_CAP, deadman_ms=SPRING_DEADMAN_MS),          # velocity brake
}
ONESHOTS = ("kick", "rumble", "pulse")                               # in firing priority
RAMPED = tuple(s for s, spec in SLOTS.items() if not spec.length_ms)
SUSTAINED = tuple(s for s in RAMPED if SLOTS[s].type == "constant")   # held by a stall: short and capped
# the riser cycles count in the sustained sum (and the budget) for their whole length
MAGNITUDE_PARAM = {"sine": "magnitude", "constant": "level", "spring": "saturation", "damper": "saturation"}


def magnitude(params: dict) -> float:
    """The capped magnitude in a backend `update` call's params."""
    for k in ("magnitude", "level", "saturation"):
        if k in params:
            return abs(params[k])
    return 0.0


def _num(v, default: float = 0.0) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _slew(value: float, target: float, rise: float, fall: float) -> float:
    """Move toward target: at most `fall` when the magnitude shrinks, at most `rise` when it grows.
    Through zero: fall to zero at the fall rate, then continue past zero by at most `rise`, so a tiny
    caller dt (small rise) never freezes a level on the wrong side."""
    delta = target - value
    if value != 0.0 and delta * value < 0.0:                  # moving toward zero
        if target * value >= 0.0:                             # stays on this side
            return value + _clamp(delta, -fall, fall)
        if fall < abs(value):                                 # not at zero yet: fall
            return value - math.copysign(fall, value)
        return math.copysign(min(abs(target), max(rise, 0.0)), target)   # through zero: then rise
    return value + _clamp(delta, -rise, rise)


class FfbUnavailable(RuntimeError):
    """No haptic device behind the joystick, or SDL could not open it."""


# --- backends ---

@runtime_checkable
class FfbBackend(Protocol):
    """A haptic device seen as named effect slots. Params by type (engine units):
    sine: magnitude 0..1, period_ms, length_ms, phase_deg (0 = sine, 90 = cosine), attack_ms, fade_ms;
    constant: level -1..1, length_ms, attack_ms, fade_ms;
    spring: coefficient 0..1, saturation 0..1, centre -1..1 of the full axis, length_ms;
    damper: coefficient 0..1, saturation 0..1, length_ms.
    `create` returns False when the device lacks the type; the engine then never
    touches that slot. `set_gain` sets the device gain 0..1 (no-op without support).
    Any call may raise; the engine handles it. Optional attributes: `reason` (why
    there is no device), `warnings` (list of str), `max_effects` (int or None)."""

    def supports(self, kind: str) -> bool: ...
    def create(self, slot: str, kind: str) -> bool: ...
    def update(self, slot: str, **params) -> None: ...
    def run(self, slot: str) -> None: ...
    def stop(self, slot: str) -> None: ...
    def stop_all(self) -> None: ...
    def set_gain(self, gain: float) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class FfbCall:
    t: float
    method: str
    slot: str | None = None
    params: dict = field(default_factory=dict)


class NullFfb:
    """Backend that drives nothing and records every call with a timestamp from
    `clock`. Used by tests and by `--no-ffb`. `supported` limits the effect types
    it claims and `max_effects` the effects it plays at once, to exercise
    degradation. `reason` says why the app got no device. Only the last
    RECORD_LIMIT calls are kept unless `record_all` (tests)."""

    RECORD_LIMIT = 2000

    def __init__(self, supported: Iterable[str] = EFFECT_TYPES,
                 clock: Callable[[], float] = time.perf_counter, reason: str = "",
                 max_effects: int | None = None, record_all: bool = False) -> None:
        self.record_all = record_all
        self.supported = frozenset(supported)
        self.clock = clock
        self.reason = reason
        self.max_effects = max_effects
        self.warnings: list[str] = []
        self.calls: list[FfbCall] = []
        self.slots: dict[str, str] = {}
        self.params: dict[str, dict] = {}
        self.running: set[str] = set()
        self.gain = 1.0
        self.closed = False

    def _rec(self, method: str, slot: str | None = None, **params) -> None:
        self.calls.append(FfbCall(self.clock(), method, slot, params))
        if not self.record_all and len(self.calls) > 2 * self.RECORD_LIMIT:
            del self.calls[:-self.RECORD_LIMIT]

    def supports(self, kind: str) -> bool:
        self._rec("supports", kind=kind)
        return kind in self.supported

    def create(self, slot: str, kind: str) -> bool:
        self._rec("create", slot, kind=kind)
        if kind not in self.supported:
            return False
        self.slots[slot] = kind
        return True

    def update(self, slot: str, **params) -> None:
        self._rec("update", slot, **params)
        self.params[slot] = params

    def run(self, slot: str) -> None:
        self._rec("run", slot)
        self.running.add(slot)

    def stop(self, slot: str) -> None:
        self._rec("stop", slot)
        self.running.discard(slot)

    def stop_all(self) -> None:
        self._rec("stop_all")
        self.running.clear()

    def set_gain(self, gain: float) -> None:
        self._rec("set_gain", gain=gain)
        self.gain = gain

    def close(self) -> None:
        self._rec("close")
        self.running.clear()
        self.closed = True


def _i16(v: float) -> int:
    return int(round(_clamp(_num(v), -1.0, 1.0) * 32767))


def _u16(v: float) -> int:
    return int(round(_clamp(_num(v), 0.0, 1.0) * 65535))


def _ms(v, default: int = DEADMAN_MS) -> int:
    return int(_clamp(_num(v, default), 1, 65535))


class SdlHapticBackend:
    """SDL2 haptics on an already opened SDL joystick (the one bound to `steer`).
    Raises FfbUnavailable when the joystick has no haptic device. Autocentre is
    turned off: the engine does its own springs. Every effect has a finite length
    (never infinite) and one direction encoding: cartesian, dir[0] = 1;
    the sign of a constant level gives the side."""

    FLAGS = {"sine": "SDL_HAPTIC_SINE", "constant": "SDL_HAPTIC_CONSTANT",
             "spring": "SDL_HAPTIC_SPRING", "damper": "SDL_HAPTIC_DAMPER"}

    def __init__(self, joystick) -> None:
        import sdl2

        self._sdl = sdl2
        if not sdl2.SDL_WasInit(sdl2.SDL_INIT_HAPTIC) and sdl2.SDL_InitSubSystem(sdl2.SDL_INIT_HAPTIC) != 0:
            raise FfbUnavailable(f"SDL haptic init failed: {sdl2.SDL_GetError()!r}")
        self._h = sdl2.SDL_HapticOpenFromJoystick(joystick)
        if not self._h:
            raise FfbUnavailable(f"no haptic device on this joystick: {sdl2.SDL_GetError()!r}")
        self._mask = int(sdl2.SDL_HapticQuery(self._h))
        self.warnings: list[str] = []
        if self._mask & sdl2.SDL_HAPTIC_GAIN and sdl2.SDL_HapticSetGain(self._h, 100) < 0:
            self.warnings.append("device gain not set")
        if self._mask & sdl2.SDL_HAPTIC_AUTOCENTER and sdl2.SDL_HapticSetAutocenter(self._h, 0) < 0:
            self.warnings.append("autocentre still on")
        counts = [int(n) for n in (sdl2.SDL_HapticNumEffects(self._h), sdl2.SDL_HapticNumEffectsPlaying(self._h))]
        self.max_effects = min((n for n in counts if n > 0), default=None)
        self._ids: dict[str, int] = {}
        self._effects: dict[str, tuple[str, object]] = {}

    def _error(self, what: str) -> RuntimeError:
        return RuntimeError(f"{what}: {self._sdl.SDL_GetError()!r}")

    def supports(self, kind: str) -> bool:
        flag = self.FLAGS.get(kind)
        return bool(flag and self._mask & getattr(self._sdl, flag))

    def _fill(self, kind: str, eff, p: dict) -> None:
        sdl2 = self._sdl
        if kind == "constant":
            eff.type = sdl2.SDL_HAPTIC_CONSTANT
            e = eff.constant
            e.level = _i16(p.get("level", 0.0))
        elif kind == "sine":
            eff.type = sdl2.SDL_HAPTIC_SINE
            e = eff.periodic
            e.period = _ms(p.get("period_ms"), 100)
            e.magnitude = _i16(p.get("magnitude", 0.0))
            e.phase = int(round(_num(p.get("phase_deg", 0.0)) * 100)) % 36000   # hundredths of a degree
        else:
            eff.type = sdl2.SDL_HAPTIC_SPRING if kind == "spring" else sdl2.SDL_HAPTIC_DAMPER
            e = eff.condition
            e.right_sat[0] = e.left_sat[0] = _u16(p.get("saturation", 0.0))
            e.right_coeff[0] = e.left_coeff[0] = _i16(p.get("coefficient", 0.0))
            e.center[0] = _i16(p.get("centre", 0.0))
        e.type = eff.type
        e.direction.type = sdl2.SDL_HAPTIC_CARTESIAN
        e.direction.dir[0] = 1
        e.length = _ms(p.get("length_ms"))
        if kind in ("constant", "sine"):
            e.attack_length, e.fade_length = int(_num(p.get("attack_ms"))), int(_num(p.get("fade_ms")))

    def create(self, slot: str, kind: str) -> bool:
        if not self.supports(kind):
            return False
        sdl2 = self._sdl
        eff = sdl2.SDL_HapticEffect()
        self._fill(kind, eff, {})
        eid = sdl2.SDL_HapticNewEffect(self._h, ctypes.byref(eff))
        if eid < 0:
            return False
        self._ids[slot], self._effects[slot] = eid, (kind, eff)
        return True

    def update(self, slot: str, **params) -> None:
        if slot not in self._ids:
            return
        kind, eff = self._effects[slot]
        self._fill(kind, eff, params)
        if self._sdl.SDL_HapticUpdateEffect(self._h, self._ids[slot], ctypes.byref(eff)) < 0:
            raise self._error(f"update {slot}")

    def run(self, slot: str) -> None:
        """Starts the effect, or restarts its length if it is playing (dead-man refresh)."""
        if slot in self._ids and self._sdl.SDL_HapticRunEffect(self._h, self._ids[slot], 1) < 0:
            raise self._error(f"run {slot}")

    def stop(self, slot: str) -> None:
        if slot in self._ids and self._sdl.SDL_HapticStopEffect(self._h, self._ids[slot]) < 0:
            raise self._error(f"stop {slot}")

    def stop_all(self) -> None:
        if self._h and self._sdl.SDL_HapticStopAll(self._h) < 0:
            raise self._error("stop all")

    def set_gain(self, gain: float) -> None:
        if self._h and self._mask & self._sdl.SDL_HAPTIC_GAIN:
            if self._sdl.SDL_HapticSetGain(self._h, int(round(_clamp(_num(gain), 0.0, 1.0) * 100))) < 0:
                raise self._error("set gain")

    def close(self) -> None:
        if not self._h:
            return
        try:
            self.stop_all()
        except RuntimeError:
            pass
        for eid in self._ids.values():
            self._sdl.SDL_HapticDestroyEffect(self._h, eid)
        self._ids.clear()
        self._effects.clear()
        self._sdl.SDL_HapticClose(self._h)
        self._h = None


def spin_active(snap: Snapshot, chart) -> bool:
    """True while a spin note is in progress (`state.spin_running`: started, not done).
    The app passes this to FfbEngine.update(spinning=...)."""
    return spin_running(snap, chart)


def rate_text(report: dict) -> str:
    """Frame rate line for the HUD and the FFB test: the mean rate and the worst frame
    (the measure the rate floor uses), the rate floor ("SPRINGS reduced" below RATE_FULL_HZ,
    "SPRINGS off" below RATE_OFF_HZ, "SPRINGS measuring" before a frame time exists) and
    LATCHED while output is latched off."""
    fps, worst = report.get("fps"), report.get("fps_worst")
    parts = [(f"FPS {fps:.0f} (worst {worst:.0f})" if worst else f"FPS {fps:.0f}") if fps else "FPS measuring"]
    if report.get("springs"):
        parts.append(f"SPRINGS {report['springs']}")
    if report.get("latched"):
        parts.append("LATCHED")
    return "   ".join(parts)


def open_backend(joystick) -> FfbBackend:
    """SdlHapticBackend on `joystick`, or a NullFfb carrying the reason when there
    is no joystick (no `steer` binding) or it has no haptics."""
    if joystick is None:
        return NullFfb(reason="no steer binding")
    try:
        return SdlHapticBackend(joystick)
    except FfbUnavailable as e:
        return NullFfb(reason=str(e))


# --- settings (own file under config_dir, not in Config) ---

SETTINGS_FILE = "ffb.json"
SIGN_ERROR = f"{SETTINGS_FILE}: cannot read ffb_sign: fix the file or delete it"


DEFAULT_STRENGTH = {**dict.fromkeys(SLOTS, 1.0), "riser": 0.0}   # the riser buzz is experimental: off


@dataclass
class FfbSettings:
    """`strength`: per effect 0..1, can only lower an effect (caps, budget and gain
    still apply after it); strength.riser is 0 unless set (experimental riser buzz).
    `ffb_sign`: +1 or -1, flips every constant force (kick, echo and weight springs)
    if a kick with dir +1 does not push the wheel to the right on the rig.
    `sign_error`: set by `load` when the file exists but ffb_sign cannot be read; the
    app then runs without force feedback."""

    strength: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_STRENGTH))
    ffb_sign: int = 1
    sign_error: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        given = self.strength if isinstance(self.strength, dict) else {}
        self.strength = {k: _clamp(_num(given.get(k, DEFAULT_STRENGTH[k]), DEFAULT_STRENGTH[k]), 0.0, 1.0)
                         for k in SLOTS}
        self.ffb_sign = -1 if _num(self.ffb_sign, 1.0) < 0 else 1

    @classmethod
    def load(cls, path: str | Path | None = None) -> FfbSettings:
        """Key by key, like config.json: defaults when the file is missing; a bad strength
        key keeps its default with one warning each; unknown keys are ignored; a UTF-8 BOM
        is accepted. Any fallback copies the file to ffb.json.bad before the first save.
        The file exists but ffb_sign cannot be read (bad JSON, not an object, ffb_sign not
        +1 or -1): `sign_error` is set and the sign is never silently reset to +1."""
        path = Path(path) if path else config_dir() / SETTINGS_FILE
        if not path.exists():
            return cls()
        try:
            d = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(d, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError) as e:
            _bad(path, f"{path} is not usable ({e})")
            return cls(sign_error=SIGN_ERROR)
        given, strength = d.get("strength", {}), {}
        if not isinstance(given, dict):
            _bad(path, f"{path}: ignored strength={given!r}: using the defaults for it")
            given = {}
        for k, v in given.items():
            if k not in SLOTS:
                continue
            if valid_like(1.0, v):
                strength[k] = v
            else:
                _bad(path, f"{path}: ignored strength.{k}={v!r}: using the default for it")
        sign = d.get("ffb_sign", 1)
        if valid_like(1, sign) and sign in (1, -1):
            return cls(strength=strength, ffb_sign=sign)
        _bad(path, f"{path}: ignored ffb_sign={sign!r}: use 1 or -1")
        return cls(strength=strength, sign_error=SIGN_ERROR)

    def save(self, path: str | Path | None = None) -> Path:
        """A file that fell back on load is first copied to ffb.json.bad."""
        path = Path(path) if path else config_dir() / SETTINGS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        backup_bad(path)
        path.write_text(json.dumps({"strength": self.strength, "ffb_sign": self.ffb_sign}, indent=2) + "\n")
        return path


def _bad(path: Path, msg: str) -> None:
    log.warning(msg)
    _fell_back.add(path.resolve())


# --- engine ---

class FfbEngine:
    """Maps GameEvents and the Snapshot to effects, once per frame, and enforces
    the SPEC 5 safety rules. Use as a context manager; `close` is also registered
    with atexit.

    `update` returns the dict for `Snapshot.ffb`: torque (signed estimate, not
    clamped), load (sum of output magnitudes now), log (newest first),
    echo_centre (lane units or None), gain (target), gain_now (effective), latched.

    Effects run while `snap.phase == "play"`; another phase stops them and they
    come back when play resumes. `stop_all()` (focus loss, errors) latches: the
    engine sends nothing until `resume()`. `set_gain()` for `--ffb-gain` and
    ffb_up/ffb_down. `wheel_range_deg` is the calibrated lock-to-lock range of the
    steer binding; None falls back to Config.wheel_range_deg.

    Threading: update, stop_all, resume, device_lost, set_gain and close run on
    the update thread only. A stop inside a frame cuts the rest of that frame."""

    def __init__(self, backend: FfbBackend, cfg: Config | None = None, settings: FfbSettings | None = None,
                 wheel_range_deg: float | None = None, clock: Callable[[], float] = time.perf_counter,
                 log_len: int = 8) -> None:
        self.backend = backend
        self.cfg = cfg or Config()
        self.settings = settings or FfbSettings()
        self.clock = clock
        self.log_len = log_len
        self._log: list[list] = []  # [msg, count], newest first
        self._gain = 0.0            # target, 0..1
        self.set_gain(self.cfg.ffb_gain, quiet=True)
        self._g = 0.0               # effective gain, slews toward the target
        self._echo_rate = _clamp(_num(self.cfg.echo_max_deg_s, ECHO_MAX_DEG_S), 0.0, ECHO_MAX_DEG_S)
        rng = _num(wheel_range_deg, 0.0)
        self.wheel_range_deg = rng if rng > 0 else _num(self.cfg.wheel_range_deg, 0.0)
        self._t = 0.0                                   # engine time, sum of limited steps
        self._now = 0.0                                 # expiry time: clock credited at most MAX_DT per update
        self._wall = 0.0                                # clock at the last update
        self._last_clock: float | None = None
        self._ran_at = dict.fromkeys(SLOTS, -math.inf)  # wall time a ramped effect was last (re)started
        self._pre_sent = dict.fromkeys(SLOTS, 0.0)      # pre-gain value of the last level the device accepted
        self._ghosts: dict[str, tuple[float, float]] = {}   # slot -> (pre-gain value, wall end): failed stop
        self._pattern: tuple[str, float, float | None] | None = None  # test pattern: name, last t, echo start
        self._mode = ""                                 # test pattern running in this update, "" in play
        self._len = {s: spec.length_ms for s, spec in SLOTS.items()}   # length of the last one-shot fired
        self._pre = dict.fromkeys(SLOTS, 0.0)           # pre-gain value (signed for constants)
        self._out = dict.fromkeys(SLOTS, 0.0)           # last value sent (after gain, before ffb_sign)
        self._fired = dict.fromkeys(SLOTS, -math.inf)   # clock time a one-shot started
        self._running: set[str] = set()
        self._fails = dict.fromkeys(SLOTS, 0)
        self._retry_at = dict.fromkeys(SLOTS, -math.inf)
        self._wind = 0.0
        self._riser_env = 0.0                           # 0..1 fade of the riser buzz
        self._riser_end = -math.inf                     # wall time the playing riser cycle ends
        self._oneshot_wall = -math.inf                  # wall time a pulse or rumble last started
        self._centre = 0.0                              # echo spring centre, wheel degrees
        self._steer: float | None = None                # last finite wheel angle; None until one is seen
        self._vel = 0.0                                 # wheel speed, deg/s: median of the last 3 samples
        self._vels: list[float] = []                    # last raw speed samples
        self._angle: float | None = None                # last finite angle for the speed; None resets it
        self._frame_times: list[tuple[float, float]] = []   # (wall, frame time) over RATE_WINDOW_S
        self._fps_times: list[tuple[float, float]] = []     # the same, also while latched, no gaps: display only
        self._floor_seen = False                        # the rate floor has been evaluated in play
        self._springs_ok = False                        # rate floor state (hysteresis)
        self._springs_logged = False
        self._echo_restart = False                      # a restart (gap, dead-man end, failure, stop) happened
        self._echo_recover = False                      # centre returning gently after a restart
        self._brake_armed = False                       # stall brake: armed by a stall, off once safe
        self._ref: float | None = None                  # weight / centre-pattern spring centre while returning
        self._ref_restart = False                       # a stall happened: fixed springs return gently
        self._springs_prev = False
        self._elapsed = 0.0                             # wall time since the previous update
        self._echo_on = False                           # the echo spring was active last frame
        self._wait = False                              # a test pattern is waiting (nothing may move)
        self._offset = 0.0                              # turn offset, degrees
        self._sign = self.settings.ffb_sign             # applied at start only
        self._listen = False
        self._torque = 0.0
        self._halted = True
        self._latched = False
        self._closed = False

        reason = getattr(backend, "reason", "")
        if reason:
            self._note(f"NO FFB - {reason}")
        for w in getattr(backend, "warnings", None) or []:
            self._note(f"WARN {w}")
        if self._strength("riser") <= 0.0:
            self._note("riser buzz off (experimental)")
        keep: dict[str, bool] = {}
        for slot, spec in SLOTS.items():
            try:
                keep[slot] = bool(backend.supports(spec.type))
            except Exception:
                keep[slot] = False
            if not keep[slot] and not reason:
                self._note(f"NO {spec.type.upper()} - {slot} off")
        if keep["echo"] and self.wheel_range_deg < MIN_WHEEL_RANGE_DEG:
            keep["echo"] = False
            self._note(f"NO ECHO - wheel range {round(self.wheel_range_deg)} deg < {round(MIN_WHEEL_RANGE_DEG)}")
        limit = getattr(backend, "max_effects", None)
        if isinstance(limit, int) and limit > 0:
            for slot in DROP_ORDER:
                if sum(keep.values()) <= limit:
                    break
                if keep[slot]:
                    keep[slot] = False
                    self._note(f"DROP {slot} - device plays {limit} effects")
        self.enabled: dict[str, bool] = {}
        for slot, spec in SLOTS.items():
            try:
                self.enabled[slot] = keep[slot] and bool(backend.create(slot, spec.type))
            except Exception:
                self.enabled[slot] = False
        atexit.register(self.close)

    # --- public ---

    def __enter__(self) -> FfbEngine:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def gain(self) -> float:
        """Target global gain 0..1; change it with set_gain."""
        return self._gain

    @property
    def latched(self) -> bool:
        return self._latched

    def set_gain(self, gain: float, quiet: bool = False) -> float:
        """Global gain, clamped to 0..1 (rule 1). The output ramps to it, up or down."""
        self._gain = _clamp(_num(gain), 0.0, 1.0)
        if not quiet:
            self._note(f"GAIN {round(self._gain * 100)}%")
        return self._gain

    def update(self, events: Iterable[GameEvent], snap: Snapshot, dt: float, spinning: bool = False) -> dict:
        """One frame. `spinning`: a spin note is in progress (`spin_active(snap, chart)`);
        the weight spring then ramps to zero so it does not fight a full turn.
        Stops everything, latches and re-raises on any error."""
        wall = _num(self.clock(), self._wall)
        elapsed = wall - self._last_clock if self._last_clock is not None else 0.0
        self._last_clock = self._wall = wall
        self._elapsed = max(0.0, elapsed)
        self._now += _clamp(elapsed, 0.0, MAX_DT)
        if 0.0 < elapsed <= GAP_LOG_S:   # a hitch or a pause gap is not the frame rate
            self._fps_times.append((wall, elapsed))
            self._fps_times = [f for f in self._fps_times if f[0] >= wall - RATE_WINDOW_S]
        if self._latched or self._closed:
            return self.report()
        if elapsed > 0.0:
            self._frame_times.append((wall, elapsed))
            self._frame_times = [f for f in self._frame_times if f[0] >= wall - RATE_WINDOW_S]
        if elapsed > GAP_LOG_S and not self._halted:
            self._note(f"GAP {round(elapsed * 1000)} ms")
            self._echo_restart = True
            self._arm_brake()
        try:
            if snap.phase != "play":
                if not self._halted:
                    self._silence("STOP pause")
            else:
                self._step(list(events), snap, _clamp(min(_num(dt), elapsed), 0.0, MAX_DT), bool(spinning))
        except BaseException:
            self.stop_all()
            raise
        return self.report()

    def stop_all(self) -> None:
        """Stop every effect now and latch (rule 4): nothing is sent until resume()."""
        self._latched = True
        self._silence("STOP all")

    def resume(self) -> None:
        """Clear the latch. Effects ramp up from zero on the next play frame."""
        if self._closed or not self._latched:
            return
        self._latched = False
        self._note("RESUME")

    PATTERNS = ("centre", "kick", "echo", "riser")

    def test_pattern(self, name: str, t: float, dt: float, steer_deg: float, phase: str) -> dict:
        """First-run check without a song (`torquehero play --ffb-test NAME`). Call once
        per frame, instead of update(), with `t` seconds since the pattern started, the
        wheel angle and the app phase. Runs through update(): every safety rule applies,
        a phase other than play stops output (Escape pauses a pattern), latch and resume
        work as in play.
        centre: a software spring on its own sustained slot, full pull at 45 deg from centre,
          plus the damper; direction is what the check reads.
        kick: a 150 ms kick right, then left 1 s later, repeated every 3 s; no spring, no damper.
        echo: once the wheel is within 5 deg of centre, the echo spring turns it
          0 -> +45 deg -> 0 at 90 deg/s, every 3 s. Until then (and again after any phase
          other than play) nothing runs and the log asks to centre the wheel.
        riser: a full riser every 4 s: the buzz grows for 3 s, then 1 s off; no spring, no damper."""
        if name not in self.PATTERNS:
            raise ValueError(f"unknown ffb test pattern {name!r}; use one of {', '.join(self.PATTERNS)}")
        t = _num(t)
        same = self._pattern is not None and self._pattern[0] == name and self._pattern[1] <= t
        last, t0 = (self._pattern[1], self._pattern[2]) if same and self._pattern else (-1.0, None)
        events: list[GameEvent] = []
        kw: dict = {"weight": 0.5 if name == "centre" else 0.0}
        if name == "kick":
            for at, d in ((0.5, 1.0), (1.5, -1.0)):
                k = math.floor((t - at) / 3.0)          # latest crossing of at + 3k at or before t
                if k >= 0 and last < at + 3.0 * k:
                    events.append(FfbCue(t, "kick", dir=d))
        elif name == "echo":
            wheel = _num(steer_deg, math.inf)
            if phase != "play":
                t0 = None                                # after a pause it waits for a centred wheel again
            elif t0 is None and abs(wheel) <= PATTERN_READY_DEG:
                t0 = t
            if t0 is None:
                if not same or (self._pattern and self._pattern[2] is not None):
                    self._note("CENTRE the wheel")   # once per wait: at the start and after a pause
            else:
                u = (t - t0) % 3.0
                deg = 90.0 * (u - 1.0) if 1.0 <= u < 1.5 else 90.0 * (2.0 - u) if 1.5 <= u < 2.0 else 0.0
                kw.update(echo="listen", echo_target=deg / max(1e-6, _num(self.cfg.play_range_deg, 90.0)))
        elif name == "riser":
            u = t % 4.0
            kw["riser"] = RiserStatus(active=u < 3.0, progress=min(1.0, u / 3.0), held=u < 3.0)
        self._pattern = (name, t, t0)
        snap = Snapshot(phase=phase, now=t, input=InputState(steer_deg=steer_deg), **kw)  # type: ignore[arg-type]
        self._mode, self._wait = name, name == "echo" and t0 is None
        try:
            return self.update(events, snap, dt)
        finally:
            self._mode, self._wait = "", False

    def device_lost(self) -> None:
        """The wheel is going away (unplug, e-stop, shutdown): stop, latch and close
        the haptic device. The joystick belongs to the input layer and is not touched.
        Register with the input layer's `on_steer_lost`."""
        self._note("DEVICE LOST")
        self.close()

    def close(self) -> None:
        """Stop everything and release the device. Safe to call twice."""
        if self._closed:
            return
        atexit.unregister(self.close)
        try:
            self.stop_all()
        finally:
            self._closed = True
            try:
                self.backend.close()
            except Exception:
                pass

    def report(self) -> dict:
        echo = self._echo_on and self.enabled["echo"]
        return {
            "torque": round(self._torque, 4),
            "load": round(self._load(), 4),
            "log": [m if n == 1 else f"{m} x{n}" for m, n in self._log],
            "echo_centre": (round((self._centre - self._offset) / max(1e-6, _num(self.cfg.play_range_deg, 90.0)), 4)
                            if echo else None),   # lane units, like echo_target
            "gain": self._gain,
            "gain_now": self._g,
            "latched": self._latched,
            "fps": round(len(self._fps_times) / sum(f[1] for f in self._fps_times), 1) if self._fps_times else None,
            "fps_worst": round(1.0 / max(f[1] for f in self._fps_times), 1) if self._fps_times else None,
            "sign": None if self.settings.sign_error else self._sign,   # the sign in use
            "springs": self._springs_state(),
        }

    def _springs_state(self) -> str | None:
        """Rate floor for the HUD: "measuring" until a frame time exists, "off" below
        RATE_OFF_HZ, "reduced" below RATE_FULL_HZ, else None (also before the first play frame)."""
        if not self._floor_seen:
            return None
        if not self._frame_times:
            return "measuring"
        if not self._springs_ok:
            return "off"
        rate = 1.0 / max(f[1] for f in self._frame_times) if self._frame_times else 0.0
        return "reduced" if rate < RATE_FULL_HZ else None

    # --- internals ---

    def _note(self, msg: str) -> None:
        if self._log and self._log[0][0] == msg:
            self._log[0][1] += 1
            return
        self._log.insert(0, [msg, 1])
        del self._log[self.log_len:]

    def _arm_brake(self) -> None:
        """A stall: brake until the springs are back. Pre-stall speed samples are dropped so the
        first brake frame cannot take its sign from before the stall."""
        self._brake_armed = True
        self._vels = []
        self._ref_restart = True

    def _silence(self, msg: str) -> None:
        self._halted = True
        self._running.clear()
        self._pre = dict.fromkeys(SLOTS, 0.0)
        self._out = dict.fromkeys(SLOTS, 0.0)
        self._fired = dict.fromkeys(SLOTS, -math.inf)
        self._g = self._wind = self._torque = self._vel = self._riser_env = 0.0
        self._riser_end = -math.inf
        self._listen = self._echo_on = False
        self._vels, self._angle, self._echo_restart, self._brake_armed = [], None, True, False
        try:
            self.backend.stop_all()
        except Exception:
            for slot in SLOTS:
                try:
                    self.backend.stop(slot)
                except Exception:
                    pass
            msg = "STOP FAILED - stopped each effect, device gain 0"
        try:
            self.backend.set_gain(0.0)
        except Exception:
            msg += " (device gain not set)"
        self._note(msg)

    def _active(self, slot: str) -> bool:
        return self._now < self._fired[slot] + self._len[slot] / 1000 + EXPIRY_MARGIN_S

    def _riser_live(self) -> float:
        """Pre-gain amplitude of the riser cycle the device is playing now."""
        return abs(self._pre["riser"]) if self._wall < self._riser_end else 0.0

    def _load(self) -> float:
        live = sum(abs(self._out[s]) for s in RAMPED) + sum(abs(self._out[s]) for s in ONESHOTS if self._active(s))
        live += abs(self._out["riser"]) if self._wall < self._riser_end else 0.0
        return live + self._ghost_load() * _clamp(self._g, 0.0, 1.0)

    def _ghost_load(self, slots: Iterable[str] = SLOTS) -> float:
        """Pre-gain level a device may still play after a failed send and a failed stop."""
        self._ghosts = {k: v for k, v in self._ghosts.items() if v[1] > self._wall}
        return sum(v for k, (v, _) in self._ghosts.items() if k in slots)

    def _available(self, slot: str) -> bool:
        return self.enabled[slot] and self._now >= self._retry_at[slot]

    def _strength(self, slot: str) -> float:
        return self.settings.strength.get(slot, DEFAULT_STRENGTH[slot])

    def _cap(self, slot: str, raw: float) -> float:
        """Rule 2 cap after the settings strength, before budget and gain."""
        return math.copysign(min(abs(_num(raw)) * self._strength(slot), SLOTS[slot].cap), raw)

    def _call(self, slot: str, method: str, ghost: tuple[float, float] = (0.0, 0.0), **params) -> bool:
        """Rule 5: a failing effect goes quiet, is retried after RETRY_S, and is off
        after MAX_FAILS failures. The other effects go on. Nothing is sent once latched.
        `ghost` (pre-gain value, wall end): what the device may still play if this call
        fails; kept in the budget until then when the stop that follows fails too."""
        if self._latched or self._closed:
            return False
        try:
            getattr(self.backend, method)(slot, **params)
            return True
        except Exception:
            self._fails[slot] += 1
            self._running.discard(slot)
            if slot == "echo":
                self._echo_restart = True
            self._pre[slot] = self._out[slot] = self._pre_sent[slot] = 0.0
            self._fired[slot] = -math.inf
            if slot == "riser":
                self._wind = self._riser_env = 0.0   # a retry fades in from zero
            try:
                self.backend.stop(slot)
            except Exception:
                if ghost[0] > 0.0 and ghost[1] > self._wall:
                    self._ghosts[slot] = (min(ghost[0], SLOTS[slot].cap), ghost[1])
            if self._fails[slot] >= MAX_FAILS:
                self.enabled[slot] = False
                self._note(f"FAIL {slot} - off")
            else:
                self._retry_at[slot] = self._now + RETRY_S
                self._note(f"FAIL {slot} - retry in {RETRY_S:g} s")
            return False

    def _send(self, slot: str, stiffness: float = 0.0, **params) -> None:
        """Send a ramped slot's level every frame; re-run it at half its length (dead-man).
        Springs and dampers: coefficient = stiffness x output / cap, so gain and the budget
        scale the force near centre as well as the saturation."""
        if self._latched or self._closed:
            return
        g = _clamp(self._g, 0.0, 1.0)
        out = self._pre[slot] * g
        if slot in SUSTAINED:
            # the sum rule at send time: a call that failed earlier in this frame leaves a ghost
            # the plan did not know about; the other slots count at what the device plays now
            others = sum(abs(self._out[s]) for s in SUSTAINED if s != slot)
            others += abs(self._out["riser"]) if self._wall < self._riser_end else 0.0
            head = max(0.0, SUSTAINED_CAP * g - others - self._ghost_load(SUSTAINED) * g)
            if abs(out) > head:
                out = math.copysign(head, out)
                self._pre[slot] = out / g if g > 0 else 0.0
        if abs(out) < 1e-6:
            out = 0.0
        live = slot in self._running
        old = abs(self._pre_sent[slot]) if live else 0.0     # what the device plays now
        end = self._ran_at[slot] + SLOTS[slot].deadman_s
        self._out[slot] = out
        if out == 0.0:
            if live:
                self._running.discard(slot)
                self._call(slot, "stop", ghost=(old, end))
                self._pre_sent[slot] = 0.0
            return
        spec = SLOTS[slot]
        params[MAGNITUDE_PARAM[spec.type]] = out * self._sign if spec.type == "constant" else abs(out)
        if spec.type in ("spring", "damper"):
            params["coefficient"] = _clamp(stiffness * abs(out) / spec.cap, 0.0, 1.0)
        if not self._call(slot, "update", ghost=(old, end), length_ms=spec.deadman_ms, **params):
            return
        self._pre_sent[slot] = self._pre[slot]
        if not live or self._wall - self._ran_at[slot] >= spec.refresh_s:   # always before it ends
            if self._call(slot, "run", ghost=(max(old, abs(self._pre[slot])), end) if live else (0.0, 0.0)):
                self._running.add(slot)
                self._ran_at[slot] = self._wall
                self._ghosts.pop(slot, None)

    def _fire(self, slot: str, pre: float, **params) -> None:
        if self._latched or self._closed:
            return
        out = pre * _clamp(self._g, 0.0, 1.0)
        if abs(out) < 1e-6:
            return
        spec = SLOTS[slot]
        params[MAGNITUDE_PARAM[spec.type]] = out * self._sign if spec.type == "constant" else abs(out)
        length = int(params.pop("length_ms", spec.length_ms))
        prev = abs(self._pre[slot]) if self._active(slot) else 0.0
        left = self._fired[slot] + self._len[slot] / 1000 + EXPIRY_MARGIN_S - self._now
        new_end = self._wall + length / 1000 + EXPIRY_MARGIN_S
        if (self._call(slot, "update", ghost=(prev, self._wall + max(0.0, left)), length_ms=length, **params)
                and self._call(slot, "run", ghost=(max(prev, abs(pre)), new_end))):
            self._pre[slot], self._out[slot], self._fired[slot], self._len[slot] = pre, out, self._now, length
            if slot in ("pulse", "rumble"):
                self._oneshot_wall = self._wall
            self._ghosts.pop(slot, None)

    def _fire_riser(self, mag: float) -> None:
        """One whole riser cycle: a device sine of one period, amplitude fixed for the cycle."""
        if self._latched or self._closed:
            return
        out = mag * _clamp(self._g, 0.0, 1.0)
        if out < 1e-6:
            return
        end = self._wall + RISER_CYCLE_MS / 1000
        if (self._call("riser", "update", magnitude=out, period_ms=RISER_CYCLE_MS, length_ms=RISER_CYCLE_MS,
                       phase_deg=RISER_PHASE_DEG)
                and self._call("riser", "run", ghost=(mag, end))):
            end = _num(self.clock(), self._wall) + RISER_CYCLE_MS / 1000   # read after run returns
            self._pre["riser"], self._out["riser"], self._riser_end = mag, out, end
            self._fired["riser"], self._len["riser"] = self._now, RISER_CYCLE_MS
            self._ghosts.pop("riser", None)

    def _step(self, events: list[GameEvent], snap: Snapshot, dt: float, spinning: bool) -> None:
        # turn offset contract (SPEC 9 rule 10); read with defaults so older Snapshots work
        offset = _num(getattr(snap, "steer_offset_deg", 0.0), self._offset)   # non-finite: hold the last
        unwind = bool(getattr(snap, "steer_unwind", False))
        # by value: the steer_offset_changed flag can be missed or repeated. Only the springs
        # have a centre: they stop now and ramp back from the next frame; damper and riser go on.
        resync = offset != self._offset
        self._offset = offset
        # per slot dead-man: a ramped effect not re-run for ENDED_S may have ended on the device,
        # whatever the gap between updates was. Stop it for certain; it starts again from zero.
        # A slot re-run more recently is still live and keeps its state, so a zero target sends stop.
        for s in RAMPED:
            if s in self._running and self._wall - self._ran_at[s] >= SLOTS[s].deadman_s * ENDED_FRACTION:
                self._running.discard(s)
                self._call(s, "stop", ghost=(abs(self._pre_sent[s]), self._ran_at[s] + SLOTS[s].deadman_s))
                if s == "echo":
                    self._echo_restart = True
                if s in SUSTAINED:
                    self._arm_brake()
                self._pre[s] = self._out[s] = self._pre_sent[s] = 0.0
        if resync:
            # the springs stop this frame (want is zero below) and ramp back at the new centre
            self._pre["echo"] = self._pre["weight_spring"] = 0.0
        if self._halted:
            self._halted = False
            try:
                self.backend.set_gain(1.0)
            except Exception:
                self._note("WARN device gain not restored")
            self._sign = self.settings.ffb_sign   # a sign change takes effect only at a start
        self._t += dt
        raw_steer = snap.input.steer_deg
        steer_ok = isinstance(raw_steer, int | float) and math.isfinite(raw_steer)
        if steer_ok:
            angle = float(raw_steer)
            if self._angle is not None and 0.0 < self._elapsed <= MAX_DT:
                self._vels = [*self._vels[-2:], _clamp((angle - self._angle) / self._elapsed, -VEL_MAX, VEL_MAX)]
            elif self._elapsed > MAX_DT:
                self._vels = []
            self._angle = self._steer = angle
        else:
            self._vels, self._angle = [], None    # a non-finite sample resets the angle history
        if len(self._vels) == 2:
            self._vel = min(self._vels, key=abs)    # symmetric: the smaller magnitude of two
        else:
            self._vel = sorted(self._vels)[len(self._vels) // 2] if self._vels else 0.0
        steer = self._steer if self._steer is not None else offset
        now = _num(snap.now)

        # one-shots: decided first, within their own budget, kick before rumble before pulse
        wants: dict[str, tuple[float, dict]] = {}
        for e in events:
            if isinstance(e, Beat) and now - e.t <= BEAT_STALE and _num(e.strength) > 0:
                wants["pulse"] = (0.5 * _clamp(_num(e.strength), 0.0, 1.0),
                                  {"period_ms": PULSE_PERIOD_MS})
            elif isinstance(e, SectionChange):
                self._note(f"WEIGHT {round(_clamp(_num(e.weight), 0.0, 1.0) * 100)}%")
            elif isinstance(e, FfbCue):
                if e.name == "kick":
                    d = 1.0 if _num(e.dir, 1.0) >= 0 else -1.0
                    if self._active("kick") and d * self._pre["kick"] < 0:
                        continue    # an opposite shove on top of a live kick would swing 1.2
                    wants["kick"] = (KICK_LEVEL * d, {"length_ms": PATTERN_KICK_MS} if self._mode == "kick" else {})
                    self._note(f"KICK {'right' if d > 0 else 'left'}")
                elif e.name == "rumble":
                    wants["rumble"] = (RUMBLE_MAG, {"period_ms": RUMBLE_PERIOD_MS,
                                                    "attack_ms": RUMBLE_EDGE_MS, "fade_ms": RUMBLE_EDGE_MS})
                    self._note("MISS rumble")
                elif e.name == "riser_start":
                    self._note("RISER wind-up")
                elif e.name == "riser_release":
                    self._note("RISER release")
        held = {s: abs(self._pre[s]) if self._active(s) else 0.0 for s in ONESHOTS}
        fire: list[tuple[str, float, dict]] = []
        for slot in ONESHOTS:
            if slot not in wants or not self._available(slot):
                continue
            raw, params = wants[slot]
            room = ONESHOT_BUDGET - sum(v for s, v in held.items() if s != slot) - self._ghost_load(ONESHOTS)
            pre = math.copysign(min(abs(self._cap(slot, raw)), max(0.0, room)), raw)
            if abs(pre) >= 1e-3:
                held[slot] = abs(pre)
                fire.append((slot, pre, params))
        ramp_budget = max(0.0, BUDGET - sum(held.values()) - self._ghost_load())

        # springs need a finite wheel angle within half a turn of the offset (whatever steer_unwind
        # says); otherwise only the damper runs
        springs = steer_ok and not unwind and abs(steer - offset) <= 180.0
        rate = 1.0 / max(f[1] for f in self._frame_times) if self._frame_times else 0.0
        if self._springs_ok and rate < RATE_OFF_HZ:
            self._springs_ok = False
            self._arm_brake()
            self._note(f"SPRINGS off - {round(rate)} Hz")
            self._springs_logged = True
        elif not self._springs_ok and rate >= RATE_ON_HZ:
            self._springs_ok = True
            if self._springs_logged:
                self._note(f"SPRINGS on - {round(rate)} Hz")
        springs = springs and self._springs_ok
        self._floor_seen = True
        rf = min(1.0, rate / RATE_FULL_HZ)
        if self._springs_prev and not springs:
            self._ref_restart = True    # springs dropped (rate floor, unwind, angle): return gently later
        self._springs_prev = springs
        # fixed-centre springs (weight, centre pattern): after a stall their centre starts at the wheel
        # and returns to the offset at ECHO_RECOVER_DEG_S, so a stopped wheel is not snapped back
        if not springs:
            self._ref = None
        elif self._ref_restart:
            self._ref, self._ref_restart = steer, False
        elif self._ref is not None:
            v = ECHO_RECOVER_DEG_S * dt
            self._ref += _clamp(offset - self._ref, -v, v)
            if abs(offset - self._ref) < 1e-9:
                self._ref = None
        ref = offset if self._ref is None else self._ref

        # echo: the base turns the wheel toward echo_target (rule 3)
        target = snap.echo_target
        listen = snap.echo == "listen" and target is not None and math.isfinite(_num(target, math.nan))
        goal = _clamp(_num(target), -1.0, 1.0) * _num(self.cfg.play_range_deg, 90.0) + offset if listen else steer
        echo_on = listen and springs
        if listen and self._echo_on and not springs:
            self._echo_restart = True   # echo dropped mid-phrase (rate floor, unwind, angle): return gently
        if resync or self._echo_restart or (not self._echo_on and self._out["echo"] == 0.0):
            # re-sync when an echo starts from zero output: after an offset change, a gap, a
            # dead-man end, a failed call or a stop (SPEC 5 rules 9 and 13)
            # a restart during a phrase (gap, dead-man end, failed call, pause) returns gently;
            # the flag holds through the frames the springs stay off
            if listen and self._echo_restart:
                self._echo_recover = True
            if not listen:
                self._echo_recover = False
            self._centre = steer
            self._echo_restart = False
        else:
            rate_limit = min(self._echo_rate, PATTERN_ECHO_DEG_S) if self._mode == "echo" else self._echo_rate
            if self._echo_recover:
                rate_limit = min(rate_limit, ECHO_RECOVER_DEG_S)
                if abs(goal - self._centre) <= rate_limit * dt or not listen:
                    self._echo_recover = False
            v = rate_limit * dt
            self._centre += _clamp(goal - self._centre, -v, v)
        self._centre = _clamp(self._centre, steer - ECHO_LEAD_DEG, steer + ECHO_LEAD_DEG)
        if listen and not self._listen:
            self._note(f"ECHO spring centre -> {round(goal)} deg")
        self._listen, self._echo_on = listen, echo_on

        # riser wind-up: grows while held; the cycle amplitude fades in and out over RISER_FADE_S
        # (start, release, retry) and the wind-up resets once faded out. Wall-clock steps only.
        wall_dt = _clamp(self._elapsed, 0.0, MAX_DT)
        r = snap.riser
        playing = r.active and r.held and self._available("riser")
        if playing:
            top = SLOTS["riser"].cap * (0.2 + 0.8 * _clamp(_num(r.progress), 0.0, 1.0))
            self._wind = min(top, self._wind + RISER_RISE * wall_dt)
            self._riser_env = min(1.0, self._riser_env + wall_dt / RISER_FADE_S)
        else:
            self._riser_env = max(0.0, self._riser_env - wall_dt / RISER_FADE_S)
            if self._riser_env == 0.0:
                self._wind = 0.0

        # section weight: the spring gives way to the echo; the damper stays to keep the echo stable
        w = _clamp(_num(snap.weight), 0.0, 1.0)
        # software springs: a constant level from the angle error, minus a velocity term
        # divided by sqrt(gain): stiffness and damping both scale with the gain, so this keeps the
        # damping ratio the same at every gain (the output is still gain x level, within every cap)
        # Below RATE_FULL_HZ: velocity term x rf, spring levels x rf^2. Each velocity term is clamped to
        # its own spring's full level.
        damp = K_V * rf * self._vel / math.sqrt(max(_clamp(self._g, 0.0, 1.0), 0.05))
        weight_on = springs and not listen and not spinning
        full_echo = SLOTS["echo"].cap * rf * rf
        full_weight = SLOTS["weight_spring"].cap * (0.2 + 0.8 * w) * rf * rf
        raw = {
            "echo": full_echo * _clamp((self._centre - steer) / ECHO_FULL_DEG, -1.0, 1.0)
            - _clamp(damp, -full_echo, full_echo) if echo_on else 0.0,
            "weight_spring": full_weight * _clamp((ref - steer) / WEIGHT_FULL_DEG, -1.0, 1.0)
            - _clamp(damp, -full_weight, full_weight) if weight_on else 0.0,
            "weight_damper": 0.1 + 0.4 * w,
            "pattern": 0.0,
        }
        if self._mode == "centre":   # centre pattern: a software spring on its own slot
            raw["weight_spring"] = 0.0
            full = SLOTS["pattern"].cap * rf * rf
            raw["pattern"] = (-full * _clamp((steer - ref) / PATTERN_SPRING_DEG, -1.0, 1.0)
                            - _clamp(damp, -full, full)) if springs else 0.0
        elif self._mode in ("kick", "echo", "riser"):
            raw["weight_spring"] = 0.0
            if self._mode in ("kick", "riser") or self._wait:
                raw["weight_damper"] = 0.0   # "nothing moves" while waiting is exact
        # stall brake: only opposes motion; off once a spring really produces output again, or
        # when all 3 speed samples are slow (one repeated angle right after a stall reads as 0 deg/s)
        spring_live = ((echo_on and self._available("echo") and self._strength("echo") > 0)
                       or (weight_on and self._mode == "" and self._available("weight_spring")
                           and self._strength("weight_spring") > 0))
        slow = len(self._vels) == 3 and all(abs(v) < BRAKE_STOP_DEG_S for v in self._vels)
        if self._brake_armed and (spring_live or slow):
            self._brake_armed = False
        raw["brake"] = 0.0
        if self._brake_armed and steer_ok and self._vels and self._vels[-1] * self._vel > 0:
            # zero when the newest sample and the median disagree in sign; otherwise the smaller of
            # the two, so the brake lets go as fast as the wheel slows (the median alone lags)
            speed = min(self._vel, self._vels[-1], key=abs)
            k = K_BRAKE * min(1.0, BRAKE_FRAME_S / max(self._elapsed, 1e-6))   # constant gain per frame
            raw["brake"] = -k * speed
        held_off = ("echo", "weight_spring") if resync else ()
        want = {s: self._cap(s, raw[s]) if self._available(s) and s not in held_off else 0.0 for s in RAMPED}
        room = max(0.0, SUSTAINED_CAP - self._ghost_load((*SUSTAINED, "riser")))   # stuck slots count too
        # a new riser cycle takes its share now, scaled with this frame's wants, and keeps it for its
        # whole length; the ramped slots get what is left
        riser_fire = 0.0
        amp = abs(self._cap("riser", self._wind * self._riser_env))   # strength.riser (0 by default) applies
        near_oneshot = (self._wall - self._oneshot_wall < RISER_ONESHOT_GAP_S
                        or any(s in ("pulse", "rumble") for s, _, _ in fire))
        if (self._wall >= self._riser_end + RISER_GAP_S and not near_oneshot and amp >= 1e-3
                and self._available("riser")):
            sus = sum(abs(want[s]) for s in SUSTAINED) + amp
            tot = sum(abs(v) for v in want.values()) + amp
            riser_fire = amp * min(1.0, room / sus, ramp_budget / tot)
        riser_hold = riser_fire if riser_fire >= 1e-3 else self._riser_live()
        room = max(0.0, room - riser_hold)
        ramp_budget = max(0.0, ramp_budget - riser_hold)
        sustained = sum(abs(want[s]) for s in SUSTAINED)
        if sustained > room:                              # the sum rule: scale them down together
            for s in SUSTAINED:
                want[s] *= room / sustained
        total = sum(abs(v) for v in want.values())
        if total > ramp_budget:
            want = {s: v * ramp_budget / total for s, v in want.items()}

        # ramp: effect values and the effective gain each slew by a quarter of the limit. Rises use
        # min(caller dt, wall step); falls toward zero use the wall step, so a caller dt of 0 or NaN
        # cannot freeze a level the device keeps re-running
        step = min(MAX_STEP, SLEW_RATE * dt)
        fall = min(MAX_STEP, SLEW_RATE * wall_dt)
        self._g = _slew(self._g, self._gain, step, fall)
        for s in RAMPED:
            self._pre[s] = _slew(self._pre[s], want[s], step, fall)
        sustained = sum(abs(self._pre[s]) for s in SUSTAINED)
        if sustained > room:                              # also after the slew: drop, never rise
            for s in SUSTAINED:
                self._pre[s] *= room / sustained
        total = sum(abs(self._pre[s]) for s in RAMPED)
        if total > ramp_budget:   # a one-shot just took its share: drop, never rise
            for s in RAMPED:
                self._pre[s] *= max(0.0, ramp_budget) / total

        # every falling call before every rising one: the device sum stays in the budget
        # after every single call of the frame
        sends: dict[str, dict] = {
            "echo": {},
            "weight_spring": {},
            "weight_damper": {"stiffness": SLOTS["weight_damper"].cap},
            "pattern": {},
            "brake": {},
        }
        g = _clamp(self._g, 0.0, 1.0)
        calls = [(abs(self._pre[s]) * g - abs(self._out[s]), 0, s, lambda s=s: self._send(s, **sends[s]))
                 for s in sends]
        # a re-fired one-shot replaces its own live level, so a lower re-fire is a falling call
        calls += [(abs(pre) * g - (abs(self._out[s]) if self._active(s) else 0.0), 1, s,
                   lambda s=s, pre=pre, params=params: self._fire(s, pre, **params)) for s, pre, params in fire]
        if riser_fire >= 1e-3:                            # the previous cycle has ended: a rising call
            calls.append((riser_fire * g, 1, "riser", lambda: self._fire_riser(riser_fire)))
        for _, _, _, send in sorted(calls, key=lambda c: (c[0], c[1])):
            send()
        self._torque = self._estimate(steer)

    def _estimate(self, steer: float) -> float:
        """Signed torque estimate for the HUD; not clamped, so an excess would show."""
        out, now = self._out, self._now
        tq = sum(out[s] for s in SUSTAINED) * self._sign   # the constant levels as sent
        if self._active("kick"):
            tq += out["kick"] * self._sign
        for slot, period in (("pulse", PULSE_PERIOD_MS), ("rumble", RUMBLE_PERIOD_MS)):
            if self._active(slot):
                tq += out[slot] * math.sin(2 * math.pi * (now - self._fired[slot]) * 1000 / period)
        if self._wall < self._riser_end:
            start = self._riser_end - RISER_CYCLE_MS / 1000
            phase = math.radians(RISER_PHASE_DEG)                        # the riser starts as a cosine
            tq += out["riser"] * math.sin(2 * math.pi * (self._wall - start) * 1000 / RISER_CYCLE_MS + phase)
        return tq
