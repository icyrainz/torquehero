"""Device bindings: which device input drives each control (SPEC 8.3).

A binding names its device by SDL GUID plus name plus a stable key (the USB
serial, else the device path), so bindings survive devices being unplugged or
re-ordered, and a binding never moves to an identical twin device. The SDL index
is a tiebreak only between devices with no serial and no path.

Learn and calibration are pure functions over device snapshots (baseline vs
current). The `probe` and `bind` commands are thin terminal loops around them.

SDL is reached only through `SdlBackend`, which imports `sdl2` on first use.
Tests use fake devices with the same shape as `SdlDevice`.
"""
from __future__ import annotations

import ctypes
import json
import logging
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol

from .config import config_dir
from .state import BIPOLAR, CONTROLS, UNIPOLAR

log = logging.getLogger(__name__)

BINDINGS_FILE = "bindings.json"
FORMAT = 1

ControlKind = Literal["bipolar", "unipolar", "button"]
SourceKind = Literal["axis", "button", "hat"]
SOURCE_KINDS = ("axis", "button", "hat")
LEVERS = ("lever0", "lever1")

# Learn thresholds in raw axis units (-1..1, so 0.3 is 15% of the travel).
LEARN_AXIS = 0.3        # pedals, handbrake, levers, axis-as-button: low enough for a stiff load cell
LEARN_STEER = 0.03      # wheel: about 14 degrees on a 900 degree base, 38 on 2520
MIN_SPAN = 0.05         # smallest usable rest-to-full travel
SETTLE = 0.1            # an input is back at rest within this of its baseline
STABLE_TOL = 0.02       # a baseline must hold within this ...
STABLE_SPAN = 0.3       # ... over this many seconds
STEER_CAL_DEG = 90.0    # the player holds the wheel here to calibrate its range
REST_NEAR_END = 0.8     # a pedal resting inside +-this was probably touched at baseline
PEAK_TIMEOUT = 15.0     # seconds to track a pedal's full travel before giving up
DRIFT_SAMPLES = 7       # an axis still moving after this many baseline samples (~2 s) is ignored

HAT_DIRS = {1: "up", 2: "right", 4: "down", 8: "left"}


def control_kind(name: str) -> ControlKind:
    if name in BIPOLAR:
        return "bipolar"
    if name in UNIPOLAR:
        return "unipolar"
    if name in CONTROLS:
        return "button"
    raise KeyError(f"unknown control {name!r}")


@dataclass(frozen=True)
class DeviceSnapshot:
    """One device's inputs at one instant. Axes are -1..1; hats are SDL bitmasks.
    `key` is "serial:..." or "path:..." or "" when SDL knows neither."""

    guid: str
    name: str
    index: int
    axes: tuple[float, ...] = ()
    buttons: tuple[bool, ...] = ()
    hats: tuple[int, ...] = ()
    key: str = ""


class _Identified(Protocol):
    @property
    def guid(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def index(self) -> int: ...
    @property
    def key(self) -> str: ...



@dataclass(frozen=True)
class Binding:
    """One control bound to one device input.

    `kind` "axis": `num` is the axis. Unipolar and button controls learn `rest`
    and `full` (the raw values at rest and at full travel); value is
    (raw - rest) / (full - rest), clamped 0..1. `sign` is the direction of travel.
    The steer axis has no rest: value is raw * sign, and `range_deg` is the
    lock-to-lock rotation the raw -1..1 spans (None: use Config.wheel_range_deg).
    "button": `num` is the button. "hat": `num` is the hat, `dir` the direction
    bit (1 up, 2 right, 4 down, 8 left)."""

    guid: str
    name: str
    index: int
    kind: SourceKind
    num: int
    rest: float | None = None
    sign: float = 1.0
    dir: int = 0
    full: float | None = None
    range_deg: float | None = None
    key: str = ""

    @property
    def digital(self) -> bool:
        return self.kind != "axis"

    @property
    def _full(self) -> float:
        return self.full if self.full is not None else self.sign

    def label(self) -> str:
        if self.kind == "button":
            src = f"button {self.num}"
        elif self.kind == "hat":
            src = f"hat {self.num} {HAT_DIRS.get(self.dir, self.dir)}"
        elif self.rest is None:
            src = f"axis {self.num}{' -' if self.sign < 0 else ' +'}"
            if self.range_deg is not None:
                src += f" range {self.range_deg:.0f} deg"
        else:
            src = f"axis {self.num} rest {self.rest:+.2f} full {self._full:+.2f}"
        return f"{self.name} #{self.index}: {src}"

    def raw(self, snap: DeviceSnapshot) -> float | None:
        """The bound input as SDL reports it: axis -1..1, button or hat 0/1."""
        try:
            if self.kind == "axis":
                return snap.axes[self.num]
            if self.kind == "button":
                return 1.0 if snap.buttons[self.num] else 0.0
            return 1.0 if snap.hats[self.num] & self.dir else 0.0
        except IndexError:
            return None

    def value(self, snap: DeviceSnapshot) -> float | None:
        """Normalised value: steer axis -1..1 (sign applied), everything else 0..1
        with 0 at rest. None when the device lacks the input."""
        raw = self.raw(snap)
        if raw is None or self.kind != "axis":
            return raw
        if self.rest is None:
            return max(-1.0, min(1.0, raw * self.sign))
        span = self._full - self.rest
        if abs(span) < MIN_SPAN:
            span = math.copysign(MIN_SPAN, span or self.sign)
        return max(0.0, min(1.0, (raw - self.rest) / span))

    def to_dict(self) -> dict:
        d: dict = {"guid": self.guid, "name": self.name, "index": self.index, "kind": self.kind, "num": self.num}
        if self.key:
            d["key"] = self.key
        if self.kind == "axis":
            d["sign"] = self.sign
            for k in ("rest", "full", "range_deg"):
                if getattr(self, k) is not None:
                    d[k] = getattr(self, k)
        if self.kind == "hat":
            d["dir"] = self.dir
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Binding:
        """Raises KeyError, TypeError, ValueError or OverflowError on a malformed
        entry; non-finite numbers are rejected."""
        kind = d["kind"]
        if kind not in SOURCE_KINDS:
            raise ValueError(f"unknown input kind {kind!r}")
        num, dir_ = int(d["num"]), int(d.get("dir", 0))
        if num < 0 or (kind == "hat" and dir_ not in HAT_DIRS):
            raise ValueError("bad input number or hat direction")

        def opt(k):
            v = d.get(k)
            if v is None:
                return None
            if not math.isfinite(v := float(v)):
                raise ValueError(f"{k} is not finite")
            return v

        range_deg, sign = opt("range_deg"), opt("sign")
        if range_deg is not None and range_deg <= 0:
            raise ValueError("range_deg must be positive")
        return cls(guid=str(d["guid"]), name=str(d["name"]), index=int(d.get("index", 0)), kind=kind, num=num,
                   rest=opt("rest"), sign=-1.0 if sign is not None and sign < 0 else 1.0, dir=dir_,
                   full=opt("full"), range_deg=range_deg, key=str(d.get("key", "")))


def match_device[D: _Identified](b: Binding, devices: Sequence[D]) -> D | None:
    """The device a binding names, from snapshots or open devices alike. Same GUID
    and name, else same GUID (renamed by a driver update). A binding with a key
    takes only the device with that key, so it never moves to an identical twin.
    Without a key, among keyless devices the bound index wins, else the first."""
    cands = [d for d in devices if d.guid == b.guid and d.name == b.name] or [d for d in devices if d.guid == b.guid]
    if b.key:
        return next((d for d in cands if d.key == b.key), None)
    cands = [d for d in cands if not d.key]
    if not cands:
        return None
    return next((d for d in cands if d.index == b.index), cands[0])


def _pairs(baseline: list[DeviceSnapshot], current: list[DeviceSnapshot]):
    base = {(s.guid, s.name, s.key, s.index): s for s in baseline}
    for cur in current:
        b0 = base.get((cur.guid, cur.name, cur.key, cur.index))
        if b0 is not None:
            yield b0, cur


def _dev(s: DeviceSnapshot) -> dict:
    return {"guid": s.guid, "name": s.name, "index": s.index, "key": s.key}


def axis_id(s: DeviceSnapshot, i: int) -> tuple:
    return (s.guid, s.name, s.key, s.index, i)


def learn(control: str, baseline: list[DeviceSnapshot], current: list[DeviceSnapshot],
          ignore: frozenset | set = frozenset()) -> Binding | None:
    """The input the player moved for `control` since `baseline`, or None yet.

    Bipolar (steer): the axis that moved furthest past LEARN_STEER; the direction
    moved becomes positive (turn right to learn). Unipolar: an axis past LEARN_AXIS
    (rest and direction learned, `full` starts at the current value and grows with
    `extend_peak`), else a newly pressed button or hat direction. Button: a newly
    pressed button or hat direction, else an axis past LEARN_AXIS. Axes in
    `ignore` (`axis_id`s of drifting axes) never count."""
    kind = control_kind(control)
    axis_hits: list[tuple[float, Binding]] = []
    digital_hits: list[Binding] = []
    limit = LEARN_STEER if kind == "bipolar" else LEARN_AXIS
    for b0, cur in _pairs(baseline, current):
        dev = _dev(cur)
        for i, (v0, v1) in enumerate(zip(b0.axes, cur.axes, strict=False)):
            d = v1 - v0
            if abs(d) > limit and axis_id(cur, i) not in ignore:
                sign = 1.0 if d > 0 else -1.0
                b = (Binding(**dev, kind="axis", num=i, sign=sign) if kind == "bipolar"
                     else Binding(**dev, kind="axis", num=i, rest=v0, sign=sign, full=v1))
                axis_hits.append((abs(d), b))
        if kind == "bipolar":
            continue
        for i, (p0, p1) in enumerate(zip(b0.buttons, cur.buttons, strict=False)):
            if p1 and not p0:
                digital_hits.append(Binding(**dev, kind="button", num=i))
        for i, (h0, h1) in enumerate(zip(b0.hats, cur.hats, strict=False)):
            new = h1 & ~h0
            if new:
                digital_hits.append(Binding(**dev, kind="hat", num=i, dir=new & -new))
    best_axis = max(axis_hits, key=lambda h: h[0])[1] if axis_hits else None
    first_digital = digital_hits[0] if digital_hits else None
    if kind == "button":
        return first_digital or best_axis
    return best_axis or first_digital


def extend_peak(b: Binding, snaps: list[DeviceSnapshot]) -> Binding:
    """Move `full` out to the furthest raw value seen in the direction of travel."""
    if b.kind != "axis" or b.rest is None:
        return b
    dev = match_device(b, snaps)
    raw = b.raw(dev) if dev else None
    full = b.full if b.full is not None else b.rest
    if raw is not None and (raw - full) * b.sign > 0:
        return replace(b, full=raw)
    return b


def learn_range(control: str, closed: list[DeviceSnapshot], opened: list[DeviceSnapshot]) -> Binding | None:
    """A spring-less axis (STECS lever) from its two ends: the axis that moved most
    between the closed and the open snapshot, rest = closed, full = open."""
    if control_kind(control) == "bipolar":
        raise ValueError("steer is not a ranged axis")
    best: tuple[float, Binding] | None = None
    for a, b in _pairs(closed, opened):
        for i, (v0, v1) in enumerate(zip(a.axes, b.axes, strict=False)):
            d = v1 - v0
            if abs(d) > LEARN_AXIS and (best is None or abs(d) > best[0]):
                best = (abs(d), Binding(**_dev(b), kind="axis", num=i, rest=v0, sign=1.0 if d > 0 else -1.0,
                                        full=v1))
    return best[1] if best else None


def steer_range(b: Binding, held: list[DeviceSnapshot], deg: float = STEER_CAL_DEG) -> float | None:
    """Lock-to-lock degrees of the steer axis, from the raw value with the wheel
    held `deg` degrees right. None when the wheel is not turned right, or the
    result is outside 180..3600 degrees."""
    dev = match_device(b, held)
    raw = b.raw(dev) if dev else None
    if raw is None or raw * b.sign <= 0:
        return None
    r = 2 * deg / (raw * b.sign)
    return r if 180.0 <= r <= 3600.0 else None


def unsettled(a: list[DeviceSnapshot], b: list[DeviceSnapshot], tol: float = STABLE_TOL) -> set[tuple]:
    """`axis_id`s of the axes that moved more than `tol` between two samples of the rig."""
    return {axis_id(s1, i) for s0, s1 in _pairs(a, b)
            for i, (x, y) in enumerate(zip(s0.axes, s1.axes, strict=False)) if abs(x - y) > tol}


def stable(a: list[DeviceSnapshot], b: list[DeviceSnapshot], tol: float = STABLE_TOL) -> bool:
    """True when no axis moved more than `tol` between two samples of the rig."""
    return not unsettled(a, b, tol)


def settled(b: Binding, baseline: list[DeviceSnapshot], current: list[DeviceSnapshot]) -> bool:
    """True when the bound input is back near its baseline (the player let go)."""
    s0, s1 = match_device(b, baseline), match_device(b, current)
    if s0 is None or s1 is None:
        return True
    r0, r1 = b.raw(s0), b.raw(s1)
    return r0 is None or r1 is None or abs(r1 - r0) <= SETTLE


def rest_at_full_travel(b: Binding, current: list[DeviceSnapshot]) -> bool:
    """An axis still at its `full` end after the player let go: the learned rest
    was full travel (held down while the baseline was taken, or an inverted pedal)."""
    if b.kind != "axis" or b.rest is None or b.full is None:
        return False
    s = match_device(b, current)
    raw = b.raw(s) if s else None
    return raw is not None and abs(raw - b.full) <= SETTLE and abs(b.full - b.rest) >= MIN_SPAN


def rest_suspicious(b: Binding) -> bool:
    """A pedal axis whose learned rest is mid-travel: probably a foot on it."""
    return b.kind == "axis" and b.rest is not None and abs(b.rest) < REST_NEAR_END


@dataclass
class Bindings:
    """Control name -> Binding, persisted as JSON under `config_dir()`."""

    controls: dict[str, Binding] = field(default_factory=dict)
    # key -> keys of devices seen present at the same time this session (a proven twin)
    _together: dict[str, set[str]] = field(default_factory=dict, compare=False, repr=False)

    def __contains__(self, name: str) -> bool:
        return name in self.controls

    def get(self, name: str) -> Binding | None:
        return self.controls.get(name)

    def set(self, name: str, b: Binding) -> None:
        control_kind(name)
        self.controls[name] = b

    def clear(self, names: list[str] | None = None) -> None:
        if names is None:
            self.controls.clear()
        for n in names or ():
            self.controls.pop(n, None)

    def resolve[T: _Identified](self, devices: Sequence[T]) -> dict[str, T]:
        """Control -> device for every binding whose device is present. A binding
        keyed by a device path whose path changed (another USB port, re-enumeration)
        takes the one present device with the same GUID and name that no other
        binding claims by key and that was never present together with the old key
        this session (that one is a different, identical device); its key is
        updated here (saved on the next save) and one warning is logged. Serial
        keys never move."""
        present = {d.key for d in devices if d.key}
        for b in self.controls.values():
            if b.key in present:
                self._together.setdefault(b.key, set()).update(present - {b.key})
        out = {n: d for n, b in self.controls.items() if (d := match_device(b, devices)) is not None}
        claimed = {b.key for b in self.controls.values() if b.key}
        for n, b in list(self.controls.items()):
            if n in out or not b.key.startswith("path:"):
                continue
            twins = self._together.get(b.key, set())
            cands = [d for d in devices
                     if d.guid == b.guid and d.name == b.name and d.key not in claimed and d.key not in twins]
            if len(cands) != 1:
                continue
            d, old = cands[0], b.key
            log.warning("%s moved from %s to %s; bindings follow it", b.name, old, d.key or "(no key)")
            for m, o in list(self.controls.items()):
                if o.key == old and o.guid == b.guid:
                    self.controls[m] = replace(o, key=d.key, index=d.index)
                    out[m] = d
            claimed.add(d.key)
            self._together[d.key] = self._together.pop(old, set())
        return out

    def to_dict(self) -> dict:
        ordered = {n: self.controls[n].to_dict() for n in CONTROLS if n in self.controls}
        d: dict = {"format": FORMAT, "bindings": ordered}
        twins = {k: sorted(v) for k, v in sorted(self._together.items()) if v}
        if twins:
            d["twins"] = twins      # keys seen together: the twin guard survives a restart
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Bindings:
        """Unknown controls and malformed entries are skipped (one warning) so the
        game still starts with the rest."""
        out = cls()
        raw = d.get("bindings") if isinstance(d, dict) else None
        if not isinstance(raw, dict):
            log.warning("bindings file has no bindings table; starting with no bindings")
            return out
        skipped = []
        for name, entry in raw.items():
            try:
                if name not in CONTROLS:
                    raise ValueError("unknown control")
                b = Binding.from_dict(entry)
                if name == "steer" and b.kind != "axis":
                    raise ValueError("steer needs an axis")
                out.controls[name] = b
            except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
                skipped.append(str(name))
        if skipped:
            log.warning("skipped bad bindings: %s", ", ".join(skipped))
        twins = d.get("twins")
        if isinstance(twins, dict):
            for k, v in twins.items():
                if isinstance(k, str) and isinstance(v, list):
                    out._together[k] = {x for x in v if isinstance(x, str)}
        return out

    @classmethod
    def load(cls, path: str | Path | None = None) -> Bindings:
        """Missing file: no bindings. Unreadable or corrupt file: one warning, no bindings."""
        path = Path(path) if path else default_path()
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            log.warning("ignoring unreadable bindings file %s: %s", path, e)
            return cls()
        return cls.from_dict(data)

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


def default_path() -> Path:
    return config_dir() / BINDINGS_FILE


# --- SDL devices ---

class Device(Protocol):
    """An open joystick. `SdlDevice` wraps SDL; tests supply fakes."""

    guid: str
    name: str
    index: int
    key: str
    haptic: bool
    handle: object          # the SDL_Joystick pointer (FFB opens its haptic from it)

    def snapshot(self) -> DeviceSnapshot: ...


@dataclass(frozen=True)
class JoyEvent:
    """A button or hat change from the SDL event queue, so presses shorter than a
    frame are not lost. `value`: button 0/1, hat bitmask."""

    device: object
    kind: Literal["button", "hat"]
    num: int
    value: int


class JoystickBackend(Protocol):
    def update(self) -> None: ...                  # pump SDL, open new devices, close detached ones
    def devices(self) -> Sequence[Device]: ...
    def events(self) -> Sequence[JoyEvent]: ...    # button/hat events from the last update, oldest first
    # cb(handle) before a detached device closes / before close() closes it; returns a remover
    def on_detach(self, cb: Callable[[object], None]) -> Callable[[], None]: ...
    def on_close(self, cb: Callable[[object], None]) -> Callable[[], None]: ...
    def close(self) -> None: ...


def notify(callbacks: list[Callable[[object], None]], handle: object) -> None:
    """Run callbacks in registration order; one failing never stops the rest."""
    for cb in callbacks:
        try:
            cb(handle)
        except Exception:
            log.exception("joystick callback failed")


class SdlError(RuntimeError):
    pass


_SDL = None


def _sdl():
    """The sdl2 module, imported once (pysdl2-dll warns on import; silence it once)."""
    global _SDL
    if _SDL is None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import sdl2
        _SDL = sdl2
    return _SDL


def _device_key(s, j) -> str:
    serial = s.SDL_JoystickGetSerial(j) if hasattr(s, "SDL_JoystickGetSerial") else None
    if serial:
        return "serial:" + serial.decode(errors="replace")
    path = s.SDL_JoystickPath(j) if hasattr(s, "SDL_JoystickPath") else None
    if path:
        return "path:" + path.decode(errors="replace")
    return ""


@dataclass(eq=False)
class SdlDevice:
    guid: str
    name: str
    index: int
    key: str
    haptic: bool
    handle: object
    instance_id: int
    n_axes: int
    n_buttons: int
    n_hats: int
    closing: bool = False       # handlers are running before SDL_JoystickClose: skip it everywhere

    def snapshot(self) -> DeviceSnapshot:
        s, j = _sdl(), self.handle
        return DeviceSnapshot(
            self.guid, self.name, self.index,
            tuple(max(-1.0, s.SDL_JoystickGetAxis(j, i) / 32767) for i in range(self.n_axes)),
            tuple(bool(s.SDL_JoystickGetButton(j, i)) for i in range(self.n_buttons)),
            tuple(int(s.SDL_JoystickGetHat(j, i)) for i in range(self.n_hats)),
            self.key,
        )


class SdlBackend:
    """SDL joystick and haptic subsystems only: no video, no signal handlers,
    background events allowed so the rig keeps working when the game window is not
    focused. Devices are tracked by instance id: a hot-plug opens only the new
    device and closes only the detached one, so the wheel's joystick (and the
    haptic device FFB opened from it) stays open while other devices come and go."""

    BUF = 64

    def __init__(self) -> None:
        s = _sdl()
        s.SDL_SetHint(s.SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, b"1")
        s.SDL_SetHint(s.SDL_HINT_NO_SIGNAL_HANDLERS, b"1")
        if s.SDL_InitSubSystem(s.SDL_INIT_JOYSTICK | s.SDL_INIT_HAPTIC) != 0:
            raise SdlError(f"SDL init failed: {s.SDL_GetError().decode(errors='replace')}")
        s.SDL_JoystickEventState(s.SDL_ENABLE)
        s.SDL_EventState(s.SDL_JOYAXISMOTION, s.SDL_IGNORE)      # axes are level-polled
        s.SDL_EventState(s.SDL_JOYBALLMOTION, s.SDL_IGNORE)
        self._shut = False
        self._detach: list[Callable[[object], None]] = []
        self._close: list[Callable[[object], None]] = []
        self._devices: dict[int, SdlDevice] = {}
        self._events: list[JoyEvent] = []
        self._buf = (s.SDL_Event * self.BUF)()
        self.update()

    def _open(self, i: int) -> SdlDevice | None:
        s = _sdl()
        j = s.SDL_JoystickOpen(i)
        if not j:
            return None
        guid = ctypes.create_string_buffer(33)
        s.SDL_JoystickGetGUIDString(s.SDL_JoystickGetGUID(j), guid, 33)
        name = s.SDL_JoystickName(j)
        return SdlDevice(
            guid=guid.value.decode(), name=name.decode(errors="replace") if name else f"joystick {i}",
            index=i, key=_device_key(s, j), haptic=s.SDL_JoystickIsHaptic(j) == 1, handle=j,
            instance_id=s.SDL_JoystickInstanceID(j), n_axes=max(0, s.SDL_JoystickNumAxes(j)),
            n_buttons=max(0, s.SDL_JoystickNumButtons(j)), n_hats=max(0, s.SDL_JoystickNumHats(j)),
        )

    def update(self) -> None:
        if self._shut:
            self._events = []
            return
        s = _sdl()
        s.SDL_PumpEvents()
        for iid, d in list(self._devices.items()):
            if not d.closing and not s.SDL_JoystickGetAttached(d.handle):
                d.closing = True                   # a handler that polls again must not see it
                notify(self._detach, d.handle)     # FFB lets go of its haptic device first
                if self._shut:                     # a handler closed the backend (and SDL with it)
                    self._events = []
                    return
                s.SDL_JoystickClose(d.handle)
                self._devices.pop(iid, None)
        opened = False
        for i in range(s.SDL_NumJoysticks()):
            iid = s.SDL_JoystickGetDeviceInstanceID(i)
            if iid < 0:
                continue
            if iid in self._devices:
                self._devices[iid].index = i
                continue
            d = self._open(i)
            if d is not None:
                self._devices[d.instance_id] = d
                opened = True
        if opened:
            s.SDL_JoystickUpdate()      # a new device reads 0 on every axis until its first update
        self._events = self._drain()

    def _drain(self) -> list[JoyEvent]:
        s = _sdl()
        out: list[JoyEvent] = []
        while True:
            n = s.SDL_PeepEvents(self._buf, self.BUF, s.SDL_GETEVENT, s.SDL_JOYAXISMOTION, s.SDL_JOYBATTERYUPDATED)
            for k in range(max(n, 0)):
                ev = self._buf[k]
                if ev.type in (s.SDL_JOYBUTTONDOWN, s.SDL_JOYBUTTONUP):
                    d = self._devices.get(ev.jbutton.which)
                    if d is not None:
                        out.append(JoyEvent(d, "button", int(ev.jbutton.button), int(ev.jbutton.state)))
                elif ev.type == s.SDL_JOYHATMOTION:
                    d = self._devices.get(ev.jhat.which)
                    if d is not None:
                        out.append(JoyEvent(d, "hat", int(ev.jhat.hat), int(ev.jhat.value)))
            if n < self.BUF:
                break
        s.SDL_FlushEvents(s.SDL_FIRSTEVENT, s.SDL_LASTEVENT)   # nothing else reads SDL's queue
        return out

    def devices(self) -> list[SdlDevice]:
        return sorted((d for d in self._devices.values() if not d.closing), key=lambda d: d.index)

    def events(self) -> list[JoyEvent]:
        return self._events

    @staticmethod
    def _add(cbs: list, cb: Callable[[object], None]) -> Callable[[], None]:
        cbs.append(cb)

        def remove() -> None:
            if cb in cbs:
                cbs.remove(cb)

        return remove

    def on_detach(self, cb: Callable[[object], None]) -> Callable[[], None]:
        return self._add(self._detach, cb)

    def on_close(self, cb: Callable[[object], None]) -> Callable[[], None]:
        return self._add(self._close, cb)

    def close(self) -> None:
        if self._shut:
            return
        self._shut = True
        s = _sdl()
        for d in self.devices():
            if d.closing:
                continue
            d.closing = True
            notify(self._close, d.handle)
            s.SDL_JoystickClose(d.handle)
        self._devices = {}
        s.SDL_QuitSubSystem(s.SDL_INIT_JOYSTICK | s.SDL_INIT_HAPTIC)


class RealOnly:
    """A backend view for learn mode that applies the input layer's unknown-axis
    rule: a device whose first snapshot has ALL axes at 0.0 may report only on
    change, so it stays hidden until any axis changes or it sends an event. Its
    first report then shows every real value at once and is never mistaken for
    movement."""

    def __init__(self, inner: JoystickBackend) -> None:
        self.inner = inner
        self._blank: dict[int, tuple[float, ...]] = {}
        self._seen: set[int] = set()
        self._real: list[Device] = []

    def update(self) -> None:
        self.inner.update()
        reported = {id(ev.device) for ev in self.inner.events()}
        self._real = []
        for d in self.inner.devices():
            axes = d.snapshot().axes
            if id(d) not in self._seen:
                self._seen.add(id(d))
                if axes and all(a == 0.0 for a in axes):
                    self._blank[id(d)] = axes
            elif id(d) in self._blank and (axes != self._blank[id(d)] or id(d) in reported):
                del self._blank[id(d)]
            if id(d) not in self._blank:
                self._real.append(d)

    def devices(self) -> list[Device]:
        return list(self._real)

    def events(self) -> Sequence[JoyEvent]:
        return self.inner.events()

    def on_detach(self, cb: Callable[[object], None]) -> Callable[[], None]:
        return self.inner.on_detach(cb)

    def on_close(self, cb: Callable[[object], None]) -> Callable[[], None]:
        return self.inner.on_close(cb)

    def close(self) -> None:
        self.inner.close()


def snapshots(backend: JoystickBackend) -> list[DeviceSnapshot]:
    return [d.snapshot() for d in backend.devices()]


# --- CLI ---

NO_DEVICES = "No joystick devices found. Plug in the wheel base, pedals and other USB devices, then run probe again."


def _open_backend() -> SdlBackend | None:
    try:
        return SdlBackend()
    except (SdlError, ImportError, OSError) as e:
        print(f"torquehero: cannot open SDL joysticks: {e}", file=sys.stderr)
        return None


def format_device(dev: Device, snap: DeviceSnapshot, bound: list[str]) -> str:
    lines = [f"[{snap.index}] {snap.name}",
             f"    guid {snap.guid}  axes {len(snap.axes)}  buttons {len(snap.buttons)}  hats {len(snap.hats)}"
             f"  haptic {'yes' if dev.haptic else 'no'}",
             f"    key     {snap.key or '(none: index is the tiebreak)'}"]
    if snap.axes:
        lines.append("    axes    " + " ".join(f"{i}:{v:+.2f}" for i, v in enumerate(snap.axes)))
    if snap.buttons:
        down = [str(i) for i, p in enumerate(snap.buttons) if p]
        lines.append("    buttons " + (" ".join(down) if down else "(none down)"))
    if snap.hats:
        lines.append("    hats    " + " ".join(f"{i}:{h}" for i, h in enumerate(snap.hats)))
    if bound:
        lines.append("    bound   " + " ".join(bound))
    return "\n".join(lines)


def probe_text(bindings: Bindings, devs: Sequence[Device]) -> str:
    found = bindings.resolve(devs)
    blocks = [format_device(d, d.snapshot(), [n for n in bindings.controls if found.get(n) is d]) for d in devs]
    missing = [f"{n} ({b.name})" for n, b in bindings.controls.items() if n not in found]
    if missing:
        blocks.append("bound but device not found: " + ", ".join(missing))
    return "\n\n".join(blocks)


def _live_ok() -> bool:
    """True when stdout takes cursor codes; enables VT processing on Windows consoles."""
    if not sys.stdout.isatty():
        return False
    if sys.platform != "win32":
        return True
    try:
        k = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h, mode = k.GetStdHandle(-11), ctypes.c_uint32()
        return bool(k.GetConsoleMode(h, ctypes.byref(mode)) and k.SetConsoleMode(h, mode.value | 0x0004))
    except (AttributeError, OSError):
        return False


def cmd_probe(args) -> int:
    backend = _open_backend()
    if backend is None:
        return 1
    try:
        bindings = Bindings.load()
        if not backend.devices():
            print(NO_DEVICES)
            return 0
        if not args.live:
            print(probe_text(bindings, backend.devices()))
            return 0
        vt, prev_lines = _live_ok(), 0
        while True:
            backend.update()
            text = probe_text(bindings, backend.devices())
            if vt:
                if prev_lines:
                    sys.stdout.write(f"\x1b[{prev_lines}F\x1b[J")
                prev_lines = text.count("\n") + 1
                sys.stdout.write(text + "\n")
            else:
                sys.stdout.write(text + "\n" + "-" * 40 + "\n")
            sys.stdout.flush()
            time.sleep(0.1 if vt else 0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        backend.close()


PROMPTS = {
    "steer": "turn the wheel RIGHT a little",
    "brake": "press the brake pedal", "clutch": "press the clutch pedal",
    "throttle": "press the throttle pedal", "handbrake": "pull the handbrake",
}


def _sample(backend: JoystickBackend) -> list[DeviceSnapshot]:
    backend.update()
    return snapshots(backend)


def _wait(backend: JoystickBackend, until, timeout: float) -> list[DeviceSnapshot] | None:
    end = time.monotonic() + timeout
    while True:
        snaps = _sample(backend)
        if until(snaps):
            return snaps
        if time.monotonic() >= end:
            return None
        time.sleep(0.01)


def _enter_pressed() -> bool:
    """Non-blocking: True when the player pressed Enter in the terminal."""
    try:
        if sys.platform == "win32":
            import msvcrt

            while msvcrt.kbhit():  # type: ignore[attr-defined]
                if msvcrt.getwch() in "\r\n":  # type: ignore[attr-defined]
                    return True
            return False
        import select

        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return True
    except (OSError, ValueError):
        pass
    return False


def _ask(question: str) -> str:
    try:
        return input(question).strip().lower()
    except EOFError:
        return ""


def stable_baseline(backend: JoystickBackend, timeout: float) -> tuple[list[DeviceSnapshot], set[tuple]] | None:
    """A baseline whose axes held still over STABLE_SPAN, and the axes to ignore:
    an axis that kept moving for DRIFT_SAMPLES samples in a row (an unconnected,
    floating input) is reported and ignored instead of blocking the baseline.
    None if the rest never held still."""
    end = time.monotonic() + timeout
    a, told = _sample(backend), False
    moving_for: dict[tuple, int] = {}
    while True:
        time.sleep(STABLE_SPAN)
        b = _sample(backend)
        moving = unsettled(a, b)
        moving_for = {ax: moving_for.get(ax, 0) + 1 for ax in moving}
        drifting = {ax for ax, n in moving_for.items() if n >= DRIFT_SAMPLES}
        if moving <= drifting:
            for _guid, name, _key, index, axis in sorted(drifting):
                print(f"    ignoring axis {axis} on {name} #{index}: it never settles", flush=True)
            return b, drifting
        if time.monotonic() >= end:
            return None
        if not told:
            print("    something is moving: feet off the pedals, hands off the controls ...", flush=True)
            told = True
        a = b


def learn_control(backend: JoystickBackend, control: str, timeout: float) -> Binding | None:
    """Terminal learn for one control (not levers): stable baseline, wait for
    movement, then track full travel until the player lets go or presses Enter."""
    kind = control_kind(control)
    print("    centre the wheel and let go ..." if kind == "bipolar"
          else "    feet off the pedals, hands off the controls ...", flush=True)
    while True:
        got = stable_baseline(backend, timeout)
        if got is None:
            print("    inputs never held still")
            return None
        base, drifting = got
        print(f"    now {PROMPTS.get(control, 'press the button (or move the axis) you want')}", flush=True)
        found: list[Binding] = []
        known = {(s.guid, s.name, s.key, s.index) for s in base}
        woke: list[bool] = []

        def moved(snaps, base=base, drifting=drifting, found=found, known=known, woke=woke):
            if {(s.guid, s.name, s.key, s.index) for s in snaps} - known:
                woke.append(True)               # a device just reported (or was plugged in): re-baseline
                return True
            b = learn(control, base, snaps, drifting)
            if b:
                found.append(b)
            return b is not None

        if _wait(backend, moved, timeout) is None:
            return None
        if not woke:
            break
        print("    a device just reported its real values: let go, then do it again", flush=True)
    b = found[0]
    if kind == "bipolar":
        return b
    if b.kind == "axis":
        print("    all the way to the end of travel, then let go (or press Enter)", flush=True)
        end = time.monotonic() + PEAK_TIMEOUT
        snaps = base
        while time.monotonic() < end:
            snaps = _sample(backend)
            b = extend_peak(b, snaps)
            if settled(b, base, snaps) or _enter_pressed():
                break
            time.sleep(0.01)
        else:
            if rest_at_full_travel(b, snaps):
                print(f"    WARNING rest {b.rest:+.2f} looks like full travel: it never came back there. "
                      "Was it held down at the start, or is the pedal inverted? answer n and retry", flush=True)
    else:
        print("    let go ...", flush=True)
        _wait(backend, lambda s: settled(b, base, s), 10.0)
    return b


def learn_lever(backend: JoystickBackend, control: str) -> Binding | None:
    """Levers have no spring, so both ends are taught explicitly."""
    _ask(f"    pull {control} fully closed (towards you), then press Enter ")
    closed = _sample(backend)
    _ask("    push it fully open, then press Enter ")
    return learn_range(control, closed, _sample(backend))


def calibrate_steer(backend: JoystickBackend, b: Binding) -> Binding:
    _ask(f"    hold the wheel at {STEER_CAL_DEG:g} degrees right, then press Enter ")
    r = steer_range(b, _sample(backend))
    if r is None:
        print("    could not measure the range (wheel not turned right?): Config.wheel_range_deg applies")
        return b
    print(f"    wheel range {r:.0f} degrees lock to lock")
    return replace(b, range_deg=r)


def _same_input(a: Binding, b: Binding) -> bool:
    return (a.guid, a.key, a.kind, a.num, a.dir) == (b.guid, b.key, b.kind, b.num, b.dir)


def bind_axis_by_hand(args, bindings: Bindings, path: Path) -> int:
    """`bind CONTROL --axis N --rest V --full V [--device GUID]`: a unipolar or
    button control on a given axis, for a stiff load cell or a noisy rig."""
    if len(args.controls) != 1 or control_kind(args.controls[0]) == "bipolar":
        print("torquehero: --axis takes exactly one control, not steer", file=sys.stderr)
        return 2
    if args.rest is None or args.full is None or not all(-1 <= v <= 1 for v in (args.rest, args.full)) \
            or abs(args.full - args.rest) < MIN_SPAN:
        print(f"torquehero: --axis needs --rest and --full in -1..1, at least {MIN_SPAN} apart", file=sys.stderr)
        return 2
    backend = _open_backend()
    if backend is None:
        return 1
    try:
        devs = [d for d in backend.devices() if args.device in (None, d.guid, d.key, str(d.index))]
        if len(devs) != 1:
            if not devs:
                msg = "no such device"
            elif args.device is None:
                msg = "name the device with --device GUID, key or index"
            else:
                msg = f"{len(devs)} devices match --device {args.device}: pass the key or index instead"
            print(f"torquehero: {msg} (see `torquehero probe`)", file=sys.stderr)
            return 2
        snap = devs[0].snapshot()
        if not 0 <= args.axis < len(snap.axes):
            print(f"torquehero: {snap.name} has axes 0..{len(snap.axes) - 1}", file=sys.stderr)
            return 2
        control = args.controls[0]
        b = Binding(**_dev(snap), kind="axis", num=args.axis, rest=args.rest, full=args.full,
                    sign=1.0 if args.full > args.rest else -1.0)
        bindings.set(control, b)
        print(f"{control} -> {b.label()}; saved {bindings.save(path)}")
        return 0
    finally:
        backend.close()


def cmd_bind(args) -> int:
    path = Path(args.file) if args.file else default_path()
    bindings = Bindings.load(path)
    unknown = [c for c in args.controls if c not in CONTROLS]
    if unknown:
        print(f"torquehero: unknown control(s): {' '.join(unknown)}\nknown: {' '.join(CONTROLS)}", file=sys.stderr)
        return 2
    if args.clear:
        bindings.clear(args.controls or None)
        print(f"cleared {' '.join(args.controls) or 'all bindings'}; saved {bindings.save(path)}")
        return 0
    if args.list:
        for n in CONTROLS:
            b = bindings.get(n)
            print(f"{n:<11} {b.label() if b else '-'}")
        print(f"({path})")
        return 0
    controls = list(args.controls)
    if args.axis is not None:
        return bind_axis_by_hand(args, bindings, path)
    if args.range is not None:
        if not 180 <= args.range <= 3600 or controls not in ([], ["steer"]):
            print("torquehero: --range takes 180..3600 degrees and applies to steer only", file=sys.stderr)
            return 2
        b = bindings.get("steer")
        if b is not None:
            bindings.set("steer", replace(b, range_deg=args.range))
            print(f"steer -> {bindings.controls['steer'].label()}; saved {bindings.save(path)}")
            return 0
        controls = ["steer"]
    backend = _open_backend()
    if backend is None:
        return 1
    try:
        if not backend.devices():
            print(NO_DEVICES)
            return 0
        before = dict(bindings.controls)
        bindings.resolve(backend.devices())
        rig = RealOnly(backend)
        changed = bindings.controls != before          # a device path moved: save the new key
        for control in controls or CONTROLS:
            print(f"{control}: ({args.timeout:g} s per step, do nothing to skip)", flush=True)
            b = learn_lever(rig, control) if control in LEVERS else learn_control(rig, control, args.timeout)
            if b is None:
                print("    skipped")
                continue
            if control == "steer":
                b = replace(b, range_deg=args.range) if args.range is not None else calibrate_steer(rig, b)
            warnings = []
            if control not in LEVERS and control_kind(control) == "unipolar" and rest_suspicious(b):
                warnings.append(f"rest {b.rest:+.2f} is mid-travel: was a foot or hand on it? answer n and retry")
            clash = [n for n, o in bindings.controls.items() if n != control and _same_input(o, b)]
            if clash:
                warnings.append(f"also bound to {', '.join(clash)}")
            for w in warnings:
                print(f"    WARNING {w}")
            if _ask(f"    {control} -> {b.label()}  keep? [Y/n] ") in ("", "y", "yes"):
                bindings.set(control, b)
                changed = True
        if changed:
            print(f"saved {bindings.save(path)}")
        return 0
    except KeyboardInterrupt:
        print("\naborted, nothing saved")
        return 130
    finally:
        backend.close()


def add_cli(sub) -> None:
    p = sub.add_parser("probe", help="list joystick devices with their GUID, inputs and haptic support")
    p.add_argument("--live", action="store_true", help="keep refreshing live values until Ctrl+C")
    p.set_defaults(func=cmd_probe)

    b = sub.add_parser("bind", help="learn device bindings in the terminal and save them")
    b.add_argument("controls", nargs="*", metavar="CONTROL", help="controls to learn (default: all)")
    g = b.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="show current bindings")
    g.add_argument("--clear", action="store_true", help="remove the named bindings (default: all)")
    b.add_argument("--range", type=float, metavar="DEG",
                   help="steer: set the wheel's lock-to-lock degrees directly instead of calibrating")
    b.add_argument("--axis", type=int, metavar="N", help="bind one pedal/lever/button control to axis N by hand")
    b.add_argument("--rest", type=float, metavar="V", help="with --axis: raw value at rest (-1..1, see probe)")
    b.add_argument("--full", type=float, metavar="V", help="with --axis: raw value at full travel (-1..1)")
    b.add_argument("--device", metavar="ID", help="with --axis: the device's GUID, key or index as probe shows it")
    b.add_argument("--timeout", type=float, default=8.0, help="seconds to wait at each step")
    b.add_argument("--file", help=f"bindings file (default: {BINDINGS_FILE} in the config dir)")
    b.set_defaults(func=cmd_bind)
