"""Input sources that fill `InputState` each frame (SPEC 8.3).

`SdlInput` reads every bound device of the rig; `KeyboardMouseInput` reads a
`KeySource` (the renderer's raylib implementation, or a fake); `CompositeInput`
merges them so an unbound control falls back to the keyboard. Each source first
produces a `Reading` (levels, plus button taps from the SDL event queue);
`StateBuilder` turns readings into edges, holds and velocities, so all sources
share one threshold, one hysteresis and one velocity map.
"""
from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .bindings import Binding, Bindings, DeviceSnapshot, JoyEvent, JoystickBackend, notify
from .config import Config
from .state import BUTTONS, SYSTEM_CONTROLS, UNIPOLAR, InputState, KeySource

# Hysteresis: a control goes down at Config.press_threshold and comes up only
# below press_threshold - RELEASE_MARGIN, so a load cell held near the threshold
# does not chatter.
RELEASE_MARGIN = 0.1

# Velocity: rise rate (travel per second) measured from the oldest sample in the
# last VEL_WINDOW seconds that had already left rest (value > REST_LEVEL), so idle
# frames before the press do not dilute it. Mapped on a log scale: RATE_GENTLE
# gives VEL_GENTLE, RATE_STOMP gives 1.0. Keys and buttons always give 1.0.
VEL_WINDOW = 0.03
REST_LEVEL = 0.02
RATE_GENTLE = 2.0       # full travel in 0.5 s
RATE_STOMP = 15.0       # full travel in about 65 ms
VEL_GENTLE = 0.3
VEL_MIN = 0.1

# A poll whose dt is above this drops the button/hat events queued meanwhile
# (song load, stall): they are stale. `flush()` does the same on demand.
STALL_FLUSH = 0.25

KB_STEER_RATE = 3.5     # lane units per second while a steer key is held
MOUSE_SPAN = 0.35       # mouse travel from window centre that reaches lane +-1

PRESSABLE = (*UNIPOLAR, *BUTTONS)

# raylib key names (KeySource) per control; any listed key serves the control.
DEFAULT_KEYMAP: dict[str, tuple[str, ...]] = {
    "steer_left": ("A", "LEFT"),
    "steer_right": ("D", "RIGHT"),
    "brake": ("S",),
    "clutch": ("SPACE",),
    "throttle": ("W",),
    "handbrake": ("LEFT_SHIFT", "RIGHT_SHIFT"),
    "gate1": ("ONE",), "gate2": ("TWO",), "gate3": ("THREE",),
    "gate4": ("FOUR",), "gate5": ("FIVE",), "gate6": ("SIX",),
    "paddle_l": ("Q",),
    "paddle_r": ("E",),
    "menu_up": ("UP",),
    "menu_down": ("DOWN",),
    "menu_ok": ("ENTER",),
    "menu_back": ("BACKSPACE",),
    "pause": ("ESCAPE", "P"),
    "trim_minus": ("MINUS",),
    "trim_plus": ("EQUAL",),
    "vol_down": ("NINE",),
    "vol_up": ("ZERO",),
    "ffb_down": ("LEFT_BRACKET",),
    "ffb_up": ("RIGHT_BRACKET",),
}


def velocity_from_rate(rate: float) -> float:
    """Rise rate (travel per second) -> hit strength 0..1."""
    if rate <= 0:
        return VEL_MIN
    v = VEL_GENTLE + (1 - VEL_GENTLE) * math.log(rate / RATE_GENTLE) / math.log(RATE_STOMP / RATE_GENTLE)
    return max(VEL_MIN, min(1.0, v))


@dataclass
class Reading:
    """Levels from one source for one frame. `values` holds unipolar and button
    controls 0..1; `steer_deg` is None when the source does not serve steer.
    `digital` names controls read from a key or button (velocity 1.0). `system`
    holds system controls held down now. `taps`/`untaps` name controls whose
    button went down/up during the frame (from events, even when the level shows
    nothing). `fresh` names controls on a device that appeared this frame: no
    press edge for them yet. `bound`/`fallback` name the performance controls this
    source serves from a real device / from keyboard and mouse."""

    values: dict[str, float] = field(default_factory=dict)
    steer_deg: float | None = None
    digital: set[str] = field(default_factory=set)
    system: set[str] = field(default_factory=set)
    taps: set[str] = field(default_factory=set)
    untaps: set[str] = field(default_factory=set)
    fresh: set[str] = field(default_factory=set)
    dt: float | None = None             # measured seconds since the previous read, when the source knows it
    bound: set[str] = field(default_factory=set)
    fallback: set[str] = field(default_factory=set)


def merge(primary: Reading, fallback: Reading) -> Reading:
    """Each performance control from `primary` (devices) when it serves it, else
    from `fallback` (keyboard/mouse). System controls held on either count."""
    taken = primary.bound
    return Reading(
        values={**{c: v for c, v in fallback.values.items() if c not in taken}, **primary.values},
        steer_deg=primary.steer_deg if "steer" in taken else fallback.steer_deg,
        digital=primary.digital | (fallback.digital - taken),
        system=primary.system | fallback.system,
        taps=primary.taps | (fallback.taps - taken),
        untaps=primary.untaps | (fallback.untaps - taken),
        fresh=primary.fresh | (fallback.fresh - taken),
        dt=primary.dt,
        bound=set(taken),
        fallback=fallback.fallback - taken,
    )


class StateBuilder:
    """Turns successive readings into `InputState`: latched holds with
    hysteresis, press/release edges (including sub-frame taps), system presses,
    rise-rate velocities, lane-unit steer. A control whose source changed this
    frame (device plugged in or out, fallback to keys) gives no press edge."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.t = 0.0
        self._down: frozenset[str] = frozenset()
        self._system: set[str] = set()
        self._source: dict[str, str] = {}
        self._hist: dict[str, deque[tuple[float, float]]] = {}

    def _velocity(self, c: str, v: float) -> float:
        h = self._hist.get(c)
        if not h:
            return 1.0
        moving = [s for s in h if s[0] >= self.t - VEL_WINDOW and s[1] > REST_LEVEL]
        t0, v0 = moving[0] if moving else h[-1]
        return velocity_from_rate((v - v0) / max(self.t - t0, 1e-3))

    def _remember(self, c: str, v: float) -> None:
        h = self._hist.setdefault(c, deque())
        h.append((self.t, v))
        while len(h) > 1 and h[1][0] <= self.t - VEL_WINDOW:
            h.popleft()

    def build(self, r: Reading, dt: float) -> InputState:
        self.t += max(r.dt if r.dt is not None else dt, 0.0)
        thr = self.cfg.press_threshold
        source = {c: "device" for c in r.bound} | {c: "keys" for c in r.fallback - r.bound}
        fresh = r.fresh | {c for c in PRESSABLE if source.get(c) != self._source.get(c)}
        prev = self._down
        down = frozenset(c for c in PRESSABLE
                         if r.values.get(c, 0.0) >= (thr - RELEASE_MARGIN if c in prev else thr))
        tapped = r.taps.intersection(PRESSABLE)
        pressed = ((down - prev) | tapped) - fresh
        # Within a frame a press comes before a release. A control that ends the
        # frame down (contact bounce: down, up, down) reports no release.
        released = ((prev - down) | (r.untaps & (prev | pressed))) - down
        velocity = {c: 1.0 if c in r.digital or c in tapped else self._velocity(c, r.values.get(c, 0.0))
                    for c in pressed}
        for c, v in r.values.items():
            self._remember(c, v)
        system = ((r.system - self._system) | r.taps.intersection(SYSTEM_CONTROLS)) - r.fresh
        self._down, self._system, self._source = down, set(r.system), source
        deg = r.steer_deg or 0.0
        play = self.cfg.play_range_deg
        return InputState(
            steer=max(-1.0, min(1.0, deg / play)) if play else 0.0,
            steer_deg=deg,
            **{c: max(0.0, min(1.0, r.values.get(c, 0.0))) for c in UNIPOLAR},
            down=down, pressed=frozenset(pressed), released=frozenset(released), system=frozenset(system),
            bound=frozenset(r.bound), fallback=frozenset(r.fallback - r.bound), velocity=velocity,
        )


@runtime_checkable
class InputSource(Protocol):
    """One call to `poll` per frame. Sources measure time between polls
    themselves (time.monotonic); `dt` is kept for compatibility and used only
    where nothing is measured (keyboard steer ramp, first poll)."""

    def poll(self, dt: float) -> InputState: ...
    def flush(self) -> None: ...                     # drop queued button edges (after a long frame)
    def close(self) -> None: ...


class SdlInput:
    """Every bound control read from the devices present now. A control whose
    device is unplugged is not in `bound` and falls back (via CompositeInput).
    Axes are level-polled; button and hat edges also come from SDL events so a
    press shorter than a frame is not lost. An axis is "unknown" until its value
    reads 0 on every axis at first: a device whose first snapshot has ALL axes at
    exactly 0.0 may report only on change, so its axes read as rest until any axis
    changes or an event arrives from it; then every axis of it is real. `backend` defaults to SDL
    (joystick + haptic only); tests pass a fake. Time between polls is measured
    with `clock` (time.monotonic), not taken from the caller's dt."""

    def __init__(self, bindings: Bindings, cfg: Config, backend: JoystickBackend | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if backend is None:
            from .bindings import SdlBackend

            backend = SdlBackend()
        self.bindings, self.cfg, self.backend = bindings, cfg, backend
        self._builder = StateBuilder(cfg)
        self._seen: Sequence[object] = ()
        self._levels: dict[str, bool] = {}
        self._blank: dict[object, tuple[float, ...]] = {}   # device -> its all-zero first axes, until real
        self._woke: list[object] = []
        self._drop_events = False
        self.clock = clock
        self._last: float | None = None
        self._steer_handle: object | None = None
        self._lost_cbs: list[Callable[[object], None]] = []
        self._unhook: list[Callable[[], None]] | None = None

    def _snapshot(self, dev, snaps: dict[int, DeviceSnapshot], reported: list) -> DeviceSnapshot:
        snap = snaps.get(id(dev))
        if snap is None:
            snap = snaps[id(dev)] = dev.snapshot()
            if not any(dev is d for d in self._seen):
                if snap.axes and all(a == 0.0 for a in snap.axes):
                    self._blank[dev] = snap.axes
                else:
                    self._blank.pop(dev, None)
            elif dev in self._blank and (snap.axes != self._blank[dev] or any(d is dev for d in reported)):
                del self._blank[dev]
                self._woke.append(dev)          # its real axis values arrive now: no axis edge this frame
        return snap

    def _taps(self, name: str, b: Binding, dev: object, events: list[JoyEvent], r: Reading) -> None:
        on = self._levels.get(name, False)
        for ev in events:
            if ev.device is not dev or ev.kind != b.kind or ev.num != b.num:
                continue
            now = bool(ev.value & b.dir) if b.kind == "hat" else bool(ev.value)
            if now and not on:
                r.taps.add(name)
            elif on and not now:
                r.untaps.add(name)
            on = now

    def read(self, dt: float = 0.0) -> Reading:
        now = self.clock()
        gap = dt if self._last is None else now - self._last
        self._last = now
        self.backend.update()
        devs = list(self.backend.devices())
        events = list(self.backend.events())
        reported = [ev.device for ev in events]
        if self._drop_events or gap > STALL_FLUSH:
            events = []
        self._drop_events = False
        found = self.bindings.resolve(devs)
        self._steer_handle = found["steer"].handle if "steer" in found else None
        snaps: dict[int, DeviceSnapshot] = {}
        self._woke: list[object] = []
        r = Reading(dt=gap)
        for name, b in list(self.bindings.controls.items()):
            dev = found.get(name)
            if dev is None:
                self._levels.pop(name, None)
                continue
            snap = self._snapshot(dev, snaps, reported)
            v = b.value(snap)
            if v is None:
                continue
            if b.kind == "axis" and dev in self._blank:
                v = 0.0
            if not any(dev is d for d in self._seen) or (b.kind == "axis" and any(dev is d for d in self._woke)):
                r.fresh.add(name)
            if name == "steer":
                r.steer_deg = v * (b.range_deg or self.cfg.wheel_range_deg) / 2
                r.bound.add(name)
                continue
            if b.digital:
                self._taps(name, b, dev, events, r)
                self._levels[name] = v >= 0.5
            if name in SYSTEM_CONTROLS:
                if v >= self.cfg.press_threshold:
                    r.system.add(name)
                continue
            r.values[name] = v
            r.bound.add(name)
            if b.digital:
                r.digital.add(name)
        self._seen = devs
        self._blank = {d: v for d, v in self._blank.items() if any(d is x for x in devs)}
        return r

    def flush(self) -> None:
        """Drop the button/hat edges queued since the last poll (call after a long
        frame such as a song load). A poll more than STALL_FLUSH seconds after the
        previous one (measured) does it itself. Levels are untouched: a button
        still held gives `pressed` on the next poll if it was up before."""
        self._drop_events = True

    def poll(self, dt: float) -> InputState:
        return self._builder.build(self.read(dt), dt)

    @property
    def steer_joystick(self) -> object | None:
        """The open SDL joystick of the device bound to `steer`, for the FFB engine
        to open its haptic device (SPEC 8.6). It stays the same object while the
        wheel stays plugged in. None without a steer binding or device."""
        dev = self.bindings.resolve(self.backend.devices()).get("steer")
        self._steer_handle = dev.handle if dev else None
        return self._steer_handle

    def on_steer_lost(self, cb: Callable[[object], None]) -> Callable[[], None]:
        """`cb(handle)` runs when the steer device is detached or input shuts down,
        BEFORE its joystick is closed: input owns the joystick, FFB borrows it and
        must close its haptic device here. The device is already hidden from
        `devices()`, so cb may poll without re-entering. Returns a function that
        removes cb (register again after a replug without being called twice)."""
        if not self._lost_cbs and self._unhook is None:
            self._unhook = [self.backend.on_detach(self._steer_lost), self.backend.on_close(self._steer_lost)]
        self._lost_cbs.append(cb)

        def remove() -> None:
            if cb in self._lost_cbs:
                self._lost_cbs.remove(cb)

        return remove

    def _steer_lost(self, handle: object) -> None:
        """One backend registration for all steer-lost handlers: the handle is
        checked once, then every handler runs in registration order, each once,
        even if one of them polls (which clears the steer handle) or fails."""
        if handle is None or handle is not self._steer_handle:
            return
        self._steer_handle = None
        notify(list(self._lost_cbs), handle)

    def close(self) -> None:
        self.backend.close()


class KeyboardMouseInput:
    """Keyboard and mouse through a `KeySource`; never imports raylib. Steer: the
    steer keys ramp at KB_STEER_RATE, otherwise the mouse x sets it. Levers have
    no keys, so the faders layer stays auto on keyboard."""

    def __init__(self, keys: KeySource, cfg: Config, keymap: Mapping[str, tuple[str, ...]] | None = None) -> None:
        self.keys, self.cfg = keys, cfg
        self.keymap = dict(DEFAULT_KEYMAP if keymap is None else keymap)
        self._kb_steer = 0.0
        self._builder = StateBuilder(cfg)

    def _held(self, name: str) -> bool:
        return any(self.keys.is_down(k) for k in self.keymap.get(name, ()))

    def read(self, dt: float = 0.0) -> Reading:
        r = Reading()
        kdir = self._held("steer_right") - self._held("steer_left")
        if kdir:
            self._kb_steer = max(-1.0, min(1.0, self._kb_steer + kdir * KB_STEER_RATE * dt))
            lane = self._kb_steer
        else:
            self._kb_steer = 0.0
            lane = max(-1.0, min(1.0, (self.keys.mouse_x_norm() - 0.5) / MOUSE_SPAN))
        r.steer_deg = lane * self.cfg.play_range_deg
        r.fallback.add("steer")
        for c in PRESSABLE:
            if c in self.keymap:
                r.values[c] = 1.0 if self._held(c) else 0.0
                r.digital.add(c)
                r.fallback.add(c)
        r.system = {c for c in SYSTEM_CONTROLS if self._held(c)}
        return r

    def poll(self, dt: float) -> InputState:
        return self._builder.build(self.read(dt), dt)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class CompositeInput:
    """Devices first; each control no device serves comes from keyboard/mouse."""

    def __init__(self, primary: SdlInput, fallback: KeyboardMouseInput, cfg: Config) -> None:
        self.primary, self.fallback = primary, fallback
        self._builder = StateBuilder(cfg)

    def poll(self, dt: float) -> InputState:
        return self._builder.build(merge(self.primary.read(dt), self.fallback.read(dt)), dt)

    @property
    def steer_joystick(self) -> object | None:
        return self.primary.steer_joystick

    def on_steer_lost(self, cb: Callable[[object], None]) -> Callable[[], None]:
        return self.primary.on_steer_lost(cb)

    def flush(self) -> None:
        self.primary.flush()
        self.fallback.flush()

    def close(self) -> None:
        self.primary.close()
        self.fallback.close()


def open_input(cfg: Config, keys: KeySource | None, *, devices: bool = True,
               bindings: Bindings | None = None) -> InputSource:
    """The app's input: devices plus keyboard fallback. Falls back to keyboard
    only when `devices` is False or SDL cannot start."""
    kb = KeyboardMouseInput(keys, cfg) if keys is not None else None
    sdl = None
    if devices:
        from .bindings import SdlError

        try:
            sdl = SdlInput(bindings if bindings is not None else Bindings.load(), cfg)
        except (SdlError, ImportError, OSError):
            sdl = None
    if sdl and kb:
        return CompositeInput(sdl, kb, cfg)
    if sdl:
        return sdl
    if kb:
        return kb
    raise RuntimeError("no input source: SDL unavailable and no KeySource given")
