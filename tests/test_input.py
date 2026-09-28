import subprocess
import sys

import pytest

from torquehero.bindings import Binding, Bindings, DeviceSnapshot, JoyEvent
from torquehero.config import Config
from torquehero.input import (
    RATE_GENTLE,
    RATE_STOMP,
    RELEASE_MARGIN,
    VEL_GENTLE,
    CompositeInput,
    InputSource,
    KeyboardMouseInput,
    SdlInput,
    open_input,
    velocity_from_rate,
)
from torquehero.state import KeySource, default_layer_modes

WHEEL, PEDALS, BOX = "wheelguid", "pedalguid", "boxguid"


class FakeJoystick:
    """A device whose inputs the test sets directly."""

    def __init__(self, guid, name, index, axes=2, buttons=4, hats=1, haptic=False, key=""):
        self.guid, self.name, self.index, self.haptic, self.key = guid, name, index, haptic, key
        self.handle = f"joy-{name}-{index}"
        self.axes, self.buttons, self.hats = [0.0] * axes, [False] * buttons, [0] * hats

    def snapshot(self):
        return DeviceSnapshot(self.guid, self.name, self.index, tuple(self.axes), tuple(self.buttons),
                              tuple(self.hats), self.key)


class FakeBackend:
    def __init__(self, *devices):
        self.devs = list(devices)
        self.pending: list[JoyEvent] = []
        self.current: list[JoyEvent] = []
        self.updates = 0
        self.closed = False
        self.detach_cbs, self.close_cbs = [], []

    def update(self):
        self.updates += 1
        self.current, self.pending = self.pending, []

    def devices(self):
        return list(self.devs)

    def events(self):
        return self.current

    def on_detach(self, cb):
        self.detach_cbs.append(cb)
        return lambda: self.detach_cbs.remove(cb)

    def on_close(self, cb):
        self.close_cbs.append(cb)
        return lambda: self.close_cbs.remove(cb)

    def unplug(self, dev):
        for cb in self.detach_cbs:
            cb(dev.handle)
        self.devs.remove(dev)

    def close(self):
        for d in self.devs:
            for cb in self.close_cbs:
                cb(d.handle)
        self.closed = True


class Clock:
    """Injectable monotonic clock; tests move it by hand."""

    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def tick(src, clock, dt):
    clock.t += dt
    return src.poll(dt)


class FakeKeys:
    def __init__(self):
        self.down: set[str] = set()
        self.mouse = 0.5

    def is_down(self, key):
        return key in self.down

    def pressed(self, key):
        return False

    def mouse_x_norm(self):
        return self.mouse


def rig_bindings() -> Bindings:
    return Bindings({
        "steer": Binding(WHEEL, "wheel", 0, "axis", 0),
        "gate1": Binding(WHEEL, "wheel", 0, "button", 0),
        "pause": Binding(WHEEL, "wheel", 0, "hat", 0, dir=1),
        "vol_up": Binding(BOX, "box", 2, "button", 1),
        "brake": Binding(PEDALS, "pedals", 1, "axis", 0, rest=-1.0, sign=1.0, full=1.0),
        "clutch": Binding(PEDALS, "pedals", 1, "axis", 1, rest=1.0, sign=-1.0),
        "paddle_l": Binding(BOX, "box", 2, "axis", 0, rest=-1.0, sign=1.0),
    })


@pytest.fixture
def rig():
    wheel = FakeJoystick(WHEEL, "wheel", 0, haptic=True)
    pedals = FakeJoystick(PEDALS, "pedals", 1)
    pedals.axes = [-1.0, 1.0]
    box = FakeJoystick(BOX, "box", 2)
    box.axes = [-1.0, 0.0]
    backend = FakeBackend(wheel, pedals, box)
    return wheel, pedals, box, backend


def sdl(backend, cfg=None, bindings=None):
    """An SdlInput that has already seen the rig once (past the fresh-device frame)."""
    src = SdlInput(bindings or rig_bindings(), cfg or Config(), backend)
    src.poll(0.01)
    return src


def test_protocols():
    assert isinstance(KeyboardMouseInput(FakeKeys(), Config()), InputSource)
    assert isinstance(FakeKeys(), KeySource)


def test_sdl_reads_every_bound_device(rig):
    wheel, pedals, box, backend = rig
    cfg = Config(wheel_range_deg=900, play_range_deg=90)
    src = SdlInput(rig_bindings(), cfg, backend)
    s = src.poll(1 / 60)
    assert s.bound == {"steer", "gate1", "brake", "clutch", "paddle_l"}
    assert s.fallback == frozenset() and s.down == frozenset() and s.brake == 0.0
    wheel.axes[0] = 0.1                         # 45 degrees right
    pedals.axes[0] = 0.0                        # brake half way
    pedals.axes[1] = -1.0                       # clutch fully in (inverted axis)
    wheel.buttons[0] = True
    s = src.poll(1 / 60)
    assert s.steer_deg == pytest.approx(45.0) and s.steer == pytest.approx(0.5)
    assert s.brake == pytest.approx(0.5) and s.clutch == 1.0
    assert s.down == {"brake", "clutch", "gate1"} and s.pressed == s.down
    assert s.velocity["gate1"] == 1.0
    assert backend.updates == 2


def test_steer_uses_binding_range(rig):
    wheel, *_, backend = rig
    b = rig_bindings()
    b.set("steer", Binding(WHEEL, "wheel", 0, "axis", 0, sign=-1.0, range_deg=2520.0))
    src = sdl(backend, Config(wheel_range_deg=900, play_range_deg=90), b)
    wheel.axes[0] = -0.05
    s = src.poll(0.01)
    assert s.steer_deg == pytest.approx(63.0) and s.steer == pytest.approx(0.7)


@pytest.mark.parametrize("raw", [-1.0, -0.3, -0.01, 0.0, 0.004, 0.05, 0.0833, 0.2, 1.0])
@pytest.mark.parametrize("play", [45.0, 90.0, 135.0])
def test_bound_steer_is_exactly_deg_over_play(rig, raw, play):
    wheel, *_, backend = rig
    src = sdl(backend, Config(wheel_range_deg=1080, play_range_deg=play))
    wheel.axes[0] = raw
    s = src.poll(0.01)
    assert s.steer_deg == raw * 540
    assert s.steer == max(-1.0, min(1.0, s.steer_deg / play))   # no centre, no deadzone


def test_steer_clamps_lane_but_not_degrees(rig):
    wheel, *_, backend = rig
    src = sdl(backend, Config(wheel_range_deg=900, play_range_deg=90))
    wheel.axes[0] = -0.5
    s = src.poll(0.01)
    assert s.steer == -1.0 and s.steer_deg == pytest.approx(-225.0)
    assert "steer" not in s.down


def test_no_edge_on_first_sample_of_a_device(rig):
    wheel, *_, backend = rig
    wheel.buttons[0] = True                     # gate already engaged when the device appears
    src = SdlInput(rig_bindings(), Config(), backend)
    s = src.poll(0.01)
    assert "gate1" in s.down and s.pressed == frozenset() and s.velocity == {}
    wheel.buttons[0] = False
    assert src.poll(0.01).released == {"gate1"}


def test_all_zero_device_reads_rest_until_it_reports(rig):
    _, pedals, _, backend = rig
    pedals.axes = [0.0, 0.0]                    # reports only on change: these zeros are not real
    src = SdlInput(rig_bindings(), Config(), backend)
    for _ in range(5):
        s = src.poll(0.01)
        assert s.brake == 0.0 and s.clutch == 0.0 and not s.down and not s.pressed
    pedals.axes = [-1.0, 0.0]                   # one report: axis 0 at rest, axis 1 now real too
    s = src.poll(0.01)
    assert s.brake == 0.0 and s.clutch == pytest.approx(0.5) and not s.pressed
    pedals.axes[0] = 1.0
    assert src.poll(0.01).pressed == {"brake"}


def test_all_zero_device_becomes_real_on_an_event(rig):
    wheel, *_, backend = rig
    b = rig_bindings()
    b.set("steer", Binding(WHEEL, "wheel", 0, "axis", 1))
    src = SdlInput(b, Config(wheel_range_deg=900), backend)
    src.poll(0.01)
    backend.pending = [JoyEvent(wheel, "button", 3, 1)]
    assert src.poll(0.01).steer_deg == 0.0     # axis 1 really is 0: real now, still centre
    wheel.axes[1] = 0.1
    assert src.poll(0.01).steer_deg == pytest.approx(45.0)


def test_parked_lever_is_real_at_once(rig):
    *_, box, backend = rig
    box.axes = [0.3, 0.0]                       # a spring-less lever parked mid-travel
    b = rig_bindings()
    b.set("lever0", Binding(BOX, "box", 2, "axis", 0, rest=-1.0, full=1.0))
    s = SdlInput(b, Config(), backend).poll(0.01)
    assert s.lever0 == pytest.approx(0.65) and "lever0" in s.down and not s.pressed


def test_replugged_device_gives_no_phantom_press(rig):
    wheel, pedals, box, backend = rig
    src = sdl(backend)
    backend.devs = [wheel, box]
    src.poll(0.01)
    pedals.axes[0] = 1.0
    backend.devs = [wheel, pedals, box]
    s = src.poll(0.01)
    assert s.brake == 1.0 and "brake" in s.down and "brake" not in s.pressed


def test_edges_and_release(rig):
    _, pedals, _, backend = rig
    src = sdl(backend)
    pedals.axes[0] = 0.0
    s = src.poll(0.01)
    assert s.pressed == {"brake"} and s.released == frozenset()
    s = src.poll(0.01)
    assert s.pressed == frozenset() and s.down == {"brake"}
    pedals.axes[0] = -1.0
    s = src.poll(0.01)
    assert s.released == {"brake"} and s.down == frozenset()


def test_hysteresis(rig):
    _, pedals, _, backend = rig
    src = sdl(backend)
    thr = Config().press_threshold

    def at(v):
        pedals.axes[0] = -1.0 + 2.0 * v
        return src.poll(0.01)

    assert at(thr).pressed == {"brake"}
    for v in (thr - 0.02, thr + 0.02, thr - RELEASE_MARGIN + 0.02, thr + 0.01):
        s = at(v)
        assert "brake" in s.down and not s.pressed and not s.released
    s = at(thr - RELEASE_MARGIN - 0.02)
    assert s.released == {"brake"} and "brake" not in s.down
    assert not at(thr - 0.02).pressed
    assert at(thr).pressed == {"brake"}


def test_axis_bound_button_uses_threshold(rig):
    *_, box, backend = rig
    src = sdl(backend)
    box.axes[0] = -0.1                          # 0.45 of travel
    assert "paddle_l" not in src.poll(0.01).down
    box.axes[0] = 0.1
    assert src.poll(0.01).pressed == {"paddle_l"}


def test_system_controls_are_edges(rig):
    wheel, *_, backend = rig
    src = sdl(backend)
    wheel.hats[0] = 1
    assert src.poll(0.01).system == {"pause"}
    assert src.poll(0.01).system == frozenset()
    wheel.hats[0] = 0
    src.poll(0.01)
    wheel.hats[0] = 1 | 2
    assert src.poll(0.01).system == {"pause"}


def test_tap_inside_one_frame_is_not_lost(rig):
    wheel, _, box, backend = rig
    src = sdl(backend)
    backend.pending = [JoyEvent(wheel, "button", 0, 1), JoyEvent(wheel, "button", 0, 0),
                       JoyEvent(box, "button", 1, 1), JoyEvent(box, "button", 1, 0),
                       JoyEvent(wheel, "button", 2, 1)]      # an unbound button
    s = src.poll(0.01)
    assert "gate1" in s.pressed and "gate1" in s.released and "gate1" not in s.down
    assert s.velocity["gate1"] == 1.0
    assert s.system == {"vol_up"}                            # encoder pulse on the button box
    s = src.poll(0.01)
    assert not s.pressed and not s.released and not s.system


def test_hat_tap_and_release_then_press_in_one_frame(rig):
    wheel, *_, backend = rig
    src = sdl(backend)
    wheel.buttons[0] = True
    src.poll(0.01)
    backend.pending = [JoyEvent(wheel, "button", 0, 0), JoyEvent(wheel, "button", 0, 1),
                       JoyEvent(wheel, "hat", 0, 1 | 2), JoyEvent(wheel, "hat", 0, 2)]
    s = src.poll(0.01)
    assert "gate1" in s.pressed and "gate1" not in s.released and "gate1" in s.down
    assert s.system == {"pause"}


def test_contact_bounce_ends_down_without_release(rig):
    wheel, *_, backend = rig
    src = sdl(backend)
    wheel.buttons[0] = True
    backend.pending = [JoyEvent(wheel, "button", 0, v) for v in (1, 0, 1)]
    s = src.poll(0.01)
    assert s.pressed == {"gate1"} and s.released == frozenset() and s.down == {"gate1"}


def test_flush_drops_queued_edges(rig):
    wheel, *_, backend = rig
    clock = Clock()
    src = SdlInput(rig_bindings(), Config(), backend, clock=clock)
    tick(src, clock, 0.01)
    tap = [JoyEvent(wheel, "button", 0, 1), JoyEvent(wheel, "button", 0, 0)]
    backend.pending = list(tap)
    src.flush()
    assert not tick(src, clock, 0.01).pressed
    backend.pending = list(tap)
    clock.t += 0.5                              # a stall the caller hides: dt says 0 (paused song clock)
    assert not src.poll(0.0).pressed
    backend.pending = list(tap)
    assert tick(src, clock, 0.01).pressed == {"gate1"}


def test_flush_keeps_levels(rig):
    wheel, *_, backend = rig
    src = sdl(backend)
    wheel.buttons[0] = True
    backend.pending = [JoyEvent(wheel, "button", 0, 1)]
    src.flush()
    assert src.poll(0.01).pressed == {"gate1"}  # still held after the load: the level gives the press


def test_steer_lost_fires_only_for_the_wheel(rig):
    wheel, pedals, box, backend = rig
    keys = FakeKeys()
    src = CompositeInput(SdlInput(rig_bindings(), Config(), backend), KeyboardMouseInput(keys, Config()), Config())
    lost = []
    remove = src.on_steer_lost(lost.append)
    src.poll(0.01)
    backend.unplug(box)
    assert lost == []
    backend.unplug(wheel)
    assert lost == [wheel.handle]
    backend.devs.append(wheel)                  # replug: FFB registers again, the old handler goes
    src.poll(0.01)
    remove()
    src.on_steer_lost(lost.append)
    src.close()
    assert lost == [wheel.handle, wheel.handle]


def test_twin_wheel_does_not_take_over_after_unplug(rig):
    wheel, pedals, box, backend = rig
    wheel.key = "path:usb-1"
    twin = FakeJoystick(WHEEL, "wheel", 3, key="path:usb-2")
    backend.devs.append(twin)
    b = rig_bindings()
    b.set("steer", Binding(WHEEL, "wheel", 0, "axis", 0, key="path:usb-1"))
    b.set("gate1", Binding(WHEEL, "wheel", 0, "button", 0, key="path:usb-1"))
    src = sdl(backend, bindings=b)
    assert src.steer_joystick == wheel.handle
    backend.devs.remove(wheel)
    s = src.poll(0.01)
    assert "steer" not in s.bound and "gate1" not in s.bound and src.steer_joystick is None


def test_path_key_follows_the_device(rig, caplog):
    _, pedals, _, backend = rig
    pedals.key = "path:usb-7"
    b = rig_bindings()
    b.set("brake", Binding(PEDALS, "pedals", 1, "axis", 0, rest=-1.0, full=1.0, key="path:usb-2"))
    src = sdl(backend, bindings=b)
    pedals.axes[0] = 1.0
    s = src.poll(0.01)
    assert "brake" in s.bound and s.pressed == {"brake"}
    assert b.get("brake").key == "path:usb-7"
    assert sum("moved" in r.message for r in caplog.records) == 1


def test_velocity_mapping_constants():
    assert velocity_from_rate(RATE_GENTLE) == pytest.approx(VEL_GENTLE)
    assert velocity_from_rate(RATE_STOMP) == pytest.approx(1.0)
    assert velocity_from_rate(RATE_STOMP * 10) == 1.0
    assert 0 < velocity_from_rate(0.0) < VEL_GENTLE


def press_velocity(backend, pedals, seconds, hz):
    """Idle frames, then the brake ramps from rest to full over `seconds`, starting
    between two frames. Returns the velocity reported on the press."""
    clock = Clock()
    src = SdlInput(rig_bindings(), Config(), backend, clock=clock)
    start = 0.2 + 0.37 / hz
    for f in range(int(hz)):
        t = f / hz
        pedals.axes[0] = -1.0 + 2.0 * max(0.0, min(1.0, (t - start) / seconds))
        s = tick(src, clock, 1 / hz)
        if s.pressed:
            return s.velocity["brake"]
    raise AssertionError("never pressed")


@pytest.mark.parametrize("hz", [60, 144, 250])
def test_velocity_is_frame_rate_independent(rig, hz):
    _, pedals, _, backend = rig
    assert press_velocity(backend, pedals, 0.065, hz) >= 0.95
    assert press_velocity(backend, pedals, 0.5, hz) == pytest.approx(0.3, abs=0.05)


def test_unplugged_device_leaves_bound_and_reorder_survives(rig):
    wheel, pedals, box, backend = rig
    src = sdl(backend)
    backend.devs = [wheel, box]
    assert "brake" not in src.poll(0.01).bound
    pedals.index = 0
    wheel.index = 1
    backend.devs = [pedals, wheel, box]
    src.poll(0.01)
    pedals.axes[0] = 1.0
    s = src.poll(0.01)
    assert {"brake", "steer"} <= s.bound and s.brake == 1.0


def test_steer_joystick(rig):
    wheel, pedals, _, backend = rig
    src = SdlInput(rig_bindings(), Config(), backend)
    assert src.steer_joystick == wheel.handle
    backend.devs = [pedals]
    assert src.steer_joystick is None
    assert SdlInput(Bindings(), Config(), backend).steer_joystick is None


def test_keyboard_fallback_values():
    keys = FakeKeys()
    kb = KeyboardMouseInput(keys, Config())
    s = kb.poll(0.01)
    assert "steer" in s.fallback and "lever0" not in s.fallback and s.bound == frozenset()
    assert s.steer == 0.0
    keys.down |= {"S", "ONE", "ESCAPE"}
    s = kb.poll(0.01)
    assert s.brake == 1.0 and s.pressed == {"brake", "gate1"} and s.system == {"pause"}
    assert s.velocity == {"brake": 1.0, "gate1": 1.0}
    assert default_layer_modes(s.bound, s.fallback)["faders"] == "auto"


def test_keyboard_steer_keys_ramp_then_mouse():
    keys = FakeKeys()
    kb = KeyboardMouseInput(keys, Config(play_range_deg=90))
    keys.down.add("D")
    s = [kb.poll(0.1) for _ in range(2)][-1]
    assert s.steer == pytest.approx(0.7) and s.steer_deg == pytest.approx(63.0)
    keys.down = {"D", "A"}
    assert kb.poll(0.1).steer == pytest.approx(0.0)     # keys cancel: mouse at centre
    keys.down = set()
    keys.mouse = 1.0
    assert kb.poll(0.1).steer == 1.0
    keys.mouse = 0.5 - 0.35 / 2
    assert kb.poll(0.1).steer == pytest.approx(-0.5)


def test_composite_prefers_devices_and_falls_back(rig):
    wheel, pedals, box, backend = rig
    keys = FakeKeys()
    cfg = Config(wheel_range_deg=900)
    src = CompositeInput(SdlInput(rig_bindings(), cfg, backend), KeyboardMouseInput(keys, cfg), cfg)
    src.poll(0.01)
    keys.down |= {"S", "W", "TWO", "P"}
    keys.mouse = 1.0
    wheel.axes[0] = 0.05
    s = src.poll(0.01)
    assert s.bound == {"steer", "gate1", "brake", "clutch", "paddle_l"}
    assert s.fallback & s.bound == frozenset()
    assert {"throttle", "gate2", "handbrake", "paddle_r"} <= s.fallback
    assert s.brake == 0.0                         # the S key does not override the pedal
    assert s.throttle == 1.0 and s.pressed == {"throttle", "gate2"}
    assert s.steer_deg == pytest.approx(22.5)     # wheel, not the mouse
    assert s.system == {"pause"}
    assert src.steer_joystick == wheel.handle
    backend.devs = [box]                          # wheel and pedals unplugged
    s = src.poll(0.01)
    assert "brake" in s.fallback and s.brake == 1.0 and "brake" in s.down
    assert "brake" not in s.pressed               # a source switch is not a press
    assert "steer" in s.fallback and s.steer == 1.0
    keys.down.discard("S")
    assert "brake" in src.poll(0.01).released
    src.close()
    assert backend.closed


def test_open_input_without_sdl(monkeypatch):
    import torquehero.bindings as bmod

    def boom():
        raise bmod.SdlError("no sdl")

    monkeypatch.setattr(bmod, "SdlBackend", boom)
    src = open_input(Config(), FakeKeys(), bindings=Bindings())
    assert isinstance(src, KeyboardMouseInput)
    assert isinstance(open_input(Config(), FakeKeys(), devices=False), KeyboardMouseInput)
    with pytest.raises(RuntimeError):
        open_input(Config(), None, devices=False)


def test_no_heavy_imports_at_module_import():
    code = ("import sys, torquehero.input, torquehero.bindings; "
            "bad = [m for m in ('sdl2', 'raylib', 'pyray') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
