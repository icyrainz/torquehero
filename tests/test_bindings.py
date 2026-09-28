import argparse
import ctypes
import json
import logging

import pytest

from torquehero import bindings as bmod
from torquehero.bindings import (
    Binding,
    Bindings,
    DeviceSnapshot,
    JoyEvent,
    SdlBackend,
    control_kind,
    extend_peak,
    learn,
    learn_range,
    match_device,
    rest_at_full_travel,
    rest_suspicious,
    settled,
    stable,
    steer_range,
    unsettled,
)
from torquehero.config import Config
from torquehero.input import SdlInput

WHEEL = "030000001b0e00000410000000000000"
PEDALS = "03000000eb0e00000a18000000000000"


def snap(guid=WHEEL, name="wheel", index=0, axes=(0.0, 0.0), buttons=(False, False), hats=(0,), key=""):
    return DeviceSnapshot(guid, name, index, tuple(axes), tuple(buttons), tuple(hats), key)


def test_control_kinds():
    assert control_kind("steer") == "bipolar"
    assert {control_kind(c) for c in ("brake", "clutch", "throttle", "handbrake", "lever0", "lever1")} == {"unipolar"}
    assert control_kind("gate3") == control_kind("paddle_l") == control_kind("pause") == "button"
    with pytest.raises(KeyError):
        control_kind("horn")


# --- learn and calibration (pure) ---

def test_learn_steer_small_turn_sets_direction():
    base = [snap(axes=(0.0, -1.0)), snap(PEDALS, "pedals", 1, axes=(-1.0, -1.0))]
    assert learn("steer", base, base) is None
    assert learn("steer", base, [snap(axes=(-0.02, -1.0)), base[1]]) is None
    b = learn("steer", base, [snap(axes=(-0.036, -1.0)), base[1]])   # 90 degrees on a 2520 base
    assert (b.guid, b.kind, b.num, b.rest, b.sign, b.range_deg) == (WHEEL, "axis", 0, None, -1.0, None)
    assert b.value(snap(axes=(-0.5, 0.0))) == pytest.approx(0.5)


def test_learn_steer_ignores_buttons():
    assert learn("steer", [snap()], [snap(buttons=(True, False))]) is None


def test_steer_range():
    b = Binding(WHEEL, "wheel", 0, "axis", 0, sign=-1.0)
    assert steer_range(b, [snap(axes=(-90 / 1260, 0))]) == pytest.approx(2520.0)
    assert steer_range(b, [snap(axes=(-0.2, 0))]) == pytest.approx(900.0)
    assert steer_range(b, [snap(axes=(0.2, 0))]) is None             # turned left
    assert steer_range(b, [snap(axes=(-0.001, 0))]) is None          # out of range
    assert steer_range(b, []) is None


def test_learn_pedal_triggers_early_and_tracks_peak():
    base = [snap(PEDALS, "pedals", 1, axes=(-1.0, 1.0))]
    assert learn("brake", base, [snap(PEDALS, "pedals", 1, axes=(-1.0, 0.8))]) is None
    b = learn("brake", base, [snap(PEDALS, "pedals", 1, axes=(-1.0, 0.6))])      # stiff load cell: 20% so far
    assert (b.kind, b.num, b.rest, b.sign, b.full) == ("axis", 1, 1.0, -1.0, 0.6)
    for raw in (0.2, -0.3, -0.1):
        b = extend_peak(b, [snap(PEDALS, "pedals", 1, axes=(-1.0, raw))])
    assert b.full == -0.3                                                        # never reached -1.0
    assert b.value(snap(PEDALS, "pedals", 1, axes=(0, -0.3))) == 1.0
    assert b.value(snap(PEDALS, "pedals", 1, axes=(0, 0.35))) == pytest.approx(0.5)
    assert b.value(snap(PEDALS, "pedals", 1, axes=(0, 1.0))) == 0.0
    assert extend_peak(Binding(WHEEL, "w", 0, "button", 0), []) .kind == "button"


def test_value_without_full_uses_the_end_of_the_axis():
    b = Binding(PEDALS, "p", 0, "axis", 0, rest=-1.0, sign=1.0)
    assert b.value(snap(PEDALS, "p", 0, axes=(0.0,))) == pytest.approx(0.5)
    tiny = Binding(PEDALS, "p", 0, "axis", 0, rest=0.5, sign=1.0, full=0.5)
    assert tiny.value(snap(PEDALS, "p", 0, axes=(0.525,))) == pytest.approx(0.5)


def test_learn_picks_the_axis_that_moved_most():
    base = [snap(axes=(-1.0, -1.0))]
    assert learn("throttle", base, [snap(axes=(-0.4, 0.9))]).num == 1


def test_learn_unipolar_accepts_a_button():
    b = learn("handbrake", [snap()], [snap(buttons=(False, True))])
    assert (b.kind, b.num) == ("button", 1)
    assert b.value(snap(buttons=(False, True))) == 1.0


def test_learn_button_prefers_button_over_axis():
    b = learn("gate2", [snap(axes=(-1.0, 0.0))], [snap(axes=(1.0, 0.0), buttons=(True, False))])
    assert (b.kind, b.num) == ("button", 0)


def test_learn_button_accepts_axis_and_hat():
    base = [snap(axes=(-1.0, 0.0), hats=(1,))]
    b = learn("paddle_r", base, [snap(axes=(1.0, 0.0), hats=(1,))])
    assert (b.kind, b.rest, b.sign, b.full) == ("axis", -1.0, 1.0, 1.0)
    h = learn("menu_down", base, [snap(axes=(-1.0, 0.0), hats=(1 | 4,))])
    assert (h.kind, h.num, h.dir) == ("hat", 0, 4)
    assert h.value(snap(hats=(4,))) == 1.0 and h.value(snap(hats=(2,))) == 0.0


def test_learn_already_held_button_does_not_count():
    assert learn("gate1", [snap(buttons=(True, False))], [snap(buttons=(True, False))]) is None


def test_learn_skips_devices_not_in_baseline():
    assert learn("gate1", [snap()], [snap(), snap(PEDALS, "pedals", 1, buttons=(True,))]) is None


def test_learn_ignores_drifting_axes():
    base = [snap(axes=(-1.0, 0.0))]
    drifting = {(WHEEL, "wheel", "", 0, 1)}
    assert learn("brake", base, [snap(axes=(-1.0, 0.8))], drifting) is None
    assert learn("brake", base, [snap(axes=(0.0, 0.8))], drifting).num == 0
    assert unsettled(base, [snap(axes=(-1.0, 0.5))]) == drifting


def test_learn_copies_device_key():
    b = learn("gate1", [snap(key="serial:A")], [snap(buttons=(True, False), key="serial:A")])
    assert b.key == "serial:A"


def test_learn_range_for_levers():
    closed = [snap(axes=(0.9, 0.3)), snap(PEDALS, "pedals", 1, axes=(-1.0,))]
    opened = [snap(axes=(-0.95, 0.32)), snap(PEDALS, "pedals", 1, axes=(-1.0,))]
    b = learn_range("lever0", closed, opened)
    assert (b.num, b.rest, b.full, b.sign) == (0, 0.9, -0.95, -1.0)
    assert b.value(snap(axes=(-0.025, 0))) == pytest.approx(0.5)
    assert learn_range("lever1", closed, closed) is None
    with pytest.raises(ValueError):
        learn_range("steer", closed, opened)


def test_stable_and_rest_suspicious():
    assert stable([snap(axes=(-1.0, 0.0))], [snap(axes=(-0.99, 0.01))])
    assert not stable([snap(axes=(-1.0, 0.0))], [snap(axes=(-0.9, 0.0))])
    assert rest_suspicious(Binding(PEDALS, "p", 0, "axis", 0, rest=-0.4))
    assert not rest_suspicious(Binding(PEDALS, "p", 0, "axis", 0, rest=-1.0))
    assert not rest_suspicious(Binding(PEDALS, "p", 0, "button", 0))


def test_rest_at_full_travel_is_an_axis_that_never_came_back():
    base = [snap(axes=(1.0, 0.0))]                       # pedal held down while the baseline was taken
    b = learn("brake", base, [snap(axes=(-1.0, 0.0))])   # the foot lets go: "full" is really the rest
    assert b.rest == 1.0 and b.full == -1.0
    assert rest_at_full_travel(b, [snap(axes=(-1.0, 0.0))])
    assert not rest_at_full_travel(b, [snap(axes=(1.0, 0.0))])   # back at rest: a normal pedal
    assert not rest_at_full_travel(Binding(PEDALS, "p", 0, "button", 0), base)


def test_settled():
    base = [snap(axes=(-1.0, 0.0))]
    b = learn("brake", base, [snap(axes=(0.8, 0.0))])
    assert not settled(b, base, [snap(axes=(0.8, 0.0))])
    assert settled(b, base, [snap(axes=(-0.95, 0.0))])
    assert settled(b, base, [])


# --- device matching ---

def test_match_device_by_guid_and_name_not_index():
    b = Binding(PEDALS, "pedals", 0, "axis", 1, rest=-1.0)
    snaps = [snap(WHEEL, "wheel", 0), snap(PEDALS, "pedals", 3)]
    assert match_device(b, snaps) is snaps[1]
    assert match_device(b, [snap(WHEEL, "wheel", 0)]) is None


def test_match_device_index_breaks_ties_only_without_keys():
    b = Binding(PEDALS, "pedals", 2, "button", 0)
    snaps = [snap(PEDALS, "pedals", 1), snap(PEDALS, "pedals", 2)]
    assert match_device(b, snaps) is snaps[1]
    assert match_device(b, snaps[:1]) is snaps[0]
    assert match_device(b, [snap(PEDALS, "pedals", 2, key="serial:X")]) is None


def test_identical_twin_never_takes_over():
    b = Binding(PEDALS, "pedals", 0, "button", 0, key="serial:A")
    a, twin = snap(PEDALS, "pedals", 1, key="serial:A"), snap(PEDALS, "pedals", 0, key="serial:B")
    assert match_device(b, [twin, a]) is a
    assert match_device(b, [twin]) is None
    serial = Bindings({"gate1": b})
    assert serial.resolve([twin]) == {} and serial.get("gate1").key == "serial:A"


def test_path_key_follows_a_moved_device(caplog):
    moved = snap(PEDALS, "pedals", 3, key="path:usb-2")
    bs = Bindings({"brake": Binding(PEDALS, "pedals", 0, "axis", 0, rest=-1.0, key="path:usb-1"),
                   "clutch": Binding(PEDALS, "pedals", 0, "axis", 1, rest=-1.0, key="path:usb-1"),
                   "gate1": Binding(WHEEL, "wheel", 0, "button", 0, key="path:w")})
    with caplog.at_level(logging.WARNING, logger="torquehero.bindings"):
        found = bs.resolve([moved, snap(WHEEL, "wheel", 0, key="path:w")])
        assert found["brake"] is moved and found["clutch"] is moved
        assert bs.resolve([moved]).keys() == {"brake", "clutch"}
    assert len(caplog.records) == 1
    assert bs.get("brake").key == bs.get("clutch").key == "path:usb-2" and bs.get("brake").index == 3


def test_path_key_never_moves_to_a_twin_seen_together():
    a, b = snap(WHEEL, "wheel", 0, key="path:usb-1"), snap(WHEEL, "wheel", 1, key="path:usb-2")
    bs = Bindings({"steer": Binding(WHEEL, "wheel", 0, "axis", 0, key="path:usb-1")})
    assert bs.resolve([a, b])["steer"] is a
    assert bs.resolve([b]) == {}                                 # A unplugged: B is a different wheel
    assert bs.get("steer").key == "path:usb-1"
    assert bs.resolve([a])["steer"] is a


def test_path_key_does_not_take_a_claimed_or_ambiguous_device():
    a = Binding(PEDALS, "pedals", 0, "button", 0, key="path:usb-1")
    other = Binding(PEDALS, "pedals", 1, "button", 0, key="path:usb-9")
    claimed = Bindings({"gate1": a, "gate2": other})
    assert "gate1" not in claimed.resolve([snap(PEDALS, "pedals", 1, key="path:usb-9")])
    two = Bindings({"gate1": a})
    assert two.resolve([snap(PEDALS, "pedals", 1, key="path:x"), snap(PEDALS, "pedals", 2, key="path:y")]) == {}
    assert two.get("gate1").key == "path:usb-1"


def test_match_device_falls_back_to_guid_when_name_changed():
    b = Binding(PEDALS, "pedals", 0, "button", 0, key="serial:A")
    snaps = [snap(PEDALS, "Pedals v2", 4, key="serial:A")]
    assert match_device(b, snaps) is snaps[0]


def test_value_none_when_input_missing():
    assert Binding(WHEEL, "wheel", 0, "axis", 7).value(snap()) is None
    assert Binding(WHEEL, "wheel", 0, "button", 9).value(snap()) is None


# --- persistence ---

def test_twin_keys_survive_a_restart(tmp_path):
    a, b = snap(WHEEL, "wheel", 0, key="path:usb-1"), snap(WHEEL, "wheel", 1, key="path:usb-2")
    bs = Bindings({"steer": Binding(WHEEL, "wheel", 0, "axis", 0, key="path:usb-1")})
    bs.resolve([a, b])
    path = bs.save(tmp_path / "b.json")
    assert json.loads(path.read_text())["twins"] == {"path:usb-1": ["path:usb-2"]}
    again = Bindings.load(path)
    assert again.resolve([b]) == {} and again.get("steer").key == "path:usb-1"


def test_bad_twins_table_is_ignored(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"bindings": {}, "twins": {"k": "notalist", "j": [1, "path:x"]}}))
    assert Bindings.load(p)._together == {"j": {"path:x"}}


def test_save_load_roundtrip(tmp_path):
    bs = Bindings()
    bs.set("steer", Binding(WHEEL, "wheel", 0, "axis", 0, sign=-1.0, range_deg=1080.0, key="serial:W"))
    bs.set("brake", Binding(PEDALS, "pedals", 1, "axis", 1, rest=1.0, sign=-1.0, full=-0.3, key="path:p"))
    bs.set("gate1", Binding(WHEEL, "wheel", 0, "button", 4))
    bs.set("menu_up", Binding(WHEEL, "wheel", 0, "hat", 0, dir=1))
    path = bs.save(tmp_path / "b.json")
    data = json.loads(path.read_text())
    assert data["format"] == 1 and list(data["bindings"]) == ["steer", "brake", "gate1", "menu_up"]
    assert "rest" not in data["bindings"]["gate1"] and "key" not in data["bindings"]["gate1"]
    assert data["bindings"]["steer"]["range_deg"] == 1080.0
    assert Bindings.load(path) == bs


def test_load_skips_bad_entries_with_one_warning(tmp_path, caplog):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"format": 1, "bindings": {
        "horn": {"guid": "x", "name": "y", "kind": "button", "num": 0},
        "brake": {"guid": "x"},
        "clutch": {"guid": "x", "name": "y", "kind": "slider", "num": 0},
        "pause": {"guid": "x", "name": "y", "kind": "hat", "num": 0, "dir": 3},
        "steer": {"guid": "x", "name": "y", "kind": "button", "num": 0},
        "gate2": {"guid": "x", "name": "y", "kind": "button", "num": "a"},
        "gate3": "nonsense",
        "gate4": {"guid": "x", "name": "y", "kind": "axis", "num": 0, "rest": float("nan")},
        "gate5": {"guid": "x", "name": "y", "kind": "axis", "num": 0, "full": float("inf")},
        "gate6": {"guid": "x", "name": "y", "kind": "axis", "num": 0, "range_deg": -5},
        "paddle_l": {"guid": "x", "name": "y", "kind": "button", "num": "OVERFLOW"},
        "gate1": {"guid": "x", "name": "y", "index": 2, "kind": "button", "num": 3},
    }}).replace('"OVERFLOW"', "1e999"))
    with caplog.at_level(logging.WARNING, logger="torquehero.bindings"):
        assert list(Bindings.load(p).controls) == ["gate1"]
    assert len(caplog.records) == 1


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", '{"bindings": 5}', "\xff\xfe"])
def test_corrupt_file_starts_empty(tmp_path, caplog, text):
    p = tmp_path / "b.json"
    p.write_bytes(text.encode("latin-1"))
    with caplog.at_level(logging.WARNING, logger="torquehero.bindings"):
        assert Bindings.load(p).controls == {}
    assert len(caplog.records) == 1


def test_load_missing_file(tmp_path):
    assert Bindings.load(tmp_path / "none.json").controls == {}


def test_default_path_under_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TORQUEHERO_CONFIG_DIR", str(tmp_path))
    Bindings({"gate1": Binding(WHEEL, "wheel", 0, "button", 0)}).save()
    assert (tmp_path / "bindings.json").exists()
    assert "gate1" in Bindings.load()


def test_clear():
    bs = Bindings({"gate1": Binding(WHEEL, "w", 0, "button", 0), "gate2": Binding(WHEEL, "w", 0, "button", 1)})
    bs.clear(["gate1"])
    assert list(bs.controls) == ["gate2"]
    bs.clear()
    assert bs.controls == {}


# --- SdlBackend against a fake sdl2 module ---

class _Btn(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("timestamp", ctypes.c_uint), ("which", ctypes.c_int),
                ("button", ctypes.c_ubyte), ("state", ctypes.c_ubyte)]


class _Hat(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("timestamp", ctypes.c_uint), ("which", ctypes.c_int),
                ("hat", ctypes.c_ubyte), ("value", ctypes.c_ubyte)]


class _Event(ctypes.Union):
    _fields_ = [("type", ctypes.c_uint), ("jbutton", _Btn), ("jhat", _Hat)]


class Phys:
    """A physical USB device as the fake SDL sees it."""

    def __init__(self, iid, guid, name, serial=None, axes=(), buttons=0, haptic=False):
        self.iid, self.guid, self.name, self.serial, self.haptic = iid, guid, name, serial, haptic
        self.axes, self.buttons, self.updated = list(axes), [False] * buttons, False


class FakeSdl:
    SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS = b"BG"
    SDL_HINT_NO_SIGNAL_HANDLERS = b"NOSIG"
    SDL_INIT_JOYSTICK, SDL_INIT_HAPTIC, SDL_ENABLE, SDL_GETEVENT = 0x200, 0x1000, 1, 2
    SDL_JOYAXISMOTION, SDL_JOYHATMOTION, SDL_JOYBUTTONDOWN, SDL_JOYBUTTONUP = 0x600, 0x602, 0x603, 0x604
    SDL_JOYBATTERYUPDATED, SDL_FIRSTEVENT, SDL_LASTEVENT = 0x607, 0, 0xFFFF
    SDL_JOYBALLMOTION, SDL_IGNORE = 0x601, 0
    SDL_Event = _Event

    def __init__(self, *plugged):
        self.plugged = list(plugged)
        self.hints, self.opened, self.closed, self.calls, self.queue = {}, [], [], [], []

    def SDL_SetHint(self, k, v): self.hints[k] = v
    def SDL_InitSubSystem(self, flags): return 0
    def SDL_QuitSubSystem(self, flags): self.calls.append("quit")
    def SDL_GetError(self): return b""
    def SDL_JoystickEventState(self, state): self.calls.append(("events", state))
    def SDL_EventState(self, kind, state): self.calls.append(("event_state", kind, state))
    def SDL_PumpEvents(self): self.calls.append("pump")
    def SDL_NumJoysticks(self): return len(self.plugged)
    def SDL_JoystickGetDeviceInstanceID(self, i): return self.plugged[i].iid
    def SDL_JoystickGetGUID(self, j): return j.guid
    def SDL_JoystickGetGUIDString(self, g, buf, n): buf.value = g.encode()
    def SDL_JoystickName(self, j): return j.name.encode()
    def SDL_JoystickIsHaptic(self, j): return 1 if j.haptic else 0
    def SDL_JoystickInstanceID(self, j): return j.iid
    def SDL_JoystickNumAxes(self, j): return len(j.axes)
    def SDL_JoystickNumButtons(self, j): return len(j.buttons)
    def SDL_JoystickNumHats(self, j): return 0
    def SDL_JoystickGetSerial(self, j): return j.serial
    def SDL_JoystickPath(self, j): return None
    def SDL_JoystickGetAttached(self, j): return j in self.plugged
    def SDL_JoystickClose(self, j):
        self.closed.append(j)
        self.calls.append(("close", j))
    def SDL_JoystickGetButton(self, j, i): return int(j.buttons[i])
    def SDL_JoystickGetHat(self, j, i): return 0
    def SDL_FlushEvents(self, lo, hi): self.calls.append("flush")

    def SDL_JoystickOpen(self, i):
        self.opened.append(self.plugged[i])
        return self.plugged[i]

    def SDL_JoystickUpdate(self):
        self.calls.append("update")
        for p in self.plugged:
            p.updated = True

    def SDL_JoystickGetAxis(self, j, i):
        return round(j.axes[i] * 32767) if j.updated else 0    # like SDL: 0 until the first update

    def SDL_PeepEvents(self, buf, n, action, lo, hi):
        k = 0
        while self.queue and k < n:
            buf[k] = self.queue.pop(0)
            k += 1
        return k


def button_event(which, button, state):
    ev = _Event()
    ev.jbutton = _Btn(FakeSdl.SDL_JOYBUTTONDOWN if state else FakeSdl.SDL_JOYBUTTONUP, 0, which, button, state)
    return ev


def hat_event(which, hat, value):
    ev = _Event()
    ev.jhat = _Hat(FakeSdl.SDL_JOYHATMOTION, 0, which, hat, value)
    return ev


@pytest.fixture
def fake_sdl(monkeypatch):
    wheel = Phys(10, WHEEL, "wheel", b"W1", axes=(0.0,), buttons=4, haptic=True)
    box = Phys(11, "boxguid", "box", None, buttons=8)
    fake = FakeSdl(wheel, box)
    monkeypatch.setattr(bmod, "_SDL", fake)
    return fake, wheel, box


def test_backend_setup(fake_sdl):
    fake, wheel, box = fake_sdl
    be = SdlBackend()
    assert fake.hints == {b"BG": b"1", b"NOSIG": b"1"}
    assert ("events", 1) in fake.calls and ("event_state", 0x600, 0) in fake.calls
    devs = be.devices()
    assert [(d.name, d.key, d.haptic) for d in devs] == [("wheel", "serial:W1", True), ("box", "", False)]
    be.close()
    assert fake.closed == [wheel, box] and fake.calls[-1] == "quit"


def test_hotplug_never_touches_the_wheel(fake_sdl):
    fake, wheel, box = fake_sdl
    be = SdlBackend()
    wheel_dev = be.devices()[0]
    handle = wheel_dev.handle
    fake.plugged = [wheel]                                       # button box unplugged
    be.update()
    assert be.devices() == [wheel_dev] and fake.closed == [box]
    pedals = Phys(12, PEDALS, "pedals", b"P1", axes=(-1.0,))
    fake.plugged = [wheel, pedals]                               # pedals plugged in
    fake.calls.clear()
    be.update()
    assert fake.opened == [wheel, box, pedals] and wheel not in fake.closed
    assert be.devices()[0] is wheel_dev and wheel_dev.handle is handle
    assert fake.calls.index("update") > fake.calls.index("pump")
    p = be.devices()[1]
    assert p.index == 1 and p.snapshot().axes == (-1.0,)         # updated before the first snapshot
    fake.plugged = [pedals, wheel]                               # re-ordered: still the same objects
    be.update()
    assert wheel_dev.index == 1 and be.devices()[1] is wheel_dev


def test_callbacks_run_before_close_in_order(fake_sdl):
    fake, wheel, box = fake_sdl
    be = SdlBackend()
    be.on_detach(lambda h: fake.calls.append(("detach1", h)))
    be.on_detach(lambda h: 1 / 0)                                # a failing callback is logged, not fatal
    be.on_detach(lambda h: fake.calls.append(("detach2", h)))
    be.on_close(lambda h: fake.calls.append(("shutdown", h)))
    fake.calls.clear()
    fake.plugged = [box]
    be.update()
    assert [c for c in fake.calls if isinstance(c, tuple)] == [("detach1", wheel), ("detach2", wheel), ("close", wheel)]
    fake.calls.clear()
    be.close()
    assert [c for c in fake.calls if isinstance(c, tuple)] == [("shutdown", box), ("close", box)]


def test_steer_lost_before_the_wheel_closes(fake_sdl):
    fake, wheel, box = fake_sdl
    bs = Bindings({"steer": Binding(WHEEL, "wheel", 0, "axis", 0, key="serial:W1")})
    src = SdlInput(bs, Config(), SdlBackend())
    src.on_steer_lost(lambda h: fake.calls.append(("ffb_lost", h)))
    fake.calls.clear()
    fake.plugged = [wheel]                                       # box unplugged: FFB not told
    src.poll(0.01)
    assert not any(c[0] == "ffb_lost" for c in fake.calls if isinstance(c, tuple))
    fake.plugged = []                                            # wheel e-stop / power cycle
    fake.calls.clear()
    src.poll(0.01)
    assert [c for c in fake.calls if isinstance(c, tuple)] == [("ffb_lost", wheel), ("close", wheel)]
    fake.plugged = [wheel]
    src.poll(0.01)
    fake.calls.clear()
    src.close()
    assert [c for c in fake.calls if isinstance(c, tuple)][:2] == [("ffb_lost", wheel), ("close", wheel)]


def test_handler_that_polls_does_not_reenter(fake_sdl):
    fake, wheel, box = fake_sdl
    bs = Bindings({"steer": Binding(WHEEL, "wheel", 0, "axis", 0, key="serial:W1")})
    src = SdlInput(bs, Config(), SdlBackend())
    seen = []

    def lost(h):
        seen.append(h)
        src.poll(0.01)                                           # FFB handler touching input again
        seen.append(src.steer_joystick)

    src.on_steer_lost(lost)
    src.poll(0.01)
    fake.plugged = [box]
    src.poll(0.01)
    assert seen == [wheel, None] and fake.closed.count(wheel) == 1
    src.close()
    assert fake.closed.count(wheel) == 1 and fake.closed.count(box) == 1


def test_two_steer_lost_handlers_both_run_before_close(fake_sdl):
    fake, wheel, box = fake_sdl
    bs = Bindings({"steer": Binding(WHEEL, "wheel", 0, "axis", 0, key="serial:W1")})
    src = SdlInput(bs, Config(), SdlBackend())
    src.on_steer_lost(lambda h: (fake.calls.append(("app", h)), src.poll(0.01)))   # polls: clears the handle
    src.on_steer_lost(lambda h: 1 / 0)
    src.on_steer_lost(lambda h: fake.calls.append(("ffb", h)))
    src.poll(0.01)
    fake.plugged = [box]
    fake.calls.clear()
    src.poll(0.01)
    calls = [c for c in fake.calls if isinstance(c, tuple)]
    assert calls == [("app", wheel), ("ffb", wheel), ("close", wheel)]
    fake.calls.clear()
    src.close()
    assert not any(c[0] in ("app", "ffb") for c in fake.calls if isinstance(c, tuple))


def test_detach_handler_that_closes_the_backend(fake_sdl):
    fake, wheel, box = fake_sdl
    be = SdlBackend()
    be.on_detach(lambda h: be.close())
    fake.plugged = []
    be.update()
    assert fake.closed.count(wheel) == 0 and fake.closed == [box]    # SDL quit closed the wheel
    assert fake.calls.count("quit") == 1 and be.devices() == [] and be.events() == []


def test_handler_that_closes_during_close(fake_sdl):
    fake, wheel, box = fake_sdl
    be = SdlBackend()
    be.on_close(lambda h: be.close())
    be.on_detach(lambda h: be.update())
    be.close()
    assert fake.closed == [wheel, box] and fake.calls.count("quit") == 1


def test_remover_unregisters(fake_sdl):
    fake, wheel, box = fake_sdl
    be = SdlBackend()
    calls = []
    remove = be.on_detach(calls.append)
    remove()
    remove()
    fake.plugged = []
    be.update()
    assert calls == []


def test_backend_events(fake_sdl):
    fake, wheel, box = fake_sdl
    be = SdlBackend()
    wheel_dev, box_dev = be.devices()
    fake.queue = [button_event(10, 3, 1), button_event(10, 3, 0), hat_event(11, 0, 4), button_event(99, 0, 1)]
    fake.queue += [button_event(11, 1, 1) for _ in range(70)]    # more than one buffer
    be.update()
    ev = be.events()
    assert ev[:3] == [JoyEvent(wheel_dev, "button", 3, 1), JoyEvent(wheel_dev, "button", 3, 0),
                      JoyEvent(box_dev, "hat", 0, 4)]
    assert len(ev) == 73 and not fake.queue and "flush" in fake.calls
    be.update()
    assert be.events() == []


def test_sdl_input_end_to_end(fake_sdl):
    fake, wheel, box = fake_sdl
    pedals = Phys(12, PEDALS, "pedals", b"P1", axes=(-1.0,))
    fake.plugged.append(pedals)
    bs = Bindings({"brake": Binding(PEDALS, "pedals", 2, "axis", 0, rest=-1.0, full=1.0, key="serial:P1"),
                   "gate1": Binding(WHEEL, "wheel", 0, "button", 3, key="serial:W1"),
                   "steer": Binding(WHEEL, "wheel", 0, "axis", 0, range_deg=1080.0, key="serial:W1")})
    src = SdlInput(bs, Config(), SdlBackend())
    s = src.poll(0.01)
    assert s.brake == 0.0 and not s.pressed                      # no phantom 0.5 from an un-updated axis
    fake.queue = [button_event(10, 3, 1), button_event(10, 3, 0)]
    s = src.poll(0.01)
    assert s.pressed == {"gate1"} and s.released == {"gate1"}
    assert src.steer_joystick is wheel
    fake.plugged = [wheel]
    src.poll(0.01)
    assert src.steer_joystick is wheel and wheel not in fake.closed


# --- CLI, with a scripted backend ---

class ScriptDevice:
    """Inputs are a function of the backend's tick, so a test scripts a whole session."""

    def __init__(self, backend, guid, name, index, script, key="", haptic=False):
        self.backend, self.script = backend, script
        self.guid, self.name, self.index, self.key, self.haptic, self.handle = guid, name, index, key, haptic, object()

    def snapshot(self):
        axes, buttons = self.script(self.backend.tick)
        return DeviceSnapshot(self.guid, self.name, self.index, tuple(axes), tuple(buttons), (), self.key)


class ScriptBackend:
    def __init__(self):
        self.tick, self.devs, self.closed = 0, [], False

    def add(self, *a, **kw):
        self.devs.append(ScriptDevice(self, *a, **kw))
        return self

    def update(self):
        self.tick += 1

    def devices(self):
        return self.devs

    def events(self):
        return []

    def on_detach(self, cb):
        pass

    def on_close(self, cb):
        pass

    def close(self):
        self.closed = True


def run_cli(argv):
    parser = argparse.ArgumentParser()
    bmod.add_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(argv)
    return args.func(args)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setenv("TORQUEHERO_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(bmod, "STABLE_SPAN", 0.0)
    monkeypatch.setattr(bmod, "_enter_pressed", lambda: False)
    monkeypatch.setattr("builtins.input", lambda q: "")
    return tmp_path


def use(monkeypatch, backend):
    monkeypatch.setattr(bmod, "_open_backend", lambda: backend)
    return backend


def test_probe_no_devices(cli, monkeypatch, capsys):
    backend = use(monkeypatch, ScriptBackend())
    assert run_cli(["probe"]) == 0
    assert "No joystick devices found" in capsys.readouterr().out
    assert backend.closed


def test_probe_lists_devices(cli, monkeypatch, capsys):
    Bindings({"steer": Binding(WHEEL, "wheel", 5, "axis", 0),
              "brake": Binding(PEDALS, "pedals", 1, "axis", 0, rest=-1.0)}).save()
    use(monkeypatch, ScriptBackend().add(WHEEL, "wheel", 0, lambda t: ((0.25, -1.0), (False, True)), haptic=True))
    assert run_cli(["probe"]) == 0
    out = capsys.readouterr().out
    assert WHEEL in out and "haptic yes" in out and "0:+0.25" in out and "buttons 1" in out
    assert "bound   steer" in out and "bound but device not found: brake (pedals)" in out


def test_probe_live_plain_output(cli, monkeypatch, capsys):
    backend = ScriptBackend().add(WHEEL, "wheel", 0, lambda t: ((0.0,), ()))
    use(monkeypatch, backend)
    monkeypatch.setattr(bmod, "_live_ok", lambda: False)

    def sleep(s):
        if backend.tick >= 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(bmod.time, "sleep", sleep)
    assert run_cli(["probe", "--live"]) == 0
    out = capsys.readouterr().out
    assert "\x1b[" not in out and out.count("[0] wheel") == 3


def test_probe_sdl_unavailable(monkeypatch):
    monkeypatch.setattr(bmod, "_open_backend", lambda: None)
    assert run_cli(["probe"]) == 1


def pedal_script(t):
    """Rest, a foot fidgeting at first, then a press that peaks at +0.7 (not +1.0), then release."""
    if t <= 4:
        return (-1.0 if t % 2 else -0.9,), ()
    if t <= 7:
        return (-1.0,), ()
    if t <= 11:
        return (-1.0 + 0.4 * (t - 7),), ()
    if t <= 13:
        return (0.7,), ()
    return (-1.0,), ()


def test_bind_pedal_calibrates_full_travel(cli, monkeypatch, capsys):
    use(monkeypatch, ScriptBackend().add(PEDALS, "pedals", 0, pedal_script, key="serial:P"))
    assert run_cli(["bind", "brake", "--timeout", "5"]) == 0
    b = Bindings.load().get("brake")
    assert (b.guid, b.key, b.kind, b.num, b.rest, b.sign, b.full) == (PEDALS, "serial:P", "axis", 0, -1.0, 1.0, 0.7)
    out = capsys.readouterr().out
    assert "feet off" in out and "something is moving" in out and "saved" in out


def test_bind_pedal_enter_finishes_tracking(cli, monkeypatch):
    use(monkeypatch, ScriptBackend().add(PEDALS, "pedals", 0, lambda t: ((-1.0 if t <= 3 else 0.2,), ())))
    monkeypatch.setattr(bmod, "_enter_pressed", lambda: True)
    assert run_cli(["bind", "throttle", "--timeout", "2"]) == 0
    assert Bindings.load().get("throttle").full == 0.2


def test_bind_warns_on_mid_travel_rest(cli, monkeypatch, capsys):
    use(monkeypatch, ScriptBackend().add(PEDALS, "pedals", 0, lambda t: ((-0.3 if t <= 3 else 0.9 if t <= 5
                                                                             else -0.3,), ())))
    monkeypatch.setattr("builtins.input", lambda q: "n")
    assert run_cli(["bind", "clutch", "--timeout", "2"]) == 0
    assert "WARNING rest -0.30 is mid-travel" in capsys.readouterr().out
    assert Bindings.load().get("clutch") is None


def test_bind_steer_calibrates_range(cli, monkeypatch, capsys):
    raw90 = 90 / 540                                              # a 1080 degree base held at 90 right
    use(monkeypatch, ScriptBackend().add(WHEEL, "wheel", 0, lambda t: ((0.0 if t <= 3 else raw90, -1.0), ())))
    assert run_cli(["bind", "steer", "--timeout", "2"]) == 0
    b = Bindings.load().get("steer")
    assert b.sign == 1.0 and b.range_deg == pytest.approx(1080.0)
    assert "1080 degrees lock to lock" in capsys.readouterr().out


def test_bind_steer_range_flag(cli, monkeypatch, capsys):
    use(monkeypatch, ScriptBackend().add(WHEEL, "wheel", 0, lambda t: ((0.0 if t <= 3 else -0.1, -1.0), ())))
    assert run_cli(["bind", "steer", "--range", "2520", "--timeout", "2"]) == 0
    b = Bindings.load().get("steer")
    assert (b.sign, b.range_deg) == (-1.0, 2520.0)
    assert run_cli(["bind", "--range", "900"]) == 0              # already bound: no devices needed
    assert Bindings.load().get("steer").range_deg == 900.0
    assert run_cli(["bind", "--range", "90"]) == 2
    assert run_cli(["bind", "brake", "--range", "900"]) == 2


def test_bind_lever_two_ends(cli, monkeypatch):
    state = {"raw": 0.1}

    def ask(q):
        state["raw"] = 0.9 if "closed" in q else -0.95 if "open" in q else state["raw"]
        return ""

    use(monkeypatch, ScriptBackend().add("stecs", "STECS", 0, lambda t: ((0.0, state["raw"]), ())))
    monkeypatch.setattr("builtins.input", ask)
    assert run_cli(["bind", "lever1"]) == 0
    b = Bindings.load().get("lever1")
    assert (b.num, b.rest, b.full, b.sign) == (1, 0.9, -0.95, -1.0)


def test_bind_button_declined_and_timeout_save_nothing(cli, monkeypatch):
    use(monkeypatch, ScriptBackend().add(WHEEL, "wheel", 0, lambda t: ((), (4 <= t <= 5,))))
    monkeypatch.setattr("builtins.input", lambda q: "n")
    assert run_cli(["bind", "gate1", "gate2", "--timeout", "0.05"]) == 0
    assert not (cli / "bindings.json").exists()


def test_bind_button_saved(cli, monkeypatch, capsys):
    Bindings({"gate2": Binding(WHEEL, "wheel", 0, "button", 0)}).save()
    use(monkeypatch, ScriptBackend().add(WHEEL, "wheel", 0, lambda t: ((), (4 <= t <= 5,))))
    assert run_cli(["bind", "gate1", "--timeout", "2"]) == 0
    assert Bindings.load().get("gate1").num == 0
    assert "also bound to gate2" in capsys.readouterr().out


def test_bind_waits_for_an_all_zero_device_to_report(cli, monkeypatch, capsys):
    def script(t):
        if t <= 3:
            return (0.0, 0.0), ()                                 # opened, nothing reported yet
        if t <= 20:
            return (-1.0, -1.0), ()                               # first report: both pedals at rest
        if t <= 22:
            return (0.8, -1.0), ()                                # brake pressed
        return (-1.0, -1.0), ()

    use(monkeypatch, ScriptBackend().add(PEDALS, "pedals", 0, script))
    assert run_cli(["bind", "brake", "--timeout", "5"]) == 0
    b = Bindings.load().get("brake")
    assert (b.num, b.rest, b.sign, b.full) == (0, -1.0, 1.0, 0.8)   # not the untouched axis 1 going 0 -> -1
    assert "just reported" in capsys.readouterr().out


def test_real_only_hides_an_all_zero_device_until_an_event():
    class Dev:
        guid, name, index, key, haptic, handle = PEDALS, "pedals", 0, "", False, None

        def snapshot(self):
            return snap(PEDALS, "pedals", 0, axes=(0.0, 0.0))

    dev = Dev()

    class Inner(ScriptBackend):
        def devices(self):
            return [dev]

        def events(self):
            return [JoyEvent(dev, "button", 0, 1)] if self.tick >= 3 else []

    rig = bmod.RealOnly(Inner())
    rig.update()
    rig.update()
    assert rig.devices() == []
    rig.update()
    assert rig.devices() == [dev]


def test_bind_ignores_a_drifting_axis(cli, monkeypatch, capsys):
    def script(t):
        noise = 0.5 if t % 2 else -0.5                            # axis 1: unconnected, floats for ever
        return (-1.0 if t <= 12 else 0.9 if t <= 14 else -1.0, noise), ()

    use(monkeypatch, ScriptBackend().add(PEDALS, "pedals", 0, script))
    assert run_cli(["bind", "handbrake", "--timeout", "5"]) == 0
    assert Bindings.load().get("handbrake").num == 0
    assert "ignoring axis 1 on pedals #0" in capsys.readouterr().out


def test_bind_axis_by_hand(cli, monkeypatch, capsys):
    backend = ScriptBackend().add(PEDALS, "pedals", 0, lambda t: ((-1.0, 1.0), ()), key="serial:P")
    use(monkeypatch, backend)
    assert run_cli(["bind", "brake", "--axis", "1", "--rest", "1.0", "--full", "-0.2"]) == 0
    b = Bindings.load().get("brake")
    assert (b.key, b.num, b.rest, b.full, b.sign) == ("serial:P", 1, 1.0, -0.2, -1.0)
    assert run_cli(["bind", "brake", "--axis", "5", "--rest", "1", "--full", "0"]) == 2       # no axis 5
    assert run_cli(["bind", "brake", "--axis", "1", "--rest", "1"]) == 2                     # no --full
    assert run_cli(["bind", "steer", "--axis", "0", "--rest", "0", "--full", "1"]) == 2
    assert run_cli(["bind", "brake", "clutch", "--axis", "0", "--rest", "-1", "--full", "1"]) == 2
    backend.add(WHEEL, "wheel", 1, lambda t: ((0.0,), ()))
    assert run_cli(["bind", "brake", "--axis", "0", "--rest", "-1", "--full", "1"]) == 2     # which device?
    assert run_cli(["bind", "brake", "--axis", "0", "--rest", "-1", "--full", "1", "--device", WHEEL]) == 0
    assert Bindings.load().get("brake").guid == WHEEL
    assert run_cli(["bind", "brake", "--axis", "0", "--rest", "-1", "--full", "1", "--device", "nope"]) == 2
    assert "see `torquehero probe`" in capsys.readouterr().err


def test_bind_axis_picks_one_of_two_twins(cli, monkeypatch, capsys):
    backend = ScriptBackend()
    backend.add(PEDALS, "pedals", 0, lambda t: ((-1.0,), ()), key="path:a")
    backend.add(PEDALS, "pedals", 1, lambda t: ((-1.0,), ()), key="path:b")
    use(monkeypatch, backend)
    args = ["bind", "brake", "--axis", "0", "--rest", "-1", "--full", "1", "--device"]
    assert run_cli([*args, PEDALS]) == 2
    assert "2 devices match --device" in capsys.readouterr().err
    assert run_cli([*args, "path:b"]) == 0
    assert Bindings.load().get("brake").key == "path:b"
    assert run_cli([*args, "0"]) == 0
    assert Bindings.load().get("brake").key == "path:a"


def test_bind_saves_a_moved_path_key(cli, monkeypatch):
    Bindings({"gate2": Binding(WHEEL, "wheel", 0, "button", 0, key="path:old")}).save()
    use(monkeypatch, ScriptBackend().add(WHEEL, "wheel", 0, lambda t: ((), (False,)), key="path:new"))
    assert run_cli(["bind", "gate1", "--timeout", "0.05"]) == 0
    assert Bindings.load().get("gate2").key == "path:new"


def test_bind_list_clear_and_unknown(cli, capsys):
    Bindings({"gate1": Binding(WHEEL, "wheel", 0, "button", 3),
              "gate2": Binding(WHEEL, "wheel", 0, "button", 4)}).save()
    assert run_cli(["bind", "--list"]) == 0
    assert "wheel #0: button 3" in capsys.readouterr().out
    assert run_cli(["bind", "--clear", "gate1"]) == 0
    assert list(Bindings.load().controls) == ["gate2"]
    assert run_cli(["bind", "--clear"]) == 0
    assert Bindings.load().controls == {}
    assert run_cli(["bind", "horn"]) == 2


def test_bind_no_devices(cli, monkeypatch, capsys):
    use(monkeypatch, ScriptBackend())
    assert run_cli(["bind", "gate1"]) == 0
    assert "No joystick devices found" in capsys.readouterr().out
