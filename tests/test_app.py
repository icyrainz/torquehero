"""App loop with fakes for every hardware module: phases, layer modes, safety wiring,
retry lead-in, flush after a long frame, shutdown order under exceptions."""
from __future__ import annotations

import importlib.machinery
import json
import sys
import types
from pathlib import Path

import pytest
from conftest import FIXTURES, inp

from torquehero import app as app_mod
from torquehero.__main__ import build_parser, main
from torquehero.app import (
    MIN_LEAD,
    App,
    AppSettings,
    Menu,
    Song,
    click_chart,
    filter_difficulty,
    find_songs,
    load_chart,
    parse_layers,
    resolve_gain,
    resolve_layer_modes,
)
from torquehero.audio import NullAudio
from torquehero.chart import Chart
from torquehero.config import Config
from torquehero.state import GATES, LAYERS, Snapshot

CHART = FIXTURES / "chart.json"
DT = 1 / 60


@pytest.fixture(autouse=True)
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TORQUEHERO_CONFIG_DIR", str(tmp_path / "cfg"))
    return tmp_path / "cfg"


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class FakeInput:
    def __init__(self, log: list):
        self.log, self.next = log, inp()
        self.queue: list = []
        self.lost: list = []
        self.polls: list = []

    def poll(self, dt):
        self.log.append("poll")
        state = self.queue.pop(0) if self.queue else self.next
        self.polls.append(state)
        return state

    def flush(self):
        self.log.append("flush")

    def on_steer_lost(self, cb):
        self.lost.append(cb)
        return lambda: self.lost.remove(cb)

    def close(self):
        self.log.append("close input")
        for cb in list(self.lost):
            cb(object())


class FakeAudio(NullAudio):
    def __init__(self, log: list, clock):
        super().__init__(clock=clock)
        self.log = log

    def handle(self, events):
        self.log.append("audio.handle")
        super().handle(events)

    def update(self, snap):
        self.log.append("audio.update")
        super().update(snap)

    def stop(self):
        self.log.append("close audio")
        super().stop()


class FakeFfb:
    def __init__(self, log: list, gain: float = 0.2):
        self.log, self.gain = log, gain
        self.phases: list[str] = []
        self.spinning: list[bool] = []
        self.stops = self.resumes = 0
        self.closed = self.lost = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def update(self, events, snap, dt, spinning=False):
        self.log.append("ffb.update")
        self.phases.append(snap.phase)
        self.spinning.append(spinning)
        return {"torque": 0.25, "log": ["TEST"], "latched": self.stops > self.resumes}

    def set_gain(self, g, quiet=False):
        self.gain = min(1.0, max(0.0, g))
        return self.gain

    def stop_all(self):
        self.log.append("ffb.stop_all")
        self.stops += 1

    def resume(self):
        self.log.append("ffb.resume")
        self.resumes += 1

    def device_lost(self):
        self.lost = True
        self.close()

    def close(self):
        if not self.closed:
            self.log.append("close ffb")
        self.closed = True


class FakeCompanion:
    def __init__(self, log: list):
        self.log, self.songs, self.snaps = log, [], []

    def publish(self, snap):
        self.log.append("companion.publish")
        self.snaps.append(snap)

    def publish_song(self, info, chart):
        self.songs.append((info, chart))

    def close(self):
        self.log.append("close companion")


class FakeView:
    def __init__(self, log: list, focused: bool = True):
        self.log, self.is_focused = log, focused
        self.drawn: list = []
        self.close_after: int | None = None

    def focused(self):
        return self.is_focused

    def should_close(self):
        return self.close_after is not None and len(self.drawn) >= self.close_after

    def draw(self, snap, chart, notice=None):
        self.log.append("draw")
        self.drawn.append((snap.phase, chart))
        self.notice = notice

    def close(self):
        self.log.append("close window")


def make_app(tmp_path=None, songs=None, start=None, focused=True, cfg=None, **kw):
    log: list = []
    clock = Clock()
    parts = dict(inp=FakeInput(log), audio=FakeAudio(log, clock), ffb=FakeFfb(log),
                 companion=FakeCompanion(log), view=FakeView(log, focused))
    a = App(cfg or Config(), songs=songs or [Song("fixture", CHART)], start=start, clock=clock,
            click_dir=lambda: FIXTURES, **parts, **kw)
    a.log, a.fake_clock = log, clock
    return a


def step(a: App, n: int = 1, dt: float = DT, **input_kw) -> None:
    """Run n frames of dt seconds; input_kw is the input for the first of them."""
    if input_kw:
        a.inp.queue.append(inp(**input_kw))
    for _ in range(n):
        a.fake_clock.t += dt
        a.frame()


def skip(a, seconds: float) -> None:
    """Move the song clock on by `seconds` without a long frame (no hitch), then step once."""
    a.fake_clock.t += seconds
    a._last = a.fake_clock.t
    step(a)


def press(a: App, *system: str, **kw) -> None:
    step(a, system=set(system), **kw)


def to_play(a: App) -> None:
    """From the menu (song row selected) through the countdown into play."""
    step(a)
    press(a, "menu_ok")
    assert a.phase == "countdown"
    step(a, int(a.countdown / DT) + 2)
    assert a.phase == "play"


# --- pure helpers ---

def test_parse_layers():
    assert parse_layers(["you:melody,kick", "auto:hat"]) == {"melody": "you", "kick": "you", "hat": "auto"}
    assert parse_layers(None) == {}
    for bad in (["kick"], ["me:kick"], ["you:drums"]):
        with pytest.raises(ValueError):
            parse_layers(bad)


def test_layer_modes_defaults_then_cli_then_bound(fixture_chart):
    c = fixture_chart
    c.default_layers = {"kick": "auto", "melody": "auto"}
    rig = inp(bound={"steer", "brake", "clutch", "handbrake"}, fallback={"gate1", "gate2", "gate3"})
    m = resolve_layer_modes(c, rig)
    assert m["kick"] == "auto"          # chart default
    assert m["melody"] == "you"         # always
    assert m["hat"] == "you" and m["riser"] == "you"
    assert m["expr"] == "auto"          # throttle not served
    m = resolve_layer_modes(c, rig, {"kick": "you", "hat": "auto", "expr": "you"})
    assert m["kick"] == "you" and m["hat"] == "auto"
    assert m["expr"] == "auto"          # --layers you: still needs the control


def test_difficulty_filter(fixture_chart_dict):
    cfg = Config()
    for level, allowed in app_mod.DIFFICULTY_LAYERS.items():
        d = filter_difficulty(fixture_chart_dict, level)
        assert {k for k, v in d["defaults"]["layers"].items() if v == "auto"} >= set(LAYERS) - set(allowed)
        Chart.from_dict(d)  # still valid
    easy = filter_difficulty(fixture_chart_dict, "easy")["notes"]
    assert not [n for n in easy if n["kind"] == "spin" or n.get("listen") or n.get("blind")]
    assert {n["layer"] for n in easy} >= {"hat", "pads", "fills"}   # auto, not deleted: the song plays them
    assert len(load_chart(CHART, cfg, "hard").notes) == len(fixture_chart_dict["notes"])
    normal = load_chart(CHART, cfg, "normal")
    assert {"expr", "fader", "spin"} <= {n.kind for n in normal.notes}
    assert normal.default_layers["expr"] == normal.default_layers["faders"] == "auto"


def test_first_run_gain():
    cfg = Config(ffb_gain=0.5)
    assert resolve_gain(cfg, AppSettings(), None) == 0.2
    assert resolve_gain(Config(ffb_gain=0.1), AppSettings(), None) == 0.1
    assert resolve_gain(cfg, AppSettings(ffb_first_run_done=True), None) == 0.5
    assert resolve_gain(cfg, AppSettings(), 0.7) == 0.7


def test_app_settings_roundtrip(config_dir):
    s = AppSettings.load()
    assert not s.ffb_first_run_done and s.charts_path() == config_dir / "charts"
    s.ffb_first_run_done, s.difficulty = True, "hard"
    s.save()
    assert AppSettings.load() == s
    (config_dir / "app.json").write_text("[1, 2")
    assert AppSettings.load() == AppSettings()


def test_find_songs(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"title": "Alpha"}))
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "chart.json").write_text(json.dumps({"title": "Beta"}))
    (tmp_path / "junk.json").write_text("{")
    songs = find_songs(tmp_path)
    assert [s.title for s in songs] == [app_mod.DEMO_TITLE, "Alpha", "Beta"]
    assert songs[0].path is None
    assert [s.title for s in find_songs(tmp_path / "missing")] == [app_mod.DEMO_TITLE]


def test_click_chart_is_valid():
    c = click_chart(FIXTURES)
    assert {n.kind for n in c.notes} == {"kick"} and len(c.beats) == len(c.notes)


# --- phases ---

def test_song_flow_countdown_play_results_retry_menu():
    a = make_app()
    step(a)
    assert a.phase == "attract" and a.view.drawn[-1][1] is a.menu.chart
    press(a, "menu_ok")
    assert a.phase == "countdown"
    assert ("start", -a.countdown) in a.audio.calls
    assert len(a.companion.songs) == 1
    info, chart_d = a.companion.songs[0]
    assert info.title == a.chart.title and chart_d["notes"]
    step(a, 10)
    assert a.phase == "countdown" and a.snapshot.now < 0
    step(a, int(a.countdown / DT))
    assert a.phase == "play" and a.snapshot.now >= 0
    skip(a, a.chart.length + 1.0)
    assert a.phase == "results" and a.game.finished
    a.audio.calls.clear()
    press(a, "menu_ok")  # retry
    assert a.phase == "countdown"
    starts = [c for c in a.audio.calls if c[0] == "start"]
    assert starts and starts[0][1] <= -MIN_LEAD
    assert len(a.companion.songs) == 2
    assert a.game.score == 0 and a.game.now == 0.0
    step(a, int(a.countdown / DT) + 2)
    assert a.phase == "play"
    press(a, "pause")
    press(a, "menu_back")
    assert a.phase == "attract" and a.game is None and ("stop",) in a.audio.calls


def test_retry_lead_in_is_at_least_15ms():
    a = make_app(countdown=0.0)
    assert a.countdown == MIN_LEAD
    to_play(a)
    skip(a, a.chart.length + 1)
    a.audio.calls.clear()
    press(a, "menu_ok")
    assert [c for c in a.audio.calls if c[0] == "start"] == [("start", -MIN_LEAD)]


def test_game_reset_at_end_of_countdown_derives_turn_offset():
    a = make_app()
    step(a)
    press(a, "menu_ok")
    game = a.game
    a.inp.next = inp(steer_deg=0.0, bound={"steer"})
    step(a, int(a.countdown / DT) - 5)
    assert a.phase == "countdown" and game.now == 0.0  # not updated during the countdown
    a.inp.next = inp(steer_deg=360.0, bound={"steer"})  # the wheel sits one turn right as play starts
    step(a, 10)
    assert a.phase == "play" and a.game is game
    assert game.steer_offset_deg == 360.0


def test_start_from_cli_uses_bound_controls_for_layer_modes():
    a = make_app(start=Song("fixture", CHART))
    a.inp.next = inp(bound={"steer", "brake"})
    step(a)
    assert a.phase == "countdown"
    assert a.game.layer_modes["kick"] == "you" and a.game.layer_modes["hat"] == "auto"
    assert a.game.layer_modes["melody"] == "you"


def test_cli_layers_apply():
    a = make_app(start=Song("fixture", CHART), layers={"kick": "auto"})
    a.inp.next = inp(bound={"steer", "brake"})
    step(a)
    assert a.game.layer_modes["kick"] == "auto"


def test_bad_chart_stays_in_menu(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    a = make_app(songs=[Song("bad", bad)])
    step(a)
    press(a, "menu_ok")
    assert a.phase == "attract" and a.menu.message.startswith("cannot load")
    step(a)
    assert a.menu.chart.artist.startswith("cannot load")


def test_menu_rows_difficulty_and_calibrate(config_dir):
    a = make_app(difficulty="normal")
    step(a)
    press(a, "menu_down")
    assert a.menu.row() == Menu.DIFFICULTY and "normal" in a.menu.chart.title
    press(a, "menu_ok")
    assert a.menu.difficulty == "hard" and "hard" in a.menu.chart.title
    assert json.loads((config_dir / "app.json").read_text())["difficulty"] == "hard"
    press(a, "menu_down")
    assert a.menu.row() == Menu.CALIBRATE
    press(a, "menu_down")
    assert a.menu.cursor == 0
    press(a, "menu_up")
    press(a, "menu_ok")
    assert a.phase == "calibrate" and a.game.is_auto("kick")
    step(a, 30)
    assert a.snapshot.phase == "calibrate" and a.snapshot.beat_phase > 0
    press(a, "trim_plus")
    press(a, "trim_plus")
    assert a.cfg.audio_offset == pytest.approx(0.01)
    assert json.loads((config_dir / "config.json").read_text())["audio_offset"] == pytest.approx(0.01)
    skip(a, app_mod.CLICK_LENGTH + 1)
    assert a.phase == "calibrate" and a.game.now < 1.0  # looped
    press(a, "menu_back")
    assert a.phase == "attract"


def test_difficulty_decides_the_layer_modes():
    a = make_app(difficulty="easy")
    step(a)
    press(a, "menu_ok", bound={"steer", "brake", "clutch", "handbrake", *GATES, "paddle_l", "paddle_r"})
    assert not [n for n in a.chart.notes if n.kind == "spin"]
    assert {k for k, v in a.game.layer_modes.items() if v == "you"} <= {"melody", "kick", "riser"}
    assert a.game.layer_modes["hat"] == a.game.layer_modes["pads"] == "auto"
    assert {n.layer for n in a.chart.notes} >= {"hat", "pads"}     # their notes still play (auto)


# --- safety wiring ---

def test_pause_stops_ffb_and_resumes_only_on_explicit_play():
    a = make_app()
    to_play(a)
    stops, resumes = a.ffb.stops, a.ffb.resumes
    press(a, "pause")
    assert a.phase == "paused" and a.ffb.stops == stops + 1
    assert a.ffb.phases[-1] == "paused" and ("pause",) in a.audio.calls
    t = a.game.now
    step(a, 30)
    assert a.game.now == t and a.ffb.resumes == resumes
    press(a, "pause")
    assert a.phase == "play" and a.ffb.resumes == resumes + 1 and ("resume",) in a.audio.calls


def test_focus_loss_stops_ffb_and_pauses_until_explicit_resume():
    a = make_app()
    to_play(a)
    stops, resumes = a.ffb.stops, a.ffb.resumes
    a.view.is_focused = False
    step(a)
    assert a.phase == "paused" and a.ffb.stops > stops
    step(a, 5)
    press(a, "menu_ok")  # a wheel button while unfocused does not resume
    assert a.phase == "paused" and a.ffb.resumes == resumes
    a.view.is_focused = True
    step(a, 5)
    assert a.phase == "paused" and a.ffb.resumes == resumes
    press(a, "menu_ok")
    assert a.phase == "play" and a.ffb.resumes == resumes + 1


def test_unfocused_at_start_is_stopped_and_song_start_pauses():
    a = make_app(focused=False)
    step(a)
    assert a.ffb.stops == 1
    press(a, "menu_ok")
    assert a.phase == "countdown" and a.ffb.resumes == 0
    step(a)
    assert a.phase == "paused" and a.ffb.resumes == 0


def test_focus_loss_in_menu_only_stops_once():
    a = make_app()
    step(a)
    a.view.is_focused = False
    step(a, 10)
    assert a.ffb.stops == 1 and a.phase == "attract"


def test_ffb_gain_steps_and_persists(config_dir):
    a = make_app(cfg=Config(ffb_gain=0.5))
    step(a)
    press(a, "ffb_up")
    assert a.ffb.gain == pytest.approx(0.25) and a.settings.ffb_first_run_done
    assert json.loads((config_dir / "app.json").read_text())["ffb_first_run_done"] is True
    assert json.loads((config_dir / "config.json").read_text())["ffb_gain"] == pytest.approx(0.25)
    press(a, "ffb_down")
    press(a, "ffb_down")
    assert a.ffb.gain == pytest.approx(0.15)


def test_ffb_down_on_first_run_does_not_mark_done(config_dir):
    a = make_app()
    step(a)
    press(a, "ffb_down")
    assert not a.settings.ffb_first_run_done and a.ffb.gain == pytest.approx(0.15)


def test_volume_steps(config_dir):
    a = make_app()
    step(a)
    press(a, "vol_up")
    assert a.audio.volume == pytest.approx(0.85)
    assert json.loads((config_dir / "config.json").read_text())["master_volume"] == pytest.approx(0.85)


def test_long_frame_flushes_and_ignores_presses_but_not_stop_keys():
    a = make_app()
    step(a)
    assert a.log[:2] == ["flush", "poll"]  # the first frame is treated as long too
    press(a, "menu_ok")
    assert a.phase == "countdown"  # the song load happened inside this frame
    a.log.clear()
    press(a, "menu_ok", "vol_up", dt=0.5, pressed={"brake"})
    assert a.log[:2] == ["flush", "poll"]
    assert a.phase == "countdown" and not a.input.pressed and not a.input.system
    assert a.audio.volume == pytest.approx(0.8)
    a.log.clear()
    press(a, "pause", dt=0.5)  # a stop press during a hitch still stops
    assert a.log[:2] == ["flush", "poll"] and a.phase == "paused"
    press(a, "pause", dt=0.5)  # but a resume press during a hitch does not resume
    assert a.phase == "paused"
    press(a, "menu_back", dt=0.5)  # quitting from pause is a stop, kept
    assert a.phase == "attract"


def test_hitch_during_play_pauses_and_stops_ffb():
    a = make_app()
    to_play(a)
    stops, resumes = a.ffb.stops, a.ffb.resumes
    press(a, "pause", dt=0.4)  # pause held through the hitch: pause once, not pause-then-resume
    assert a.phase == "paused" and a.ffb.stops > stops and a.ffb.resumes == resumes
    assert ("pause",) in a.audio.calls and a.ffb.phases[-1] == "paused"
    step(a, 5)
    assert a.phase == "paused"
    press(a, "pause")
    assert a.phase == "play" and a.ffb.resumes == resumes + 1
    step(a, dt=0.3)
    assert a.phase == "paused"


def test_hitch_in_countdown_or_results_does_not_pause():
    a = make_app()
    step(a)
    press(a, "menu_ok")
    step(a, dt=0.4)
    assert a.phase == "countdown"


def test_paused_screen_says_why_it_cannot_resume():
    a = make_app()
    to_play(a)
    press(a, "pause")
    assert a.view.notice is None
    a.view.is_focused = False
    step(a)
    assert a.view.notice == app_mod.FOCUS_HINT == "Click the game window, then press P or Enter"
    a.view.is_focused = True
    step(a)
    assert a.view.notice is None


def test_frame_order_and_ffb_report_in_snapshot():
    a = make_app()
    to_play(a)
    a.log.clear()
    step(a)
    assert a.log == ["poll", "audio.handle", "audio.update", "ffb.update", "companion.publish", "draw"]
    assert a.snapshot.ffb["torque"] == 0.25 and a.companion.snaps[-1] is a.snapshot
    assert a.ffb.phases[-1] == "play"


def test_spinning_flag_reaches_ffb():
    a = make_app()
    to_play(a)
    spin = next(n for n in a.chart.notes if n.kind == "spin")
    skip(a, spin.t + 0.1 + a.cfg.audio_offset - a.game.now)
    assert a.ffb.spinning[-1] is True


def test_countdown_ffb_phase_is_not_play():
    a = make_app()
    step(a)
    press(a, "menu_ok")
    step(a)
    assert a.ffb.phases[-1] == "countdown"


def test_run_frames_exits_zero_and_closes_in_order():
    a = make_app(start=Song("fixture", CHART))
    assert a.run(120) == 0
    assert a.frames == 120
    closes = [x for x in a.log if x.startswith("close")]
    assert closes == ["close ffb", "close audio", "close companion", "close input", "close window"]


def test_window_close_ends_the_loop():
    a = make_app()
    a.view.close_after = 5
    assert a.run() == 0 and a.frames == 5


@pytest.mark.parametrize("where", ["poll", "audio.handle", "audio.update", "ffb.update",
                                   "companion.publish", "draw", "game"])
def test_exception_in_a_frame_runs_every_close_in_order(where, monkeypatch):
    a = make_app()
    to_play(a)
    a.log.clear()
    obj, name = {"poll": (a.inp, "poll"), "audio.handle": (a.audio, "handle"),
                 "audio.update": (a.audio, "update"), "ffb.update": (a.ffb, "update"),
                 "companion.publish": (a.companion, "publish"), "draw": (a.view, "draw"),
                 "game": (a.game, "update")}[where]

    def boom(*args, **kw):
        raise RuntimeError(where)

    monkeypatch.setattr(obj, name, boom)
    with pytest.raises(RuntimeError, match=where.replace(".", r"\.")):
        a.run(10)
    closes = [x for x in a.log if x.startswith("close")]
    assert closes == ["close ffb", "close audio", "close companion", "close input", "close window"]
    assert a.log.index("ffb.stop_all") < a.log.index("close ffb")


def test_failing_close_does_not_skip_later_closes(monkeypatch):
    a = make_app()

    def boom():
        raise OSError("audio device gone")

    monkeypatch.setattr(a.audio, "stop", boom)
    with pytest.raises(OSError):
        a.run(3)
    closes = [x for x in a.log if x.startswith("close")]
    assert closes == ["close ffb", "close companion", "close input", "close window"]


# --- CLI wiring with fake hardware modules ---

def test_play_help_lists_flags(capsys):
    with pytest.raises(SystemExit):
        main(["play", "--help"])
    out = capsys.readouterr().out
    for flag in ("--demo", "--layers", "--span", "--fullscreen", "--hidden", "--frames", "--no-ffb",
                 "--ffb-gain", "--no-audio", "--kb", "--companion", "--companion-host", "--difficulty"):
        assert flag in out


@pytest.mark.parametrize("argv", [["--layers", "loud:kick"], ["--ffb-gain", "1.5"], ["missing.json"],
                                  [str(CHART), "--demo"]])
def test_bad_arguments_exit_2(argv):
    assert main(["play", *argv]) == 2


def test_play_parses_all_flags():
    args = build_parser().parse_args(
        ["play", str(CHART), "--layers", "you:melody,kick", "auto:hat", "--span", "5760x1080+0+0",
         "--hidden", "--frames", "3", "--no-ffb", "--ffb-gain", "0.3", "--no-audio", "--kb",
         "--companion", "0", "--companion-host", "127.0.0.1", "--difficulty", "hard"])
    assert args.layers == ["you:melody,kick", "auto:hat"] and args.frames == 3 and args.difficulty == "hard"


def fake_modules(rec: dict) -> dict:
    """Fake render, input and ffb modules, so the real `_cmd` wiring runs without
    hardware. Everything they do is appended to rec["log"]."""
    log = rec["log"]

    class Window:
        def __init__(self, span=None, fullscreen=False, hidden=False, settings=None):
            rec["window"] = dict(span=span, fullscreen=fullscreen, hidden=hidden, settings=settings)
            self.hidden, self.rl, self.settings = hidden, None, settings

        def should_close(self):
            return False

        def close(self):
            log.append("close window")

    render = types.ModuleType("torquehero.render")
    render.Window = Window
    render.RaylibKeys = lambda: "keys"
    render.RenderSettings = types.SimpleNamespace(load=lambda: "render.json settings")
    render.parse_span = lambda s: tuple(int(v) for v in s.replace("x", "+").split("+"))

    class Input(FakeInput):
        steer_joystick = "wheel"

    def open_input(cfg, keys, devices=True):
        rec["devices"] = devices
        rec["input"] = Input(log)
        return rec["input"]

    input_mod = types.ModuleType("torquehero.input")
    input_mod.open_input = open_input

    class Engine(FakeFfb):
        def __init__(self, backend, cfg, settings, wheel_range_deg=None):
            if rec.get("engine_error"):
                raise RuntimeError("engine failed")
            super().__init__(log, cfg.ffb_gain)
            rec["ffb"], rec["backend"] = self, backend
            self.patterns: list = []

        def test_pattern(self, name, t, dt, steer_deg, phase):
            self.patterns.append((name, t, steer_deg, phase))
            return {"gain": self.gain, "log": ["PATTERN"]}

    class Backend:
        def __init__(self, joystick=None, reason=""):
            self.joystick, self.reason = joystick, reason

        def close(self):
            log.append("close backend")

    ffb_mod = types.ModuleType("torquehero.ffb")
    ffb_mod.FfbEngine, ffb_mod.NullFfb = Engine, lambda reason="": Backend(reason=reason)
    ffb_mod.FfbSettings = types.SimpleNamespace(
        load=lambda: types.SimpleNamespace(sign_error=rec.get("sign_error"), ffb_sign=1))
    ffb_mod.open_backend = lambda js: Backend(js)
    mods = {}
    for name, mod in (("render", render), ("input", input_mod), ("ffb", ffb_mod)):
        mod.__spec__ = importlib.machinery.ModuleSpec(mod.__name__, None)
        mods[f"torquehero.{name}"] = mod
    return mods


def _view_draw(self, snap, chart, notice=None):
    self.win_log.append("draw")


@pytest.fixture
def fake_hw(monkeypatch):
    rec: dict = {"log": []}
    for name, mod in fake_modules(rec).items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(app_mod.RaylibView, "win_log", rec["log"], raising=False)
    monkeypatch.setattr(app_mod.RaylibView, "draw", _view_draw)
    monkeypatch.setattr(app_mod.RaylibView, "draw_lines", lambda self, lines: rec.setdefault("lines", lines))
    monkeypatch.setattr(app_mod.RaylibView, "focused", lambda self: True)
    return rec


def test_cmd_wiring_hidden_smoke(fake_hw):
    assert main(["play", str(CHART), "--hidden", "--frames", "5", "--no-audio", "--kb",
                 "--no-companion", "--ffb-gain", "0.3"]) == 0
    assert fake_hw["window"]["hidden"] and fake_hw["devices"] is False
    assert fake_hw["window"]["settings"] == "render.json settings"  # play loads render.json
    assert fake_hw["backend"].reason == "--hidden"  # --hidden: no FFB
    assert fake_hw["ffb"].gain == 0.3
    assert len(fake_hw["input"].lost) == 1
    closes = [x for x in fake_hw["log"] if x.startswith("close")]
    assert closes == ["close ffb", "close input", "close window"]
    assert "close backend" not in fake_hw["log"]  # the engine owns it once built


def test_cmd_wiring_rig_first_run_gain_and_steer_lost(fake_hw, monkeypatch):
    monkeypatch.setattr(app_mod, "_steer_range_deg", lambda kb: 900.0)
    assert main(["play", str(CHART), "--frames", "2", "--no-audio", "--no-companion",
                 "--span", "5760x1080+0+0"]) == 0
    assert fake_hw["backend"].joystick == "wheel" and fake_hw["devices"] is True
    assert fake_hw["window"]["span"] == (5760, 1080, 0, 0)
    assert fake_hw["ffb"].gain == 0.2  # first run
    assert fake_hw["ffb"].lost  # input.close fired on_steer_lost -> ffb.device_lost


def test_cmd_unreadable_ffb_sign_runs_without_ffb(fake_hw, caplog):
    fake_hw["sign_error"] = "ffb.json: cannot read ffb_sign: fix the file or delete it"
    assert main(["play", str(CHART), "--frames", "2", "--no-audio", "--no-companion"]) == 0
    assert fake_hw["backend"].reason == fake_hw["sign_error"]
    assert fake_hw["backend"].joystick is None                 # the wheel was never opened
    assert "ffb_sign" not in [r.getMessage().split()[0] for r in caplog.records]


def test_cmd_logs_the_ffb_sign_once(fake_hw, caplog):
    caplog.set_level("INFO")
    assert main(["play", str(CHART), "--frames", "2", "--no-audio", "--no-companion", "--hidden"]) == 0
    assert [r.getMessage() for r in caplog.records].count("ffb_sign +1") == 1


def test_steer_lost_registered_before_anything_else(fake_hw):
    order = []
    orig = FakeInput.on_steer_lost

    def spy(self, cb):
        order.append(len(self.lost))
        return orig(self, cb)

    FakeInput.on_steer_lost = spy
    try:
        main(["play", str(CHART), "--frames", "1", "--no-audio", "--no-companion"])
    finally:
        FakeInput.on_steer_lost = orig
    assert order == [0]


def test_raylib_view_focus_needs_an_event():
    class RL:
        key = 0

        def get_key_pressed(self):
            return self.key

        def is_mouse_button_pressed(self, b):
            return False

    class Win:
        hidden, rl, flag = False, RL(), True

        def focused(self):
            return self.flag

    w = Win()
    v = app_mod.RaylibView(w, Config())
    assert not v.focused()          # reads focused, but no event seen yet
    w.rl.key = 65
    assert v.focused()              # a key reached the window
    w.rl.key = 0
    w.flag = False
    assert not v.focused()
    w.flag = True
    assert v.focused()
    w2 = Win()
    v2 = app_mod.RaylibView(w2, Config())
    w2.flag = False
    assert not v2.focused()
    w2.flag = True
    assert v2.focused()             # false -> true is a focus event
    w3 = Win()
    w3.hidden = True
    assert app_mod.RaylibView(w3, Config()).focused()


def test_app_module_imports_no_hardware():
    assert Path(app_mod.__file__).read_text().count("import pyray") == 0


# --- fix round 2: stops, construction failure, second Ctrl-C, signals, --ffb-test ---

def test_ffb_stopped_at_results_and_calibrate():
    a = make_app()
    to_play(a)
    stops = a.ffb.stops
    skip(a, a.chart.length + 1.0)
    assert a.phase == "results" and a.ffb.stops == stops + 1
    press(a, "menu_back")
    stops = a.ffb.stops
    a.menu.cursor = len(a.menu.rows) - 1
    press(a, "menu_ok")
    assert a.phase == "calibrate" and a.ffb.stops == stops + 1


def test_engine_failure_closes_backend_before_input(fake_hw):
    fake_hw["engine_error"] = True
    with pytest.raises(RuntimeError, match="engine failed"):
        main(["play", str(CHART), "--frames", "1", "--no-audio", "--no-companion"])
    closes = [x for x in fake_hw["log"] if x.startswith("close")]
    assert closes == ["close backend", "close input", "close window"]


def test_second_ctrl_c_in_a_close_does_not_skip_the_rest(monkeypatch):
    a = make_app()

    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(a.audio, "stop", interrupted)
    with pytest.raises(KeyboardInterrupt):
        a.run(3)
    closes = [x for x in a.log if x.startswith("close")]
    assert closes == ["close ffb", "close companion", "close input", "close window"]


def test_close_all_runs_every_close():
    seen = []

    def bad():
        seen.append("bad")
        raise SystemExit(1)

    errors = app_mod.close_all([("a", bad), ("b", lambda: seen.append("b"))])
    assert seen == ["bad", "b"] and isinstance(errors[0], SystemExit)


SIGTERM_SCRIPT = r"""
import sys
sys.path.insert(0, {tests!r})
import test_app
from torquehero import app as app_mod

class Log(list):
    def append(self, x):
        super().append(x)
        print(x, flush=True)

rec = {{"log": Log()}}
sys.modules.update(test_app.fake_modules(rec))
app_mod.RaylibView.win_log = rec["log"]
app_mod.RaylibView.draw = test_app._view_draw
app_mod.RaylibView.focused = lambda self: True
from torquehero.__main__ import main
sys.exit(main(["play", {chart!r}, "--no-audio", "--no-companion"]))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_sigterm_mid_play_stops_ffb_and_closes_in_order(tmp_path):
    import os
    import signal
    import subprocess

    script = tmp_path / "run.py"
    script.write_text(SIGTERM_SCRIPT.format(tests=str(Path(__file__).parent), chart=str(CHART)))
    env = {**os.environ, "TORQUEHERO_CONFIG_DIR": str(tmp_path / "cfg")}
    p = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env)
    lines = []
    for line in p.stdout:
        lines.append(line.strip())
        if lines.count("draw") >= 20:
            break
    p.send_signal(signal.SIGTERM)
    out, err = p.communicate(timeout=10)
    lines += out.splitlines()
    assert p.returncode == 128 + signal.SIGTERM, err
    closes = [x for x in lines if x.startswith("close")]
    assert closes == ["close ffb", "close input", "close window"]
    last_stop = len(lines) - 1 - lines[::-1].index("ffb.stop_all")
    assert last_stop < lines.index("close ffb")


def test_signal_handlers_install_and_restore():
    import signal
    import threading

    before = signal.getsignal(signal.SIGTERM)
    restore = app_mod.install_signal_handlers(lambda: None, threading.Event())
    assert signal.getsignal(signal.SIGTERM) is app_mod._raise_exit
    with pytest.raises(SystemExit) as e:
        app_mod._raise_exit(signal.SIGTERM, None)
    assert e.value.code == 128 + signal.SIGTERM
    restore()
    assert signal.getsignal(signal.SIGTERM) is before


class PatternFfb(FakeFfb):
    def __init__(self, log):
        super().__init__(log)
        self.patterns: list = []

    def test_pattern(self, name, t, dt, steer_deg, phase):
        self.log.append("ffb.test_pattern")
        self.patterns.append((name, t, steer_deg, phase))
        return {"gain": self.gain, "gain_now": self.gain, "log": ["PATTERN"], **getattr(self, "report", {})}


class LinesViewFake(FakeView):
    def draw_lines(self, lines):
        self.log.append("draw")
        self.drawn.append(("lines", lines))


def make_test(focused=True, name="centre", **kw):
    log: list = []
    clock = Clock()
    t = app_mod.FfbTest(name, inp=FakeInput(log), ffb=PatternFfb(log), view=LinesViewFake(log, focused),
                        clock=clock, **kw)
    t.log, t.fake_clock = log, clock
    return t


def test_ffb_test_starts_stopped_then_runs_pattern_with_wheel_angle():
    t = make_test()
    step(t)
    assert t.phase == "paused" and t.ffb.patterns[-1][3] == "paused"
    press(t, "pause")
    assert t.phase == "play" and t.ffb.resumes == 1
    t.inp.next = inp(steer_deg=42.0, bound={"steer"})
    step(t, 30)
    name, pt, deg, phase = t.ffb.patterns[-1]
    assert name == "centre" and deg == 42.0 and pt == pytest.approx(31 * DT) and phase == "play"
    lines = t.view.drawn[-1][1]
    assert lines[0] == "FFB TEST: CENTRE" and "+42 deg" in lines[1] and "GAIN 20%" in lines[2]
    assert "PATTERN" in lines


def test_ffb_test_gain_keys_step_and_persist_like_play(config_dir):
    t = make_test(save_gain=True)       # started with --ffb-gain
    step(t)
    press(t, "ffb_up")
    assert t.ffb.gain == pytest.approx(0.25) and t.settings.ffb_first_run_done
    assert json.loads((config_dir / "config.json").read_text())["ffb_gain"] == pytest.approx(0.25)
    assert json.loads((config_dir / "app.json").read_text())["ffb_first_run_done"] is True
    press(t, "ffb_down")
    press(t, "ffb_down")
    assert t.ffb.gain == pytest.approx(0.15)
    assert "GAIN 15%" in t.view.drawn[-1][1][2] and "5% per press" in t.view.drawn[-1][1][2]


def test_ffb_test_shows_the_frame_rate_and_the_rate_floor():
    t = make_test()
    step(t)
    assert t.view.drawn[-1][1][4] == "FPS measuring"
    t.ffb.report = {"fps": 143.6, "springs": None}
    step(t)
    assert t.view.drawn[-1][1][4] == "FPS 144"
    t.ffb.report = {"fps": 72.0, "springs": "reduced"}
    step(t)
    assert t.view.drawn[-1][1][4] == "FPS 72   SPRINGS reduced"
    t.ffb.report = {"fps": 72.0, "springs": "off", "latched": True}
    step(t)
    assert t.view.drawn[-1][1][4] == "FPS 72   SPRINGS off   LATCHED"
    t.ffb.report = {"fps": 143.6, "fps_worst": 71.8, "springs": "reduced", "sign": -1}
    step(t)
    assert t.view.drawn[-1][1][4] == "FPS 144 (worst 72)   SPRINGS reduced"
    assert t.view.drawn[-1][1][1].endswith("ffb_sign -1")


def test_ffb_test_without_a_gain_flag_keeps_the_tuned_gain(config_dir, caplog):
    config_dir.mkdir(parents=True)
    Config(ffb_gain=0.35).save()
    t = make_test(cfg=Config.load())    # save_gain False: started without --ffb-gain
    step(t)
    with caplog.at_level("INFO"):
        press(t, "ffb_up")
        press(t, "ffb_up")
    assert t.ffb.gain == pytest.approx(0.3)
    assert Config.load().ffb_gain == pytest.approx(0.35) and not (config_dir / "app.json").exists()
    assert len([r for r in caplog.records if "tuned gain 0.35 is kept" in r.message]) == 1
    assert "not saved" in t.view.drawn[-1][1][2]


def test_cmd_ffb_test_without_gain_flag_never_exceeds_the_tuned_gain(config_dir, fake_hw):
    config_dir.mkdir(parents=True)
    Config(ffb_gain=0.15).save()
    AppSettings(ffb_first_run_done=True).save()
    main(["play", "--ffb-test", "kick", "--frames", "1"])
    assert fake_hw["ffb"].gain == pytest.approx(0.15)
    Config(ffb_gain=0.6).save()
    main(["play", "--ffb-test", "kick", "--frames", "1"])
    assert fake_hw["ffb"].gain == pytest.approx(0.2)


def test_cmd_ffb_test_saves_gain_only_with_the_gain_flag(fake_hw, monkeypatch):
    made = []

    class Recorded(app_mod.FfbTest):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self.save_gain)

    monkeypatch.setattr(app_mod, "FfbTest", Recorded)
    main(["play", "--ffb-test", "kick", "--frames", "1"])
    main(["play", "--ffb-test", "kick", "--ffb-gain", "0.2", "--frames", "1"])
    assert made == [False, True]


@pytest.mark.parametrize("content", ["{not json", '{"ffb_gain": 0.5, "play_range_deg": "wide"}', '{"ffb_gain": 15}'])
def test_config_fallback_caps_the_gain_even_after_first_run(config_dir, fake_hw, content):
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(content)
    AppSettings(ffb_first_run_done=True).save()
    assert main(["play", str(CHART), "--hidden", "--frames", "2", "--no-audio", "--kb", "--no-companion"]) == 0
    assert fake_hw["ffb"].gain <= 0.2


def test_config_fallback_keeps_valid_keys_and_backs_up_before_saving(config_dir):
    config_dir.mkdir(parents=True)
    bad = '{"ffb_gain": 0.15, "master_volume": 0.6, "play_range_deg": "wide"}'
    (config_dir / "config.json").write_text(bad)
    cfg = app_mod.load_config()
    assert (cfg.ffb_gain, cfg.master_volume, cfg.play_range_deg) == (0.15, 0.6, Config().play_range_deg)
    cfg.save()
    assert (config_dir / "config.json.bad").read_text() == bad
    assert Config.load().ffb_gain == 0.15
    (config_dir / "config.json.bad").unlink()
    cfg.save()                                          # only before the first save
    assert not (config_dir / "config.json.bad").exists()


def test_app_passes_the_calibrated_wheel_range_to_the_game():
    a = make_app(wheel_range_deg=900.0)
    to_play(a)
    assert a.game.spin_lock_deg == 445.0
    assert make_app().wheel_range_deg is None


def test_shared_rules_have_one_source():
    from torquehero import audio, chart, generate, state

    assert audio.ALLOWED_MODES is chart.ALLOWED_MODES
    assert app_mod.DIFFICULTY_LAYERS is generate.DIFFICULTY_LAYERS is state.DIFFICULTY_LAYERS
    assert app_mod.DIFFICULTIES is generate.DIFFICULTIES is state.DIFFICULTIES


def test_ffb_test_pause_focus_loss_and_quit():
    t = make_test()
    step(t)
    press(t, "menu_ok")
    step(t, 5)
    stops = t.ffb.stops
    press(t, "pause")
    assert t.phase == "paused" and t.ffb.stops == stops + 1
    frozen = t.t
    step(t, 10)
    assert t.t == frozen and {p[3] for p in t.ffb.patterns[-10:]} == {"paused"}
    press(t, "pause")
    assert t.phase == "play"
    t.view.is_focused = False
    step(t)
    assert t.phase == "paused" and t.ffb.stops == stops + 2
    press(t, "pause")  # unfocused: no resume
    assert t.phase == "paused" and app_mod.FOCUS_HINT in t.view.drawn[-1][1][3]
    press(t, "menu_back")
    assert t.quit


def test_ffb_test_hitch_stops_the_pattern():
    t = make_test()
    step(t)
    press(t, "pause")
    step(t, 3)
    stops = t.ffb.stops
    step(t, dt=0.5)
    assert t.phase == "paused" and t.ffb.stops == stops + 1 and t.ffb.patterns[-1][3] == "paused"
    press(t, "pause", dt=0.5)  # resume is not kept through a hitch
    assert t.phase == "paused"
    press(t, "menu_back", dt=0.5)  # quit is
    assert t.quit


def test_signal_after_construction_still_closes_in_order(fake_hw, monkeypatch):
    def interrupted(self, frames=None):
        raise SystemExit(143)

    monkeypatch.setattr(app_mod.App, "run", interrupted)
    with pytest.raises(SystemExit):
        main(["play", str(CHART), "--no-audio", "--no-companion"])
    closes = [x for x in fake_hw["log"] if x.startswith("close")]
    assert closes == ["close ffb", "close input", "close window"]


def test_sighup_is_handled_like_sigterm():
    import signal
    import threading

    if not hasattr(signal, "SIGHUP"):
        pytest.skip("no SIGHUP")
    before = signal.getsignal(signal.SIGHUP)
    restore = app_mod.install_signal_handlers(lambda: None, threading.Event())
    assert signal.getsignal(signal.SIGHUP) is app_mod._raise_exit
    restore()
    assert signal.getsignal(signal.SIGHUP) is before


def test_ffb_test_exception_stops_and_closes_in_order(monkeypatch):
    t = make_test()
    step(t)
    press(t, "pause")

    def boom(*a, **k):
        raise RuntimeError("pattern")

    monkeypatch.setattr(t.ffb, "test_pattern", boom)
    with pytest.raises(RuntimeError):
        t.run(5)
    closes = [x for x in t.log if x.startswith("close")]
    assert closes == ["close ffb", "close input", "close window"]
    assert t.log.index("ffb.stop_all", len(t.log) - 5) < t.log.index("close ffb")


def test_cmd_ffb_test_wiring(fake_hw):
    assert main(["play", "--ffb-test", "kick", "--frames", "3"]) == 0
    assert fake_hw["ffb"].gain == 0.2 and fake_hw["devices"] is True
    assert fake_hw["backend"].joystick == "wheel" and len(fake_hw["input"].lost) == 1
    assert fake_hw["lines"][0] == "FFB TEST: KICK"
    closes = [x for x in fake_hw["log"] if x.startswith("close")]
    assert closes == ["close ffb", "close input", "close window"]  # no audio, no companion


def test_cmd_ffb_test_gain_flag(fake_hw):
    assert main(["play", "--ffb-test", "echo", "--ffb-gain", "0.3", "--frames", "1"]) == 0
    assert fake_hw["ffb"].gain == 0.3


@pytest.mark.parametrize("argv", [["--ffb-test", "centre", "--demo"], [str(CHART), "--ffb-test", "kick"]])
def test_cmd_ffb_test_bad_arguments(fake_hw, argv):
    assert main(["play", *argv]) == 2


def test_cmd_ffb_test_patterns_come_from_the_engine(fake_hw):
    FfbEngine = app_mod.FfbEngine  # the real engine; fake_hw only replaces what _play imports
    assert "riser" in FfbEngine.PATTERNS
    for name in FfbEngine.PATTERNS:
        assert main(["play", "--ffb-test", name, "--frames", "1"]) == 0
    with pytest.raises(SystemExit):
        main(["play", "--ffb-test", "spin"])


# --- the real modules: every attribute app.py uses exists with a compatible signature ---

def _callable_with(obj, *args, **kw) -> None:
    import inspect

    inspect.signature(obj).bind(*args, **kw)  # TypeError if the call app.py makes would not bind


def test_real_module_api_matches_what_app_uses():
    import dataclasses

    from torquehero import audio, bindings, companion, ffb, input, render, synthsong

    _callable_with(render.Window, span=None, fullscreen=False, hidden=True, settings=render.RenderSettings())
    for name in ("focused", "should_close", "close", "frame"):
        assert callable(getattr(render.Window, name))
    assert isinstance(render.Window.layout, property)
    _callable_with(render.Window.frame, None, lambda: None)
    _callable_with(render.Renderer, None, Config(), render.RenderSettings())
    _callable_with(render.Renderer.draw, None, None, None)
    assert callable(render.Renderer.close)
    _callable_with(render.RaylibKeys)
    assert render.parse_span("5760x1080+0+0") == (5760, 1080, 0, 0)
    _callable_with(render.Painter.text, None, "notice", 0.0, 0.0, 10.0, render.COL["amber"], "center")
    lay = render.Layout(1920, 1080)
    assert all(isinstance(getattr(lay, k), int | float) for k in ("cx", "h", "unit"))

    _callable_with(input.open_input, Config(), None, devices=False)
    for cls in (input.SdlInput, input.CompositeInput, input.KeyboardMouseInput):
        _callable_with(cls.poll, None, 0.0)
        _callable_with(cls.flush, None)
        _callable_with(cls.close, None)
    for cls in (input.SdlInput, input.CompositeInput):
        assert isinstance(cls.steer_joystick, property)
        _callable_with(cls.on_steer_lost, None, lambda h: None)
    assert "range_deg" in {f.name for f in dataclasses.fields(bindings.Binding)}
    _callable_with(bindings.Bindings.load)
    _callable_with(bindings.Bindings.get, None, "steer")

    E = ffb.FfbEngine
    _callable_with(E, None, Config(), None, wheel_range_deg=None)
    _callable_with(E.update, None, [], Snapshot(), 0.0, spinning=False)
    _callable_with(E.set_gain, None, 0.2)
    _callable_with(E.test_pattern, None, "centre", 0.0, 0.0, 0.0, "play")
    for name in ("__enter__", "__exit__", "stop_all", "resume", "close", "device_lost"):
        assert callable(getattr(E, name))
    assert isinstance(E.gain, property) and set(app_mod.FfbEngine.PATTERNS) >= {"centre", "kick", "echo", "riser"}
    _callable_with(ffb.NullFfb, reason="--no-ffb")
    _callable_with(ffb.open_backend, None)
    _callable_with(ffb.FfbSettings.load)
    assert app_mod.spin_active is ffb.spin_active

    for cls in (audio.AudioEngine, audio.NullAudio):
        _callable_with(cls, audio.AudioSettings(), volume=0.8)
        for name, args in (("load", (None, {})), ("start", (-0.015,)), ("pause", ()), ("resume", ()),
                           ("stop", ()), ("time", ()), ("handle", ([],)), ("update", (Snapshot(),))):
            _callable_with(getattr(cls, name), None, *args)
    assert isinstance(audio.AudioEngine.volume, property) and audio.AudioEngine.volume.fset
    _callable_with(audio.AudioSettings.load)

    _callable_with(companion.start_companion, None, None)
    for cls in (companion.CompanionServer, companion.NullCompanion):
        _callable_with(cls.publish, None, Snapshot())
        _callable_with(cls.publish_song, None, {}, {})
        _callable_with(cls.close, None)
    _callable_with(synthsong.demo_chart_path)


class Keys:
    """state.KeySource over a set of held keys; `pressed` is true once per tap."""

    def __init__(self):
        self.down: set[str] = set()
        self.taps: set[str] = set()

    def is_down(self, key):
        return key in self.down or key in self.taps

    def pressed(self, key):
        if key in self.taps:
            self.taps.discard(key)
            return True
        return False

    def mouse_x_norm(self):
        return 0.5


def test_song_runs_on_the_real_engine_input_and_audio_modules():
    from torquehero.ffb import FfbEngine, NullFfb
    from torquehero.input import KeyboardMouseInput

    log: list = []
    clock = Clock()
    cfg = Config()
    keys = Keys()
    backend = NullFfb(reason="test", clock=clock)
    engine = FfbEngine(backend, cfg, clock=clock)
    a = App(cfg, inp=KeyboardMouseInput(keys, cfg), audio=NullAudio(clock=clock), ffb=engine,
            companion=FakeCompanion(log), view=FakeView(log), songs=[Song("fixture", CHART)],
            clock=clock, persist=False)
    a.fake_clock = clock
    step(a, 2)
    keys.taps.add("ENTER")
    step(a)
    assert a.phase == "countdown"
    step(a, int(a.countdown / DT) + 2)
    assert a.phase == "play" and not a.snapshot.ffb["latched"]
    for _ in range(int(a.chart.length / DT) + 30):
        step(a)
        if a.phase == "results":
            break
    assert a.phase == "results" and a.snapshot.ffb["latched"]
    assert any(c.method == "run" for c in backend.calls)  # effects ran during play
    keys.taps.add("BACKSPACE")
    step(a)
    assert a.phase == "attract"
    a.shutdown()
    assert backend.closed


def test_ffb_test_autostart_for_hidden_runs(fake_hw):
    assert main(["play", "--ffb-test", "echo", "--hidden", "--frames", "3"]) == 0
    assert [p[3] for p in fake_hw["ffb"].patterns] == ["play"] * 3
    assert fake_hw["backend"].reason == "--hidden"
