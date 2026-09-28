"""The app loop (SPEC 4): `torquehero play`.

Every frame: input poll -> game.update(audio clock) -> audio.handle(events) and
audio.update(snapshot) -> ffb.update(events, snapshot, dt) -> companion.publish(snapshot)
-> render. `App` owns the phases (attract, countdown, play, paused, calibrate,
results) and the safety wiring of SPEC 5; the hardware modules sit behind small
interfaces so tests drive the loop with fakes. Heavy modules (pyray, sdl2,
sounddevice) are imported inside `_cmd` only.
"""
from __future__ import annotations

import json
import logging
import math
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from .app_settings import AppSettings
from .chart import Chart, ChartError
from .config import Config, settings_fallback
from .ffb import FfbEngine, rate_text, spin_active
from .game import Game
from .state import DIFFICULTIES, DIFFICULTY_LAYERS, LAYERS, InputState, Snapshot, default_layer_modes

log = logging.getLogger(__name__)

COUNTDOWN = 3.0          # seconds of lead-in before a song, counted on the song clock
MIN_LEAD = 0.015         # SPEC 10: a restart on a playing engine leads in at least this much
LONG_FRAME = 0.25        # a longer frame flushes queued input and ignores the next presses
GAIN_STEP = 0.05         # ffb_up / ffb_down
VOL_STEP = 0.05          # vol_up / vol_down
TRIM_STEP = 0.005        # trim_plus / trim_minus, seconds of audio offset
FIRST_RUN_GAIN = 0.2     # SPEC 5 rule 12: FFB gain until the player raises it once
CLICK_BPM = 120.0
CLICK_LENGTH = 32.0      # the calibration click track loops after this many seconds
DEMO_TITLE = "Torque Hero Demo"
FOCUS_HINT = "Click the game window, then press P or Enter"
CONSOLE_CLOSE_WAIT = 4.0  # seconds a Windows console close waits for the shutdown


# --- songs and charts ---

@dataclass(frozen=True)
class Song:
    """A menu entry. `path` None is the built-in demo song, rendered on first use."""

    title: str
    path: Path | None = None

    def chart_path(self) -> Path:
        if self.path is not None:
            return self.path
        from .synthsong import demo_chart_path

        return demo_chart_path()


def _chart_title(path: Path) -> str | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    title = d.get("title") if isinstance(d, dict) else None
    return title if isinstance(title, str) and title else None


def find_songs(folder: Path | None) -> list[Song]:
    """The demo song, then every chart in `folder` (`*.json` and `*/chart.json`) that has a title."""
    songs = [Song(DEMO_TITLE)]
    if folder is not None and folder.is_dir():
        for p in sorted({*folder.glob("*.json"), *folder.glob("*/chart.json")}):
            if (title := _chart_title(p)) is not None:
                songs.append(Song(title, p))
    return songs


def filter_difficulty(d: dict, difficulty: str) -> dict:
    """The chart dict as `difficulty` plays it (SPEC 10). Layers the difficulty does not
    give the player become auto in defaults.layers, so the song still plays their notes
    (--layers can still override). Easy also drops spins and echo notes."""
    keep = DIFFICULTY_LAYERS[difficulty]

    def ok(n) -> bool:
        if not isinstance(n, dict):
            return False
        return difficulty != "easy" or not (n.get("kind") == "spin" or n.get("listen") or n.get("blind"))

    out = {**d, "notes": [n for n in d.get("notes", []) if ok(n)]}
    defaults = d.get("defaults", {})
    if isinstance(defaults, dict) and isinstance(defaults.get("layers", {}), dict):
        layers = {**defaults.get("layers", {}), **{k: "auto" for k in LAYERS if k not in keep}}
        out["defaults"] = {**defaults, "layers": layers}
    return out


def load_chart(path: str | Path, cfg: Config, difficulty: str = "hard") -> Chart:
    path = Path(path)
    try:
        d = json.loads(path.read_text())
    except ValueError as e:
        raise ChartError([f"not valid JSON: {e}"], str(path)) from e
    if not isinstance(d, dict):
        raise ChartError(["not a chart object"], str(path))
    return Chart.from_dict(filter_difficulty(d, difficulty), path.parent.resolve(),
                           play_range_deg=cfg.play_range_deg, echo_max_deg_s=cfg.echo_max_deg_s)


def click_chart(base_dir: Path, bpm: float = CLICK_BPM, length: float = CLICK_LENGTH) -> Chart:
    """Calibration click track: an auto kick on every beat, accented on the bar."""
    beat = 60.0 / bpm
    n = int(length / beat)
    d = {
        "format": 1, "title": "Calibrate", "bpm": bpm, "length": length,
        "sections": [{"t": 0.0, "name": "calibrate", "weight": 0.0}],
        "beats": [{"t": i * beat, "s": 1.0 if i % 4 == 0 else 0.6} for i in range(n)],
        "road": [{"t": 0.0, "x": 0.0}],
        "notes": [{"t": i * beat, "kind": "kick", "layer": "kick", "vel": 1.0 if i % 4 == 0 else 0.7}
                  for i in range(n)],
        "audio": {"sr": 48000, "backing": [], "stems": {}, "oneshots": {"kick": "oneshots/kick.wav"},
                  "layers": {"kick": {"oneshot": "kick", "mode": "trigger"}}},
    }
    return Chart.from_dict(d, base_dir)


# --- layer modes and gain ---

def parse_layers(tokens: Iterable[str] | None) -> dict[str, str]:
    """`["you:melody,kick", "auto:hat"]` -> {layer: mode}. ValueError on a bad token."""
    out: dict[str, str] = {}
    for tok in tokens or ():
        mode, sep, names = tok.partition(":")
        if not sep or mode not in ("you", "auto"):
            raise ValueError(f"--layers: {tok!r} is not you:LAYER,... or auto:LAYER,...")
        for name in filter(None, names.split(",")):
            if name not in LAYERS:
                raise ValueError(f"--layers: unknown layer {name!r} (layers: {', '.join(LAYERS)})")
            out[name] = mode
    return out


def resolve_layer_modes(chart: Chart, inp: InputState, cli: dict[str, str] | None = None) -> dict[str, str]:
    """Chart defaults, then --layers, then what the controls actually served allow.
    Melody is always played by the player."""
    wanted = {**chart.default_layers, **(cli or {})}
    modes = default_layer_modes(inp.bound, inp.fallback, chart.controls_used(), wanted)
    modes["melody"] = "you"
    return dict(modes)


def load_config(path: str | Path | None = None) -> Config:
    """Config.load. After any fallback (a bad file or a bad key) the saved gain cannot be
    trusted against the first-run limit: the gain is at most FIRST_RUN_GAIN this session,
    and in any save made from it."""
    cfg = Config.load(path)
    if settings_fallback(cfg):
        cfg.ffb_gain = min(cfg.ffb_gain, FIRST_RUN_GAIN)
        log.warning("config.json fell back: force feedback gain %.2f this session", cfg.ffb_gain)
    return cfg


def resolve_gain(cfg: Config, settings: AppSettings, flag: float | None) -> float:
    """--ffb-gain wins; before the player first raises the gain, at most FIRST_RUN_GAIN."""
    if flag is not None:
        return min(1.0, max(0.0, flag))
    return cfg.ffb_gain if settings.ffb_first_run_done else min(cfg.ffb_gain, FIRST_RUN_GAIN)


# --- the modules the loop drives ---

class InputLike(Protocol):
    def poll(self, dt: float) -> InputState: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class AudioLike(Protocol):
    @property
    def volume(self) -> float: ...
    @volume.setter
    def volume(self, v: float) -> None: ...

    def load(self, chart: Chart, layer_modes: dict[str, str]) -> None: ...
    def start(self, at: float = 0.0) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...
    def time(self) -> float: ...
    def handle(self, events: list) -> None: ...
    def update(self, snap: Snapshot) -> None: ...


class FfbLike(Protocol):
    gain: float

    def __enter__(self): ...
    def __exit__(self, *exc) -> None: ...
    def update(self, events, snap: Snapshot, dt: float, spinning: bool = False) -> dict: ...
    def set_gain(self, gain: float, quiet: bool = False) -> float: ...
    def stop_all(self) -> None: ...
    def resume(self) -> None: ...
    def close(self) -> None: ...


class CompanionLike(Protocol):
    def publish(self, snapshot) -> None: ...
    def publish_song(self, info, chart) -> None: ...
    def close(self) -> None: ...


class View(Protocol):
    def focused(self) -> bool: ...
    def should_close(self) -> bool: ...
    def draw(self, snap: Snapshot, chart: Chart, notice: str | None = None) -> None: ...
    def close(self) -> None: ...


# --- menu ---

class Menu:
    """Attract menu rows: every song, then difficulty, then calibrate. The renderer
    draws the attract screen from a chart title, so the menu shows itself through
    a placeholder chart whose title and artist describe the selected row."""

    DIFFICULTY, CALIBRATE = "difficulty", "calibrate"

    def __init__(self, songs: list[Song], difficulty: str):
        self.songs, self.difficulty, self.cursor = songs, difficulty, 0
        self.message = ""
        self.chart = Chart(title="", bpm=120.0, length=1.0)
        self.refresh(0.0)

    @property
    def rows(self) -> list[Song | str]:
        return [*self.songs, self.DIFFICULTY, self.CALIBRATE]

    def row(self) -> Song | str:
        return self.rows[self.cursor]

    def move(self, d: int) -> None:
        self.cursor = (self.cursor + d) % len(self.rows)
        self.message = ""

    def cycle_difficulty(self) -> None:
        self.difficulty = DIFFICULTIES[(DIFFICULTIES.index(self.difficulty) + 1) % len(DIFFICULTIES)]

    def refresh(self, audio_offset: float) -> None:
        row, c = self.row(), self.chart
        if row == self.DIFFICULTY:
            c.title, c.artist = f"Difficulty: {self.difficulty}", "OK to change"
        elif row == self.CALIBRATE:
            c.title, c.artist = "Calibrate audio offset", f"offset {audio_offset * 1000:+.0f} ms"
        else:
            i = self.songs.index(row) + 1  # type: ignore[arg-type]
            c.title = row.title  # type: ignore[union-attr]
            c.artist = f"{i}/{len(self.songs)} · {self.difficulty} · up/down to choose"
        if self.message:
            c.artist = self.message


# --- the loop ---

def close_all(closers: Iterable[tuple[str, Callable[[], object]]]) -> list[BaseException]:
    """Run every close in order. A close that raises, even KeyboardInterrupt or
    SystemExit (a second Ctrl-C), is logged and the rest still run."""
    errors: list[BaseException] = []
    for name, close in closers:
        try:
            close()
        except BaseException as e:
            log.error("closing %s: %r", name, e)
            errors.append(e)
    return errors


class _Loop:
    """Frame loop shared by the game and the FFB test: the FFB engine wraps the run,
    any exception stops FFB first, then `closers()` run in order even when the frame
    or an earlier close raised."""

    inp: InputLike
    ffb: FfbLike
    view: View
    clock: Callable[[], float]
    cfg: Config
    settings: AppSettings
    persist: bool
    save_gain: bool = True    # ffb_up / ffb_down save the gain (the FFB test: only with --ffb-gain)

    def _init_loop(self) -> None:
        self.quit = False
        self.shut_down = False
        self.frames = 0
        self.input = InputState()
        self._last: float | None = None
        self._t0: float | None = None
        self._focused: bool | None = None

    def closers(self) -> list[tuple[str, Callable[[], object]]]:
        raise NotImplementedError

    def frame(self) -> None:
        raise NotImplementedError

    def run(self, frames: int | None = None) -> int:
        exc: BaseException | None = None
        try:
            with self.ffb:
                while not self.quit and (frames is None or self.frames < frames) and not self.view.should_close():
                    try:
                        self.frame()
                    except BaseException:
                        self.ffb.stop_all()
                        raise
                    self.frames += 1
        except BaseException as e:
            exc = e
            raise
        finally:
            errors = self.shutdown()
            if errors and exc is None:
                raise errors[0]
        return 0

    def shutdown(self) -> list[BaseException]:
        self.shut_down = True
        return close_all(self.closers())

    def stop_keys(self) -> frozenset[str]:
        """System controls that stop something in the current phase; they survive a
        stale poll, because a hitch is exactly when the player hits stop."""
        return frozenset()

    def _poll(self) -> tuple[float, float, InputState, bool]:
        """(now, dt, input, hitch). After a long frame (`hitch`) or on the first frame,
        queued input is flushed and this poll's presses are ignored, except `stop_keys()`."""
        t = self.clock()
        if self._t0 is None:
            self._t0 = t
        first = self._last is None
        dt = 0.0 if first else t - self._last
        hitch = not first and dt > LONG_FRAME
        self._last = t
        if first or hitch:
            self.inp.flush()
        inp = self.inp.poll(dt)
        if first or hitch:  # presses queued during a long frame (song load) are not the player's
            inp = replace(inp, pressed=frozenset(), system=inp.system & self.stop_keys(), velocity={})
        self.input = inp
        return t, dt, inp, hitch

    def _step_gain(self, sys_: frozenset[str]) -> None:
        """ffb_up / ffb_down: GAIN_STEP per press, saved in the config; the first raise
        ends the first-run gain (SPEC 5 rule 12)."""
        if "ffb_up" not in sys_ and "ffb_down" not in sys_:
            return
        up = "ffb_up" in sys_
        g = self.ffb.set_gain(round(self.ffb.gain + (GAIN_STEP if up else -GAIN_STEP), 3))
        if not self.save_gain:
            if not getattr(self, "_gain_kept_logged", False):
                self._gain_kept_logged = True
                log.info("gain %.2f for this test only: the tuned gain %.2f is kept (start the test with "
                         "--ffb-gain to save changes)", g, self.cfg.ffb_gain)
            return
        self.cfg.ffb_gain = g
        if up and not self.settings.ffb_first_run_done:
            self.settings.ffb_first_run_done = True
            self._save_settings()
        self._save_cfg()

    def _save_cfg(self) -> None:
        if self.persist:
            try:
                self.cfg.save()
            except OSError as e:
                log.warning("cannot save the config: %s", e)

    def _save_settings(self) -> None:
        if self.persist:
            try:
                self.settings.save()
            except OSError as e:
                log.warning("cannot save app settings: %s", e)

    def _check_focus(self) -> bool:
        """Stop FFB once per focus loss (latched until an explicit resume)."""
        focused = bool(self.view.focused())
        if not focused and self._focused is not False:
            log.info("window not focused: force feedback stopped until you resume")
            self.ffb.stop_all()
        self._focused = focused
        return focused


class App(_Loop):
    """One game session. `run(frames)` owns the frame loop and the shutdown order:
    FFB, audio, companion, input (fires on_steer_lost), window."""

    def __init__(self, cfg: Config, *, inp: InputLike, audio: AudioLike, ffb: FfbLike,
                 companion: CompanionLike, view: View, songs: list[Song],
                 settings: AppSettings | None = None, difficulty: str | None = None,
                 layers: dict[str, str] | None = None, start: Song | None = None,
                 countdown: float = COUNTDOWN, clock: Callable[[], float] = time.perf_counter,
                 click_dir: Callable[[], Path] | None = None, persist: bool = True,
                 wheel_range_deg: float | None = None):
        self.cfg, self.inp, self.audio, self.ffb = cfg, inp, audio, ffb
        self.wheel_range_deg = wheel_range_deg   # calibrated steer range (spin lock limit)
        self.companion, self.view = companion, view
        self.settings = settings or AppSettings()
        self.menu = Menu(songs, difficulty or self.settings.difficulty)
        self.layers = dict(layers or {})
        self.countdown = max(countdown, MIN_LEAD)
        self.clock = clock
        self.click_dir = click_dir or (lambda: Song(DEMO_TITLE).chart_path().parent)
        self.persist = persist
        self.phase = "attract"
        self.game: Game | None = None
        self.chart: Chart | None = None
        self.song: Song | None = None
        self.snapshot = Snapshot()
        self._init_loop()
        self._pending = start
        self._resume_to = "play"
        self._song_now = 0.0

    def closers(self) -> list[tuple[str, Callable[[], object]]]:
        return [("ffb", self.ffb.close), ("audio", self.audio.stop), ("companion", self.companion.close),
                ("input", self.inp.close), ("window", self.view.close)]

    # --- one frame ---

    def stop_keys(self) -> frozenset[str]:
        if self.phase in ("countdown", "play"):
            return frozenset({"pause"})
        return frozenset({"menu_back"}) if self.phase in ("paused", "results") else frozenset()

    def frame(self) -> None:
        t, dt, inp, hitch = self._poll()
        if hitch and self.phase == "play":  # the game hung: stop until the player resumes
            log.info("frame took %.2f s: paused", dt)
            self._pause()
            inp = self.input = replace(inp, system=inp.system - {"pause"})
        if not self._check_focus() and self.phase in ("countdown", "play"):
            self._pause()

        if self._pending is not None:
            song, self._pending = self._pending, None
            self.start_song(song)
        self._system(inp.system)

        events = self._advance(dt, inp)
        snap = self._build_snapshot(inp, t - self._t0)
        if self.game is not None:
            self.audio.handle(events)
            self.audio.update(snap)
        spinning = self.chart is not None and spin_active(snap, self.chart)
        snap.ffb = self.ffb.update(events, snap, dt, spinning=spinning)
        self.companion.publish(snap)
        self.snapshot = snap
        notice = FOCUS_HINT if self.phase == "paused" and not self._focused else None
        self.view.draw(snap, self.chart if self.game is not None and self.chart else self.menu.chart, notice)

    def _advance(self, dt: float, inp: InputState) -> list:
        game = self.game
        if game is None:
            return []
        if self.phase == "countdown":
            self._song_now = self.audio.time()
            if self._song_now < 0.0:
                return []
            game.reset()  # SPEC 10: the turn offset comes from the wheel as play starts
            self.phase = "play"
        if self.phase == "play":
            self._song_now = self.audio.time()
            events = game.update(self._song_now, dt, inp)
            if game.finished:
                self.ffb.stop_all()
                self.phase = "results"
                log.info("results: score %d, accuracy %.0f%%", game.score, game.accuracy * 100)
            return events
        if self.phase == "calibrate":
            self._song_now = self.audio.time()
            events = game.update(self._song_now, dt, inp)
            if game.finished:  # loop the click track
                game.reset()
                self.audio.start(-MIN_LEAD)
            return events
        return []

    def _build_snapshot(self, inp: InputState, uptime: float) -> Snapshot:
        if self.game is None:
            self.menu.refresh(self.cfg.audio_offset)
            return Snapshot(phase="attract", now=uptime, input=inp)
        snap = self.game.snapshot()
        snap.phase = self.phase  # type: ignore[assignment]
        if self.phase in ("countdown", "paused", "results", "calibrate"):
            snap.input = inp
        if self.phase == "countdown" or (self.phase == "paused" and self._resume_to == "countdown"):
            snap.now = self._song_now
        return snap

    # --- system controls ---

    def _system(self, sys_: frozenset[str]) -> None:
        if "vol_up" in sys_ or "vol_down" in sys_:
            v = self.audio.volume + (VOL_STEP if "vol_up" in sys_ else -VOL_STEP)
            self.audio.volume = self.cfg.master_volume = round(min(1.0, max(0.0, v)), 3)
            self._save_cfg()
        self._step_gain(sys_)
        if self.phase in ("play", "paused", "calibrate") and ("trim_plus" in sys_ or "trim_minus" in sys_):
            step = TRIM_STEP if "trim_plus" in sys_ else -TRIM_STEP
            self.cfg.audio_offset = round(self.cfg.audio_offset + step, 4)
            log.info("audio offset %+.0f ms", self.cfg.audio_offset * 1000)
            self._save_cfg()

        match self.phase:
            case "attract":
                self._menu(sys_)
            case "countdown" | "play":
                if "pause" in sys_:
                    self._pause()
            case "paused":
                if "menu_back" in sys_:
                    self.to_menu()
                elif "pause" in sys_ or "menu_ok" in sys_:
                    self._unpause()
            case "results":
                if "menu_ok" in sys_:
                    self.retry()
                elif "menu_back" in sys_:
                    self.to_menu()
            case "calibrate":
                if sys_ & {"menu_back", "menu_ok", "pause"}:
                    self.to_menu()

    def _menu(self, sys_: frozenset[str]) -> None:
        m = self.menu
        if "menu_up" in sys_:
            m.move(-1)
        if "menu_down" in sys_:
            m.move(1)
        if "menu_ok" not in sys_:
            return
        row = m.row()
        if row == Menu.DIFFICULTY:
            m.cycle_difficulty()
            self.settings.difficulty = m.difficulty
            self._save_settings()
        elif row == Menu.CALIBRATE:
            self.start_calibrate()
        else:
            self.start_song(row)  # type: ignore[arg-type]

    # --- phase changes ---

    def start_song(self, song: Song) -> bool:
        try:
            chart = load_chart(song.chart_path(), self.cfg, self.menu.difficulty)
        except (ChartError, OSError) as e:
            log.error("cannot load %s: %s", song.title, e)
            self.menu.message = f"cannot load: {str(e).splitlines()[0][:60]}"
            return False
        modes = resolve_layer_modes(chart, self.input, self.layers)
        self.song, self.chart = song, chart
        self.game = Game(chart, self.cfg, modes, self.wheel_range_deg)
        self.audio.load(chart, modes)
        log.info("song %r (%s): %s", chart.title, self.menu.difficulty,
                 " ".join(f"{k}={v}" for k, v in modes.items()))
        self._begin()
        return True

    def retry(self) -> None:
        if self.game is None:
            return
        self.game.reset()
        self._begin()

    def _begin(self) -> None:
        """Count in on the song clock: the song starts at -countdown (at least MIN_LEAD)."""
        game = self.game
        assert game is not None and self.chart is not None
        self.companion.publish_song(game.song_info(), self.chart.to_dict())
        self.audio.start(-self.countdown)
        self._song_now = -self.countdown
        self.phase = "countdown"
        if self._focused:
            self.ffb.resume()

    def start_calibrate(self) -> None:
        try:
            chart = click_chart(self.click_dir())
        except (ChartError, OSError) as e:
            log.error("cannot build the click track: %s", e)
            self.menu.message = "cannot build the click track"
            return
        modes = {k: "auto" for k in LAYERS}
        modes["melody"] = "you"
        self.song, self.chart = None, chart
        self.game = Game(chart, self.cfg, modes, self.wheel_range_deg)
        self.audio.load(chart, modes)
        self.ffb.stop_all()
        self.audio.start(-MIN_LEAD)
        self.phase = "calibrate"

    def _pause(self) -> None:
        self.ffb.stop_all()
        self.audio.pause()
        if self.phase in ("countdown", "play"):
            self._resume_to = self.phase
        self.phase = "paused"

    def _unpause(self) -> None:
        """Resume only on an explicit play, and only with the window focused."""
        if not self._focused:
            log.info("click the game window to give it focus, then resume")
            return
        self.ffb.resume()
        self.audio.resume()
        self.phase = self._resume_to

    def to_menu(self) -> None:
        self.ffb.stop_all()
        self.audio.stop()
        self.game = self.chart = self.song = None
        self.phase = "attract"


# --- FFB test patterns ---

class LinesView(Protocol):
    def focused(self) -> bool: ...
    def should_close(self) -> bool: ...
    def draw_lines(self, lines: list[str]) -> None: ...
    def close(self) -> None: ...


class FfbTest(_Loop):
    """`play --ffb-test NAME`: input and FFB only, no song and no audio. Drives
    `FfbEngine.test_pattern` each frame with the live wheel angle. Pause, focus
    loss, exceptions and exit stop it exactly like play; it starts paused until the
    player starts it with a focused window (`autostart` for hidden smoke runs, which
    have no FFB device). BACK while paused quits. ffb_up / ffb_down change the gain
    as in play; they save it only when `save_gain` (the test started with --ffb-gain),
    so a test at the default 0.2 never overwrites a tuned gain."""

    def __init__(self, name: str, *, inp: InputLike, ffb, view: LinesView,
                 clock: Callable[[], float] = time.perf_counter, autostart: bool = False,
                 cfg: Config | None = None, settings: AppSettings | None = None, persist: bool = True,
                 save_gain: bool = False):
        self.name, self.inp, self.ffb, self.view, self.clock = name, inp, ffb, view, clock
        self.cfg, self.settings, self.persist = cfg or Config(), settings or AppSettings(), persist
        self.save_gain = save_gain
        self.phase = "play" if autostart else "paused"
        self.t = 0.0          # pattern time; frozen while paused
        self.report: dict = {}
        self._init_loop()

    def closers(self) -> list[tuple[str, Callable[[], object]]]:
        return [("ffb", self.ffb.close), ("input", self.inp.close), ("window", self.view.close)]

    def stop_keys(self) -> frozenset[str]:
        return frozenset({"pause"}) if self.phase == "play" else frozenset({"menu_back"})

    def frame(self) -> None:
        _, dt, inp, hitch = self._poll()
        focused = self._check_focus()
        sys_ = inp.system
        self._step_gain(sys_)
        if self.phase == "play":
            if hitch or "pause" in sys_:
                self.ffb.stop_all()
                self.phase = "paused"
            elif not focused:  # _check_focus stopped FFB
                self.phase = "paused"
        elif "menu_back" in sys_:
            self.quit = True
        elif sys_ & {"pause", "menu_ok"} and focused:
            self.ffb.resume()
            self.phase = "play"
        if self.phase == "play":
            self.t += dt
        self.report = self.ffb.test_pattern(self.name, self.t, dt, inp.steer_deg, self.phase)
        self.view.draw_lines(self.lines(inp, focused))

    def lines(self, inp: InputState, focused: bool) -> list[str]:
        rep = self.report
        wheel = "wheel" if "steer" in inp.bound else "keyboard/mouse (no steer binding)"
        if self.phase == "play":
            state = "RUNNING  -  ESC or P to stop"
        elif focused:
            state = "STOPPED  -  P or ENTER to start, BACKSPACE to quit"
        else:
            state = f"STOPPED  -  {FOCUS_HINT}"
        sign = f"   ffb_sign {rep['sign']:+d}" if rep.get("sign") else ""
        return [f"FFB TEST: {self.name.upper()}",
                f"WHEEL {inp.steer_deg:+.0f} deg   ({wheel}){sign}",
                f"GAIN {rep.get('gain', self.ffb.gain) * 100:.0f}%   now {rep.get('gain_now', 0.0) * 100:.0f}%"
                f"   ([ ] or ffb_up/ffb_down: {GAIN_STEP * 100:.0f}% per press"
                f"{'' if self.save_gain else ', not saved'})",
                state,
                rate_text(rep),
                *([str(rep["status"]).upper()] if rep.get("status") else []),
                *[str(m) for m in rep.get("log", [])[:6]]]


# --- raylib view ---

class RaylibView:
    """Window plus a Renderer per chart. Focus: a window counts as focused only after
    a focus event has been seen (the focus flag going from false to true, or keyboard
    or mouse input reaching the window), because raylib reports a window that opened
    without focus as focused. A hidden window is never on the desktop and counts as
    focused; the app runs it without force feedback."""

    def __init__(self, window, cfg: Config):
        self.win, self.cfg = window, cfg
        self.rl = window.rl
        self._renderer = None
        self._chart: Chart | None = None
        self._seen = False
        self._prev: bool | None = None

    def focused(self) -> bool:
        if self.win.hidden:
            return True
        f = bool(self.win.focused())
        if f and not self._seen and (self._prev is False or self._input_reached()):
            self._seen = True
        self._prev = f
        return f and self._seen

    def _input_reached(self) -> bool:
        rl = self.rl
        return bool(rl.get_key_pressed()) or any(rl.is_mouse_button_pressed(b) for b in (0, 1, 2))

    def should_close(self) -> bool:
        return self.win.should_close()

    def draw(self, snap: Snapshot, chart: Chart, notice: str | None = None) -> None:
        """The renderer's frame, plus `notice` under the phase overlay (e.g. why a
        pause cannot be resumed)."""
        from .render import COL, Renderer

        if chart is not self._chart or self._renderer is None:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer, self._chart = Renderer(chart, self.cfg, getattr(self.win, "settings", None)), chart
        r, lay = self._renderer, self.win.layout

        def draw() -> None:
            r.draw(snap, lay)
            if notice:
                r.paint.text(notice.upper(), lay.cx, lay.h * 0.36 + lay.unit * 8.4, lay.unit * 0.9,
                             COL["amber"], "center")

        self.win.frame(draw)

    def draw_lines(self, lines: list[str]) -> None:
        """Plain text on a dark screen (the FFB test)."""
        rl = self.rl

        def draw() -> None:
            rl.clear_background(rl.Color(12, 10, 28, 255))
            y = 40
            for i, text in enumerate(lines):
                size = 64 if i == 0 else 32
                rl.draw_text(text, 40, y, size, rl.RAYWHITE if i < 5 else rl.GRAY)
                y += size + 16

        self.win.frame(draw)

    def close(self) -> None:
        try:
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
        finally:
            self.win.close()


# --- CLI ---

def add_cli(sub) -> None:
    p = sub.add_parser("play", help="play a chart on the rig (or keyboard and mouse)")
    p.add_argument("chart", nargs="?", metavar="CHART", help="chart.json to play (default: the menu)")
    p.add_argument("--demo", action="store_true", help="play the built-in demo song")
    p.add_argument("--layers", nargs="+", metavar="MODE:LAYERS",
                   help="layer modes, e.g. you:melody,kick auto:hat")
    p.add_argument("--span", metavar="WxH+X+Y", help="borderless window at this size and position (triples)")
    p.add_argument("--fullscreen", action="store_true", help="borderless fullscreen on the current monitor")
    p.add_argument("--hidden", action="store_true", help="render off screen (smoke tests); no force feedback")
    p.add_argument("--frames", type=int, metavar="N", help="exit 0 after N frames")
    p.add_argument("--no-ffb", action="store_true", help="no force feedback")
    p.add_argument("--ffb-gain", type=float, metavar="G", help="force feedback gain 0..1 (first run: 0.2)")
    p.add_argument("--ffb-test", choices=FfbEngine.PATTERNS, metavar="PATTERN",
                   help=f"first-run check: play one FFB pattern ({', '.join(FfbEngine.PATTERNS)}), "
                        "no song, no audio (docs/FFB-CHECKLIST.md)")
    p.add_argument("--no-audio", action="store_true", help="no sound; the song clock is a timer")
    p.add_argument("--kb", action="store_true", help="keyboard and mouse only, ignore the rig")
    p.add_argument("--companion", type=int, metavar="PORT", help="companion pages port, this run only")
    p.add_argument("--companion-host", metavar="H", help="companion pages bind address (saved setting)")
    p.add_argument("--no-companion", action="store_true", help="do not serve the companion pages")
    p.add_argument("--difficulty", choices=DIFFICULTIES, help="easy, normal or hard (default: last used)")
    p.set_defaults(func=_cmd)


def _steer_range_deg(kb: bool) -> float | None:
    if kb:
        return None
    try:
        from .bindings import Bindings

        b = Bindings.load().get("steer")
    except Exception:
        return None
    return getattr(b, "range_deg", None) if b else None


def _raise_exit(signum, frame) -> None:
    raise SystemExit(128 + signum)


def install_signal_handlers(request_quit: Callable[[], None], done: threading.Event) -> Callable[[], None]:
    """SIGTERM, SIGHUP (macOS, Linux) and SIGBREAK (Windows) raise SystemExit in the main thread, so the
    normal exception path runs: FFB stop, then the ordered shutdown. Closing the
    Windows console gives the process a few seconds: the handler asks the loop to
    quit and waits for `done`. Returns a function that restores the old handlers."""
    old = {}
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
            if (sig := getattr(signal, name, None)) is not None:
                old[sig] = signal.signal(sig, _raise_exit)
    console = _win_console_handler(request_quit, done) if sys.platform == "win32" else None

    def restore() -> None:
        for sig, h in old.items():
            signal.signal(sig, h)
        if console is not None:
            import ctypes

            ctypes.windll.kernel32.SetConsoleCtrlHandler(console, False)  # type: ignore[attr-defined]

    return restore


def _win_console_handler(request_quit: Callable[[], None], done: threading.Event):
    import ctypes
    from ctypes import wintypes

    close_events = (2, 5, 6)  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)  # type: ignore[attr-defined]
    def handler(event: int) -> bool:
        if event not in close_events:
            return False
        request_quit()
        done.wait(CONSOLE_CLOSE_WAIT)  # Windows ends the process when this returns
        return True

    ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, True)  # type: ignore[attr-defined]
    return handler


def _cmd(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    holder: list[_Loop] = []
    done = threading.Event()
    restore = install_signal_handlers(lambda: [setattr(x, "quit", True) for x in holder], done)
    try:
        return _play(args, holder)
    except KeyboardInterrupt:
        return 130
    finally:
        done.set()
        restore()


def _play(args, holder: list) -> int:
    cfg = load_config()
    settings = AppSettings.load()
    if settings.difficulty not in DIFFICULTIES:
        settings.difficulty = "normal"
    difficulty = args.difficulty or settings.difficulty
    test = args.ffb_test
    try:
        layers = parse_layers(args.layers)
        if args.ffb_gain is not None and not (0.0 <= args.ffb_gain <= 1.0 and math.isfinite(args.ffb_gain)):
            raise ValueError("--ffb-gain must be between 0 and 1")
        if args.frames is not None and args.frames < 0:
            raise ValueError("--frames must be 0 or more")
        if args.chart and args.demo:
            raise ValueError("give CHART or --demo, not both")
        if test and (args.chart or args.demo):
            raise ValueError("--ffb-test plays a pattern, not a song: drop CHART and --demo")
        start = None
        if args.chart and not test:
            path = Path(args.chart)
            load_chart(path, cfg, difficulty)  # fail before any window opens
            start = Song(_chart_title(path) or path.stem, path)
        elif args.demo and not test:
            start = Song(DEMO_TITLE)
            start.chart_path()  # render the demo song now, not inside a frame
        from .render import RaylibKeys, RenderSettings, Window, parse_span

        span = parse_span(args.span) if args.span else None
    except (ValueError, ChartError, OSError) as e:
        print(f"play: {e}", file=sys.stderr)
        return 2

    from .ffb import FfbEngine, FfbSettings, NullFfb, open_backend
    from .input import open_input

    opened: list = []  # closed in reverse if construction fails
    loop: _Loop | None = None
    try:
        window = Window(span=span, fullscreen=args.fullscreen, hidden=args.hidden, settings=RenderSettings.load())
        view = RaylibView(window, cfg)
        opened.append(view)
        inp = open_input(cfg, RaylibKeys(), devices=not args.kb)
        opened.append(inp)
        ffb_settings = FfbSettings.load()
        if ffb_settings.sign_error:   # never guess the sign: run without force feedback
            log.warning("%s: no force feedback", ffb_settings.sign_error)
            backend = NullFfb(reason=ffb_settings.sign_error)
        elif args.no_ffb or args.hidden:
            backend = NullFfb(reason="--hidden" if args.hidden and not args.no_ffb else "--no-ffb")
        else:
            backend = open_backend(getattr(inp, "steer_joystick", None))
        opened.append(backend)  # the haptic device closes before the joystick if the engine fails
        wheel_range = _steer_range_deg(args.kb)
        ffb = FfbEngine(backend, cfg, ffb_settings, wheel_range_deg=wheel_range)
        opened.append(ffb)
        if not ffb_settings.sign_error:
            log.info("ffb_sign %+d", ffb_settings.ffb_sign)
        if hasattr(inp, "on_steer_lost"):  # first registration: FFB lets go before the wheel closes
            inp.on_steer_lost(lambda _handle: ffb.device_lost())
        if test:
            # without --ffb-gain: 0.2 at most, and never above the gain the player tuned
            ffb.set_gain(min(resolve_gain(cfg, settings, None), FIRST_RUN_GAIN) if args.ffb_gain is None
                         else args.ffb_gain)
            loop = FfbTest(test, inp=inp, ffb=ffb, view=view, autostart=args.hidden, cfg=cfg, settings=settings,
                           save_gain=args.ffb_gain is not None)
        else:
            ffb.set_gain(resolve_gain(cfg, settings, args.ffb_gain))
            loop = _game(args, cfg, settings, difficulty, layers, start, inp, ffb, view, opened, wheel_range)
        holder.append(loop)
        return loop.run(args.frames)
    except BaseException:
        if loop is None:
            close_all((type(obj).__name__, obj.close) for obj in reversed(opened))
        elif not loop.shut_down:  # a signal before run() took over: the ordered shutdown
            loop.shutdown()
        raise


def _game(args, cfg, settings, difficulty, layers, start, inp, ffb, view, opened, wheel_range=None) -> App:
    from .audio import AudioEngine, AudioSettings, NullAudio
    from .companion import NullCompanion, start_companion

    if args.no_audio:
        audio = NullAudio(volume=cfg.master_volume)
    else:
        audio = AudioEngine(AudioSettings.load(), volume=cfg.master_volume)
    companion = NullCompanion() if args.no_companion else start_companion(args.companion_host, args.companion)
    opened.append(companion)
    return App(cfg, inp=inp, audio=audio, ffb=ffb, companion=companion, view=view,
               songs=find_songs(settings.charts_path()), settings=settings, difficulty=difficulty,
               layers=layers, start=start, countdown=MIN_LEAD if args.hidden else COUNTDOWN,
               wheel_range_deg=wheel_range)
