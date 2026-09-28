"""`torquehero companion --replay FIXTURE`: serve the pages without the game.

Starts from the sample snapshot (layer modes, bound controls, FFB log) and, when
the fixture chart sits next to it, drives the pure `Game` on that chart with a
scripted player on a looping clock, so every panel moves. Without a chart, only
the snapshot's clock and beat advance.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

from ..chart import Chart
from ..config import Config
from ..game import Game
from ..state import GATES, InputState

HZ = 30.0
MISS_EVERY = 6  # the scripted player skips every 6th note so lamps go dark sometimes


def default_fixture() -> Path:
    """A copy of tests/fixtures shipped as package data, so an installed wheel can replay."""
    return Path(__file__).resolve().parent / "sample" / "snapshot.json"


class Replay:
    def __init__(self, snapshot: dict, chart: Chart | None = None):
        self.base = snapshot
        self.chart = chart
        self.length = float(chart.length if chart else snapshot.get("length") or 20.0)
        inp = snapshot.get("input", {})
        self.bound = frozenset(inp.get("bound", ()))
        self.fallback = frozenset(inp.get("fallback", ()))
        self.game = None
        if chart:
            modes = {k: v["mode"] for k, v in snapshot.get("layers", {}).items()}
            self.game = Game(chart, Config(), modes)
        self._prev_down: frozenset[str] = frozenset()
        self._last_t = -1.0

    def song(self) -> tuple[dict, dict] | None:
        if not self.game or not self.chart:
            return None
        return self.game.song_info().to_dict(), self.chart.to_dict()

    def _input(self, t: float, dt: float) -> InputState:
        chart = self.chart
        assert chart is not None
        steer = chart.road_x(t)
        vals = {"throttle": 0.0, "lever0": 0.3 + 0.1 * math.sin(t * 0.7), "lever1": 0.6 + 0.1 * math.sin(t * 0.5)}
        down: set[str] = set()
        spin_deg = 0.0
        for i, n in enumerate(chart.notes):
            if i % MISS_EVERY == MISS_EVERY - 1 or n.t - 0.02 > t:
                continue
            if n.kind == "spin" and n.dur:  # a completed turn stays on the wheel
                spin_deg += (-360.0 if n.dir == "ccw" else 360.0) * min(1.0, (t - n.t) / (0.9 * n.dur))
                continue
            if t > n.end + 0.06:
                continue
            if n.kind == "kick":
                down.add("brake")
            elif n.kind == "hat":
                down.add("clutch")
            elif n.kind == "stab" and n.gate:
                down.add(GATES[n.gate - 1])
            elif n.kind == "tom":
                down.add("paddle_l" if n.side == "L" else "paddle_r")
            elif n.kind == "riser" and t < n.end + 0.03:
                down.add("handbrake")
            elif n.kind == "expr" and t <= n.end:
                vals["throttle"] = n.value_at(t) + 0.04 * math.sin(t * 9)
            elif n.kind == "fader" and n.lever in (0, 1) and t <= n.end:
                vals[f"lever{n.lever}"] = n.value_at(t) + 0.05 * math.sin(t * 7 + n.lever)
        if self.game and self.game.echo_at(t) == "listen":
            steer = self.game.echo_target_at(t) or steer
        for c in ("brake", "clutch", "handbrake"):
            vals[c] = 1.0 if c in down else 0.0
        for c in ("lever0", "lever1"):
            if vals[c] >= 0.5:
                down.add(c)
        frozen = frozenset(down)
        pressed, released = frozen - self._prev_down, self._prev_down - frozen
        self._prev_down = frozen
        return InputState(
            steer=steer, steer_deg=steer * 90.0 + spin_deg, **vals,
            down=frozen, pressed=pressed, released=released,
            bound=self.bound, fallback=self.fallback,
        )

    def frame(self, elapsed: float) -> dict:
        t = elapsed % self.length
        dt = 1.0 / HZ if self._last_t < 0 or t < self._last_t else t - self._last_t
        if t < self._last_t:
            self._restart()
        self._last_t = t
        ffb = copy.deepcopy(self.base.get("ffb") or {"torque": 0.0, "log": []})
        if not self.game:
            d = copy.deepcopy(self.base)
            beat = 60.0 / (self.base.get("bpm") or 120.0)
            d.update(phase="play", now=t, beat_phase=(t % beat) / beat)
            ffb["torque"] = 0.35 * math.sin(t * 2.0)
            d["ffb"] = ffb
            return d
        self.game.update(t, dt, self._input(t, dt))
        snap = self.game.snapshot()
        snap.phase = "play"
        ffb["torque"] = max(-1.0, min(1.0, snap.weight * (1.0 - snap.beat_phase) ** 2 * math.copysign(1, math.sin(t))
                                      + 0.3 * (snap.echo_target or 0.0)))
        snap.ffb = ffb
        d = json.loads(snap.to_json())
        return d

    def _restart(self) -> None:
        if self.game:
            self.game.reset()
        self._prev_down = frozenset()


def load(fixture: str | Path, chart: str | Path | None = None) -> Replay:
    fixture = Path(fixture)
    snap = json.loads(fixture.read_text())
    chart_path = Path(chart) if chart else fixture.with_name("chart.json")
    return Replay(snap, Chart.load(chart_path) if chart_path.exists() else None)


def run(server, replay: Replay, seconds: float | None = None) -> None:
    """Publish the replay to `server` at HZ until `seconds` pass (forever when None)."""
    song = replay.song()
    if song:
        server.publish_song(*song)
    t0 = time.monotonic()
    while True:
        elapsed = time.monotonic() - t0
        if seconds is not None and elapsed >= seconds:
            return
        server.publish(replay.frame(elapsed))
        time.sleep(1.0 / HZ)
