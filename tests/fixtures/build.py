"""Builds the shared fixtures: `chart.json` and `snapshot.json`.

Run `uv run python tests/fixtures/build.py` after a contract change;
tests/test_fixtures.py fails when the committed files drift from this output.
"""
from __future__ import annotations

import json
from pathlib import Path

from torquehero.chart import Chart
from torquehero.config import Config
from torquehero.game import Game
from torquehero.state import InputState

HERE = Path(__file__).parent
BPM = 120.0
BEAT = 60.0 / BPM


def audio_dict() -> dict:
    """The full SPEC 8.2 manifest the fixture chart uses; tests reuse it."""
    oneshots = {k: f"oneshots/{k}.wav" for k in
                ("kick", "hat", "tom_l", "tom_r", "riser", "impact", "dud", *(f"stab{i}" for i in range(1, 7)))}
    return {
        "sr": 48000,
        "backing": ["stems/backing.wav"],
        "stems": {k: f"stems/{k}.wav" for k in ("lead", "pad", "bass", "drums")},
        "oneshots": oneshots,
        "layers": {
            "melody": {"stem": "lead", "mode": "filter"},
            "expr": {"stem": "pad", "mode": "level"},
            "faders": {"stem": "bass", "mode": "levers"},
            "kick": {"oneshot": "kick", "mode": "trigger"},
            "hat": {"oneshot": "hat", "mode": "trigger"},
            "pads": {"oneshot": "stab{gate}", "mode": "trigger"},
            "fills": {"oneshot": "tom_{side}", "mode": "trigger"},
            "riser": {"oneshot": "riser", "mode": "riser"},
        },
    }


def chart_dict() -> dict:
    gates = [(1.0, 0.0), (2.0, 0.5), (3.0, -0.5)]
    listen = [(8.0, 0.0), (8.5, 0.3), (9.0, 0.0)]
    blind = [(10.0, 0.0), (10.5, 0.3), (11.0, 0.0)]
    notes = [{"t": t, "kind": "gate", "layer": "melody", "x": x} for t, x in gates]
    notes += [
        {"t": 4.0, "kind": "kick", "layer": "kick", "vel": 1.0},
        {"t": 4.5, "kind": "hat", "layer": "hat"},
        {"t": 5.0, "kind": "kick", "layer": "kick", "vel": 0.8},
        {"t": 5.5, "kind": "hat", "layer": "hat", "open": True},
        {"t": 6.0, "kind": "expr", "layer": "expr", "dur": 2.0, "curve": [[0.0, 0.2], [1.0, 0.8], [2.0, 0.5]]},
    ]
    notes += [{"t": t, "kind": "gate", "layer": "melody", "x": x, "listen": True} for t, x in listen]
    notes += [{"t": t, "kind": "gate", "layer": "melody", "x": x, "blind": True} for t, x in blind]
    notes += [
        {"t": 12.0, "kind": "stab", "layer": "pads", "gate": 1},
        {"t": 12.0, "kind": "fader", "layer": "faders", "dur": 2.0, "lever": 0, "curve": [[0.0, 0.0], [2.0, 1.0]]},
        {"t": 13.0, "kind": "stab", "layer": "pads", "gate": 3},
        {"t": 14.5, "kind": "fader", "layer": "faders", "dur": 1.0, "lever": 1, "curve": [[0.0, 0.5], [1.0, 0.5]]},
        {"t": 16.0, "kind": "tom", "layer": "fills", "side": "L"},
        {"t": 16.5, "kind": "tom", "layer": "fills", "side": "R"},
        {"t": 17.5, "kind": "spin", "layer": "melody", "dur": 1.5, "dir": "cw"},
        {"t": 19.5, "kind": "spin", "layer": "melody", "dur": 1.5, "dir": "ccw"},
        {"t": 21.5, "kind": "riser", "layer": "riser", "dur": 2.0},
    ]
    notes.sort(key=lambda n: n["t"])
    road = [{"t": 0.0, "x": 0.0}] + [{"t": t, "x": x} for t, x in gates + listen + blind]
    return {
        "format": 1, "title": "Fixture", "artist": "Torque Hero tests", "bpm": BPM, "length": 24.0,
        "sections": [
            {"t": 0.0, "name": "intro", "weight": 0.2},
            {"t": 4.0, "name": "verse", "weight": 0.6},
            {"t": 8.0, "name": "echo", "weight": 0.3},
            {"t": 12.0, "name": "chorus", "weight": 0.9},
        ],
        "beats": [{"t": i * BEAT, "s": 1.0 if i % 4 == 0 else 0.5} for i in range(48)],
        "road": road,
        "notes": notes,
        "audio": audio_dict(),
    }


def perfect_input(chart: Chart, t: float, prev: set[str], hz: float = 60.0) -> InputState:
    """What a perfect player does at `t`: on the road, pedals pressed on their notes."""
    down = set()
    for n in chart.notes:
        if n.kind in ("kick", "hat") and abs(t - n.t) < 0.5 / hz:
            down.add("brake" if n.kind == "kick" else "clutch")
    x = chart.road_x(t)
    expr = next((n.value_at(t) for n in chart.notes if n.kind == "expr" and n.t <= t <= n.end), 0.0)
    return InputState(
        steer=x, steer_deg=x * 90.0, throttle=expr,
        brake=1.0 if "brake" in down else 0.0, clutch=1.0 if "clutch" in down else 0.0,
        pressed=frozenset(down - prev), released=frozenset(prev - down),
        bound=frozenset({"steer", "brake", "clutch"}), fallback=frozenset({"throttle"}),
    )


def snapshot_dict(chart: Chart, until: float = 8.3, hz: float = 60.0) -> dict:
    modes = {"melody": "you", "kick": "you", "hat": "you", "expr": "you"}
    game = Game(chart, Config(), modes)
    prev: set[str] = set()
    for f in range(int(until * hz) + 1):
        t = f / hz
        inp = perfect_input(chart, t, prev, hz)
        prev = {c for c in ("brake", "clutch") if inp.value(c) >= 0.5}
        game.update(t, 1 / hz, inp)
    snap = game.snapshot()
    snap.ffb = {"torque": 0.35, "log": ["ECHO spring centre -> 0 deg", "WEIGHT 30%"]}
    return json.loads(snap.to_json())


def main() -> None:
    d = chart_dict()
    (HERE / "chart.json").write_text(json.dumps(d, indent=1) + "\n")
    chart = Chart.load(HERE / "chart.json")
    (HERE / "snapshot.json").write_text(json.dumps(snapshot_dict(chart), indent=1) + "\n")


if __name__ == "__main__":
    main()
