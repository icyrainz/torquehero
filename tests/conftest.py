from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from torquehero.chart import Chart
from torquehero.config import Config
from torquehero.game import Game
from torquehero.state import InputState

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_chart_path() -> Path:
    return FIXTURES / "chart.json"


@pytest.fixture
def fixture_chart_dict() -> dict:
    return json.loads((FIXTURES / "chart.json").read_text())


@pytest.fixture
def fixture_chart(fixture_chart_path) -> Chart:
    return Chart.load(fixture_chart_path)


def make_chart(notes: list[dict], length: float = 10.0, **extra) -> Chart:
    """A minimal valid chart around `notes` (layer filled in from kind), with the
    fixture's full audio manifest plus any note samples."""
    from fixtures.build import audio_dict

    from torquehero.chart import KIND_LAYER

    audio = audio_dict()
    audio["oneshots"].update({n["sample"]: f"oneshots/{n['sample']}.wav" for n in notes if "sample" in n})
    d = {
        "format": 1, "title": "t", "bpm": 120.0, "length": length,
        "sections": [], "beats": [], "road": [],
        "notes": [{"layer": KIND_LAYER[n["kind"]], **n} for n in sorted(notes, key=lambda n: n["t"])],
        "audio": audio,
    }
    d.update(copy.deepcopy(extra))
    return Chart.from_dict(d)


def inp(**kw) -> InputState:
    """InputState with list/set arguments turned into frozensets."""
    for k in ("down", "pressed", "released", "system", "bound", "fallback"):
        if k in kw:
            kw[k] = frozenset(kw[k])
    return InputState(**kw)


def game_for(notes: list[dict], modes: dict | None = None, cfg: Config | None = None, **extra) -> Game:
    chart = make_chart(notes, **extra)
    layers = {n["kind"] for n in notes}
    from torquehero.chart import KIND_LAYER

    return Game(chart, cfg or Config(), modes or {KIND_LAYER[k]: "you" for k in layers})


def run(game: Game, t0: float, t1: float, fn=lambda t: inp(), hz: float = 100.0) -> list:
    """Step the game from t0 to t1 at `hz`, input from `fn(t)`. Returns all events."""
    events = []
    n = round((t1 - t0) * hz)
    for f in range(n + 1):
        t = t0 + f / hz
        events += game.update(t, 1 / hz, fn(t))
    return events


def kinds(events, kind: str) -> list:
    return [e for e in events if e.kind == kind]
