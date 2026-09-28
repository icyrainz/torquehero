import json
from dataclasses import fields
from pathlib import Path

from torquehero.chart import Chart
from torquehero.state import Snapshot

FIXTURES = Path(__file__).parent / "fixtures"


def test_committed_fixtures_match_builder():
    from fixtures.build import chart_dict, snapshot_dict

    assert json.loads((FIXTURES / "chart.json").read_text()) == chart_dict()
    chart = Chart.load(FIXTURES / "chart.json")
    assert json.loads((FIXTURES / "snapshot.json").read_text()) == snapshot_dict(chart)


def test_sample_snapshot_has_every_field():
    d = json.loads((FIXTURES / "snapshot.json").read_text())
    assert set(d) == {f.name for f in fields(Snapshot)}
    assert d["echo"] == "listen" and d["score"] > 0 and d["ffb"]["log"]
