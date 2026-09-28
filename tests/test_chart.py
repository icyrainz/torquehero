import copy
import json

import pytest

from torquehero.chart import (
    ECHO_PEAK_FACTOR,
    NOTE_KINDS,
    Chart,
    ChartError,
    curve_at,
    phrase_x,
    playability,
    validate_dict,
)
from torquehero.state import LAYERS


def test_fixture_is_valid_and_covers_contract(fixture_chart_dict, fixture_chart):
    assert validate_dict(fixture_chart_dict) == []
    assert {n.kind for n in fixture_chart.notes} == set(NOTE_KINDS)
    assert {n.layer for n in fixture_chart.notes} == set(LAYERS)
    assert any(n.listen for n in fixture_chart.notes) and any(n.blind for n in fixture_chart.notes)
    assert len({s.weight for s in fixture_chart.sections}) >= 2


def test_roundtrip(tmp_path, fixture_chart, fixture_chart_dict):
    fixture_chart.save(tmp_path / "c.json")
    again = Chart.load(tmp_path / "c.json")
    assert again == fixture_chart
    assert json.loads((tmp_path / "c.json").read_text()) == fixture_chart_dict


def test_paths_relative_to_chart(fixture_chart_path, fixture_chart):
    assert fixture_chart.path("stems/lead.wav") == (fixture_chart_path.parent / "stems/lead.wav").resolve()


def test_oneshot_templates(fixture_chart):
    a = fixture_chart.audio
    stab = next(n for n in fixture_chart.notes if n.kind == "stab" and n.gate == 3)
    assert a.oneshot_for("pads", stab) == "stab3"
    assert a.oneshot_for("pads", gate=5) == "stab5"
    assert a.oneshot_for("fills", side="R") == "tom_r"
    assert a.oneshot_for("melody") is None


def test_road_x():
    c = Chart.from_dict({"format": 1, "title": "r", "bpm": 120, "length": 10, "audio": {"sr": 48000},
                         "road": [{"t": 1.0, "x": 0.0}, {"t": 2.0, "x": 1.0}]})
    assert c.road_x(0.5) == 0.0              # before the first keyframe
    assert c.road_x(1.0) == 0.0
    assert c.road_x(1.54) == 0.0             # held for the first 55%
    assert c.road_x(1.775) == pytest.approx(0.5)  # smoothstep midpoint
    assert 0.5 < c.road_x(1.9) < 1.0
    assert c.road_x(2.0) == 1.0
    assert c.road_x(5.0) == 1.0              # after the last keyframe
    assert Chart.from_dict({"format": 1, "title": "r", "bpm": 1, "length": 1, "audio": {"sr": 1}}).road_x(0.3) == 0


def test_curve_sampling():
    curve = [[0.0, 0.2], [1.0, 0.8], [2.0, 0.5]]
    assert curve_at(curve, -1) == 0.2
    assert curve_at(curve, 0.5) == pytest.approx(0.5)
    assert curve_at(curve, 1.5) == pytest.approx(0.65)
    assert curve_at(curve, 9) == 0.5


def _mutate(d, path, value):
    d = copy.deepcopy(d)
    obj = d
    for k in path[:-1]:
        obj = obj[k]
    if value is KeyError:
        del obj[path[-1]]
    else:
        obj[path[-1]] = value
    return d


def _note(d, kind):
    return next(i for i, n in enumerate(d["notes"]) if n["kind"] == kind)


@pytest.mark.parametrize("case, expect", [
    (lambda d: _mutate(d, ["format"], 2), "format: expected 1"),
    (lambda d: _mutate(d, ["title"], KeyError), "title: required"),
    (lambda d: _mutate(d, ["bpm"], -1), "bpm: required positive"),
    (lambda d: _mutate(d, ["sections", 1, "weight"], 1.5), "sections[1]: 'weight' must be 0..1"),
    (lambda d: _mutate(d, ["beats", 0, "s"], 2), "beats[0]: 's' must be 0..1"),
    (lambda d: _mutate(d, ["road", 1, "x"], 3), "road[1]: x 3 outside -1..1"),
    (lambda d: _mutate(d, ["notes", 0, "t"], 99), "must be sorted by t"),
    (lambda d: _mutate(d, ["notes", 0, "kind"], "slide"), "unknown kind 'slide'"),
    (lambda d: _mutate(d, ["notes", 0, "layer"], "kick"), "layer must be 'melody'"),
    (lambda d: _mutate(d, ["notes", 0, "x"], KeyError), "missing field(s) x"),
    (lambda d: _mutate(d, ["notes", 0, "bogus"], 1), "unknown field(s) bogus"),
    (lambda d: _mutate(d, ["notes", _note(d, "stab"), "gate"], 7), "gate must be an integer 1..6"),
    (lambda d: _mutate(d, ["notes", _note(d, "tom"), "side"], "C"), "side must be 'L' or 'R'"),
    (lambda d: _mutate(d, ["notes", _note(d, "fader"), "lever"], 2), "lever must be 0 or 1"),
    (lambda d: _mutate(d, ["notes", _note(d, "spin"), "dur"], 0), "'dur' must be a positive finite number"),
    (lambda d: _mutate(d, ["notes", _note(d, "riser"), "dur"], 5.0), "after the song length"),
    (lambda d: _mutate(d, ["bpm"], float("nan")), "bpm: required positive finite"),
    (lambda d: _mutate(d, ["notes", 0, "x"], float("inf")), "x must be -1..1"),
    (lambda d: _mutate(d, ["road", 1, "t"], float("nan")), "road[1]: needs finite"),
    (lambda d: _mutate(d, ["notes", 0, "blind"], 1), "'blind' must be true or false"),
    (lambda d: _mutate(d, ["notes", _note(d, "hat"), "open"], "yes"), "'open' must be true or false"),
    (lambda d: _mutate(d, ["notes", _note(d, "stab"), "sample"], 3), "sample must be a non-empty string"),
    (lambda d: _mutate(d, ["notes", _note(d, "kick"), "x"], 0.5), "field(s) x do not belong to kind 'kick'"),
    (lambda d: _mutate(d, ["notes", _note(d, "stab"), "sample"], "stab_big"), "missing 'stab_big'"),
    (lambda d: _mutate(d, ["audio", "layers", "fills"], KeyError), "audio.layers.fills: required"),
    (lambda d: _mutate(d, ["audio", "oneshots", "dud"], KeyError), "missing 'dud'"),
    (lambda d: _mutate(d, ["audio", "oneshots", "impact"], KeyError), "missing 'impact'"),
    (lambda d: _mutate(d, ["audio", "oneshots", "stab3"], KeyError), "missing 'stab3'"),
    (lambda d: _mutate(d, ["audio", "oneshots", "tom_r"], KeyError), "missing 'tom_r'"),
    (lambda d: _mutate(d, ["audio", "layers", "pads", "oneshot"], "stab{lever}"), "only {gate} and {side}"),
    (lambda d: _mutate(d, ["notes", _note(d, "expr"), "curve"], [[0, 1.5]]), "v 1.5 outside 0..1"),
    (lambda d: _mutate(d, ["notes", _note(d, "expr"), "curve"], [[0, 0.5], [5, 0.5]]), "outside 0..dur"),
    (lambda d: _mutate(d, ["notes", _note(d, "kick"), "vel"], 2), "vel must be 0..1"),
    (lambda d: _mutate(d, ["audio"], KeyError), "audio: required manifest"),
    (lambda d: _mutate(d, ["audio", "layers", "kick", "mode"], "boom"), "audio.layers.kick: mode must be"),
    (lambda d: _mutate(d, ["audio", "layers", "melody", "stem"], "nope"), "stem 'nope' is not in audio.stems"),
    (lambda d: _mutate(d, ["audio", "layers", "kick", "oneshot"], "nope"), "missing 'nope'"),
    (lambda d: _mutate(d, ["audio", "layers", "bongo"], {"mode": "trigger", "oneshot": "kick"}), "unknown layer"),
])
def test_validation_errors(fixture_chart_dict, case, expect):
    bad = case(fixture_chart_dict)
    errors = validate_dict(bad)
    assert any(expect in e for e in errors), errors
    with pytest.raises(ChartError) as ei:
        Chart.from_dict(bad)
    assert expect in str(ei.value)


def test_listen_and_blind_exclusive(fixture_chart_dict):
    d = copy.deepcopy(fixture_chart_dict)
    n = next(n for n in d["notes"] if n.get("listen"))
    n["blind"] = True
    assert any("exclusive" in e for e in validate_dict(d))


def test_echo_rate_limit_checks_peak(fixture_chart_dict):
    d = copy.deepcopy(fixture_chart_dict)
    listen = [n for n in d["notes"] if n.get("listen")]
    listen[1]["x"] = 0.6  # 54 deg in 0.5 s: mean 108, peak 162 deg/s
    assert validate_dict(d) == []
    listen[1]["x"] = 1.0  # 90 deg in 0.5 s: mean 180 (the old average check passed), peak 270
    errors = validate_dict(d)
    assert any("peaks at 270 deg/s" in e for e in errors), errors


def test_phrase_x_peak_matches_validator():
    pts = [(0.0, 0.0), (0.5, 1.0)]
    assert phrase_x(pts, -1) == 0.0 and phrase_x(pts, 0.5) == 1.0 and phrase_x(pts, 9) == 1.0
    h = 1e-4
    peak = max(abs(phrase_x(pts, t + h) - phrase_x(pts, t)) / h for t in [i / 1000 for i in range(500)])
    assert peak * 90 == pytest.approx(ECHO_PEAK_FACTOR * 90 / 0.5, rel=1e-2)


def test_fixture_echo_within_limit(fixture_chart):
    listen = [(n.t, n.x) for n in fixture_chart.notes if n.listen]
    for (ta, xa), (tb, xb) in zip(listen, listen[1:], strict=False):
        assert ECHO_PEAK_FACTOR * abs(xb - xa) * 90 / (tb - ta) <= 180


def test_load_reports_bad_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{nope")
    with pytest.raises(ChartError, match="not valid JSON"):
        Chart.load(p)


def test_load_names_file_and_all_errors(tmp_path, fixture_chart_dict):
    d = _mutate(_mutate(fixture_chart_dict, ["bpm"], 0), ["format"], 0)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ChartError) as ei:
        Chart.load(p)
    assert str(p) in str(ei.value) and len(ei.value.errors) == 2


def test_oneshots_required_without_notes(fixture_chart_dict):
    d = copy.deepcopy(fixture_chart_dict)
    d["notes"] = [n for n in d["notes"] if n["kind"] == "gate"]
    assert validate_dict(d) == []
    for name in ("kick", "hat", "tom_l", "tom_r", "riser", "impact", "dud"):
        bad = _mutate(d, ["audio", "oneshots", name], KeyError)
        assert any(f"missing {name!r}" in e for e in validate_dict(bad)), name
    # the documented exception: stabs for gates no note uses
    assert validate_dict(_mutate(d, ["audio", "oneshots", "stab6"], KeyError)) == []


@pytest.mark.parametrize("case, expect", [
    (lambda d: _mutate(d, ["notes", _note(d, "tom"), "side"], 5), "side must be 'L' or 'R'"),
    (lambda d: _mutate(d, ["notes", _note(d, "stab"), "gate"], "x"), "gate must be an integer"),
    (lambda d: _mutate(d, ["audio", "layers", "pads", "oneshot"], "stab{gate:>q}"), "cannot be filled"),
    (lambda d: _mutate(d, ["audio", "layers", "pads", "oneshot"], "stab{gate"), "malformed"),
    (lambda d: _mutate(d, ["audio", "layers", "pads", "oneshot"], "stab{}"), "only {gate} and {side}"),
    (lambda d: _mutate(d, ["notes", _note(d, "fader"), "lever"], "a"), "lever must be 0 or 1"),
    (lambda d: _mutate(d, ["notes"], [None, 3, {"t": "x"}]), "must be an object"),
])
def test_validate_never_raises(fixture_chart_dict, case, expect):
    errors = validate_dict(case(fixture_chart_dict))
    assert any(expect in e for e in errors), errors


def test_listen_gates_same_time_different_x(fixture_chart_dict):
    d = copy.deepcopy(fixture_chart_dict)
    i = next(i for i, n in enumerate(d["notes"]) if n.get("listen"))
    d["notes"].insert(i + 1, {**d["notes"][i], "x": 0.1})
    assert any("same time as the previous listen gate" in e for e in validate_dict(d))


def test_echo_limit_from_config(tmp_path, fixture_chart_dict):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(fixture_chart_dict))
    Chart.load(p)                                   # peak 81 deg/s at play range 90
    with pytest.raises(ChartError, match="over the limit of 60"):
        Chart.load(p, echo_max_deg_s=60)
    with pytest.raises(ChartError, match="peaks at 243"):
        Chart.load(p, play_range_deg=270)
    Chart.load(p).validate(play_range_deg=180, echo_max_deg_s=200)


# --- additions after the wave-2 reviews ---

@pytest.mark.parametrize("case, expect", [
    (lambda d: _mutate(d, ["notes", _note(d, "spin"), "dir"], "left"), "dir must be 'cw' or 'ccw'"),
    (lambda d: _mutate(d, ["notes", _note(d, "gate"), "dir"], "cw"), "do not belong to kind 'gate'"),
    (lambda d: _mutate(d, ["audio", "layers", "kick", "mode"], "level"), "not allowed for kick"),
    (lambda d: _mutate(d, ["audio", "layers", "melody", "mode"], "level"), "not allowed for melody"),
    (lambda d: _mutate(d, ["defaults"], []), "defaults: must be an object"),
    (lambda d: _mutate(d, ["defaults"], {"layers": {"kick": "maybe"}}), "must be 'you' or 'auto'"),
    (lambda d: _mutate(d, ["defaults"], {"layers": {"bongo": "you"}}), "unknown layer 'bongo'"),
    (lambda d: _mutate(d, ["defaults"], {"layer": {}}), "unknown field 'layer'"),
])
def test_validation_errors_added(fixture_chart_dict, case, expect):
    errors = validate_dict(case(fixture_chart_dict))
    assert any(expect in e for e in errors), errors


def test_gate_mode_allowed_for_percussion(fixture_chart_dict):
    d = _mutate(fixture_chart_dict, ["audio", "layers", "kick"], {"mode": "gate", "stem": "drums", "oneshot": "kick"})
    assert validate_dict(d) == []


@pytest.mark.parametrize("case, expect", [
    (lambda d: _mutate(d, ["notes", 0, "kind"], ["gate"]), "unknown kind"),
    (lambda d: _mutate(d, ["notes", 0, "kind"], {"a": 1}), "unknown kind"),
    (lambda d: _mutate(d, ["audio", "layers", "melody", "stem"], ["lead"]), "stem must be a string"),
    (lambda d: _mutate(d, ["bpm"], 10 ** 400), "bpm: required positive finite"),
    (lambda d: _mutate(d, ["notes", 0, "t"], 10 ** 400), "'t' must be a finite number"),
    (lambda d: _mutate(d, ["notes", 0, "x"], -(10 ** 400)), "x must be -1..1"),
])
def test_validate_never_raises_added(fixture_chart_dict, case, expect):
    errors = validate_dict(case(fixture_chart_dict))
    assert any(expect in e for e in errors), errors


def _random_json(rng, depth=0):
    pick = rng.randrange(10 if depth < 3 else 7)
    if pick == 0:
        return None
    if pick == 1:
        return rng.choice([True, False])
    if pick == 2:
        return rng.choice([0, -1, 7, 10 ** 400, -(10 ** 309), 2 ** 64])
    if pick == 3:
        return rng.choice([0.5, -3.2, 1e308, float("inf"), float("nan")])
    if pick in (4, 5, 6):
        return rng.choice(["", "gate", "cw", "melody", "you", "trigger", "stab{gate}", "{", "x", "lead"])
    if pick in (7, 8):
        return [_random_json(rng, depth + 1) for _ in range(rng.randrange(4))]
    keys = ["t", "kind", "layer", "x", "dur", "dir", "gate", "side", "lever", "curve", "mode", "stem",
            "oneshot", "layers", "stems", "sr", "defaults", "listen", "sample", "vel"]
    return {rng.choice(keys): _random_json(rng, depth + 1) for _ in range(rng.randrange(5))}


def test_validate_and_playability_never_raise_on_random_input(fixture_chart_dict):
    import random

    rng = random.Random(1)
    paths = [["notes", i] for i in range(len(fixture_chart_dict["notes"]))]
    paths += [["audio", "layers", k] for k in fixture_chart_dict["audio"]["layers"]]
    paths += [["audio"], ["road"], ["sections"], ["beats"], ["defaults"], ["bpm"], ["length"], ["notes"]]
    for _ in range(3000):
        d = copy.deepcopy(fixture_chart_dict)
        for _ in range(rng.randrange(1, 4)):
            path = rng.choice(paths)
            value = _random_json(rng)
            try:  # an earlier mutation may have replaced part of the path
                obj = d
                for k in path[:-1]:
                    obj = obj[k]
                target = obj[path[-1]]
            except (LookupError, TypeError):
                continue
            if not isinstance(obj, dict | list):
                continue
            if isinstance(target, dict) and rng.random() < 0.7:
                target[rng.choice(list(target) or ["t"])] = value
            else:
                obj[path[-1]] = value
        assert isinstance(validate_dict(d), list)
        assert isinstance(playability(d), list)
    for v in (None, 3, "x", [], [1, {}]):
        assert validate_dict(v) == ["chart must be a JSON object"]
        assert len(playability(v)) == 1


def test_defaults_roundtrip(tmp_path, fixture_chart_dict):
    d = _mutate(fixture_chart_dict, ["defaults"], {"layers": {"hat": "auto", "melody": "you"}})
    c = Chart.from_dict(d)
    assert c.default_layers == {"hat": "auto", "melody": "you"}
    assert c.to_dict() == d
    assert "defaults" not in Chart.from_dict(fixture_chart_dict).to_dict()


# --- playability (SPEC 9) ---

def _play(notes, road=None, length=30.0):
    from conftest import make_chart

    return playability(make_chart(notes, length=length, road=[{"t": 0.0, "x": 0.0}] if road is None else road))


def _rules(violations):
    return sorted({v.split(":")[0] for v in violations})


def test_fixture_is_playable_and_has_both_spin_directions(fixture_chart, fixture_chart_dict):
    for difficulty in ("easy", "normal", "hard"):
        assert playability(fixture_chart, difficulty) == []
    assert playability(fixture_chart_dict) == []
    assert sorted(n.dir for n in fixture_chart.notes if n.kind == "spin") == ["ccw", "cw"]


@pytest.mark.parametrize("notes, rule", [
    ([{"t": 1.0, "kind": "riser", "dur": 2.0}, {"t": 3.3, "kind": "stab", "gate": 1}], "rule 3"),
    ([{"t": 1.0, "kind": "tom", "side": "L"}, {"t": 1.3, "kind": "stab", "gate": 1}], "rule 3"),
    ([{"t": 1.0, "kind": "fader", "dur": 1.0, "lever": 0, "curve": [[0, 0]]},
      {"t": 2.2, "kind": "tom", "side": "L"}], "rule 3"),
    ([{"t": 1.0, "kind": "spin", "dur": 1.0, "dir": "cw"}, {"t": 2.3, "kind": "riser", "dur": 1.0},
      {"t": 5.0, "kind": "spin", "dur": 1.0, "dir": "ccw"}], "rule 3"),
    ([{"t": 1.0, "kind": "fader", "dur": 2.0, "lever": 0, "curve": [[0, 0]]},
      {"t": 2.0, "kind": "fader", "dur": 2.0, "lever": 1, "curve": [[0, 0]]}], "rule 4"),
    ([{"t": 1.0, "kind": "fader", "dur": 1.0, "lever": 0, "curve": [[0, 0.1], [1, 0.9]]},
      {"t": 2.2, "kind": "fader", "dur": 1.0, "lever": 0, "curve": [[0, 0.1]]}], "rule 4"),
    ([{"t": 1.0, "kind": "gate", "x": 0.0}, {"t": 2.0, "kind": "fader", "dur": 2.0, "lever": 0, "curve": [[0, 0]]},
      {"t": 3.0, "kind": "stab", "gate": 1}, {"t": 3.2, "kind": "gate", "x": 0.5}], "rule 5"),
    ([{"t": 1.0, "kind": "expr", "dur": 2.0, "curve": [[0, 0.5]]}, {"t": 1.5, "kind": "kick"},
      {"t": 1.7, "kind": "hat"}], "rule 7"),
    ([{"t": 1.0, "kind": "gate", "x": -0.5}, {"t": 1.5, "kind": "gate", "x": 0.5}], "rule 8"),
    ([{"t": 1.0, "kind": "spin", "dur": 1.0}], "rule 10"),
    ([{"t": 1.0, "kind": "spin", "dur": 1.0, "dir": "cw"}, {"t": 4.0, "kind": "spin", "dur": 1.0, "dir": "cw"}],
     "rule 10"),
    ([{"t": 1.0, "kind": "spin", "dur": 1.0, "dir": "cw"}, {"t": 2.3, "kind": "gate", "x": 0.0},
      {"t": 4.0, "kind": "spin", "dur": 1.0, "dir": "ccw"}], "rule 11"),
])
def test_playability_rules(notes, rule):
    assert _rules(_play(notes)) == [rule], _play(notes)


@pytest.mark.parametrize("notes", [
    [{"t": 1.0, "kind": "riser", "dur": 2.0}, {"t": 3.4, "kind": "stab", "gate": 1}],
    [{"t": 1.0, "kind": "fader", "dur": 1.0, "lever": 0, "curve": [[0, 0]]},
     {"t": 2.0, "kind": "fader", "dur": 1.0, "lever": 1, "curve": [[0, 0]]}],
    [{"t": 1.0, "kind": "kick"}, {"t": 1.0, "kind": "hat"}],
    [{"t": 1.0, "kind": "fader", "dur": 1.0, "lever": 0, "curve": [[0, 0.1], [1, 0.9]]},
     {"t": 2.4, "kind": "fader", "dur": 1.0, "lever": 0, "curve": [[0, 0.1]]}],
    # margins of exactly one beat and one limb_travel, with float rounding (0.7 + 1.1 = 1.8000000000000003)
    [{"t": 0.2, "kind": "gate", "x": 0.0}, {"t": 0.3, "kind": "stab", "gate": 1},
     {"t": 0.7, "kind": "spin", "dur": 1.1, "dir": "cw"}, {"t": 2.2, "kind": "stab", "gate": 1},
     {"t": 2.3, "kind": "gate", "x": 0.0}, {"t": 4.0, "kind": "spin", "dur": 1.0, "dir": "ccw"}],
    [{"t": 1.0, "kind": "expr", "dur": 2.0, "curve": [[0, 0.5]]}, {"t": 1.5, "kind": "kick"},
     {"t": 1.9, "kind": "hat"}],
    [{"t": 1.0, "kind": "gate", "x": -0.5}, {"t": 2.0, "kind": "gate", "x": 0.5}],
    [{"t": 1.0, "kind": "spin", "dur": 1.0, "dir": "cw"}, {"t": 2.5, "kind": "gate", "x": 0.0},
     {"t": 4.0, "kind": "spin", "dur": 1.0, "dir": "ccw"}],
])
def test_playability_accepts(notes):
    assert _play(notes) == []


def test_playability_road_and_difficulty():
    notes = [{"t": 1.0, "kind": "gate", "x": -0.5}, {"t": 1.5, "kind": "gate", "x": 0.5}]
    assert _rules(_play(notes, road=[])) == ["rule 8", "rule 9"]
    assert _rules(_play([], road=[{"t": 0.0, "x": 0.0}, {"t": 1.0, "x": 0.0}, {"t": 1.0, "x": 0.5}])) == ["rule 9"]
    from conftest import make_chart

    c = make_chart(notes, road=[{"t": 0.0, "x": 0.0}])
    assert _rules(playability(c, "easy")) == ["rule 8"] and _rules(playability(c, "hard")) == []
    assert "unknown difficulty" in playability(c, "extreme")[0]


def test_validate_does_not_call_playability(fixture_chart_dict):
    d = _mutate(fixture_chart_dict, ["notes", _note(fixture_chart_dict, "spin"), "dir"], KeyError)
    assert validate_dict(d) == [] and _rules(playability(d)) == ["rule 10"]
