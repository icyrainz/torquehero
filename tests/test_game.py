import ast
from pathlib import Path

import pytest
from conftest import game_for, inp, kinds, make_chart, run

import torquehero.game
from torquehero.chart import phrase_x
from torquehero.config import Config
from torquehero.game import Game
from torquehero.state import default_layer_modes


def judged(events):
    return [(e.note_kind, e.result) for e in kinds(events, "judgement")]


def press_at(ctrl, at, value_name=None, **held):
    """Input fn: press `ctrl` on the frame nearest `at`."""
    def fn(t):
        on = abs(t - at) < 0.005
        kw = dict(held)
        if on:
            kw["pressed"] = {ctrl}
            if value_name:
                kw[value_name] = 1.0 if value_name != "brake" else 0.9
            if ctrl.startswith("gate") or ctrl.startswith("paddle"):
                kw["down"] = {ctrl}
        return inp(**kw)
    return fn


# --- purity ---

def test_game_module_is_pure():
    tree = ast.parse(Path(torquehero.game.__file__).read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(("." * node.level) + (node.module or ""))
    assert mods <= {"__future__", "math", "bisect", "dataclasses", ".chart", ".config", ".state"}, mods


# --- gate ---

@pytest.mark.parametrize("steer, result", [(0.5, "perfect"), (0.62, "good"), (0.25, "miss")])
def test_gate(steer, result):
    g = game_for([{"t": 1.0, "kind": "gate", "x": 0.5}])
    ev = run(g, 0.8, 1.3, lambda t: inp(steer=steer))
    assert judged(ev) == [("gate", result)]


def test_gate_perfect_needs_position_inside_perfect_window():
    g = game_for([{"t": 1.0, "kind": "gate", "x": 0.5}])
    # on target only 0.08..0.12 s late: good, not perfect
    ev = run(g, 0.8, 1.3, lambda t: inp(steer=0.5 if 1.08 <= t <= 1.12 else 0.0))
    assert judged(ev) == [("gate", "good")]


# --- kick / hat ---

@pytest.mark.parametrize("kind, ctrl, value", [("kick", "brake", "brake"), ("hat", "clutch", "clutch")])
@pytest.mark.parametrize("at, result", [(1.02, "perfect"), (0.9, "good"), (1.1, "good"), (None, "miss")])
def test_kick_hat(kind, ctrl, value, at, result):
    g = game_for([{"t": 1.0, "kind": kind}])
    fn = press_at(ctrl, at, value) if at else (lambda t: inp())
    ev = run(g, 0.7, 1.3, fn)
    assert judged(ev) == [(kind, result)]
    if result != "miss":
        (s,) = kinds(ev, "sound")
        assert (s.name, s.cause) == (kind, "hit")


def test_kick_velocity_from_input_velocity():
    g = game_for([{"t": 1.0, "kind": "kick"}])
    fn = lambda t: inp(pressed={"brake"}, brake=0.5, velocity={"brake": 0.3}) if abs(t - 1.0) < 0.005 else inp()  # noqa: E731
    ev = run(g, 0.9, 1.2, fn)
    (j,) = kinds(ev, "judgement")
    assert j.vel == pytest.approx(0.3)                   # not the 0.5 pressure on the crossing frame
    assert kinds(ev, "sound")[0].vel == pytest.approx(0.3)
    ev = run(game_for([{"t": 1.0, "kind": "kick"}]), 0.9, 1.2, press_at("brake", 1.0, "brake"))
    assert kinds(ev, "judgement")[0].vel == 1.0          # no velocity supplied: 1.0


def test_free_play_outside_window_still_sounds():
    g = game_for([{"t": 1.0, "kind": "kick"}, {"t": 1.0, "kind": "stab", "gate": 1},
                  {"t": 1.0, "kind": "tom", "side": "L"}])
    ev = run(g, 0.0, 0.2, lambda t: inp(pressed={"brake", "gate4", "paddle_r"}, brake=1.0) if t == 0.1 else inp())
    assert judged(ev) == []
    assert {(s.name, s.cause) for s in kinds(ev, "sound")} == {("kick", "free"), ("stab4", "free"), ("tom_r", "free")}


# --- stab / tom ---

@pytest.mark.parametrize("ctrl, at, result, sound", [
    ("gate3", 1.0, "perfect", "stab3"), ("gate3", 1.1, "good", "stab3"),
    ("gate2", 1.0, "miss", "dud"), (None, None, "miss", None),
])
def test_stab(ctrl, at, result, sound):
    g = game_for([{"t": 1.0, "kind": "stab", "gate": 3}])
    ev = run(g, 0.8, 1.3, press_at(ctrl, at) if ctrl else (lambda t: inp()))
    assert judged(ev) == [("stab", result)]
    assert [s.name for s in kinds(ev, "sound")] == ([sound] if sound else [])


@pytest.mark.parametrize("ctrl, at, result, sound", [
    ("paddle_l", 1.0, "perfect", "tom_l"), ("paddle_l", 0.9, "good", "tom_l"),
    ("paddle_r", 1.0, "miss", "dud"), (None, None, "miss", None),
])
def test_tom(ctrl, at, result, sound):
    g = game_for([{"t": 1.0, "kind": "tom", "side": "L"}])
    ev = run(g, 0.8, 1.3, press_at(ctrl, at) if ctrl else (lambda t: inp()))
    assert judged(ev) == [("tom", result)]
    assert [s.name for s in kinds(ev, "sound")] == ([sound] if sound else [])


def test_stab_sample_override():
    g = game_for([{"t": 1.0, "kind": "stab", "gate": 2, "sample": "stab_big"}])
    ev = run(g, 0.9, 1.1, press_at("gate2", 1.0))
    assert kinds(ev, "sound")[0].name == "stab_big"


# --- spin ---

def spin_fn(total_deg, start=1.0, dur=1.0, back_and_forth=False):
    def fn(t):
        u = min(1.0, max(0.0, (t - start) / (dur * 0.8)))
        deg = total_deg * u
        if back_and_forth:  # travel total_deg but end where it started
            deg = total_deg / 2 - abs(total_deg / 2 - deg)
        return inp(steer_deg=deg)
    return fn


@pytest.mark.parametrize("deg, result", [(370, "perfect"), (-370, "perfect"), (250, "good"), (100, "miss")])
def test_spin(deg, result):
    g = game_for([{"t": 1.0, "kind": "spin", "dur": 1.0}])
    ev = run(g, 0.9, 2.3, spin_fn(deg))
    assert judged(ev) == [("spin", result)]


def test_spin_net_vs_abs():
    notes = [{"t": 1.0, "kind": "spin", "dur": 1.0}]
    wheel = lambda t: inp(steer_deg=spin_fn(400, back_and_forth=True)(t).steer_deg, bound={"steer"})  # noqa: E731
    kb = lambda t: inp(steer_deg=spin_fn(400, back_and_forth=True)(t).steer_deg)  # noqa: E731
    assert judged(run(game_for(notes), 0.9, 2.3, wheel)) == [("spin", "miss")]      # net: back where it started
    assert judged(run(game_for(notes), 0.9, 2.3, kb)) == [("spin", "perfect")]      # abs: 400 deg of travel
    forced = game_for(notes, cfg=Config(spin_mode="abs"))
    assert judged(run(forced, 0.9, 2.3, wheel)) == [("spin", "perfect")]


# --- expr / fader ---

@pytest.mark.parametrize("offset, result", [(0.0, "perfect"), (0.2, "good"), (0.4, "miss")])
def test_expr(offset, result):
    note = {"t": 1.0, "kind": "expr", "dur": 1.0, "curve": [[0.0, 0.2], [1.0, 0.6]]}
    g = game_for([note])
    target = lambda t: 0.2 + 0.4 * min(1, max(0, t - 1.0))  # noqa: E731
    ev = run(g, 0.9, 2.1, lambda t: inp(throttle=min(1.0, target(t) + offset)))
    assert judged(ev) == [("expr", result)]


@pytest.mark.parametrize("lever", [0, 1])
def test_fader_tracks_its_lever(lever):
    note = {"t": 1.0, "kind": "fader", "dur": 1.0, "lever": lever, "curve": [[0.0, 0.8], [1.0, 0.8]]}
    right = run(game_for([note]), 0.9, 2.1, lambda t: inp(**{f"lever{lever}": 0.8}))
    wrong = run(game_for([note]), 0.9, 2.1, lambda t: inp(**{f"lever{1 - lever}": 0.8}))
    assert judged(right) == [("fader", "perfect")]
    assert judged(wrong) == [("fader", "miss")]


# --- riser ---

def riser_fn(pull_from, release_at):
    def fn(t):
        pulled = pull_from <= t < release_at
        was = pull_from - 1e-9 <= t - 0.01 < release_at - 1e-9  # state on the previous 100 Hz frame
        return inp(handbrake=1.0 if pulled else 0.0, down={"handbrake"} if pulled else set(),
                   pressed={"handbrake"} if pulled and not was else set(),
                   released={"handbrake"} if was and not pulled else set())
    return fn


@pytest.mark.parametrize("pull, release, result", [
    (1.0, 2.0, "perfect"), (1.0, 2.1, "good"), (1.8, 2.0, "miss"), (1.0, 1.4, "miss"), (9, 9, "miss"),
])
def test_riser(pull, release, result):
    g = game_for([{"t": 1.0, "kind": "riser", "dur": 1.0}])
    ev = run(g, 0.9, 2.3, riser_fn(pull, release))
    assert judged(ev) == [("riser", result)]


def test_riser_sounds_and_ffb():
    g = game_for([{"t": 1.0, "kind": "riser", "dur": 1.0}])
    ev = run(g, 0.9, 2.3, riser_fn(1.0, 2.0))
    assert [(s.name, s.action) for s in kinds(ev, "sound")] == [("riser", "start"), ("riser", "stop"),
                                                                 ("impact", "play")]
    assert [c.name for c in kinds(ev, "ffb")] == ["riser_start", "riser_release"]


def test_riser_snapshot():
    g = game_for([{"t": 1.0, "kind": "riser", "dur": 1.0}])
    run(g, 0.9, 1.5, riser_fn(1.0, 2.0))
    r = g.snapshot().riser
    assert r.active and r.held and r.progress == pytest.approx(0.5, abs=0.02)
    assert r.held_frac == pytest.approx(0.5, abs=0.03)


# --- auto layers ---

def test_auto_layers_play_and_are_not_judged():
    notes = [{"t": 1.0, "kind": "kick", "vel": 0.7}, {"t": 1.5, "kind": "stab", "gate": 2},
             {"t": 2.0, "kind": "riser", "dur": 1.0}, {"t": 1.0, "kind": "gate", "x": 0.0}]
    g = game_for(notes, modes={"melody": "you"})
    ev = run(g, 0.0, 3.5, lambda t: inp())
    assert judged(ev) == [("gate", "perfect")]
    assert [(s.name, s.action, s.cause) for s in kinds(ev, "sound")] == [
        ("kick", "play", "auto"), ("stab2", "play", "auto"),
        ("riser", "start", "auto"), ("riser", "stop", "auto"), ("impact", "play", "auto")]
    assert kinds(ev, "sound")[0].vel == 0.7
    snap = g.snapshot()
    assert snap.layers["kick"].mode == "auto" and snap.layers["kick"].alive and snap.layers["kick"].total == 0


def test_auto_layer_ignores_presses():
    g = game_for([{"t": 1.0, "kind": "kick"}], modes={"melody": "you", "kick": "auto"})
    ev = run(g, 0.9, 1.1, press_at("brake", 0.95, "brake"))
    assert [s.cause for s in kinds(ev, "sound")] == ["auto"]


def test_default_layer_modes():
    modes = default_layer_modes({"brake", "gate1"})
    assert modes["melody"] == "you" and modes["kick"] == "you"
    assert modes["pads"] == "auto"                       # without `needed`, all six gates are needed
    assert modes["hat"] == "auto" and modes["faders"] == "auto"
    assert default_layer_modes({"gate1"}, needed={"gate1", "brake"})["pads"] == "you"
    g = Game(game_for([{"t": 1.0, "kind": "kick"}]).chart, Config(), {})
    assert g.layer_modes["melody"] == "you" and g.layer_modes["kick"] == "auto"


# --- alive, combo, multiplier ---

def test_layer_alive_until_next_hit():
    g = game_for([{"t": 1.0, "kind": "kick"}, {"t": 2.0, "kind": "kick"}])
    ev = run(g, 0.9, 1.5, lambda t: inp())
    assert judged(ev) == [("kick", "miss")]
    assert "rumble" in [c.name for c in kinds(ev, "ffb")]
    assert not g.snapshot().layers["kick"].alive
    run(g, 1.51, 2.2, press_at("brake", 2.0, "brake"))
    assert g.snapshot().layers["kick"].alive


def test_combo_multiplier_and_score():
    notes = [{"t": 1.0 + i * 0.5, "kind": "kick"} for i in range(35)]
    g = game_for(notes, length=30.0)
    hit = lambda t: inp(pressed={"brake"}, brake=1.0) if any(abs(t - n["t"]) < 0.005 for n in notes) else inp()  # noqa: E731
    run(g, 0.9, 1.0 + 9 * 0.5 + 0.01, hit)
    assert (g.combo, g.multiplier, g.score) == (10, 2, 9 * 100 + 200)
    run(g, 1.0 + 9 * 0.5 + 0.02, 1.0 + 34 * 0.5 + 0.01, hit)
    assert g.combo == 35 and g.multiplier == 4 and g.max_combo == 35
    # 1-9 x1, 10-19 x2, 20-29 x3, 30-35 x4
    assert g.score == 100 * (9 * 1 + 10 * 2 + 10 * 3 + 6 * 4)
    g2 = game_for(notes[:3])
    run(g2, 0.9, 2.3, lambda t: hit(t) if t < 1.7 else inp())
    assert (g2.combo, g2.max_combo, g2.counts) == (0, 2, {"perfect": 2, "good": 0, "miss": 1})
    assert g2.accuracy == pytest.approx(2 / 3)


def test_perfect_melody_hit_cues_ffb_kick():
    g = game_for([{"t": 1.0, "kind": "gate", "x": 0.0}])
    ev = run(g, 0.9, 1.1, lambda t: inp(steer=0.05))
    (cue,) = [c for c in kinds(ev, "ffb") if c.name == "kick"]
    assert cue.dir == 1.0


def test_audio_offset_shifts_judgement():
    g = game_for([{"t": 1.0, "kind": "kick"}], cfg=Config(audio_offset=0.1))
    ev = run(g, 0.9, 1.3, press_at("brake", 1.1, "brake"))
    assert judged(ev) == [("kick", "perfect")]


# --- echo ---

def test_echo_listen_is_demonstrated_then_blind_is_judged(fixture_chart):
    g = Game(fixture_chart, Config(), {"melody": "you"})
    ev = run(g, 7.0, 9.3, lambda t: inp(steer=-1.0))
    listen = [e for e in kinds(ev, "judgement") if fixture_chart.notes[e.note_index].listen]
    assert listen == [] and g.counts["miss"] == 3    # listen gates never judged; only gates 1-3 missed
    assert g.echo_at(8.2) == "listen" and g.echo_at(10.2) == "repeat" and g.echo_at(6.0) == "none"
    run(g, 7.5, 8.2, lambda t: inp())
    snap = g.snapshot()
    listen = [(n.t, n.x) for n in fixture_chart.notes if n.listen]
    assert snap.echo == "listen" and snap.echo_target == pytest.approx(phrase_x(listen, 8.2)) and snap.on_road
    assert g.echo_target_at(7.6) == 0.0                          # before the first gate: its x
    assert g.echo_target_at(8.5) == pytest.approx(0.3)           # reaches each gate's x at its time
    assert g.echo_target_at(8.25) == pytest.approx(0.15)         # smoothstep midpoint
    ev = run(g, 9.31, 11.3, lambda t: inp(steer=fixture_chart.road_x(t)))
    assert judged(ev) == [("gate", "perfect")] * 3
    assert g.snapshot().echo == "none" and g.snapshot().echo_target is None


# --- clock and snapshot ---

def test_beats_and_sections(fixture_chart):
    g = Game(fixture_chart, Config(), {})
    ev = run(g, 0.0, 12.0, lambda t: inp())
    assert len(kinds(ev, "beat")) == 25
    assert [(s.name, s.weight) for s in kinds(ev, "section")] == [
        ("intro", 0.2), ("verse", 0.6), ("echo", 0.3), ("chorus", 0.9)]
    snap = g.snapshot()
    assert (snap.section, snap.weight, snap.beat_strength) == ("chorus", 0.9, 1.0)


def test_snapshot_targets_and_upcoming(fixture_chart):
    g = Game(fixture_chart, Config(), {"melody": "you", "pads": "you"})
    run(g, 0.0, 6.5, lambda t: inp(steer=fixture_chart.road_x(t)))
    s = g.snapshot()
    assert s.expr_target == pytest.approx(0.5)
    assert s.upcoming and all(u["kind"] in ("spin", "expr", "stab", "tom", "riser", "fader") for u in s.upcoming)
    assert all(6.5 <= u["t"] <= 14.5 for u in s.upcoming)
    assert 0 <= s.beat_phase < 1 and s.length == 24.0
    run(g, 6.51, 12.5, lambda t: inp(steer=fixture_chart.road_x(t)))
    s = g.snapshot()
    assert s.fader_targets[0] == pytest.approx(0.25) and s.fader_targets[1] is None
    assert [(p.text, p.layer) for p in s.popups] == [("MISS", "pads")]


def test_on_road_uses_road_tolerance():
    g = game_for([{"t": 5.0, "kind": "gate", "x": 0.0}], road=[{"t": 0.0, "x": 0.0}])
    g.update(1.0, 0.01, inp(steer=0.19))
    assert g.snapshot().on_road
    g.update(1.01, 0.01, inp(steer=0.25))
    assert not g.snapshot().on_road


def test_snapshot_json_roundtrip(fixture_chart):
    import json

    g = Game(fixture_chart, Config(), {})
    g.update(1.0, 0.01, inp(pressed={"brake"}, bound={"brake"}))
    d = json.loads(g.snapshot().to_json())
    assert d["input"]["pressed"] == ["brake"] and d["phase"] == "play"


# --- press matching ---

def test_tom_roll_skipped_left_then_right_on_time():
    g = game_for([{"t": 1.0, "kind": "tom", "side": "L"}, {"t": 1.1, "kind": "tom", "side": "R"}])
    ev = run(g, 0.8, 1.4, press_at("paddle_r", 1.1))
    assert {e.note_t: e.result for e in kinds(ev, "judgement")} == {1.0: "miss", 1.1: "perfect"}


def test_early_wrong_gate_then_right_gate():
    # lone note: the early wrong gate is the only candidate, so it is a miss; the right
    # gate on time then finds nothing left and is free play (plays its stab)
    g = game_for([{"t": 1.0, "kind": "stab", "gate": 3}])
    ev = run(g, 0.8, 1.3, lambda t: press_at("gate2", 0.92)(t) if t < 0.95 else press_at("gate3", 1.0)(t))
    assert judged(ev) == [("stab", "miss")]
    assert [(s.name, s.cause) for s in kinds(ev, "sound")] == [("dud", "miss"), ("stab3", "free")]
    # two notes: a press matching the later note judges it, not a wrong press on the earlier one
    g = game_for([{"t": 1.0, "kind": "stab", "gate": 1}, {"t": 1.1, "kind": "stab", "gate": 2}])
    ev = run(g, 0.8, 1.4, lambda t: press_at("gate2", 1.0)(t) if t < 1.01 else press_at("gate1", 1.03)(t))
    assert {e.note_t: e.result for e in kinds(ev, "judgement")} == {1.1: "good", 1.0: "perfect"}


# --- event fields ---

def test_sound_and_judgement_carry_note():
    g = game_for([{"t": 1.0, "kind": "kick"}, {"t": 2.0, "kind": "hat"}], modes={"melody": "you", "kick": "you"})
    ev = run(g, 0.0, 2.2, lambda t: press_at("brake", 1.0, "brake")(t) if t > 0.5 else press_at("brake", 0.2)(t))
    free, hit, auto = kinds(ev, "sound")
    assert (free.cause, free.note_index, free.note_t) == ("free", None, None)
    assert (hit.cause, hit.note_index, hit.note_t) == ("hit", 0, 1.0)
    assert (auto.cause, auto.note_index, auto.note_t) == ("auto", 1, 2.0)
    (j,) = kinds(ev, "judgement")
    assert (j.note_index, j.note_t) == (0, 1.0)


# --- snapshot views ---

def test_snapshot_notes_window_and_progress():
    notes = [{"t": 0.5, "kind": "kick"}, {"t": 1.0, "kind": "spin", "dur": 1.0},
             {"t": 1.2, "kind": "kick"}, {"t": 5.0, "kind": "kick"}]
    g = game_for(notes)
    run(g, 0.0, 1.5, spin_fn(180, start=1.0, dur=0.625))
    views = {v.index: v for v in g.snapshot().notes}
    assert set(views) == {1, 2}                  # kick@0.5 is > 0.5 s old, kick@5 beyond lookahead
    assert views[1].progress == pytest.approx(0.5, abs=0.02) and views[1].result is None
    assert views[2].done and views[2].result == "miss" and views[2].progress is None


def test_song_info(fixture_chart):
    d = Game(fixture_chart, Config(), {"melody": "you", "kick": "you"}).song_info().to_dict()
    assert (d["title"], d["bpm"], d["length"]) == ("Fixture", 120.0, 24.0)
    assert d["layer_modes"]["kick"] == "you" and d["layer_modes"]["hat"] == "auto"
    assert d["layers_used"] == ["melody", "kick", "hat", "expr", "pads", "fills", "riser", "faders"]
    assert d["sections"][1] == {"t": 4.0, "name": "verse", "weight": 0.6}


# --- riser cue pairing ---

def _cues(ev):
    return [c.name for c in kinds(ev, "ffb") if c.name.startswith("riser")]


@pytest.mark.parametrize("fn, cues", [
    (riser_fn(1.0, 2.0), ["riser_start", "riser_release"]),                           # clean
    (lambda t: riser_fn(1.0, 1.4)(t) if t < 1.5 else riser_fn(1.5, 2.0)(t),
     ["riser_start", "riser_release", "riser_start", "riser_release"]),              # early release, re-pull
    (riser_fn(1.0, 9.0), ["riser_start", "riser_release"]),                           # held past the drop
    (riser_fn(1.0, 1.3), ["riser_start", "riser_release"]),                           # early release, miss
    (riser_fn(9, 9), []),                                                             # never pulled
])
def test_riser_cues_pair(fn, cues):
    g = game_for([{"t": 1.0, "kind": "riser", "dur": 1.0}])
    ev = run(g, 0.9, 2.5, fn)
    assert _cues(ev) == cues and len(kinds(ev, "judgement")) == 1


def test_auto_riser_cues_pair():
    g = game_for([{"t": 1.0, "kind": "riser", "dur": 1.0}], modes={"melody": "you"})
    assert _cues(run(g, 0.9, 2.5)) == ["riser_start", "riser_release"]


def test_riser_pull_sound_is_hit_during_note():
    g = game_for([{"t": 1.0, "kind": "riser", "dur": 1.0}])
    ev = run(g, 0.0, 2.3, lambda t: riser_fn(0.2, 0.4)(t) if t < 0.6 else riser_fn(1.0, 2.0)(t))
    starts = [(s.cause, s.note_index) for s in kinds(ev, "sound") if s.action == "start"]
    assert starts == [("free", None), ("hit", 0)]


# --- bound vs fallback ---

def test_spin_mode_uses_bound_not_fallback():
    notes = [{"t": 1.0, "kind": "spin", "dur": 1.0}]
    fb = lambda t: inp(steer_deg=spin_fn(400, back_and_forth=True)(t).steer_deg, fallback={"steer"})  # noqa: E731
    assert judged(run(game_for(notes), 0.9, 2.3, fb)) == [("spin", "perfect")]   # abs on keyboard/mouse
    g = game_for(notes)
    assert g.spin_mode(inp(bound={"steer"})) == "net" and g.spin_mode(inp(fallback={"steer"})) == "abs"


def test_default_layer_modes_counts_fallback():
    modes = default_layer_modes({"brake"}, {"clutch", "gate1"}, needed={"brake", "clutch", "gate1"})
    assert modes["kick"] == modes["hat"] == modes["pads"] == "you"
    assert modes["riser"] == "auto"


# --- JSON safety ---

def test_snapshot_json_has_no_nan(fixture_chart):
    import json

    g = Game(fixture_chart, Config(), {})
    g.update(1.0, 0.01, inp(steer=float("nan"), steer_deg=float("inf")))
    s = g.snapshot()
    s.ffb = {"torque": float("-inf"), "log": []}
    text = s.to_json()
    assert "NaN" not in text and "Infinity" not in text
    d = json.loads(text)
    assert d["input"]["steer"] is None and d["ffb"]["torque"] is None


# --- long-note entry frame ---

def test_spin_entry_frame_jump_does_not_count():
    g = game_for([{"t": 1.0, "kind": "spin", "dur": 1.0}])
    ev = run(g, 0.9, 2.3, lambda t: inp(steer_deg=400.0 if t >= 1.0 else 0.0))
    assert judged(ev) == [("spin", "miss")]


def test_expr_entry_frame_counts_only_time_since_start():
    note = {"t": 1.0, "kind": "expr", "dur": 1.0, "curve": [[0.0, 0.5], [1.0, 0.5]]}
    g = game_for([note])
    g.update(0.9, 0.01, inp(throttle=0.5))
    g.update(1.005, 0.105, inp(throttle=0.5))
    assert g._st[0].tot == pytest.approx(0.005)


def test_half_bound_layer_is_auto(fixture_chart):
    needed = fixture_chart.controls_used()
    assert {"lever0", "lever1", "gate1", "gate3", "paddle_l", "paddle_r", "steer"} <= needed
    assert "gate2" not in needed
    modes = default_layer_modes({"lever0", "gate1", "gate3"}, needed=needed)
    assert modes["faders"] == "auto"                     # lever1 notes would all miss
    assert modes["pads"] == "you"                        # every gate the chart uses is served
    assert default_layer_modes(set(), {"paddle_l"}, needed={"steer"})["fills"] == "you"  # unused layer: free play


# --- end of song ---

def test_finished_waits_for_note_at_length():
    g = game_for([{"t": 2.0, "kind": "kick"}], length=2.0)
    run(g, 1.9, 2.05)
    assert not g.finished
    ev = run(g, 2.06, 2.2)
    assert judged(ev) == [("kick", "miss")] and g.finished


def test_finished_waits_for_held_riser_ending_at_length():
    g = game_for([{"t": 1.0, "kind": "riser", "dur": 1.0}], length=2.0)
    run(g, 0.9, 2.1, riser_fn(1.0, 9.0))
    assert not g.finished and g.snapshot().riser.active
    ev = run(g, 2.11, 2.2, riser_fn(1.0, 9.0))
    assert judged(ev) == [("riser", "miss")] and "riser_release" in _cues(ev) and g.finished


# --- riser status and release sounds ---

def test_riser_status_active_until_judged():
    g = game_for([{"t": 1.0, "kind": "riser", "dur": 1.0}])
    run(g, 0.9, 2.08, riser_fn(1.0, 9.0))
    r = g.snapshot().riser
    assert r.active and r.held and r.progress == 1.0     # past the drop, not judged yet
    run(g, 2.09, 2.2, riser_fn(1.0, 9.0))
    assert not g.snapshot().riser.active


@pytest.mark.parametrize("release, cause", [(2.0, "hit"), (2.1, "hit")])
def test_judged_release_sounds_carry_note(release, cause):
    g = game_for([{"t": 1.0, "kind": "riser", "dur": 1.0}])
    ev = run(g, 0.9, 2.3, riser_fn(1.0, release))
    stop = [(s.name, s.action, s.cause, s.note_index, s.note_t) for s in kinds(ev, "sound") if s.action != "start"]
    assert stop == [("riser", "stop", cause, 0, 1.0), ("impact", "play", cause, 0, 1.0)]


def test_judged_release_miss_and_free_release():
    g = game_for([{"t": 1.0, "kind": "riser", "dur": 1.0}])
    ev = run(g, 0.9, 2.3, riser_fn(1.8, 2.0))           # held too little: judged miss at release
    assert judged(ev) == [("riser", "miss")]
    assert {s.cause for s in kinds(ev, "sound") if s.action != "start"} == {"miss"}
    g = game_for([{"t": 5.0, "kind": "riser", "dur": 1.0}])
    ev = run(g, 0.0, 1.0, riser_fn(0.2, 0.5))
    assert {(s.cause, s.note_index) for s in kinds(ev, "sound")} == {("free", None)}


# --- spin direction and turn offset (SPEC 9 rule 10) ---

def wheel(path):
    """Wheel input (net mode) following `path(t)` in degrees; steer is the raw lane, clamped."""
    def fn(t):
        d = path(t)
        return inp(steer_deg=d, steer=max(-1.0, min(1.0, d / 90)), bound={"steer"})
    return fn


def turn(deg, start=1.0, dur=1.0, base=0.0):
    """Degrees: `base` until `start`, then `deg` more over 0.8 dur, then held."""
    return lambda t: base + deg * min(1.0, max(0.0, (t - start) / (dur * 0.8)))


def spins(*dirs, gap=2.0):
    return [{"t": 1.0 + k * gap, "kind": "spin", "dur": 1.0, **({"dir": d} if d else {})} for k, d in enumerate(dirs)]


def state(g):
    s = g.snapshot()
    return s.steer_offset_deg, round(s.steer_lane, 3), s.steer_unwind


@pytest.mark.parametrize("dir_, deg, result, offset", [
    ("cw", 370, "perfect", 360.0), ("cw", -370, "miss", -360.0), ("ccw", -370, "perfect", -360.0),
    ("ccw", 370, "miss", 360.0), (None, -370, "perfect", -360.0), ("cw", 100, "miss", 0.0),
])
def test_spin_dir_and_offset_follows_wheel(dir_, deg, result, offset):
    g = game_for(spins(dir_))
    ev = run(g, 0.9, 2.3, wheel(turn(deg)))
    assert judged(ev) == [("spin", result)]
    assert g.steer_offset_deg == offset and g.snapshot().steer_offset_deg == offset
    assert not g.snapshot().steer_unwind


def test_clean_cw_then_ccw():
    g = game_for(spins("cw", "ccw"))
    ev = run(g, 0.9, 2.3, wheel(turn(360)))
    assert state(g) == (360.0, 0.0, False)
    ev += run(g, 2.31, 4.3, wheel(turn(-360, start=3.0, base=360)))
    assert judged(ev) == [("spin", "perfect")] * 2
    assert state(g) == (0.0, 0.0, False)


def test_skipped_ccw_then_cw_again_is_a_miss_beyond_one_turn():
    g = game_for(spins("cw", "ccw", "cw"))
    ev = run(g, 0.9, 2.3, wheel(turn(360)))
    ev += run(g, 2.31, 4.3, wheel(lambda t: 360.0))                     # skips the ccw spin
    assert state(g) == (360.0, 0.0, False)
    ev += run(g, 4.31, 6.3, wheel(turn(360, start=5.0, base=360)))      # obeys the next cw: wheel at 720
    assert judged(ev) == [("spin", "perfect"), ("spin", "miss"), ("spin", "miss")]
    assert state(g) == (360.0, 1.0, True)                               # one turn out: unwind


def test_good_spin_offset_matches_wheel():
    g = game_for(spins("cw"))
    ev = run(g, 0.9, 2.3, wheel(turn(250)))
    assert judged(ev) == [("spin", "good")]
    assert state(g) == (360.0, -1.0, False)   # wheel at 250: 110 deg short of the offset, lane clamped


def test_missed_spin_offset_matches_wheel():
    g = game_for(spins("cw"))
    ev = run(g, 0.9, 2.3, wheel(turn(200)))
    assert judged(ev) == [("spin", "miss")]
    assert state(g) == (360.0, -1.0, False)
    g = game_for(spins("cw"))
    run(g, 0.9, 2.3, wheel(turn(170)))
    assert state(g) == (0.0, 1.0, False)


def test_wheel_returned_to_centre_instead_of_completing():
    g = game_for(spins("cw"))
    ev = run(g, 0.9, 2.3, wheel(lambda t: turn(200)(t) if t < 1.7 else 0.0))
    assert judged(ev) == [("spin", "miss")]
    assert state(g) == (0.0, 0.0, False)


def test_reset_derives_offset_from_wheel():
    g = game_for(spins("cw") + [{"t": 3.0, "kind": "gate", "x": 0.5}])
    g.update(0.0, 0.01, wheel(lambda t: 360.0)(0.0))
    assert state(g) == (360.0, 0.0, False) and g.snapshot().steer_offset_changed
    g.update(0.01, 0.01, wheel(lambda t: 360.0)(0.01))
    assert not g.snapshot().steer_offset_changed
    g.reset()
    g.update(0.0, 0.01, wheel(lambda t: 10.0)(0.0))
    assert state(g) == (0.0, 0.111, False) and g.snapshot().steer_offset_changed


def test_offset_never_changes_between_spins():
    g = game_for(spins("cw", "ccw"))
    run(g, 0.9, 2.3, wheel(turn(360)))
    run(g, 2.31, 2.9, wheel(lambda t: 360 - 250 * (t - 2.31) / 0.59))   # steer far back, no spin finalized
    s = g.snapshot()
    assert s.steer_offset_deg == 360.0 and s.steer_unwind and not s.steer_offset_changed


def test_offset_changed_flag_on_spin_frame():
    g = game_for(spins("cw"))
    flags = []
    for f in range(140):
        t = 0.9 + f / 100
        g.update(t, 0.01, wheel(turn(370))(t))
        flags.append(g.snapshot().steer_offset_changed)
    assert flags[0] and flags[1:].count(True) == 1   # first frame after construction, then the spin frame


def test_spin_abs_mode_keeps_offset_zero_and_ignores_dir():
    g = game_for(spins("cw"))
    ev = run(g, 0.9, 2.3, lambda t: inp(steer_deg=turn(-370)(t)))
    assert judged(ev) == [("spin", "perfect")] and g.steer_offset_deg == 0.0
    assert not g.snapshot().steer_unwind


def test_gates_judged_relative_to_offset():
    g = game_for(spins("cw") + [{"t": 3.0, "kind": "gate", "x": 0.5}])
    run(g, 0.9, 2.3, wheel(turn(360)))
    # wheel one turn right plus 45 deg: inp.steer is clamped at 1, the effective lane is 0.5
    ev = run(g, 2.31, 3.3, lambda t: inp(steer_deg=405.0, steer=1.0, bound={"steer"}))
    assert judged(ev) == [("gate", "perfect")]
    assert g.snapshot().steer_lane == pytest.approx(0.5)



# --- a spin in progress (CONTRACT: added after the wave-2 reviews, spin fixes) ---

def test_spin_frame_by_frame_no_unwind_lane_on_road():
    """Spin t=2 dur 2, wheel 0 to 360 over 1.5 s, road at 0.3: every Snapshot field through the spin."""
    g = game_for([{"t": 2.0, "kind": "spin", "dur": 2.0, "dir": "cw"}], road=[{"t": 0.0, "x": 0.3}])
    path = turn(360, start=2.0, dur=1.5 / 0.8)
    done_at = None
    for f in range(301):
        t = 1.5 + f / 100
        ev = g.update(t, 0.01, wheel(path)(t))
        s, deg = g.snapshot(), path(t)
        assert not s.steer_unwind, t
        assert s.steer_lane_jump == (f == 0 or abs(t - 2.0) < 1e-9 or bool(judged(ev))), t
        if judged(ev):
            assert judged(ev) == [("spin", "perfect")] and done_at is None
            assert deg >= 345.0 and deg - 4.0 < 345.0              # finalized on the first frame past 360 - 15
            done_at = t
            assert s.steer_offset_deg == 360.0 and s.steer_offset_changed
            assert s.steer_lane == pytest.approx((deg - 360.0) / 90)
        elif t < 2.0 or done_at is not None:
            assert s.steer_lane == pytest.approx(max(-1.0, min(1.0, (deg - s.steer_offset_deg) / 90)))
            assert s.on_road == (abs(s.steer_lane - 0.3) <= 0.2)
            assert s.steer_offset_deg == (0.0 if done_at is None else 360.0)
            assert s.steer_offset_changed == (f == 0)
        else:                                                      # spin active: lane holds the road
            assert s.steer_lane == 0.3 and s.on_road and s.steer_offset_deg == 0.0
            assert not s.steer_offset_changed
    assert done_at is not None and done_at < 3.5                   # before the wheel reached 360


@pytest.mark.parametrize("deg, result, before_end", [(350, "perfect", True), (340, "good", False),
                                                     (200, "miss", False)])
def test_spin_perfect_inside_tolerance_else_judged_at_note_end(deg, result, before_end):
    g = game_for(spins("cw"))
    ev, at = [], None
    for f in range(141):
        t = 0.9 + f / 100
        e = g.update(t, 0.01, wheel(turn(deg))(t))
        if judged(e) and at is None:
            at = t
        ev += e
    assert judged(ev) == [("spin", result)]
    assert (at <= 2.0) == before_end


def test_spin_tolerance_is_configurable():
    g = game_for(spins("cw"), cfg=Config(spin_tolerance_deg=0.0))
    assert judged(run(g, 0.9, 2.3, wheel(turn(355)))) == [("spin", "good")]


@pytest.mark.parametrize("wheel_range, base, result", [
    (1080.0, 180.0, "perfect"), (1080.0, 195.0, "miss"),   # lock limit min(540, 535) = 535
    (900.0, 90.0, "perfect"), (900.0, 105.0, "miss"),      # lock limit 445
])
def test_spin_lock_limit_from_wheel_range(wheel_range, base, result):
    g = game_for(spins("cw"), cfg=Config(wheel_range_deg=wheel_range))
    assert g.spin_lock_deg == min(540.0, wheel_range / 2 - 5)
    assert judged(run(g, 0.9, 2.3, wheel(turn(360, base=base)))) == [("spin", result)]


def test_spin_lock_limit_uses_the_calibrated_wheel_range():
    cfg = Config(wheel_range_deg=1080.0)
    g = Game(make_chart(spins("cw")), cfg, {"melody": "you"}, wheel_range_deg=900.0)
    assert g.spin_lock_deg == 445.0
    assert judged(run(g, 0.9, 2.3, wheel(turn(360, base=105.0)))) == [("spin", "miss")]
    assert Game(make_chart(spins("cw")), cfg, wheel_range_deg=None).spin_lock_deg == 535.0


def test_spinning_only_from_the_spin_start_until_it_is_judged():
    g = game_for(spins("cw"))
    g.update(0.95, 0.01, inp(steer_deg=0.0, bound={"steer"}))
    assert not g.spinning and g.snapshot().notes      # in view, not started
    g.update(1.05, 0.01, inp(steer_deg=10.0, bound={"steer"}))
    assert g.spinning
    run(g, 1.06, 2.3, wheel(turn(360)))
    assert not g.spinning


def test_offset_cleared_when_steer_leaves_bound():
    g = game_for(spins("cw"))
    run(g, 0.9, 2.3, wheel(turn(360)))
    assert g.steer_offset_deg == 360.0
    g.update(2.31, 0.01, inp(steer_deg=360.0, steer=1.0, fallback={"steer"}))   # wheel unplugged
    s = g.snapshot()
    assert s.steer_offset_deg == 0.0 and s.steer_offset_changed and not s.steer_unwind
    g.update(2.32, 0.01, inp(steer=0.2, fallback={"steer"}))
    s = g.snapshot()
    assert s.steer_offset_deg == 0.0 and not s.steer_offset_changed and s.steer_lane == 0.2



def test_gate_during_spin_judged_on_real_lane():
    g = game_for([{"t": 1.0, "kind": "spin", "dur": 1.0}, {"t": 1.5, "kind": "gate", "x": 0.5}])
    ev = run(g, 0.9, 1.6, lambda t: inp(steer_deg=45.0, steer=0.5, bound={"steer"}))
    assert judged(ev) == [("gate", "perfect")]
    s = g.snapshot()
    assert s.steer_lane == 0.0 and g.steer_lane(s.input) == 0.5   # shown on the road, judged at 0.5


def test_early_perfect_offset_from_start_plus_one_turn():
    g = game_for(spins("cw"))
    ev = run(g, 0.9, 2.3, wheel(turn(360, base=-175.0)))   # finalized near 172: rounding would give 0
    assert judged(ev) == [("spin", "perfect")]
    assert g.steer_offset_deg == 360.0
    g = game_for(spins("ccw"))
    run(g, 0.9, 2.3, wheel(turn(-360, base=175.0)))
    assert g.steer_offset_deg == -360.0


# --- one drawn road (SPEC 10) ---

BLIND_ROAD = [{"t": 0.0, "x": 0.0}, {"t": 2.0, "x": 0.0}, {"t": 4.0, "x": 0.9}, {"t": 4.5, "x": 0.5},
              {"t": 6.0, "x": 0.0}]


def blind_game():
    # first blind gate 0.9 from the previous keyframe, 4 beats (2 s at 120 bpm) after it
    return game_for([{"t": 4.0, "kind": "gate", "x": 0.9, "blind": True},
                     {"t": 4.5, "kind": "gate", "x": 0.5, "blind": True}], road=BLIND_ROAD)


def test_blind_zone_and_drawn_road():
    g = blind_game()
    c = g.chart
    assert c.blind_zones() == [(2.0, 6.0)]
    assert c.road_x(3.9) > 0.5
    for f in range(401):
        t = 2.0 + f / 100
        assert c.drawn_road_x(t) == 0.0 == g.drawn_road_x(t), t
    assert c.drawn_road_x(1.0) == c.road_x(1.0) and c.drawn_road_x(7.0) == c.road_x(7.0)


@pytest.mark.parametrize("steer", [-1.0, 0.0, 1.0])
def test_on_road_anywhere_in_blind_zone(steer):
    g = blind_game()
    for f in range(401):
        t = 2.0 + f / 100
        g.update(t, 0.01, inp(steer=steer))
        s = g.snapshot()
        assert s.on_road and s.road_x == 0.0, t
    g.update(6.5, 0.01, inp(steer=-1.0))
    assert not g.snapshot().on_road


@pytest.mark.parametrize("steer, result", [(0.9, "perfect"), (0.0, "miss")])
def test_blind_gate_judged_at_real_x(steer, result):
    g = blind_game()
    ev = run(g, 3.8, 4.2, lambda t: inp(steer=steer))
    assert judged(ev) == [("gate", result)]


def test_blind_run_never_opens_before_listen_closes():
    road = [{"t": 0.0, "x": 0.0}, {"t": 3.0, "x": 0.0}, {"t": 3.5, "x": 0.3}, {"t": 3.7, "x": 0.0},
            {"t": 4.2, "x": 0.3}, {"t": 5.0, "x": 0.0}]
    notes = [{"t": 3.0, "kind": "gate", "x": 0.0, "listen": True}, {"t": 3.5, "kind": "gate", "x": 0.3, "listen": True},
             {"t": 3.7, "kind": "gate", "x": 0.0, "blind": True}, {"t": 4.2, "kind": "gate", "x": 0.3, "blind": True}]
    c = game_for(notes, road=road).chart
    assert c.blind_zones() == [(3.5, 5.0)]    # opens at 3.62 (listen close), not 3.2 (one beat before)
    assert c.drawn_road_x(3.5) == 0.3          # the listen phrase is drawn in full


# --- hysteresis: held means in `down` (SPEC 10) ---

def test_riser_held_reads_down_not_level():
    note = [{"t": 1.0, "kind": "riser", "dur": 1.0}]
    level_only = lambda t: inp(handbrake=0.9)                            # noqa: E731
    latched = lambda t: inp(handbrake=0.45, down={"handbrake"})          # noqa: E731
    g = game_for(note)
    run(g, 0.9, 1.5, level_only)
    assert not g.snapshot().riser.held and g.snapshot().riser.held_frac == 0.0
    g = game_for(note)
    run(g, 0.9, 1.5, latched)
    assert g.snapshot().riser.held and g.snapshot().riser.held_frac > 0.4


def test_default_layer_modes_with_chart_defaults():
    all_bound = {"steer", "brake", "clutch", "lever0", "lever1"}
    m = default_layer_modes(all_bound, defaults={"kick": "auto", "hat": "you", "pads": "you"})
    assert m["kick"] == "auto" and m["hat"] == "you"
    assert m["pads"] == "auto"                # "you" in defaults, but no gate is served
    assert default_layer_modes(all_bound, defaults={"melody": "auto"})["melody"] == "auto"
