import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FIXTURES, make_chart

from torquehero import hud
from torquehero.chart import Chart, RoadKey
from torquehero.config import Config
from torquehero.game import Game
from torquehero.render import (
    BEHIND,
    FADER_BASE,
    FADER_SWING,
    FONT_CODEPOINTS,
    HIT_FRAC,
    HUD_SYMBOLS,
    RISER_LANE,
    ROAD_HALF,
    SAMPLE_CHART,
    SHOULDER_HALF,
    SIDE_LANE,
    Layout,
    RenderSettings,
    RoadPath,
    SceneGeo,
    alive_fraction,
    displayable,
    harness_snapshots,
    marker_on_road,
    needs_frame_cap,
    parse_span,
    scripted_input,
    sky_state,
    snapshot_from_dict,
    span_on_monitors,
    spin_dir,
    steer_lane,
    turn_back_dir,
    turn_offset,
)
from torquehero.state import LAYERS, KeySource, LayerStatus, NoteView, Popup, Snapshot

LOOK = Config().lookahead


def test_parse_span():
    assert parse_span("5760x1080+0+0") == (5760, 1080, 0, 0)
    assert parse_span("1920x1080-1920+0") == (1920, 1080, -1920, 0)
    assert parse_span("1920x1080+-1920+20") == (1920, 1080, -1920, 20)
    for bad in ("5760x1080", "x1080+0+0", "0x1080+0+0", "5760*1080+0+0"):
        with pytest.raises(ValueError):
            parse_span(bad)


def test_span_on_monitors():
    triples = [(0, 0, 1920, 1080), (1920, 0, 1920, 1080), (3840, 0, 1920, 1080)]
    assert span_on_monitors((5760, 1080, 0, 0), triples)
    assert not span_on_monitors((5760, 1080, 100, 0), triples)
    assert not span_on_monitors((5760, 1080, 0, 0), triples[:2])
    assert span_on_monitors((1920, 1080, 1920, 0), triples)


def test_projection_monotonic_in_depth():
    lay = Layout(1920, 1080)
    trel = np.linspace(lay.min_trel, LOOK, 200)
    _, sy = lay.proj(0.0, trel)
    assert np.all(np.diff(sy) < 0)                      # further ahead = higher on screen
    assert np.all(np.diff(lay.scale(trel)) < 0)         # and smaller
    assert np.all(sy > lay.horizon)
    assert lay.proj(0.0, 0.0)[1] == pytest.approx(HIT_FRAC * 1080)
    assert lay.proj(0.0, lay.min_trel)[1] <= 1080 + 1e-6
    sx, _ = lay.proj(np.array([-1.0, 0.0, 1.0]), 1.0)
    assert sx[0] < sx[1] < sx[2] and sx[1] == pytest.approx(lay.cx)


@pytest.mark.parametrize("size", [(5760, 1080), (1920, 1080), (3840, 1080), (1280, 720)])
def test_road_and_judged_notes_stay_in_centre(size):
    lay = Layout(*size)
    lo, hi = lay.cx0, lay.cx0 + lay.cw
    trel = np.linspace(lay.min_trel, LOOK, 100)
    ahead = trel[trel >= 0]  # notes are pinned at the hit line once passed
    for road in np.linspace(-1, 1, 21):
        for off in (-ROAD_HALF, ROAD_HALF):
            sx, _ = lay.proj(road + off, trel)
            assert np.all((sx >= lo) & (sx <= hi)), (size, road, off)
        for off in (SIDE_LANE + 0.13, RISER_LANE + 0.1, FADER_BASE + FADER_SWING + 0.03):
            for sign in (-1, 1):
                sx, _ = lay.proj(road + sign * off, ahead)
                assert np.all((sx >= lo) & (sx <= hi)), (size, road, off)
    if size == (5760, 1080):
        assert (lo, hi) == (1920, 3840)


def test_fader_lanes_clear_the_shoulders():
    assert FADER_BASE - 0.03 > SHOULDER_HALF > RISER_LANE + 0.045 and SIDE_LANE + 0.13 < RISER_LANE


def test_wide_window_widens_world_without_stretch():
    single, triple = Layout(1920, 1080), Layout(5760, 1080)
    assert triple.focal == single.focal and triple.cam_h == single.cam_h
    a, b = single.proj(0.7, 1.0), triple.proj(0.7, 1.0)
    assert b[0] - triple.cx == pytest.approx(a[0] - single.cx) and b[1] == pytest.approx(a[1])


def _road_chart(keys: list[tuple[float, float]]) -> Chart:
    return Chart(title="r", bpm=120.0, length=10.0, road=[RoadKey(t, x) for t, x in keys])


@pytest.mark.parametrize("keys", [
    [], [(2.0, 0.5)], [(0.0, 0.0), (1.0, 0.5), (1.0, -0.5), (3.0, 0.2)],
    [(0.0, 0.0), (2.0, 1.0)], [(0.0, 0.0), (1.0, -1.0), (1.0, -1.0), (4.0, 0.3)],
])
def test_road_path_matches_chart_edge_cases(keys):
    chart = _road_chart(keys)
    ts = [-1.0, 0.0, 0.55 * 2.0, 1.0, 1.5, 2.0, 2.5, 5.0, 12.0]
    if len(keys) >= 2:
        a, b = keys[0][0], keys[1][0]
        ts += [a + 0.55 * (b - a), a + 0.55 * (b - a) + 1e-9]   # u exactly 0.55
    np.testing.assert_allclose(RoadPath(chart).at(ts), [chart.road_x(t) for t in ts], atol=1e-12)


def test_road_path_matches_chart(fixture_chart):
    road = RoadPath(fixture_chart)
    ts = np.linspace(-1, fixture_chart.length + 1, 997)
    np.testing.assert_allclose(road.at(ts), [fixture_chart.road_x(t) for t in ts], atol=1e-12)


def test_note_positions_match_road_x(fixture_chart):
    geo = SceneGeo(fixture_chart, LOOK)
    for i, n in enumerate(fixture_chart.notes):
        x = geo.note_x(i)
        rx = fixture_chart.road_x(n.t)
        match n.kind:
            case "gate":
                assert x is None if n.blind else x == n.x
            case "kick" | "hat":
                assert x == pytest.approx(rx - SIDE_LANE)
            case "stab" | "tom":
                assert x == pytest.approx(rx + SIDE_LANE)
            case "spin":
                assert x == pytest.approx(rx)
            case "riser":
                assert x == pytest.approx(rx + RISER_LANE)
            case "fader":
                side = -1 if n.lever == 0 else 1
                assert x == pytest.approx(rx + side * (FADER_BASE + FADER_SWING * n.value_at(n.t)))
            case _:
                assert x is None


def test_drawn_road_is_the_chart_road_away_from_blind_runs(sample_chart):
    geo = SceneGeo(sample_chart, LOOK)
    (o, c), = geo.runs
    keys = [k.t for k in sample_chart.road]
    before = max(t for t in keys if t < o)
    after = min((t for t in keys if t > c), default=sample_chart.length)
    ts = np.linspace(0, sample_chart.length, 2001)
    away = (ts <= before) | (ts >= after)
    np.testing.assert_allclose(geo.road.at(ts[away]), [sample_chart.road_x(t) for t in ts[away]], atol=1e-12)
    inside = (ts >= o) & (ts <= c)
    assert np.all(geo.road.at(ts[inside]) == 0.0)
    steepest = np.abs(np.diff([sample_chart.road_x(t) for t in ts])).max()
    assert np.abs(np.diff(geo.road.at(ts))).max() <= steepest + 1e-9   # no snap into or out of the run


def test_curve_cache_matches_chart(fixture_chart):
    geo = SceneGeo(fixture_chart, LOOK)
    for i, n in enumerate(fixture_chart.notes):
        if n.curve:
            ts = np.linspace(n.t - 0.5, n.end + 0.5, 50)
            np.testing.assert_allclose(geo.value(i, ts), [n.value_at(t) for t in ts])


@pytest.fixture
def sample_chart() -> Chart:
    return Chart.load(SAMPLE_CHART)


def _moved_blind_gates(chart: Chart) -> Chart:
    """The same chart with every blind gate, and the road keyframes on them, elsewhere."""
    other = copy.deepcopy(chart)
    moved = {}
    for n in other.notes:
        if n.kind == "gate" and n.blind:
            n.x = -0.9 if (n.x or 0.0) >= 0 else 0.9
            moved[n.t] = n.x
    other.road = [RoadKey(k.t, moved.get(k.t, k.x)) for k in other.road]
    return other


def test_sample_chart_has_shoulder_notes_in_a_blind_region(sample_chart):
    blind = [n.t for n in sample_chart.notes if n.kind == "gate" and n.blind]
    inside = {n.kind for n in sample_chart.notes if min(blind) - 0.5 <= n.t <= max(blind) and n.kind != "gate"}
    assert {"kick", "hat", "stab", "fader"} <= inside
    assert any(sample_chart.road_x(t) != 0 for t in blind)


def _wide_jump_chart() -> Chart:
    """First blind gate 0.9 lane units from the previous keyframe, four beats after it,
    with shoulder notes inside the run."""
    notes = [{"t": 1.0, "kind": "gate", "x": 0.0}, {"t": 4.0, "kind": "gate", "x": 0.0},
             {"t": 6.0, "kind": "gate", "x": 0.9, "blind": True}, {"t": 6.5, "kind": "gate", "x": 0.5, "blind": True},
             {"t": 7.0, "kind": "gate", "x": 0.3, "blind": True}, {"t": 6.25, "kind": "kick"},
             {"t": 6.75, "kind": "hat"}, {"t": 9.0, "kind": "gate", "x": 0.0}]
    road = [{"t": 0.0, "x": 0.0}] + [{"t": n["t"], "x": n["x"]} for n in notes if n["kind"] == "gate"]
    beats = [{"t": i * 0.5, "s": 1.0 if i % 4 == 0 else 0.5} for i in range(20)]
    return make_chart(notes, length=10.0, road=road, beats=beats)


def _assert_blind_hidden(chart: Chart) -> None:
    """Harness case: from lookahead seconds before a blind run opens until it closes,
    nothing drawn depends on the blind gates' x (checked against a chart whose blind
    gates and road keyframes are elsewhere)."""
    cfg = Config()
    geo = SceneGeo(chart, cfg.lookahead, cfg.good_window)
    moved = SceneGeo(_moved_blind_gates(chart), cfg.lookahead, cfg.good_window)
    (o, c), = geo.runs
    checked = 0
    for _, t, snap in harness_snapshots(chart, cfg, 900, -2.0, chart.length + 1.0):
        if not o - cfg.lookahead - BEHIND <= t <= c or snap.phase != "play":
            continue
        a, b = geo.positions(snap), moved.positions(snap)
        assert a == b, t
        checked += 1
        if geo.in_blind(t):
            assert all(x == 0.0 for k, x in a if k in ("road", "hit", "beat"))
    assert checked > 50
    assert all(geo.note_x(i) is None for i, n in enumerate(chart.notes) if n.kind == "gate" and n.blind)


def test_blind_phrase_is_never_revealed(sample_chart):
    _assert_blind_hidden(sample_chart)


def test_blind_phrase_is_never_revealed_after_a_wide_jump():
    _assert_blind_hidden(_wide_jump_chart())


@pytest.mark.parametrize("which", ["sample", "wide_jump"])
def test_renderer_road_agrees_with_chart_drawn_road(which, sample_chart):
    """The road built once from chart.drawn_road(good_window), and the renderer's own
    fallback construction, both match chart.drawn_road_x everywhere."""
    chart = sample_chart if which == "sample" else _wide_jump_chart()
    gw = Config().good_window
    ts = np.linspace(-1, chart.length + 1, 3001)
    ref = [chart.drawn_road_x(t, gw) for t in ts]
    for use_chart in (True, False):
        geo = SceneGeo(chart, LOOK, gw, use_chart_road=use_chart)
        assert geo.chart_drawn is use_chart and isinstance(geo.road, RoadPath)
        np.testing.assert_allclose(geo.road.at(ts), ref, atol=1e-12)
        assert geo.zones == pytest.approx(chart.blind_zones(gw))


def test_drawn_road_keyframes_come_from_the_chart(sample_chart, monkeypatch):
    calls = []
    real = type(sample_chart).drawn_road

    def spy(self, good_window=0.12):
        calls.append(good_window)
        return real(self, good_window)

    monkeypatch.setattr(type(sample_chart), "drawn_road", spy)
    monkeypatch.setattr(type(sample_chart), "drawn_road_x", lambda *a: pytest.fail("sampled per point"))
    geo = SceneGeo(sample_chart, LOOK, 0.09)
    geo.positions(Snapshot(now=9.0, phase="play"))
    assert calls == [0.09]


def test_listen_gate_just_before_the_run_keeps_the_real_road():
    notes = [{"t": 1.0, "kind": "gate", "x": 0.0},
             {"t": 3.0, "kind": "gate", "x": 0.2, "listen": True}, {"t": 3.5, "kind": "gate", "x": 0.4, "listen": True},
             {"t": 3.8, "kind": "gate", "x": 0.2, "blind": True}, {"t": 4.3, "kind": "gate", "x": 0.4, "blind": True}]
    road = [{"t": 0.0, "x": 0.0}] + [{"t": n["t"], "x": n["x"]} for n in notes]
    chart = make_chart(notes, length=6.0, road=road)
    geo = SceneGeo(chart, LOOK, 0.12)
    (o, c), = geo.runs
    assert o == pytest.approx(3.5 + 0.12) and c == pytest.approx(4.3 + 0.12)
    assert geo.road.at(3.5)[0] == pytest.approx(chart.road_x(3.5)) == 0.4


def test_blind_popups_and_marker(sample_chart):
    geo = SceneGeo(sample_chart, LOOK)
    (o, c), = geo.runs
    t = (o + c) / 2
    assert geo.popup_x(Popup("MISS", "melody", 0.7, t)) == 0.0
    assert geo.popup_x(Popup("MISS", "kick", 0.7 - SIDE_LANE, t)) == -SIDE_LANE
    assert geo.popup_x(Popup("MISS", "pads", 0.7 + SIDE_LANE, t)) == SIDE_LANE
    assert geo.popup_x(Popup("MISS", "pads", 0.7, 1.0)) == 0.7
    late = Popup("MISS", "melody", -0.35, c + 0.05)       # last blind gate finalized after the close
    assert geo.in_zone(late.t0) and not geo.in_blind(late.t0)
    assert geo.popup_x(late) == pytest.approx(float(geo.road.at(late.t0)[0]))
    zone_a, zone_b = geo.zones[0]
    assert zone_a < o and zone_b > c
    snap = Snapshot(on_road=False, phase="play", now=(zone_a + o) / 2)   # before open, inside the zone
    assert geo.chart_drawn and not marker_on_road(snap, geo)            # the game's on_road decides
    snap.on_road = True
    assert marker_on_road(snap, geo)
    old = SceneGeo(sample_chart, LOOK, use_chart_road=False)           # a chart without drawn_road
    snap.on_road = False
    assert marker_on_road(snap, old)
    snap.now = zone_a - 0.1
    assert not marker_on_road(snap, old)


def test_game_reports_on_road_through_blind_zones(sample_chart):
    cfg = Config()
    geo = SceneGeo(sample_chart, cfg.lookahead, cfg.good_window)
    seen = 0
    for _, t, snap in harness_snapshots(sample_chart, cfg, 900, -2.0, sample_chart.length + 1.0):
        if t >= 0 and snap.phase == "play" and geo.in_zone(t):
            assert snap.on_road and snap.road_x == pytest.approx(float(geo.road.at(t)[0]))
            seen += 1
    assert seen > 20


def test_spin_direction_and_turn_offset_read_with_defaults():
    assert spin_dir(SimpleNamespace(dir="cw")) == 1 and spin_dir(SimpleNamespace(dir="ccw")) == -1
    assert spin_dir(SimpleNamespace()) == 0 and spin_dir(SimpleNamespace(dir=None)) == 0
    old = SimpleNamespace(input=SimpleNamespace(steer=0.3))          # a snapshot without the new fields
    assert steer_lane(old) == 0.3 and turn_offset(old) == 0 and hud.turn_tag(0) == ""
    snap = Snapshot(steer_lane=-0.2, steer_offset_deg=360.0)
    assert steer_lane(snap) == -0.2 and turn_offset(snap) == 1 and hud.turn_tag(1) == "+1 TURN"
    assert hud.turn_tag(turn_offset(Snapshot(steer_offset_deg=-360.0))) == "-1 TURN"


def test_harness_exercises_spins_turn_offset_and_unwind(sample_chart):
    """The packaged chart has a cw and a ccw spin; the scripted run completes both,
    shows the +1 turn tag between them, and passes through the unwind state, where the
    hint points toward the offset. On the frame the offset changes, the marker is
    simply at steer_lane (nothing is smoothed)."""
    assert sorted(spin_dir(n) for n in sample_chart.notes if n.kind == "spin") == [-1, 1]
    from torquehero.chart import playability

    assert playability(sample_chart) == []
    cfg = Config()
    offsets, hints, changed = set(), set(), 0
    for _, t, snap in harness_snapshots(sample_chart, cfg, 1200, -2.0, sample_chart.length + 1.0):
        if t < 0 or snap.phase != "play":
            continue
        offsets.add(turn_offset(snap))
        d = turn_back_dir(snap, sample_chart)
        if d:
            hints.add(d)
            assert d == (1 if snap.steer_offset_deg > snap.input.steer_deg else -1)
        if snap.steer_offset_changed or snap.steer_lane_jump:
            changed += 1
            assert steer_lane(snap) == snap.steer_lane   # placed, never smoothed
    assert offsets == {0, 1} and hints == {1} and changed >= 2
    spins = [st for k, st in snap.layers.items() if k == "melody"]
    assert spins and spins[0].hit > 0


def test_turn_back_waits_while_a_spin_runs(sample_chart):
    i = next(i for i, n in enumerate(sample_chart.notes) if n.kind == "spin")
    n = sample_chart.notes[i]
    snap = Snapshot(now=n.t + 0.5, steer_unwind=True, steer_offset_deg=0.0, notes=[NoteView(i)])
    snap.input.steer_deg = 200.0
    assert turn_back_dir(snap, sample_chart) == 0
    snap.notes = [NoteView(i, done=True, result="perfect")]
    assert turn_back_dir(snap, sample_chart) == -1
    snap.steer_unwind = False
    assert turn_back_dir(snap, sample_chart) == 0


def test_glyph_fallback_is_per_character():
    charset = frozenset(range(32, 127)) | {0xE9}
    assert displayable("Café → Señor", charset) == "Café ? Se?or"
    assert set(range(0xA0, 0x180)) <= set(FONT_CODEPOINTS) and set(range(32, 127)) <= set(FONT_CODEPOINTS)
    assert {ord(c) for c in HUD_SYMBOLS} <= set(FONT_CODEPOINTS) and 0x2192 not in FONT_CODEPOINTS


def _snap(alive: int) -> Snapshot:
    layers = {k: LayerStatus(alive=i < alive) for i, k in enumerate(LAYERS)}
    return Snapshot(layers=layers)


def test_world_reacts_to_mix_and_beat():
    assert alive_fraction(_snap(8)) == 1.0 and alive_fraction(_snap(2)) == 0.25
    snap = _snap(0)
    snap.layers["hat"].mode = "auto"
    assert alive_fraction(snap) == 1 / 8
    dark, bright = sky_state(0.0, 0.5, 1.0), sky_state(1.0, 0.5, 1.0)
    assert bright.sun_r > dark.sun_r and bright.star_alpha > dark.star_alpha
    assert sum(bright.horizon[:3]) > sum(dark.horizon[:3])
    on_beat, off_beat = sky_state(1.0, 0.0, 1.0), sky_state(1.0, 0.99, 1.0)
    assert on_beat.sun_r > off_beat.sun_r and on_beat.star_alpha >= off_beat.star_alpha


def test_snapshot_round_trip():
    d = json.loads((FIXTURES / "snapshot.json").read_text())
    snap = snapshot_from_dict(d)
    assert json.loads(snap.to_json()) == d
    assert isinstance(snap.input.down, frozenset) and snap.echo == "listen"


def test_settings_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("TORQUEHERO_CONFIG_DIR", str(tmp_path))
    assert RenderSettings.load() == RenderSettings()
    RenderSettings(fps=240, font="x.ttf").save()
    assert (tmp_path / "render.json").exists()
    assert RenderSettings.load() == RenderSettings(fps=240, font="x.ttf")


@pytest.mark.parametrize("which", ["fixture", "sample"])
def test_scripted_player_plays_every_layer(which, fixture_chart, sample_chart):
    chart = fixture_chart if which == "fixture" else sample_chart
    game = Game(chart, Config(), {k: "you" for k in LAYERS})
    prev: frozenset[str] = frozenset()
    dt = 1 / 30
    for f in range(int((chart.length + 1) / dt)):
        inp = scripted_input(chart, f * dt, dt, prev)
        prev = inp.down
        game.update(f * dt, dt, inp)
    snap = game.snapshot()
    for k, st in snap.layers.items():
        assert st.total > 0 and st.hit > 0, k
    assert snap.layers["pads"].hit < snap.layers["pads"].total  # the scripted wrong gate
    assert game.finished


def test_hud_helpers():
    assert hud.grade(0.97) == "S" and hud.grade(0.1) == "D"
    assert hud.layer_accuracy(3, 4) == 0.75 and hud.layer_accuracy(0, 0) is None
    assert hud.ffb_log_lines(["a", {"s": "b"}, {"text": "c"}]) == ["a", "b", "c"]
    snap = Snapshot(now=10.0, upcoming=[{"t": 9.0, "kind": "spin"}, {"t": 12.0, "kind": "stab", "gate": 3}])
    label, dt = hud.next_special(snap)
    assert label == "STAB 3" and dt == pytest.approx(2.0)


def test_modules_import_without_pyray():
    code = ("import sys, torquehero.render, torquehero.hud; "
            "assert 'pyray' not in sys.modules and 'raylib' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_raylib_keys_is_a_key_source():
    from torquehero.render import RaylibKeys

    keys = RaylibKeys()
    assert isinstance(keys, KeySource)
    assert keys._code("SPACE") is not None and keys._code("MOUSE_LEFT") == (True, keys.rl.MOUSE_BUTTON_LEFT)
    assert keys._code("NOT_A_KEY") is None and not keys.is_down("NOT_A_KEY")


# --- window code, against a fake pyray/raylib (no display) ---

class _Any:
    """Permissive stand-in for raylib structs and handles."""

    x = y = width = height = glyphCount = 0

    def __getattr__(self, name):
        return _Any()


class FakeRaylib:
    """Records every call in order. Constants are ints; unknown functions return _Any."""

    def __init__(self, calls: list, screen=(1920, 1080), render=None):
        import cffi

        self.calls, self.screen, self.render = calls, list(screen), render or screen
        self.ffi = cffi.FFI()
        self.ffi.cdef("typedef struct { void *data; int width; int height; int mipmaps; int format; } Image;"
                      "typedef struct { float x; float y; } Vector2;")
        self.keep: list = []
        self.ready, self.minimised, self.fail_on = True, False, None
        self.font_loads: list[bytes] = []

    def __getattr__(self, name):
        if name.isupper():
            return sum(map(ord, name))           # constants: stable ints
        if name[:1].isupper() and "_" + name in type(self).__dict__:
            return getattr(self, "_" + name)     # raw raylib functions with behaviour

        def call(*args):
            self.calls.append(name)
            if name == self.fail_on:
                raise RuntimeError(f"fake failure in {name}")
            return _Any()
        return call

    # values the window code reads
    def is_window_ready(self):
        return self.ready

    def get_screen_width(self):
        return self.screen[0]

    def get_screen_height(self):
        return self.screen[1]

    def get_render_width(self):
        return self.render[0]

    def get_render_height(self):
        return self.render[1]

    def get_monitor_count(self):
        return 1

    def get_monitor_position(self, i):
        return SimpleNamespace(x=0, y=0)

    def get_monitor_width(self, i):
        return 1920

    def get_monitor_height(self, i):
        return 1080

    def get_monitor_refresh_rate(self, i):
        return 60

    def get_current_monitor(self):
        return 0

    def is_window_fullscreen(self):
        return False

    def is_window_minimized(self):
        return self.minimised

    def window_should_close(self):
        return False

    def set_target_fps(self, fps):
        self.calls.append(("set_target_fps", fps))

    def set_exit_key(self, key):
        self.calls.append(("set_exit_key", key))

    def _image(self, w, h):
        buf = self.ffi.new("unsigned char[]", w * h * 4)
        self.keep.append(buf)
        return self.ffi.new("Image *", {"data": buf, "width": w, "height": h, "mipmaps": 1})[0]

    def rl_read_screen_pixels(self, w, h):
        self.calls.append(("rl_read_screen_pixels", w, h))
        buf = self.ffi.new("unsigned char[]", w * h * 4)
        self.keep.append(buf)
        return buf

    def load_image_from_texture(self, tex):
        self.calls.append("load_image_from_texture")
        return self._image(4, 4)

    def gen_image_color(self, w, h, col):
        return self._image(w, h)

    def get_font_default(self):
        return _Any()

    def measure_text_ex(self, *args):
        return SimpleNamespace(x=10.0, y=10.0)

    def _ExportImageToMemory(self, img, ext, n):
        self.calls.append("ExportImageToMemory")
        data = self.ffi.new("unsigned char[]", b"\x89PNGfake")
        self.keep.append(data)
        n[0] = 8
        return data

    def _MemFree(self, p):
        pass

    def _LoadFontFromMemory(self, ext, data, size, font_size, cps, n):
        self.font_loads.append(bytes(self.ffi.buffer(data, size)))
        return SimpleNamespace(glyphCount=0)


@pytest.fixture
def fake_rl(monkeypatch, tmp_path):
    monkeypatch.setenv("TORQUEHERO_CONFIG_DIR", str(tmp_path / "config"))   # never the user's config
    calls: list = []
    fake = FakeRaylib(calls)
    monkeypatch.setitem(sys.modules, "pyray", fake)
    monkeypatch.setitem(sys.modules, "raylib", fake)
    return fake


def _names(calls):
    return [c if isinstance(c, str) else c[0] for c in calls]


def test_window_context_manager_and_idempotent_close(fake_rl):
    from torquehero.render import Window

    closed = []
    with Window(size=(640, 360)) as win:
        win.own(SimpleNamespace(close=lambda: closed.append(1)))
    assert _names(fake_rl.calls).count("close_window") == 1 and closed == [1]
    win.close()
    win.close()
    assert _names(fake_rl.calls).count("close_window") == 1 and closed == [1]


def test_window_failure_after_init_closes(fake_rl):
    from torquehero.render import Window

    fake_rl.fail_on = "rl_disable_backface_culling"
    with pytest.raises(RuntimeError, match="fake failure"):
        Window(size=(640, 360))
    names = _names(fake_rl.calls)
    assert names.index("init_window") < names.index("close_window")


def test_window_without_display_raises_cleanly(fake_rl):
    from torquehero.render import Window

    fake_rl.ready = False
    with pytest.raises(RuntimeError, match="no display"):
        Window(size=(640, 360))


def test_exit_key_disabled_and_fps_follows_vsync(fake_rl):
    from torquehero.render import Window

    Window(settings=RenderSettings(vsync=True, fps=90)).close()
    Window(settings=RenderSettings(vsync=False, fps=90)).close()
    fps = [c[1] for c in fake_rl.calls if isinstance(c, tuple) and c[0] == "set_target_fps"]
    assert fps == [0, 90]
    assert all(c[1] == fake_rl.KEY_NULL for c in fake_rl.calls if isinstance(c, tuple) and c[0] == "set_exit_key")


def test_span_and_fullscreen_rejected(fake_rl):
    from torquehero.render import Window

    with pytest.raises(ValueError, match="exclude"):
        Window(span=(5760, 1080, 0, 0), fullscreen=True)
    assert "init_window" not in _names(fake_rl.calls)


def test_draw_skipped_when_minimised_or_zero_size(fake_rl, monkeypatch):
    from torquehero import render
    from torquehero.render import Window

    sleeps = []
    monkeypatch.setattr(render.time, "sleep", sleeps.append)
    drawn = []
    with Window(size=(640, 360)) as win:
        fake_rl.screen = [0, 0]
        assert win.frame(lambda: drawn.append(1)) is False
        fake_rl.screen, fake_rl.minimised = [640, 360], True
        assert win.frame(lambda: drawn.append(1)) is False
        fake_rl.minimised = False
        assert win.frame(lambda: drawn.append(1)) is True
    assert drawn == [1] and sleeps == [render.MINIMISED_SLEEP] * 2


def test_capture_after_frame_is_complete_at_framebuffer_size(fake_rl, tmp_path):
    from torquehero.render import Window

    fake_rl.render = (2400, 1350)                 # 125% display scaling
    path = tmp_path / "écran ünï" / "shot.png"
    with Window(size=(1920, 1080)) as win:
        win.frame(lambda: fake_rl.calls.append("DRAW"), capture=path)
    names = _names(fake_rl.calls)
    order = [names.index(k) for k in ("DRAW", "rl_draw_render_batch_active", "rl_read_screen_pixels",
                                      "ExportImageToMemory")]
    assert order == sorted(order) and order[-1] < len(names) - names[::-1].index("end_drawing") - 1
    assert ("rl_read_screen_pixels", 2400, 1350) in fake_rl.calls
    assert path.read_bytes() == b"\x89PNGfake"


def test_hidden_capture_and_non_ascii_font_path(fake_rl, tmp_path):
    from torquehero.render import Painter, Window

    font = tmp_path / "pölice" / "fönt.ttf"
    font.parent.mkdir()
    font.write_bytes(b"not really a font")
    Painter(RenderSettings(font=str(font)))
    assert fake_rl.font_loads[0] == b"not really a font"   # tried first, read by Python
    with Window(size=(64, 36), hidden=True) as win:
        win.frame(lambda: None, capture=tmp_path / "ß.png")
    assert (tmp_path / "ß.png").read_bytes() == b"\x89PNGfake"


def test_frame_cap_when_vsync_is_ignored(fake_rl, monkeypatch):
    from torquehero import render
    from torquehero.render import VSYNC_PROBE_FRAMES, Window

    assert needs_frame_cap([0.001] * 10, 60) and not needs_frame_cap([1 / 60] * 10, 60)
    assert not needs_frame_cap([], 60)
    clock = iter(i * 0.001 for i in range(10_000))
    monkeypatch.setattr(render.time, "perf_counter", lambda: next(clock))
    with Window(settings=RenderSettings(vsync=True, fps=144)) as win:
        for _ in range(VSYNC_PROBE_FRAMES + 5):
            win.frame(lambda: None)
    fps = [c[1] for c in fake_rl.calls if isinstance(c, tuple) and c[0] == "set_target_fps"]
    assert fps == [0, 144]


def test_bench_prints_three_numbers(fake_rl, capsys):
    from torquehero.render import main

    assert main(["--bench", "--frames", "5", "--snapshot", ""]) == 0
    out = capsys.readouterr().out
    import re

    m = re.search(r"mean ([\d.]+) ms, p99 ([\d.]+) ms, max ([\d.]+) ms", out)
    assert m and "5760x1080" in out


def test_span_with_fullscreen_rejected_by_harness(fake_rl, capsys):
    from torquehero.render import main

    assert main(["--span", "5760x1080+0+0", "--fullscreen", "--frames", "3"]) == 2
    assert "exclude each other" in capsys.readouterr().err


def _raylib_uses() -> list[tuple[str, str, int | None, int]]:
    """(module, name, positional arg count or None, line) of every pyray/raylib name
    render.py and hud.py use: `rl.x`, `self.rl.x`, `r.rl.x` (pyray) and `self.raw.X`,
    `raylib.X` (raw). The fake accepts anything; this pins the real API."""
    import ast
    from pathlib import Path

    import torquehero

    out = []
    for f in ("render.py", "hud.py"):
        tree = ast.parse((Path(torquehero.__file__).parent / f).read_text())
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr == "ffi":
                continue
            base = node.value
            owner = base.attr if isinstance(base, ast.Attribute) else base.id if isinstance(base, ast.Name) else ""
            mod = {"rl": "pyray", "raw": "raylib", "raylib": "raylib"}.get(owner)
            if mod is None:
                continue
            call = parents.get(node)
            argc = None
            if isinstance(call, ast.Call) and call.func is node and not any(
                    isinstance(a, ast.Starred) for a in call.args) and not call.keywords:
                argc = len(call.args)
            out.append((mod, node.attr, argc, node.lineno))
    return out


def _c_name(snake: str) -> str:
    if snake.startswith("rl_"):
        return "rl" + "".join(w.title() for w in snake[3:].split("_"))
    return "".join(w.title() for w in snake.split("_"))


def test_every_raylib_name_exists_with_its_arity():
    import pyray
    import raylib

    uses = _raylib_uses()
    assert len(uses) > 60
    for mod, name, argc, line in uses:
        real = pyray if mod == "pyray" else raylib
        assert hasattr(real, name), f"{mod}.{name} (line {line}) does not exist"
        if argc is None or name[:1].isupper() and mod == "pyray":
            continue
        c = getattr(raylib, name if mod == "raylib" else _c_name(name), None)
        if c is None or not callable(c):
            continue
        n_args = len(raylib.ffi.typeof(c).args)
        assert argc == n_args, f"{mod}.{name} (line {line}) called with {argc} args, takes {n_args}"
