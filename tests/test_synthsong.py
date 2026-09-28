import json
import os
import time
import wave
from collections import Counter

import numpy as np
import pytest

from torquehero import synthsong
from torquehero.__main__ import main
from torquehero.chart import DEFAULT_LAYER_AUDIO, NOTE_KINDS, Chart, playability, validate_dict

HAND_MOVE = 0.4  # SPEC 9: seconds to move a hand or a foot between controls


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    return synthsong.build(out)


@pytest.fixture(scope="module")
def chart(built) -> Chart:
    return Chart.load(built)


def test_chart_is_valid(built):
    assert validate_dict(json.loads(built.read_text())) == []


def test_chart_is_playable(built):
    assert playability(json.loads(built.read_text()), "normal") == []


def test_every_layer_defaults_to_you(chart):
    assert chart.default_layers == dict.fromkeys(DEFAULT_LAYER_AUDIO, "you")


def test_note_counts_per_kind(chart):
    counts = Counter(n.kind for n in chart.notes)
    assert counts == {"gate": 186, "spin": 2, "kick": 96, "hat": 63, "expr": 12, "stab": 16,
                      "tom": 16, "riser": 2, "fader": 3}
    assert set(counts) == set(NOTE_KINDS)


def test_echo_phrase_repeats_blind(chart):
    listen = [(n.t, n.x) for n in chart.notes if n.listen]
    blind = [(n.t, n.x) for n in chart.notes if n.blind]
    assert len(listen) == len(blind) == 8
    assert [x for _, x in listen] == [x for _, x in blind]
    assert min(t for t, _ in blind) > max(t for t, _ in listen)


def test_road_meets_every_gate(chart):
    for n in chart.notes:
        if n.kind == "gate":
            assert chart.road_x(n.t) == pytest.approx(n.x), n


def test_faders_use_both_levers(chart):
    faders = [n for n in chart.notes if n.kind == "fader"]
    assert {n.lever for n in faders} == {0, 1}
    assert all(len(n.curve) > 2 for n in faders)


def test_riser_tone_has_no_partials_near_nyquist():
    x = synthsong.shepard(synthsong._n(synthsong.BAR))
    power = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / 48000)
    assert power[f > 20000].sum() < 1e-6 * power.sum()


def test_kick_velocity_accents_downbeats(chart):
    kicks = [n for n in chart.notes if n.kind == "kick"]
    downbeats = {round(b.t, 6) for b in chart.beats if b.s == 1.0}
    assert {n.vel for n in kicks if n.t in downbeats} == {1.0}
    assert {n.vel for n in kicks if n.t not in downbeats} == {0.7, 0.85}


def test_sections_and_beats(chart):
    assert [s.name for s in chart.sections] == [name for name, _, _ in synthsong.STRUCT]
    assert len(chart.beats) == synthsong.BARS * 4
    assert chart.bpm == 118


def test_manifest_follows_spec(chart, built):
    a = json.loads(built.read_text())["audio"]
    assert a["sr"] == 48000
    assert a["backing"] == ["stems/backing.wav"]
    assert set(a["stems"]) == {"lead", "pad", "bass"}
    assert set(a["oneshots"]) == {"kick", "hat", "tom_l", "tom_r", "riser", "impact", "dud",
                                  *(f"stab{g}" for g in range(1, 7))}
    assert a["layers"] == DEFAULT_LAYER_AUDIO
    for layer in ("kick", "hat", "pads", "fills"):
        assert a["layers"][layer]["mode"] == "trigger"
    for rel in [*a["backing"], *a["stems"].values(), *a["oneshots"].values()]:
        assert chart.path(rel).is_file(), rel


def _wavs(chart):
    a = chart.audio
    return [*a.backing, *a.stems.values(), *a.oneshots.values()]


def test_stems_have_chart_length(chart):
    for rel in [*chart.audio.backing, *chart.audio.stems.values()]:
        with wave.open(str(chart.path(rel))) as w:
            assert (w.getframerate(), w.getnchannels()) == (48000, 1)
            assert w.getnframes() == round(chart.length * 48000), rel


def test_no_file_clips(chart):
    for rel in _wavs(chart):
        x = synthsong.read_wav(chart.path(rel))
        assert np.abs(x).max() < synthsong.PEAK_CEILING, rel


def test_oneshots_start_and_end_silent(chart):
    for name, rel in chart.audio.oneshots.items():
        x = synthsong.read_wav(chart.path(rel))
        if name == "riser":  # a loop: the seam must be as smooth as any other step
            assert abs(x[0] - x[-1]) < 3 * np.abs(np.diff(x)).mean()
        else:
            assert abs(x[0]) < 1e-3 and abs(x[-1]) < 1e-3, name


def test_auto_mix_does_not_clip(chart):
    """Every layer on auto, as the audio engine would play it."""
    wav = {rel: synthsong.read_wav(chart.path(rel)) for rel in _wavs(chart)}
    mix = np.sum([wav[rel] for rel in [*chart.audio.backing, *chart.audio.stems.values()]], axis=0)

    def put(t, x, gain=1.0):
        i = round(t * 48000)
        j = min(len(mix), i + len(x))
        mix[i:j] += gain * x[: j - i]

    for n in chart.notes:
        name = chart.audio.oneshot_for(n.layer, n)
        if n.kind == "riser":
            loop = wav[chart.audio.oneshots[name]]
            k = round(n.dur * 48000)
            put(n.t, np.tile(loop, k // len(loop) + 1)[:k])
            put(n.end, wav[chart.audio.oneshots["impact"]])
        elif name is not None:
            put(n.t, wav[chart.audio.oneshots[name]], n.vel or 1.0)
    assert np.abs(mix).max() < synthsong.PEAK_CEILING


def test_render_is_deterministic(built, tmp_path):
    again = synthsong.build(tmp_path)
    assert again.read_bytes() == built.read_bytes()
    for p in sorted(built.parent.rglob("*.wav")):
        assert (tmp_path / p.relative_to(built.parent)).read_bytes() == p.read_bytes(), p


@pytest.fixture
def fake_build(monkeypatch, built):
    """synthsong.build that writes only chart.json; `calls` lists its output dirs."""
    calls = []

    def fake(out):
        calls.append(out)
        out.mkdir(parents=True)
        (out / "chart.json").write_text(built.read_text())
        return out / "chart.json"

    monkeypatch.setattr(synthsong, "build", fake)
    return calls


def _target(root):
    return root / f"demo-song-v{synthsong.VERSION}"


def test_demo_chart_path_renders_once(tmp_path, fake_build):
    first = synthsong.demo_chart_path(tmp_path)
    assert first == _target(tmp_path) / "chart.json"
    assert synthsong.demo_chart_path(tmp_path) == first
    assert len(fake_build) == 1
    assert [p.name for p in tmp_path.iterdir()] == [first.parent.name]


def test_demo_chart_path_keeps_complete_target(tmp_path, fake_build):
    _target(tmp_path).mkdir()
    (_target(tmp_path) / "chart.json").write_text("{}")
    assert synthsong.demo_chart_path(tmp_path).read_text() == "{}"
    assert fake_build == []


def test_demo_chart_path_replaces_partial_target(tmp_path, fake_build):
    (_target(tmp_path) / "stems").mkdir(parents=True)
    (_target(tmp_path) / "stems" / "lead.wav").write_bytes(b"half")
    chart = synthsong.demo_chart_path(tmp_path)
    assert chart.read_text() != ""
    assert not (_target(tmp_path) / "stems").exists()
    assert [p.name for p in tmp_path.iterdir()] == [_target(tmp_path).name]


def test_demo_chart_path_lost_race_keeps_winner(tmp_path, monkeypatch):
    def racing_build(out):
        winner = _target(tmp_path)
        winner.mkdir()
        (winner / "chart.json").write_text("winner")
        out.mkdir()
        (out / "chart.json").write_text("loser")
        return out / "chart.json"

    monkeypatch.setattr(synthsong, "build", racing_build)
    assert synthsong.demo_chart_path(tmp_path).read_text() == "winner"
    assert [p.name for p in tmp_path.iterdir()] == [_target(tmp_path).name]


def test_demo_chart_path_reports_the_first_move_error(tmp_path, fake_build, monkeypatch):
    def refuse(src, dst):
        raise PermissionError(13, "locked", str(dst))

    monkeypatch.setattr(synthsong.os, "replace", refuse)
    with pytest.raises(OSError, match="cannot move the demo song render.*locked") as e:
        synthsong.demo_chart_path(tmp_path)
    assert isinstance(e.value.__cause__, PermissionError)
    assert list(tmp_path.iterdir()) == []


def test_demo_chart_path_removes_stale_temp_dirs(tmp_path, fake_build):
    stale, fresh = tmp_path / "demo-song-v1.tmp99", tmp_path / f"demo-song-v{synthsong.VERSION}.tmp98"
    stale.mkdir()
    fresh.mkdir()
    old = time.time() - 2 * synthsong.STALE_TMP
    os.utime(stale, (old, old))
    synthsong.demo_chart_path(tmp_path)
    assert not stale.exists()
    assert fresh.exists()


def test_cache_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TORQUEHERO_CACHE_DIR", str(tmp_path))
    assert synthsong.cache_dir() == tmp_path


def test_cli_demo_song(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(synthsong, "build", lambda out: out / "chart.json")
    assert main(["demo-song", "-o", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "chart.json")


# --- playability: the limb budget (SPEC 9) ---

def _notes(chart, kind):
    return [n for n in chart.notes if n.kind == kind]


def _overlaps(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def _tom_runs(chart):
    runs = []
    for n in _notes(chart, "tom"):
        if runs and n.t - runs[-1][1] <= synthsong.BEAT:
            runs[-1][1] = n.t
        else:
            runs.append([n.t, n.t])
    return runs


def _both_hands_on_wheel(chart):
    return [(n.t, n.end) for n in _notes(chart, "spin")] + [tuple(r) for r in _tom_runs(chart)]


def test_rule_3_stab_clear_of_riser(chart):
    for s in _notes(chart, "stab"):
        for r in _notes(chart, "riser"):
            assert not r.t - HAND_MOVE < s.t < r.end + HAND_MOVE, (s, r)


def test_rule_3_right_hand_and_faders_clear_of_spins_and_toms(chart):
    busy = [(n.t, n.end, n) for kind in ("stab", "riser", "fader") for n in _notes(chart, kind)]
    for a, b in _both_hands_on_wheel(chart):
        for t0, t1, n in busy:
            assert not _overlaps(t0, max(t1, t0 + 1e-9), a - HAND_MOVE, b + HAND_MOVE), (n, a, b)


def test_rule_4_levers_take_turns_without_jumps(chart):
    faders = _notes(chart, "fader")
    for i, a in enumerate(faders):
        for b in faders[i + 1:]:
            assert not _overlaps(a.t, a.end, b.t, b.end), (a, b)
    for lever in (0, 1):
        mine = [n for n in faders if n.lever == lever]
        for a, b in zip(mine, mine[1:], strict=False):
            assert a.curve[-1][1] == pytest.approx(b.curve[0][1]), (a, b)


def test_rule_5_faders_do_not_overlap_right_hand_notes(chart):
    for f in _notes(chart, "fader"):
        for n in _notes(chart, "stab") + _notes(chart, "riser"):
            assert not _overlaps(f.t, f.end, n.t, max(n.end, n.t + 1e-9)), (f, n)


def test_rules_6_7_feet(chart):
    """A kick inside an expr note is the left foot's, so no hat within 0.4 s of it,
    inside the note or outside."""
    hats = [n.t for n in _notes(chart, "hat")]
    for e in _notes(chart, "expr"):
        for k in [n.t for n in _notes(chart, "kick") if e.t <= n.t <= e.end]:
            assert all(abs(k - h) >= HAND_MOVE for h in hats), (e, k)


def test_continuous_layers_end_on_their_held_level(chart):
    for n in _notes(chart, "expr"):
        assert n.curve[-1][1] == pytest.approx(0.55)
    held = {}  # the song opens with lever 1 closed on purpose: the intro filter sweep
    for n in _notes(chart, "fader"):
        if n.lever in held:
            assert n.curve[0][1] == pytest.approx(held[n.lever]), n
        held[n.lever] = n.curve[-1][1]
    assert held[0] == pytest.approx(1.0)


def test_rule_8_steering_speed(chart):
    gates = _notes(chart, "gate")
    for a, b in zip(gates, gates[1:], strict=False):
        assert abs(b.x - a.x) / (b.t - a.t) <= 1.5, (a, b)


def test_rule_9_road_starts_at_zero_and_never_steps(chart):
    assert (chart.road[0].t, chart.road[0].x) == (0.0, 0.0)
    ts = [k.t for k in chart.road]
    assert len(ts) == len(set(ts))
    xs = [chart.road_x(i / 1000) for i in range(int(chart.length * 1000))]
    assert max(abs(b - a) for a, b in zip(xs, xs[1:], strict=False)) < 0.02


def test_rules_10_11_spins(chart):
    spins = _notes(chart, "spin")
    assert [n.dir for n in spins] == ["cw", "ccw"]
    for s in spins:
        for g in _notes(chart, "gate"):
            assert not s.t - synthsong.BEAT < g.t < s.end + synthsong.BEAT, (s, g)
