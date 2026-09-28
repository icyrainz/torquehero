import json
import subprocess
import sys
import types
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from torquehero import generate as gen
from torquehero.chart import Chart, playability, validate_dict
from torquehero.config import Config

SR = 48000
MELODY = [60, 64, 67, 72, 67, 64, 62, 65]
HELD = {*range(40, 46), *range(52, 58)}  # beats that hold one pitch (spin candidates)
STEER_LIMIT = {"easy": 1.0, "normal": 1.5, "hard": 2.5}
DENSITY = {"easy": {"gate": 1.0, "kick": 0.5, "hat": 0.0},
           "normal": {"gate": 1.5, "kick": 1.0, "hat": 1.0},
           "hard": {"gate": 2.5, "kick": 2.0, "hat": 2.0}}
LAYERS_IN = {"easy": {"melody", "kick", "riser"},
             "normal": {"melody", "kick", "riser", "hat", "pads", "fills"},
             "hard": {"melody", "kick", "riser", "hat", "pads", "fills", "expr", "faders"}}


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def beat_times(bpm: float, length: float, drift: float = 0.0, rubato: tuple[float, float] | None = None) -> np.ndarray:
    """Beats of a tempo that rises linearly by `drift` (fraction) over `length` seconds;
    inside `rubato` (a, b) every beat is 25 % shorter to 35 % longer at random."""
    rng = np.random.default_rng(5)
    ts, t = [], 0.0
    while t < length:
        ts.append(t)
        p = 60.0 / (bpm * (1 + drift * t / length))
        t += p * rng.uniform(0.75, 1.35) if rubato and rubato[0] <= t < rubato[1] else p
    return np.array(ts)


def saw(f: float, tt: np.ndarray, phase: float = 0.0) -> np.ndarray:
    """Band-limited sawtooth (harmonics up to 10 kHz)."""
    return sum(np.sin(k * (phase + 2 * np.pi * f * tt)) / k for k in range(1, max(2, int(10000 / f))))


def synth(bpm=120.0, length=32.0, drums=True, meter=4, drift=0.0, pad=0.0, sr=SR, width=0.0,
          lead="sine", bass="sine", rubato=None, held=HELD, bursts=0) -> dict:
    """Stems of a test song: drums (kick on every beat, accented every `meter` beats,
    noise hat on the offbeats), a melody (`lead` "sine" or "saw" with a 1 ms attack),
    a bass (`bass` "sine" or "pluck": a saw note per beat with a fast decay), a pad;
    quiet first half, loud second half; `pad` seconds of silence each side."""
    beats = beat_times(bpm, length, drift, rubato)
    n = int(length * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(1)
    loud = np.where(t < length / 2, 0.25, 1.0)
    click = np.zeros(n)
    edges = [*beats, length]
    for k, bt in enumerate(beats if drums else []):
        i = int(bt * sr)
        m = min(n - i, int(0.3 * sr))
        tt = np.arange(m) / sr
        f = 50 + 100 * np.exp(-tt / 0.03)
        click[i:i + m] += (1.0 if k % meter == 0 else 0.7) * np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt / 0.12)
        click[i:i + int(0.005 * sr)] += rng.standard_normal(len(click[i:i + int(0.005 * sr)])) * 0.3
        j = int((bt + edges[k + 1]) / 2 * sr)
        h = min(n - j, int(0.06 * sr))
        if h > 0:
            noise = np.diff(np.diff(rng.standard_normal(h + 2)))
            click[j:j + h] += 0.3 * noise * np.exp(-np.arange(h) / sr / 0.015)
    vocals = np.zeros(n)
    phase = 0.0
    for k in range(len(beats)):
        a, b = int(edges[k] * sr), int(edges[k + 1] * sr)
        m = (69 if k < 48 else 71) if k in held else MELODY[k % len(MELODY)]
        tt = np.arange(b - a) / sr
        attack = 0.001 if lead == "saw" else 0.01
        env = np.ones_like(tt) if k - 1 in held and k in held else np.minimum(tt / attack, 1)
        tone = saw(midi_hz(m), tt, phase) * 0.35 if lead == "saw" else np.sin(phase + 2 * np.pi * midi_hz(m) * tt)
        vocals[a:b] = 0.4 * env * tone
        phase += 2 * np.pi * midi_hz(m) * len(tt) / sr
    if bass == "pluck":
        bass = np.zeros(n)
        for k in range(len(beats)):
            a, b = int(edges[k] * sr), int(edges[k + 1] * sr)
            tt = np.arange(b - a) / sr
            note = saw((55, 73.4, 82.4, 65.4)[k // 4 % 4], tt)
            bass[a:b] = 0.5 * note * np.exp(-tt / 0.25) * np.minimum(tt / 0.002, 1)
    else:
        bass = 0.3 * np.sin(2 * np.pi * 55 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * t / 4))
    other = 0.1 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * t / 8))
    if bursts:  # noise bursts at random times: transients with no beat
        for t0 in np.sort(rng.uniform(0.5, length - 0.5, bursts)):
            i = int(t0 * sr)
            m = min(n - i, int(0.08 * sr))
            click[i:i + m] += 0.5 * rng.standard_normal(m) * np.exp(-np.arange(m) / sr / 0.02)
    parts = {"drums": click if drums or bursts else np.zeros(n), "vocals": vocals, "bass": bass, "other": other}
    z = np.zeros(int(pad * sr))
    return {k: np.stack([np.concatenate([z, v * loud * (1 - width), z]),
                         np.concatenate([z, v * loud * (1 + width), z])]) for k, v in parts.items()}


def write_song(path: Path, parts: dict, sr: int = SR) -> Path:
    import soundfile as sf

    sf.write(path, sum(parts.values()).T, sr)
    return path


def analyse(parts: dict, **kw):
    return gen.analyse(sum(parts.values()), SR, **kw)


def charts_for(an) -> dict[str, dict]:
    return {d: gen.build_chart(an, d, title="synth") for d in gen.DIFFICULTIES}


# --- SPEC 9, written from the rule text ---

def spec9_violations(chart: dict, difficulty: str) -> list[str]:
    v = []
    notes = chart["notes"]
    beat = 60.0 / chart["bpm"]
    road = chart["road"]
    if not road or road[0] != {"t": 0.0, "x": 0.0}:
        v.append("rule 9: road does not start at t=0, x=0")
    if any(b["t"] <= a["t"] for a, b in zip(road, road[1:], strict=False)):
        v.append("rule 9: road steps")
    prev = (0.0, 0.0)
    for g in (n for n in notes if n["kind"] == "gate"):
        if abs(g["x"] - prev[1]) > STEER_LIMIT[difficulty] * (g["t"] - prev[0]) + 1e-6:
            v.append(f"rule 8: gate at {g['t']} moves too fast")
        prev = (g["t"], g["x"])

    def end(n):
        return n["t"] + n.get("dur", 0.0)

    hands = []
    for n in notes:
        k = n["kind"]
        if k == "stab":
            hands.append(("R", "shifter", n["t"], n["t"], n))
        elif k == "riser":
            hands.append(("R", "handbrake", n["t"], end(n), n))
        elif k in ("spin", "tom"):
            hands += [("L", "wheel", n["t"], end(n), n), ("R", "wheel", n["t"], end(n), n)]
        elif k == "fader":
            hands.append(("L", "lever", n["t"], end(n), n))
    for (l1, c1, a1, b1, n1), (l2, c2, a2, b2, n2) in combinations(hands, 2):
        if l1 == l2 and c1 != c2 and not (b1 + 0.4 <= a2 + 1e-6 or b2 + 0.4 <= a1 + 1e-6):
            v.append(f"rules 1-3: {n1['kind']}@{n1['t']} and {n2['kind']}@{n2['t']} need one hand in two places")
    faders = [n for n in notes if n["kind"] == "fader"]
    for f in faders:
        for n in notes:
            if n["kind"] in ("stab", "riser") and n["t"] < end(f) and end(n) > f["t"]:
                v.append(f"rule 5: fader@{f['t']} overlaps {n['kind']}@{n['t']}")
    for a, b in combinations(faders, 2):
        if a["lever"] != b["lever"] and a["t"] < end(b) and b["t"] < end(a):
            v.append(f"rule 4: faders@{a['t']},{b['t']} overlap on both levers")

    exprs = [(n["t"], end(n)) for n in notes if n["kind"] == "expr"]
    left = [(n["t"], "clutch") for n in notes if n["kind"] == "hat"]
    for n in (n for n in notes if n["kind"] == "kick"):
        t = n["t"]
        right_free = all(t < a - 0.4 - 1e-6 or t > b + 0.4 + 1e-6 for a, b in exprs)
        if not right_free:
            left.append((t, "brake"))
    for (t1, p1), (t2, p2) in combinations(sorted(left), 2):
        if p1 != p2 and abs(t2 - t1) < 0.4 - 1e-6:
            v.append(f"rules 6-7: left foot on {p1}@{t1} and {p2}@{t2}")

    spins = [n for n in notes if n["kind"] == "spin"]
    if len(spins) % 2:
        v.append("rule 10: odd number of spins")
    if [s.get("dir") for s in spins] != ["cw", "ccw"] * (len(spins) // 2):
        v.append("rule 10: spin directions do not alternate")
    for s in spins:
        for g in (n for n in notes if n["kind"] == "gate"):
            if s["t"] - beat + 1e-6 < g["t"] < end(s) + beat - 1e-6:
                v.append(f"rule 11: gate@{g['t']} within a beat of spin@{s['t']}")
    return v


def density_violations(chart: dict, difficulty: str) -> list[str]:
    v = []
    secs = chart["sections"]
    bounds = [s["t"] for s in secs] + [chart["length"]]
    for kind, per_s in DENSITY[difficulty].items():
        for a, b in zip(bounds, bounds[1:], strict=False):
            n = sum(1 for x in chart["notes"] if x["kind"] == kind and a <= x["t"] < b and not x.get("listen"))
            if n > per_s * (b - a) + 1e-6:
                v.append(f"{kind}: {n} notes in section {a}..{b}, budget {per_s * (b - a):.1f}")
    return v


def rest_violations(chart: dict, meter: int = 4) -> list[str]:
    """Each foot layer rests at least one bar in any window of 8 bars."""
    v = []
    bar = meter * 60.0 / chart["bpm"]
    for kind in ("kick", "hat"):
        ts = sorted(n["t"] for n in chart["notes"] if n["kind"] == kind)
        if not ts:
            continue
        for w in np.arange(ts[0], ts[-1] - 8 * bar, bar / 4):
            pts = [w, *[t for t in ts if w < t < w + 8 * bar], w + 8 * bar]
            if max(np.diff(pts)) < 0.95 * bar:
                v.append(f"{kind}: no bar of rest in 8 bars from {w:.2f}")
                break
    return v


def assert_good(chart: dict, difficulty: str, meter: int = 4) -> None:
    assert validate_dict(chart) == []
    ts = [n["t"] for n in chart["notes"]]
    assert ts == sorted(ts) and all(0 <= t <= chart["length"] for t in ts)
    assert spec9_violations(chart, difficulty) == []
    assert playability(chart, difficulty) == []
    assert chart["defaults"]["layers"] == {k: "you" if k in LAYERS_IN[difficulty] else "auto" for k in gen.LAYERS}
    assert density_violations(chart, difficulty) == []
    assert rest_violations(chart, meter) == []
    assert {n["layer"] for n in chart["notes"]} <= LAYERS_IN[difficulty]
    c = Chart.from_dict(chart)
    for n in c.notes:
        if n.kind == "gate":
            assert c.road_x(n.t) == pytest.approx(n.x)
    listen = [n for n in c.notes if n.kind == "gate" and n.listen]
    for a, b in zip(listen, listen[1:], strict=False):
        assert 1.5 * abs(b.x - a.x) * 90 / (b.t - a.t) <= 180


# --- fixtures ---

@pytest.fixture(scope="module")
def parts():
    return synth()


@pytest.fixture(scope="module")
def song(parts, tmp_path_factory) -> Path:
    return write_song(tmp_path_factory.mktemp("song") / "synth.wav", parts)


@pytest.fixture(scope="module")
def analysis(parts):
    return analyse(parts)


@pytest.fixture(scope="module")
def charts(analysis) -> dict[str, dict]:
    return charts_for(analysis)


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TORQUEHERO_CONFIG_DIR", str(tmp_path / "cfg"))
    return tmp_path / "cfg"


def run_cli(*argv) -> int:
    from torquehero.__main__ import main

    return main(["gen", *map(str, argv)])


# --- analysis ---

def test_import_is_light():
    code = "import sys, torquehero.generate; print(sorted({'librosa', 'torch', 'demucs'} & set(sys.modules)))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "[]"


def test_tempo_within_2_percent(charts):
    for chart in charts.values():
        assert chart["bpm"] == pytest.approx(120.0, rel=0.02)


def test_beats_align_with_clicks(charts):
    beats = np.array([b["t"] for b in charts["normal"]["beats"]])
    assert len(beats) >= 62
    assert (np.abs(beats / 0.5 - np.round(beats / 0.5)) * 0.5).max() <= 0.040


def test_melody_pitch_tracked(analysis):
    for k in (2, 10, 50):
        assert analysis.pitch_at(k * 0.5 + 0.2) == pytest.approx(MELODY[k % len(MELODY)], abs=0.5)


def test_drifting_tempo_is_tracked():
    parts = synth(length=30.0, drift=0.03)
    an = analyse(parts)
    truth = beat_times(120.0, 30.0, 0.03)
    inside = an.beat_times[(an.beat_times > an.start + 0.5) & (an.beat_times < an.end - 0.5)]
    err = np.abs(inside[:, None] - truth[None, :]).min(axis=1)
    assert err.max() <= 0.040
    assert_good(gen.build_chart(an, "normal", title="drift"), "normal")


def test_three_four_time():
    parts = synth(bpm=150.0, length=30.0, meter=3)
    an = analyse(parts, meter=3)
    assert an.bpm == pytest.approx(150.0, rel=0.02)
    bar = 3 * 0.4
    assert len(an.bar_times) >= 20
    assert (np.abs(an.bar_times / bar - np.round(an.bar_times / bar)) * bar).max() <= 0.040
    for d in gen.DIFFICULTIES:
        assert_good(gen.build_chart(an, d, title="waltz"), d, meter=3)


def test_bpm_override_fixes_half_time():
    an = analyse(synth(length=20.0), bpm=60.0)
    assert an.bpm == pytest.approx(60.0, rel=0.02)


def test_silence_is_trimmed():
    parts = synth(length=24.0, pad=5.0)
    an = analyse(parts)
    assert an.start == pytest.approx(5.0, abs=0.1) and an.end == pytest.approx(29.0, abs=0.1)
    for d, chart in charts_for(an).items():
        assert_good(chart, d)
        assert all(4.96 <= n["t"] and n["t"] + n.get("dur", 0) <= 29.04 for n in chart["notes"])
        assert all(b["s"] == 0 for b in chart["beats"] if b["t"] < 4.9 or b["t"] > 29.1)
        assert chart["sections"][0]["weight"] == 0 and chart["sections"][-1]["weight"] == 0
    plain = sorted(s["weight"] for s in analyse(synth(length=24.0)).sections)
    padded = sorted(s["weight"] for s in an.sections if s["music"])
    assert padded == pytest.approx(plain, abs=0.1)


@pytest.mark.parametrize("kw", [dict(), dict(lead="saw", bass="pluck"), dict(lead="saw", bass="pluck", bpm=176.0)],
                         ids=["sine", "saw-lead-plucked-bass", "saw-176bpm"])
def test_no_drums_no_percussion(kw):
    an = analyse(synth(drums=False, length=24.0, **kw))
    assert an.kick_conf < gen.CONFIDENT and an.hat_conf < gen.CONFIDENT
    for d, chart in charts_for(an).items():
        assert_good(chart, d)
        assert not {"kick", "hat"} & {n["kind"] for n in chart["notes"]}


@pytest.fixture(scope="module")
def demo_lead():
    """The built-in demo song's lead stem: a plucked square-wave synth lead, no drums."""
    from torquehero import synthsong

    return gen.analyse(synthsong.render()["stems/lead.wav"], synthsong.SR)


def test_demo_lead_has_no_percussion_and_a_spin_pair(demo_lead):
    for d, chart in charts_for(demo_lead).items():
        assert_good(chart, d)
        assert not {"kick", "hat"} & {n["kind"] for n in chart["notes"]}, d
        if d != "easy":
            spins = [n for n in chart["notes"] if n["kind"] == "spin"]
            assert len(spins) >= 2 and len(spins) % 2 == 0, d
            assert [s["dir"] for s in spins] == ["cw", "ccw"] * (len(spins) // 2)


def test_report_prints_both_confidences(demo_lead):
    report: list[str] = []
    gen.build_chart(demo_lead, "normal", title="lead", report=report)
    assert any(line.startswith("kick: none") and "confidence" in line for line in report)
    assert any(line.startswith("hat: none") and "confidence" in line for line in report)


@pytest.mark.parametrize("bpm", [100.0, 140.0, 170.0])
def test_random_noise_bursts_are_not_drums(bpm):
    an = analyse(synth(drums=False, bursts=60, bpm=bpm, length=24.0))
    assert an.kick_conf < gen.CONFIDENT and an.hat_conf < gen.CONFIDENT
    for d in ("normal", "hard"):
        chart = gen.build_chart(an, d, title="bursts")
        assert_good(chart, d)
        assert not {"kick", "hat"} & {n["kind"] for n in chart["notes"]}


def test_real_drums_at_170_bpm_are_found():
    an = analyse(synth(bpm=170.0, length=24.0))
    assert an.kick_conf >= gen.CONFIDENT and an.hat_conf >= gen.CONFIDENT
    chart = gen.build_chart(an, "hard", title="fast kit")
    assert_good(chart, "hard")
    assert {"kick", "hat"} <= {n["kind"] for n in chart["notes"]}


def test_drums_under_a_saw_lead_and_plucked_bass():
    an = analyse(synth(length=24.0, lead="saw", bass="pluck"))
    assert an.kick_conf >= gen.CONFIDENT and an.hat_conf >= gen.CONFIDENT
    chart = gen.build_chart(an, "normal", title="kit")
    assert_good(chart, "normal")
    assert {"kick", "hat"} <= {n["kind"] for n in chart["notes"]}


# --- charts ---

@pytest.mark.parametrize("difficulty", gen.DIFFICULTIES)
def test_charts_obey_spec9(charts, difficulty):
    assert_good(charts[difficulty], difficulty)


def test_difficulty_decides_content(charts):
    def count(d, kind):
        return sum(n["kind"] == kind for n in charts[d]["notes"])

    assert {n["kind"] for n in charts["easy"]["notes"]} <= {"gate", "kick", "riser"}
    assert count("normal", "hat") > 0 and count("normal", "spin") > 0
    for kind in ("gate", "kick"):
        assert count("easy", kind) < count("normal", kind) <= count("hard", kind)
    assert count("easy", "kick") > 0


def test_structure(charts):
    chart = charts["normal"]
    notes = chart["notes"]
    listen = [n for n in notes if n.get("listen")]
    blind = [n for n in notes if n.get("blind")]
    assert len(listen) >= 3 and [n["x"] for n in listen] == [n["x"] for n in blind]
    echo = next(s for s in chart["sections"] if s["name"] == "echo")
    assert echo["weight"] == min(s["weight"] for s in chart["sections"])
    drops = {round(n["t"] + n["dur"], 3) for n in notes if n["kind"] == "riser"}
    assert drops and drops <= {round(s["t"], 3) for s in chart["sections"]}


def test_feature_fallbacks():
    parts = synth(bpm=150.0, length=36.0, held={*range(30, 33), *range(50, 53)})  # 3-beat holds: 1.2 s
    report: list[str] = []
    chart = gen.build_chart(analyse(parts), "normal", title="short holds", report=report)
    assert_good(chart, "normal")
    spins = [n for n in chart["notes"] if n["kind"] == "spin"]
    assert len(spins) == 2 and [s["dir"] for s in spins] == ["cw", "ccw"]
    assert any(line.startswith("spins: 2") for line in report)
    assert any(line.startswith("risers: drops") for line in report)
    assert any(line.startswith("echo: ") and "none" not in line for line in report)


def test_report_names_missing_percussion():
    report: list[str] = []
    gen.build_chart(analyse(synth(drums=False, length=24.0)), "normal", title="x", report=report)
    assert any(line.startswith("kick: none") for line in report)
    assert any(line.startswith("hat: none") for line in report)


def test_steering_uses_the_range(charts):
    gates = [n for n in charts["hard"]["notes"] if n["kind"] == "gate"]
    assert max(abs(b["x"] - a["x"]) for a, b in zip(gates, gates[1:], strict=False)) >= 0.5


def test_limbs_model():
    limbs = gen.Limbs()
    assert limbs.place([("R", "handbrake", 1.0, 3.0)])
    assert not limbs.place([("R", "shifter", 3.2, 3.2)])
    assert limbs.place([("R", "shifter", 3.4, 3.4)])
    assert limbs.place([("R", "handbrake", 3.7, 4.0)]) is False
    assert limbs.place([("L", "wheel", 2.0, 2.0)])


# --- files and the CLI ---

def test_gen_cli_without_stems(song, tmp_path, isolated_config, capsys):
    import soundfile as sf

    out = tmp_path / "out"
    assert run_cli(song, "-o", out, "--difficulty", "easy") == 0
    assert "--layers you:melody,kick," in capsys.readouterr().out
    chart = Chart.load(out / "chart.json")
    assert chart.audio.backing == ["stems/backing.wav"]
    assert chart.audio.layers["melody"].stem == "lead" and chart.audio.layers["melody"].mode == "filter"
    for layer in ("kick", "hat", "pads", "fills"):
        assert chart.audio.layers[layer].mode == "trigger"
    assert "expr" not in chart.audio.layers and "faders" not in chart.audio.layers
    for rel in [*chart.audio.backing, *chart.audio.stems.values(), *chart.audio.oneshots.values()]:
        assert chart.path(rel).is_file(), rel
    assert all(n.vel <= 0.6 for n in chart.notes if n.kind == "kick")
    meta = json.loads((out / "gen.json").read_text())
    assert meta["source"] == "synth.wav" and meta["layer_modes"]["melody"] == "you"
    stems = [sf.read(chart.path(r))[0] for r in [*chart.audio.backing, *chart.audio.stems.values()]]
    assert np.max(np.abs(sum(stems))) <= gen.PEAK_SUM + 1e-3
    assert max(np.max(np.abs(s)) for s in stems) <= gen.PEAK_STEM + 1e-3
    assert sf.info(chart.path("stems/lead.wav")).samplerate == 48000
    assert not list(tmp_path.glob(".out.tmp-*"))


def test_force_replaces_only_our_own_output(song, tmp_path, isolated_config, capsys):
    out = tmp_path / "out"
    assert run_cli(song, "-o", out) == 0
    (out / "stems" / "old.wav").write_text("stale")
    assert run_cli(song, "-o", out) == 2
    assert "--force" in capsys.readouterr().err
    assert run_cli(song, "-o", out, "--force") == 0
    assert not (out / "stems" / "old.wav").exists() and (out / "chart.json").is_file()


def test_force_never_deletes_other_directories(song, tmp_path, isolated_config, capsys):
    charts = tmp_path / "charts"
    (charts / "other-song").mkdir(parents=True)
    keep = charts / "other-song" / "chart.json"
    keep.write_text("{}")
    assert run_cli(song, "-o", charts, "--force") == 2
    assert "not a torquehero gen output" in capsys.readouterr().err
    assert keep.read_text() == "{}"
    inside = charts / "song.wav"
    inside.write_bytes(song.read_bytes())
    (charts / "gen.json").write_text(json.dumps({"generator": gen.GENERATOR}))
    assert run_cli(inside, "-o", charts, "--force") == 2
    assert "contains the source file" in capsys.readouterr().err
    assert keep.exists() and inside.exists()
    afile = tmp_path / "afile"
    afile.write_text("x")
    assert run_cli(song, "-o", afile, "--force") == 2
    assert "is a file" in capsys.readouterr().err and afile.read_text() == "x"


def test_free_tempo_is_refused(tmp_path, isolated_config, capsys):
    song = write_song(tmp_path / "rubato.wav", synth(length=30.0, rubato=(0.0, 99.0)))
    assert run_cli(song, "-o", tmp_path / "o") == 2
    err = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("gen:")]
    assert len(err) == 1 and "no steady beat" in err[0] and "--bpm" in err[0]
    assert run_cli(song, "-o", tmp_path / "o", "--bpm", "120") == 0
    assert "charting anyway" in capsys.readouterr().err


def test_free_stretch_warns(tmp_path, isolated_config, capsys):
    song = write_song(tmp_path / "part.wav", synth(length=30.0, rubato=(10.0, 18.0)))
    assert run_cli(song, "-o", tmp_path / "o") == 0
    assert "fits poorly around" in capsys.readouterr().err


def test_cli_prints_the_feature_report(song, tmp_path, isolated_config, capsys):
    assert run_cli(song, "-o", tmp_path / "o") == 0
    err = capsys.readouterr().err
    for key in ("risers:", "spins:", "echo:", "notes:"):
        assert key in err


def test_stems_path_with_fake_separator(tmp_path, isolated_config):
    parts = synth(length=48.0)
    song = write_song(tmp_path / "s.wav", parts)
    path = gen.generate(song, tmp_path / "stems", "hard", stems=True, separator=lambda y, sr: (parts, SR))
    chart = json.loads(path.read_text())
    assert_good(chart, "hard")
    layers = chart["audio"]["layers"]
    assert layers["melody"] == {"stem": "vocals", "mode": "filter"}
    assert layers["kick"] == {"stem": "drums", "mode": "gate", "oneshot": "kick"}
    assert layers["hat"] == {"stem": "drums", "mode": "gate", "oneshot": "hat"}
    kinds = {n["kind"] for n in chart["notes"]}
    assert {"expr", "fader"} <= kinds
    assert layers["expr"] == {"stem": "other", "mode": "level"}
    assert layers["faders"] == {"stem": "bass", "mode": "levers"}
    for kind in ("expr", "fader"):
        covered = sum(n["dur"] for n in chart["notes"] if n["kind"] == kind)
        assert covered <= 0.5 * chart["length"], kind
    names = set(chart["audio"]["backing"]) | set(chart["audio"]["stems"].values())
    assert names == {f"stems/{s}.wav" for s in gen.DEMUCS_STEMS}
    import soundfile as sf

    stems = [sf.read(path.parent / rel)[0] for rel in sorted(names)]
    assert np.max(np.abs(sum(stems))) <= gen.PEAK_SUM + 1e-3
    assert max(np.max(np.abs(s)) for s in stems) <= gen.PEAK_STEM + 1e-3


def test_extra_separator_source_goes_to_backing(tmp_path, isolated_config):
    parts = synth(length=12.0)
    extra = {**parts, "piano": parts["other"] * 0.5}
    song = write_song(tmp_path / "s.wav", parts)
    path = gen.generate(song, tmp_path / "o", "easy", stems=True, separator=lambda y, sr: (extra, SR))
    chart = json.loads(path.read_text())
    assert "stems/piano.wav" in chart["audio"]["backing"]
    assert validate_dict(chart) == []


def test_play_range_passes_through(song, tmp_path, isolated_config):
    cfg = Config(play_range_deg=120.0)
    chart = json.loads(gen.generate(song, tmp_path / "o", "normal", cfg=cfg).read_text())
    assert validate_dict(chart, 120.0, cfg.echo_max_deg_s) == []


def test_44k_stereo_input(tmp_path, isolated_config):
    import librosa
    import soundfile as sf

    parts = synth(length=16.0, width=0.3)
    mix44 = librosa.resample(sum(parts.values()), orig_sr=SR, target_sr=44100)
    song = tmp_path / "s44.wav"
    sf.write(song, mix44.T, 44100)
    path = gen.generate(song, tmp_path / "o", "normal")
    chart = json.loads(path.read_text())
    assert chart["bpm"] == pytest.approx(120.0, rel=0.02)
    assert sf.info(path.parent / "stems/backing.wav").samplerate == 48000
    assert sf.info(path.parent / "stems/backing.wav").channels == 2


def test_short_files(tmp_path, isolated_config, capsys):
    one = write_song(tmp_path / "one.wav", synth(length=1.0))
    assert run_cli(one, "-o", tmp_path / "o1") == 2
    err = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("gen:")]
    assert len(err) == 1 and "non-silent" in err[0]
    three = write_song(tmp_path / "three.wav", synth(length=3.0))
    assert run_cli(three, "-o", tmp_path / "o3") == 0
    assert validate_dict(json.loads((tmp_path / "o3" / "chart.json").read_text())) == []


@pytest.mark.parametrize("case", ["undecodable", "settings", "missing"])
def test_expected_failures_are_one_line(case, song, tmp_path, isolated_config, capsys):
    target = song
    if case == "undecodable":
        target = tmp_path / "junk.wav"
        target.write_bytes(b"not audio at all" * 10)
    elif case == "settings":
        isolated_config.mkdir(parents=True)
        (isolated_config / "generate.json").write_text("{not json")
    else:
        target = tmp_path / "nope.wav"
    assert run_cli(target, "-o", tmp_path / "o") == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err and err.strip().splitlines()[-1].startswith("gen: ")
    assert sum(ln.startswith("gen:") for ln in err.splitlines()) == 1
    assert not (tmp_path / "o").exists()


def test_invalid_own_output_exits_2(song, tmp_path, isolated_config, monkeypatch, capsys):
    monkeypatch.setattr(gen, "validate_dict", lambda *a, **k: ["notes[0]: boom"])
    assert run_cli(song, "-o", tmp_path / "o") == 2
    assert "boom" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_missing_stems_extra(song, tmp_path, isolated_config, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "demucs", None)
    assert run_cli(song, "-o", tmp_path / "x", "--stems") == 2
    err = capsys.readouterr().err
    assert "uv sync --extra stems" in err and "cu128" in err
    assert not (tmp_path / "x").exists()


def test_cuda_requested_without_cuda(monkeypatch):
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "demucs", types.ModuleType("demucs"))
    monkeypatch.setitem(sys.modules, "demucs.apply", types.SimpleNamespace(apply_model=None))
    monkeypatch.setitem(sys.modules, "demucs.pretrained", types.SimpleNamespace(get_model=None))
    with pytest.raises(gen.GenError, match="no CUDA GPU"):
        gen.demucs_separator(gen.GenSettings(device="cuda"))


class FakeTensor(np.ndarray):
    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def fake_demucs(monkeypatch, sources):
    """Fake torch and demucs modules; returns the list of apply_model keyword arguments."""
    import contextlib

    calls = []
    model = types.SimpleNamespace(sources=list(sources), samplerate=44100)
    model.to = lambda device: model
    model.eval = lambda: model

    def apply_model(m, x, **kw):
        calls.append(kw)
        return np.stack([np.stack([x[0]] * len(m.sources))]).view(FakeTensor)

    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
                                  from_numpy=lambda a: a, no_grad=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "demucs", types.ModuleType("demucs"))
    monkeypatch.setitem(sys.modules, "demucs.apply", types.SimpleNamespace(apply_model=apply_model))
    monkeypatch.setitem(sys.modules, "demucs.pretrained", types.SimpleNamespace(get_model=lambda name: model))
    return calls


def test_separate_is_deterministic_and_resamples_once(monkeypatch):
    calls = fake_demucs(monkeypatch, gen.DEMUCS_STEMS)
    y = np.random.default_rng(0).standard_normal((2, 48000)).astype(np.float32) * 0.1
    stems, sr = gen.demucs_separator(gen.GenSettings())(y, 48000)
    assert calls[0]["shifts"] == 0 and sr == 44100
    assert set(stems) == set(gen.DEMUCS_STEMS) and stems["drums"].shape == (2, 44100)


def test_separate_rejects_a_model_without_four_stems(monkeypatch):
    fake_demucs(monkeypatch, ("drums", "bass", "other"))
    with pytest.raises(gen.GenError, match="missing vocals"):
        gen.demucs_separator(gen.GenSettings())(np.zeros((2, 4800), np.float32), 48000)


def test_settings_file(tmp_path):
    path = gen.GenSettings(lane_span=0.7).save(tmp_path / "generate.json")
    assert gen.GenSettings.load(path).lane_span == 0.7
    path.write_text(json.dumps({"demucs_model": "htdemucs_6s"}))
    with pytest.raises(gen.GenError, match="demucs_model"):
        gen.GenSettings.load(path)
