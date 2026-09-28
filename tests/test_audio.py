import logging
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from conftest import make_chart

from torquehero import audio as audio_mod
from torquehero.audio import (
    KEEP_RECENT,
    MAX_VOICES,
    AudioEngine,
    AudioSettings,
    Mixer,
    NullAudio,
    SongAudio,
    clock_time,
    load_song_audio,
)
from torquehero.chart import AudioManifest
from torquehero.state import InputState, LayerStatus, Snapshot, Sound

SR = 48000
MS = SR // 1000


@pytest.fixture(autouse=True)
def fresh_warnings():
    audio_mod._warned.clear()


def const(v=1.0, secs=2.0):
    return np.full((int(SR * secs), 2), v, np.float32)


def sine(hz, secs=1.0):
    t = np.arange(int(SR * secs)) / SR
    return np.repeat(np.sin(2 * np.pi * hz * t)[:, None], 2, axis=1).astype(np.float32)


def impulse(n=64):
    a = np.zeros((n, 2), np.float32)
    a[0] = 1.0
    return a


def rms(a):
    return float(np.sqrt(np.mean(a[:, 0] ** 2)))


def manifest(**layers):
    """The fixture manifest with some layers replaced; a None value removes one."""
    m = make_chart([]).audio.to_dict()
    for k, v in layers.items():
        if v is None:
            m["layers"].pop(k, None)
        else:
            m["layers"][k] = v
    return AudioManifest.from_dict(m)


def mixer(man=None, *, stems=None, shots=None, backing=None, notes=(), modes=None, volume=1.0):
    m = Mixer(SR, volume=volume)
    m.setup(man or make_chart([]).audio, SongAudio(SR, backing or [], stems or {}, shots or {}),
            list(notes), modes or {})
    return live(m)


def live(m):
    """Resume a mixer and skip its fade-in, so levels are exact from the first sample."""
    m.set_paused(False)
    m._drain()
    m.transport.value = 1.0
    return m


def first_nonzero(out):
    return int(np.flatnonzero(np.abs(out[:, 0]) > 1e-6)[0])


# --- gains and ramps ---

def test_backing_and_master_volume():
    m = mixer(backing=[const(0.5)], volume=0.8)
    assert np.allclose(m.render(256), 0.4)


def test_master_volume_ramps():
    m = mixer(backing=[const(1.0)], volume=1.0)
    m.set_volume(0.0)
    out = m.render(SR // 5)[:, 0]
    assert 0.9 < out[0] <= 1.0 and out[-1] < 0.01
    assert np.all(np.diff(out) <= 0)


def test_unused_stem_plays_at_unity_and_disallowed_only_stem_is_silent(caplog):
    man = manifest(melody={"stem": "lead", "mode": "level"})
    with caplog.at_level(logging.WARNING):
        m = mixer(man, stems={"lead": const(1.0), "drums": const(0.5)})
    assert np.allclose(m.render(128), 0.5)   # lead silent, drums named by no layer
    assert sum("cannot use mode" in r.message for r in caplog.records) == 1


def test_gate_ducks_over_60ms_and_recovers_over_30ms():
    man = manifest(kick={"stem": "drums", "mode": "gate"})
    m = mixer(man, stems={"drums": const()})
    m.set_layer("kick", alive=False)
    down = m.render(60 * MS)[:, 0]
    assert down[30 * MS - 1] == pytest.approx(0.6, abs=0.01)
    assert down[-1] == pytest.approx(0.2, abs=1e-4)
    assert np.allclose(m.render(64), 0.2)
    m.set_layer("kick", alive=True)
    up = m.render(30 * MS)[:, 0]
    assert up[15 * MS - 1] == pytest.approx(0.6, abs=0.01)
    assert up[-1] == pytest.approx(1.0, abs=1e-4)


def test_gate_many_layers_one_stem():
    man = manifest(kick={"stem": "drums", "mode": "gate"}, hat={"stem": "drums", "mode": "gate", "oneshot": "hat"})
    m = mixer(man, stems={"drums": const()})
    m.set_layer("kick", alive=False)
    m.render(SR // 10)
    assert np.allclose(m.render(64), 0.6)          # 0.2 + 0.8 * 1/2
    m.set_layer("hat", alive=False)
    m.render(SR // 10)
    assert np.allclose(m.render(64), 0.2)


def test_auto_gate_layer_counts_alive():
    chart = make_chart([])
    chart.audio = manifest(kick={"stem": "drums", "mode": "gate"}, hat={"stem": "drums", "mode": "gate"})
    eng = AudioEngine(stream_factory=FakeStream, volume=1.0)
    eng.load(chart, {"kick": "auto", "hat": "you"}, SongAudio(SR, [], {"drums": const()}, {}))
    snap = Snapshot(layers={"kick": LayerStatus("auto", False), "hat": LayerStatus("you", False)})
    eng.update(snap)
    live(eng.mixer)
    eng.mixer.render(SR // 10)
    assert np.allclose(eng.mixer.render(64), 0.6)


def test_filter_mode_lowpasses_and_ducks_off_road():
    for hz, want_on, want_off in ((100, 1.0, 0.35), (4000, 1.0, 0.02)):
        m = mixer(stems={"lead": sine(hz)})
        on = rms(m.render(SR // 4)[SR // 8:]) / rms(sine(hz))
        m.set_layer("melody", on_road=False)
        m.render(SR // 4)
        off = rms(m.render(SR // 4)) / rms(sine(hz))
        assert on == pytest.approx(want_on, abs=0.02)
        assert off == pytest.approx(want_off, abs=0.02)


def test_filter_gain_ramps_with_30ms_time_constant():
    m = mixer(stems={"lead": const()})
    m.set_layer("melody", on_road=False)
    g = m.render(30 * MS)[:, 0]
    # gain 1 -> 0.35 with tau 30 ms: 1/e of the way left after 30 ms (DC passes the low-pass)
    assert g[-1] == pytest.approx(0.35 + 0.65 / np.e, abs=0.02)


def test_level_mode():
    m = mixer(stems={"pad": const()})
    m.render(SR // 2)
    assert np.allclose(m.render(64), 0.55)          # SPEC 10: before the first note
    m.set_layer("expr", v=1.0)
    m.render(SR // 2)
    assert np.allclose(m.render(64), 1.0, atol=1e-3)
    m.set_layer("expr", v=0.0)
    m.render(SR // 2)
    assert np.allclose(m.render(64), 0.15, atol=1e-3)


def test_levers_gain_and_cutoff():
    m = mixer(stems={"bass": sine(2000, 2.0)})
    open_ = rms(m.render(SR // 4)[SR // 8:])
    m.set_layer("faders", v=0.0, v1=1.0)
    m.render(SR // 4)
    assert rms(m.render(SR // 4)) / open_ == pytest.approx(0.2, abs=0.02)
    m.set_layer("faders", v=1.0, v1=0.0)            # cutoff 200 Hz
    m.render(SR // 4)
    assert rms(m.render(SR // 4)) / open_ < 0.02


def continuous_engine(modes):
    eng = AudioEngine(stream_factory=FakeStream, volume=1.0)
    eng.load(make_chart([]), {"melody": "you", **modes},
             SongAudio(SR, [], {"pad": const(), "bass": const(), "lead": const()}, {}))
    return eng, eng.mixer._layers


def test_engine_update_maps_snapshot():
    eng, ctl = continuous_engine({"expr": "auto", "faders": "you"})
    assert (ctl["expr"].v, ctl["faders"].v, ctl["faders"].v1) == (pytest.approx(0.4 / 0.85), 1.0, 1.0)
    eng.update(Snapshot(expr_target=0.4, fader_targets=[0.9, 0.8], on_road=False,
                        input=InputState(lever0=0.25, lever1=0.5, throttle=1.0)))
    assert (ctl["expr"].v, ctl["melody"].on_road, ctl["faders"].v, ctl["faders"].v1) == (0.4, False, 0.25, 0.5)


@pytest.mark.parametrize("mode", ["you", "auto"])
def test_level_follows_only_during_notes_then_holds_last_target(mode):
    eng, ctl = continuous_engine({"expr": mode})
    eng.update(Snapshot(expr_target=None, input=InputState(throttle=1.0)))
    assert ctl["expr"].v == pytest.approx(0.4 / 0.85)       # before the first note: gain 0.55
    eng.update(Snapshot(expr_target=0.7, input=InputState(throttle=0.2)))
    assert ctl["expr"].v == (0.2 if mode == "you" else 0.7)
    eng.update(Snapshot(expr_target=None, input=InputState(throttle=0.0)))  # foot moves to the brake
    assert ctl["expr"].v == 0.7                               # holds the ended note's last target


@pytest.mark.parametrize("mode", ["you", "auto"])
def test_levers_follow_only_during_notes_then_hold_last_target(mode):
    eng, ctl = continuous_engine({"faders": mode})
    lev = InputState(lever0=0.1, lever1=0.2)
    eng.update(Snapshot(fader_targets=[None, None], input=lev))
    assert (ctl["faders"].v, ctl["faders"].v1) == (1.0, 1.0)                # open before the first note
    eng.update(Snapshot(fader_targets=[0.6, None], input=lev))
    assert (ctl["faders"].v, ctl["faders"].v1) == ((0.1 if mode == "you" else 0.6), 1.0)
    eng.update(Snapshot(fader_targets=[None, 0.3], input=lev))
    assert (ctl["faders"].v, ctl["faders"].v1) == (0.6, (0.2 if mode == "you" else 0.3))
    eng.update(Snapshot(fader_targets=[None, None], input=InputState()))
    assert (ctl["faders"].v, ctl["faders"].v1) == (0.6, 0.3)


# --- one-shots ---

def test_player_oneshot_starts_at_next_block():
    m = mixer(shots={"kick": impulse()})
    m.render(256)
    m.play("kick", 0.5)
    out = m.render(256)
    assert first_nonzero(out) == 0 and out[0, 0] == pytest.approx(0.5)


def test_auto_layer_scheduled_sample_accurately():
    notes = make_chart([{"t": 0.5, "kind": "kick", "vel": 0.8}, {"t": 0.75, "kind": "stab", "gate": 3}]).notes
    m = mixer(shots={"kick": impulse(), "stab3": impulse()}, notes=notes, modes={"kick": "auto", "pads": "auto"})
    out = np.concatenate([m.render(333) for _ in range(150)])
    hits = np.flatnonzero(out[:, 0])
    assert list(hits) == [round(0.5 * SR), round(0.75 * SR)]
    assert out[hits[0], 0] == pytest.approx(0.8)


def test_you_layer_not_scheduled():
    notes = make_chart([{"t": 0.1, "kind": "kick"}]).notes
    m = mixer(shots={"kick": impulse()}, notes=notes, modes={"kick": "you"})
    assert not m.render(SR // 5).any()


def test_seek_repositions_schedule():
    notes = make_chart([{"t": 0.1, "kind": "kick"}]).notes
    m = mixer(shots={"kick": impulse()}, notes=notes, modes={"kick": "auto"})
    m.render(SR // 5)
    m.seek(0)
    fade = m.render(10 * MS)                          # 10 ms fade-out, then the jump
    assert not fade[-1].any()
    # the jump lands 10 ms into the song: the song kept time while the old audio faded
    assert first_nonzero(m.render(SR // 5)) == round(0.1 * SR) - 10 * MS


def test_auto_riser_loops_then_fades_and_impacts():
    notes = make_chart([{"t": 0.1, "kind": "riser", "dur": 0.2}]).notes
    loop = const(0.5, 0.01)                          # 10 ms loop, wraps many times
    m = mixer(shots={"riser": loop, "impact": impulse() * 0.25}, notes=notes, modes={"riser": "auto"})
    out = m.render(SR // 2)[:, 0]
    start, drop = round(0.1 * SR), round(0.3 * SR)
    assert not out[:start].any() and np.allclose(out[start:drop], 0.5)
    assert out[drop] == pytest.approx(0.75)          # impact + loop at the start of its fade
    fade = out[drop + 1: drop + 20 * MS]
    assert np.all(np.diff(fade) <= 1e-7) and fade[-1] < 0.01
    assert not out[drop + 20 * MS + 1:].any()


def test_many_to_one_gate_with_reinforcement_oneshot():
    man = manifest(kick={"stem": "drums", "mode": "gate", "oneshot": "kick"})
    eng = AudioEngine(stream_factory=FakeStream, volume=1.0)
    chart = make_chart([])
    chart.audio = man
    eng.load(chart, {"kick": "you"}, SongAudio(SR, [], {}, {"kick": impulse()}))
    live(eng.mixer)
    eng.handle([Sound(1.0, "kick", "kick", vel=0.7, cause="hit", note_index=0, note_t=1.0)])
    assert eng.mixer.render(64)[0, 0] == pytest.approx(0.7)


# --- Sound events ---

def engine(shots, modes=None, notes=()):
    eng = AudioEngine(stream_factory=FakeStream, volume=1.0)
    eng.load(make_chart(list(notes)), modes or {}, SongAudio(SR, [], {}, shots))
    live(eng.mixer)
    return eng


def test_handle_ignores_auto_and_plays_other_causes():
    eng = engine({"kick": impulse(), "dud": impulse()})
    eng.handle([Sound(0.0, "kick", "kick", cause="auto")])
    assert not eng.mixer.render(64).any()
    eng.handle([Sound(0.0, "kick", "kick", cause="free", vel=0.3), Sound(0.0, "pads", "dud", cause="miss", vel=0.4)])
    assert eng.mixer.render(64)[0, 0] == pytest.approx(0.7)


def test_unknown_sound_skipped_and_logged_once(caplog):
    eng = engine({})
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            eng.handle([Sound(0.0, "pads", "stab6", cause="free")])
    assert not eng.mixer.render(64).any()
    assert sum("stab6" in r.message for r in caplog.records) == 1


def test_player_riser_start_stop():
    eng = engine({"riser": const(0.5, 0.01), "impact": impulse()})
    eng.handle([Sound(0.0, "riser", "riser", "start", cause="hit")])
    assert np.allclose(eng.mixer.render(SR // 10), 0.5)
    eng.handle([Sound(0.1, "riser", "riser", "stop", cause="free"),
                Sound(0.1, "riser", "impact", cause="free", vel=0.25)])
    out = eng.mixer.render(SR // 10)[:, 0]
    assert out[0] == pytest.approx(0.75) and not out[25 * MS:].any()


def test_disallowed_layer_sounds_are_silent():
    chart = make_chart([])
    chart.audio = manifest(kick={"oneshot": "kick", "mode": "riser"})
    eng = AudioEngine(stream_factory=FakeStream, volume=1.0)
    eng.load(chart, {"kick": "you"}, SongAudio(SR, [], {}, {"kick": impulse()}))
    live(eng.mixer)
    eng.handle([Sound(0.0, "kick", "kick", cause="free")])
    assert not eng.mixer.render(64).any()




def test_filter_transitions_do_not_click():
    m = mixer(stems={"lead": sine(200, 2.0) * 0.5})
    blocks = []
    for on, secs in ((True, 0.1), (False, 0.3), (True, 0.3)):
        m.set_layer("melody", on_road=on)
        for _ in range(int(secs * SR) // 333):
            blocks.append(m.render(333))
    out = np.concatenate(blocks)[:, 0]
    # a 0.5 amplitude 200 Hz sine moves at most 0.013 per sample
    assert np.abs(np.diff(out)).max() < 0.02
    assert m._stem_tracks["lead"].filtering is False  # back to bypass at rest


def test_filter_bypassed_when_open():
    m = mixer(stems={"lead": sine(200)})
    calls = []
    m._lfilter = lambda *a, **k: calls.append(1)
    m.render(1024)
    assert calls == []


# --- voice pool ---

def test_pool_exhaustion_keeps_held_riser():
    m = mixer(shots={"riser": const(0.1, 0.01), "pad": const(0.01, 1.0)})
    m.start_loop("riser", "riser")
    m.render(64)
    riser = m._loops["riser"]
    for _ in range(40):
        m.play("pad")
        m.render(16)
    assert m._loops["riser"] is riser and riser.active and riser.loop
    assert any(v.active for v in m._reserve)          # stolen one-shots fade in the reserve
    m.render(10 * MS)
    assert not any(v.active for v in m._reserve)      # 5 ms steal fade finished
    m.stop_loop("riser")
    m.render(30 * MS)
    assert not riser.active


def test_newest_voices_are_never_stolen():
    m = mixer(shots={"pad": const(0.01, 1.0)})
    for _ in range(MAX_VOICES + 3):
        m.play("pad")
    m.render(16)
    newest = sorted(v.seq for v in m._voices)[-KEEP_RECENT:]
    assert newest == list(range(MAX_VOICES + 3 - KEEP_RECENT + 1, MAX_VOICES + 4))


def test_reused_loop_voice_is_released_from_its_layer():
    m = mixer(shots={"empty": np.zeros((0, 2), np.float32), "kick": const(0.5, 0.1)})
    m.start_loop("riser", "empty")
    m.render(16)                                      # zero-length loop ends; still listed
    m.play("kick")
    m.render(16)
    assert "riser" not in m._loops
    m.stop_loop("riser")                              # must not fade the kick
    assert np.allclose(m.render(40 * MS), 0.5)


# --- contract ---

def test_mono_and_int16_arrays_are_normalised():
    mono = (np.ones(SR // 10) * 16384).astype(np.int16)
    m = mixer(stems={"lead": mono}, backing=[np.full(SR // 10, 0.25, np.float32)], shots={"kick": mono[:64]})
    m.play("kick")
    out = m.render(256)
    assert out.shape == (256, 2) and np.allclose(out[:64], 1.0)   # 0.25 + 0.5 + 0.5, clipped
    assert np.allclose(out[64:], 0.75)


def test_null_audio_does_not_load_scipy():
    import subprocess
    import sys

    code = ("import sys; from torquehero.audio import NullAudio; NullAudio().time(); "
            "sys.exit('scipy.signal' in sys.modules)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# --- clock and stream ---

class FakeStream:
    """Stands in for sd.OutputStream: no device; tests drive the callback."""

    def __init__(self, callback, **kw):
        self.callback, self.kw = callback, kw
        self.time, self.latency = 0.0, 0.02
        self.lead = None                              # DAC lead the host reports; None = latency
        self.started = self.closed = False
        self.zero_times = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def pull(self, frames=480, underflow=False):
        """One callback: the block reaches the DAC one latency after `time`, which then advances."""
        out = np.zeros((frames, 2), np.float32)
        lead = self.latency if self.lead is None else self.lead
        dac, cur = (0.0, 0.0) if self.zero_times else (self.time + lead, self.time)
        self.callback(out, frames, SimpleNamespace(outputBufferDacTime=dac, currentTime=cur),
                      SimpleNamespace(output_underflow=underflow))
        self.time += frames / SR
        return out


def running_engine(at=0.0, secs=0.5, **kw):
    eng = AudioEngine(AudioSettings(**kw), stream_factory=FakeStream, volume=1.0)
    eng.load(make_chart([]), {}, SongAudio(SR, [const(0.5, secs)], {}, {}))
    eng.start(at)
    st = eng._stream
    assert isinstance(st, FakeStream)
    return eng, st


def test_clock_time_interpolates_and_clamps():
    assert clock_time(48000, 480, SR, dac_time=10.02, now=10.0) == pytest.approx(0.98)
    assert clock_time(48000, 480, SR, dac_time=10.0, now=10.005) == pytest.approx(1.005)
    assert clock_time(48000, 480, SR, dac_time=10.0, now=11.0) == pytest.approx(1.01)


def test_engine_clock_is_frames_minus_latency():
    eng, st = running_engine()
    assert st.started and st.kw["samplerate"] == SR and st.kw["channels"] == 2 and st.kw["blocksize"] == 256
    assert eng.time() == pytest.approx(-0.02)         # before the first callback
    for _ in range(10):
        st.pull()
    # 10 blocks rendered (0.1 s); the last reaches the DAC at 0.09 + 0.02, stream time is 0.1
    assert eng.time() == pytest.approx(0.1 - 0.02)
    assert eng.underruns == 0
    st.pull(underflow=True)
    assert eng.underruns == 1


def test_clock_is_monotonic_across_start_and_seek():
    eng, st = running_engine(at=1.0)
    seen = [eng.time()]
    assert seen[0] == pytest.approx(0.98)
    for _ in range(20):
        st.pull(frames=240)
        for _ in range(3):
            st.time += 0.0015
            seen.append(eng.time())
    eng.seek(0.0)
    back = [eng.time()]
    assert back[0] == pytest.approx(-0.02)
    for _ in range(20):
        st.pull(frames=240)
        back.append(eng.time())
    assert np.all(np.diff(seen) >= 0) and np.all(np.diff(back) >= 0)
    assert back[-1] > 0.05


def reaches(eng, st, lo, hi, pulls=12):
    """Pull blocks; time() must end in [lo, hi] and advance on every block after the fades."""
    seen = []
    for _ in range(pulls):
        st.pull(frames=240)
        seen.append(eng.time())
    assert np.all(np.diff(seen) >= 0) and seen[-1] > seen[-4] and lo <= seen[-1] <= hi, seen
    return seen


def test_seek_between_render_and_clock_assignment():
    eng, st = running_engine(at=60.0, secs=0.1)
    for _ in range(5):
        st.pull()
    m, render, fired = eng.mixer, eng.mixer.render_into, []

    def racing_render(out):
        r = render(out)
        if not fired:                                 # seek lands after render, before the clock tuple
            fired.append(1)
            eng.seek(0.0)
        return r

    m.render_into = racing_render
    reaches(eng, st, 0.0, 0.1)


def test_start_on_a_playing_engine_restarts_the_clock():
    eng, st = running_engine(at=60.0)
    for _ in range(5):
        st.pull()
    eng.start(0.0)
    assert eng.time() == pytest.approx(-0.02)
    reaches(eng, st, 0.0, 0.1)


def test_seek_then_resume_inside_the_fade():
    eng, st = running_engine(at=60.0)
    for _ in range(5):
        st.pull()
    eng.seek(0.0)
    st.pull(frames=120)                               # fade-out under way
    eng.resume()
    reaches(eng, st, 0.0, 0.1)


def test_seek_pause_resume_seek_in_one_block():
    eng, st = running_engine(at=60.0)
    for _ in range(5):
        st.pull()
    eng.seek(5.0)
    eng.pause()
    eng.resume()
    eng.seek(0.0)
    reaches(eng, st, 0.0, 0.1)


def test_pause_resume_does_not_step_the_clock():
    eng, st = running_engine(secs=2.0)
    for _ in range(10):
        st.pull()
    eng.pause()
    frozen = eng.time()
    for _ in range(5):
        st.pull()                                     # fade-out, then silence
    eng.resume()
    st.pull()
    assert abs(eng.time() - (frozen + 0.01)) < 0.001  # only the block just played


def test_seek_and_pause_complete_during_render_errors(monkeypatch):
    eng, st = running_engine(at=60.0)
    st.pull()
    monkeypatch.setattr(eng.mixer, "_fire_schedule", lambda f0, n: 1 / 0)
    eng.seek(0.0)
    reaches(eng, st, 0.0, 0.1)
    eng.pause()
    for _ in range(4):
        st.pull()
    assert eng.mixer.paused


def test_load_on_a_playing_engine_fades_out():
    import threading
    import time

    eng, st = running_engine(secs=2.0)
    for _ in range(5):
        st.pull()
    outs, done = [], threading.Event()

    def pump():
        while not done.is_set():
            outs.append(st.pull(frames=48))
            time.sleep(0.0005)

    th = threading.Thread(target=pump)
    th.start()
    eng.load(make_chart([]), {}, SongAudio(SR, [const(0.25, 1.0)], {}, {}))
    done.set()
    th.join()
    out = np.concatenate(outs)[:, 0]
    assert out[0] == pytest.approx(0.5) and out[-1] == 0.0
    assert np.abs(np.diff(out)).max() < 0.01          # 10 ms fade, no cut


def test_cutoff_coefficients_are_cached(monkeypatch):
    calls = []
    real = audio_mod.lowpass_coeffs
    monkeypatch.setattr(audio_mod, "lowpass_coeffs", lambda fc, sr: calls.append(fc) or real(fc, sr))
    m = mixer(stems={"lead": sine(200, 2.0)})
    m.set_layer("melody", on_road=False)
    m.render(SR // 2)                                 # settles at 450 Hz
    before = len(calls)
    m.render(512)
    m.render(512)
    assert len(calls) == before


def test_moving_cutoff_updates_every_64_frames():
    m = mixer(stems={"lead": sine(200, 2.0)})
    m.set_layer("melody", on_road=False)
    calls = []
    real = m._lfilter
    m._lfilter = lambda *a, **k: calls.append(len(a[2])) or real(*a, **k)
    m.render(512)
    assert calls == [64] * 8


def test_clock_keeps_running_past_end_of_audio():
    eng, st = running_engine(secs=0.05)
    for _ in range(100):                              # 1 s; audio is 50 ms
        out = st.pull()
    assert not out.any() and eng.time() == pytest.approx(1.0 - 0.02)


def test_start_at_negative_is_lead_in_silence():
    eng, st = running_engine(at=-0.1)
    outs = np.concatenate([st.pull() for _ in range(20)])
    assert first_nonzero(outs) == round(0.1 * SR)
    assert eng.time() == pytest.approx(0.2 - 0.1 - 0.02)


def test_pause_fades_freezes_and_drops_queued_sounds():
    eng, st = running_engine(secs=2.0)
    eng.mixer._shots["kick"] = impulse()
    for _ in range(10):
        st.pull()
    eng.pause()
    frozen = eng.time()
    fade = st.pull()[:, 0]
    assert fade[0] > 0.45 and fade[-1] == 0.0 and np.all(np.diff(fade) <= 1e-7)
    eng.handle([Sound(0.0, "kick", "kick", cause="free")])
    st.time += 1.0
    assert not st.pull().any() and eng.time() == frozen
    eng.resume()
    assert eng.time() == frozen                       # until the next block
    back = st.pull()[:, 0]
    assert back[0] < 0.01 and back[-1] == pytest.approx(0.5)   # 10 ms fade-in, no kick
    assert eng.time() >= frozen
    eng.stop()
    assert st.closed and eng._stream is None


def test_render_error_silences_but_clock_advances(monkeypatch, caplog):
    eng, st = running_engine()
    st.pull()
    monkeypatch.setattr(eng.mixer, "_fire_schedule", lambda f0, n: 1 / 0)
    with caplog.at_level(logging.ERROR):
        outs = [st.pull() for _ in range(10)]
    assert not any(o.any() for o in outs)
    assert eng.render_errors == 10 and sum("render failed" in r.message for r in caplog.records) == 1
    assert eng.time() == pytest.approx(0.11 - 0.02)


def test_zero_timestamps_fall_back_to_perf_counter(caplog):
    eng, st = running_engine()
    perf = [5.0]
    eng._perf = lambda: perf[0]
    st.zero_times = True
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            st.pull()
            perf[0] += 0.01
    perf[0] += 0.005                                  # 15 ms after the last callback
    assert eng.time() == pytest.approx(0.01 + 0.015 - 0.02)
    assert sum("timestamps" in r.message for r in caplog.records) == 1


def test_perf_counter_fallback_spaces_bursts():
    eng, st = running_engine()
    perf = [5.0]
    eng._perf = lambda: perf[0]
    st.zero_times = True
    st.pull()
    st.pull()                                         # same instant: a burst
    assert eng._prev_stamp[0] == pytest.approx(5.01)
    perf[0] = 5.02
    assert eng.time() == pytest.approx(0.01 + (5.02 - 5.01) - 0.02)   # no staircase back to 5.0


def test_load_while_running_swaps_mixer():
    eng, st = running_engine()
    for _ in range(5):
        st.pull()
    old = eng.mixer
    old_tracks, old_frame = old._tracks, old.frame
    eng.load(make_chart([]), {}, SongAudio(SR, [const(0.25, 1.0)], {}, {}))
    assert eng.mixer is not old and old._tracks is old_tracks and old.frame == old_frame
    assert not st.pull().any() and eng.time() == pytest.approx(-0.02)   # new song waits for start
    eng.start(0.0)
    st.pull()
    assert st.pull()[-1, 0] == pytest.approx(0.25)


def test_callback_renders_blocks_larger_than_max_block():
    eng, st = running_engine()
    out = st.pull(frames=eng.mixer.max_block + 100)
    assert np.allclose(out[10 * MS: int(0.5 * SR)][: eng.mixer.max_block], 0.5)


# --- loading and settings ---

def test_load_reads_and_resamples_files(tmp_path, caplog):
    (tmp_path / "stems").mkdir()
    sf.write(tmp_path / "stems" / "lead.wav", np.zeros(44100, np.float32), 44100)
    sf.write(tmp_path / "stems" / "pad.flac", np.zeros((22050, 2), np.float32), 22050)
    chart = make_chart([])
    chart.base_dir = tmp_path
    chart.audio.stems = {"lead": "stems/lead.wav", "pad": "stems/pad.flac"}
    with caplog.at_level(logging.WARNING):
        a = load_song_audio(chart, SR)
    assert a.stems["lead"].shape == (SR, 2) and a.stems["pad"].shape == (SR, 2)
    assert a.stems["lead"].dtype == np.float32 and a.oneshots == {}   # fixture one-shots do not exist
    assert any("cannot read" in r.message for r in caplog.records)


def test_setup_resamples_array_audio():
    m = Mixer(SR)
    m.setup(make_chart([]).audio, SongAudio(24000, [np.zeros((24000, 2), np.float32)], {}, {}), [], {})
    assert len(m._tracks[0].data) == SR


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TORQUEHERO_CONFIG_DIR", str(tmp_path))
    assert AudioSettings.load() == AudioSettings()
    s = AudioSettings(device="USB", blocksize=512, latency=0.05)
    assert s.save() == tmp_path / "audio.json"
    assert AudioSettings.load() == s
    assert "offset" not in {f for f in AudioSettings.__dataclass_fields__}


# --- NullAudio ---

def test_null_audio_clock_and_calls():
    now = [100.0]
    a = NullAudio(clock=lambda: now[0])
    a.load(make_chart([]), {"kick": "you"})
    assert a.time() == 0.0
    a.start(-1.0)
    now[0] += 1.5
    assert a.time() == pytest.approx(0.5)
    a.pause()
    now[0] += 5
    assert a.time() == pytest.approx(0.5)
    a.resume()
    now[0] += 0.5
    assert a.time() == pytest.approx(1.0)
    a.seek(0)
    assert a.time() == pytest.approx(0.0)
    a.handle([Sound(0, "kick", "kick", cause="auto"), Sound(0, "kick", "kick", cause="free")])
    a.update(Snapshot(now=0.0))
    a.stop()
    assert [c[0] for c in a.calls] == ["load", "start", "pause", "resume", "seek", "update", "stop"]
    assert [s.cause for s in a.played] == ["free"]


def test_engine_and_null_share_interface():
    public = {"load", "start", "pause", "resume", "seek", "stop", "time", "handle", "update"}
    for cls in (AudioEngine, NullAudio):
        assert public <= set(dir(cls))
    assert {"volume", "underruns", "render_errors"} <= set(dir(NullAudio()))


# --- round 4: transport orderings ---

def test_resume_while_playing_is_a_noop():
    eng, st = running_engine(secs=2.0)
    for _ in range(10):
        st.pull()
    t = eng.time()
    eng.resume()
    assert eng.time() == pytest.approx(t)
    st.pull()
    assert eng.time() == pytest.approx(t + 0.01)


def test_pause_resume_resume_does_not_step_back():
    eng, st = running_engine(secs=2.0)
    for _ in range(10):
        st.pull()
    eng.pause()
    frozen = eng.time()
    eng.resume()
    eng.resume()
    assert eng.time() == frozen
    st.pull()
    assert abs(eng.time() - (frozen + 0.01)) < 0.001


def test_resume_inside_the_pause_fade_does_not_step():
    eng, st = running_engine(secs=2.0)
    for _ in range(10):
        st.pull()
    eng.pause()
    frozen = eng.time()
    st.pull(frames=240)                               # 5 ms into the 10 ms fade
    eng.resume()
    assert eng.time() == frozen
    st.pull(frames=240)                               # fade finishes, the mixer lands 5 ms on
    assert abs(eng.time() - (frozen + 0.005)) < 0.001
    st.pull(frames=240)
    assert abs(eng.time() - (frozen + 0.010)) < 0.001
    out = np.concatenate([st.pull(frames=240) for _ in range(4)])[:, 0]
    assert out[-1] == pytest.approx(0.5)              # faded back in


def test_resume_rewind_uses_the_measured_dac_lead():
    eng, st = running_engine(secs=2.0)
    st.lead = 0.035                                   # the host's real lead differs from stream.latency
    for _ in range(10):
        st.pull()
    eng.pause()
    frozen = eng.time()
    for _ in range(3):
        st.pull()
    eng.resume()
    st.pull()
    assert abs(eng.time() - (frozen + 0.01)) < 0.001
    eng.seek(0.0)
    assert eng.time() == pytest.approx(-0.035)


def test_seek_and_start_reset_held_targets():
    eng, ctl = continuous_engine({"expr": "auto", "faders": "auto"})
    eng._stream = FakeStream(eng._callback)
    eng.update(Snapshot(expr_target=0.7, fader_targets=[0.2, 0.3]))
    eng.update(Snapshot())
    assert (ctl["expr"].v, ctl["faders"].v, ctl["faders"].v1) == (0.7, 0.2, 0.3)
    eng.seek(0.0)
    assert (ctl["expr"].v, ctl["faders"].v, ctl["faders"].v1) == (pytest.approx(0.4 / 0.85), 1.0, 1.0)
    eng.update(Snapshot(expr_target=0.7))
    eng.start(0.0)
    eng.update(Snapshot())
    assert ctl["expr"].v == pytest.approx(0.4 / 0.85)


def test_render_errors_drop_past_auto_sounds():
    notes = make_chart([{"t": 0.05, "kind": "kick"}, {"t": 0.1, "kind": "kick"}, {"t": 0.3, "kind": "kick"}]).notes
    m = mixer(shots={"kick": impulse()}, notes=notes, modes={"kick": "auto"})
    real = m._fire_schedule
    m._fire_schedule = lambda f0, n: 1 / 0
    m.render(SR // 5)                                 # 0.2 s of errors
    m._fire_schedule = real
    out = m.render(SR // 5)
    assert list(np.flatnonzero(out[:, 0])) == [round(0.3 * SR) - SR // 5]


SEQUENCES = {
    "pause-resume": [("start", 0.0), ("pause",), ("resume",)],
    "resume-mid-fade": [("start", 0.0), ("pause!",), ("pull5",), ("resume",)],
    "resume-at-once": [("start", 0.0), ("pause!",), ("resume",)],
    "pause-mid-seek-fade": [("start", 0.0), ("seek!", 2.0), ("pull5",), ("pause!",), ("pull5",), ("resume",)],
    "resume-twice": [("start", 0.0), ("resume",), ("resume",), ("pause",), ("pause",), ("resume",)],
    "seek-pause-resume-seek": [("start", 0.0), ("seek", 2.0), ("pause",), ("resume",), ("seek", 0.5)],
    "seek-mid-fade-resume": [("start", 0.0), ("seek!", 1.0), ("pull5",), ("resume",)],
    "seek-while-paused": [("start", 0.0), ("pause",), ("seek", 3.0), ("resume",)],
    "stop-resume": [("start", 0.0), ("stop",), ("resume",), ("pause",)],
    "stop-start": [("start", 0.0), ("stop",), ("start", 1.0)],
    "load-resume-start": [("start", 0.0), ("load",), ("resume",), ("start", 0.5)],
    "before-start": [("pause",), ("resume",), ("seek", 3.0), ("start", 0.25)],
    "restart": [("start", 0.0), ("start", 0.0), ("start", 2.0)],
}


@pytest.mark.parametrize("name", SEQUENCES)
def test_engine_and_null_agree_on_transport(name):
    st = FakeStream(None)
    st.latency = 0.0
    eng = AudioEngine(stream_factory=lambda **kw: st, volume=1.0)
    st.callback = eng._callback
    chart = make_chart([])
    song = SongAudio(SR, [const(0.5, 4.0)], {}, {})
    eng.load(chart, {}, song)
    null = NullAudio(clock=lambda: st.time)
    null.load(chart, {})
    for op, *args in SEQUENCES[name]:
        if op == "pull5":
            for _ in range(5):
                st.pull(frames=48)
            continue
        settle = not op.endswith("!")                 # "op!" = next op follows at once
        op = op.rstrip("!")
        if op == "load":
            eng.load(chart, {}, song)
            null.load(chart, {})
        else:
            getattr(eng, op)(*args)
            getattr(null, op)(*args)
        assert abs(eng.time() - null.time()) < 0.001, (op, "at once", eng.time(), null.time())
        for _ in range(20 if settle else 0):          # 20 ms settle, 1 ms blocks
            st.pull(frames=48)
            assert abs(eng.time() - null.time()) < 0.001, (op, eng.time(), null.time())


def test_pause_that_drops_a_pending_resume_forgets_its_frames():
    eng, st = running_engine(secs=2.0)
    for _ in range(10):
        st.pull()
    eng.pause()
    st.pull(frames=240)                               # mid fade
    eng.resume()                                      # pending: rewind after the fade
    st.pull(frames=120)
    eng.pause()                                       # drops that resume
    st.pull(frames=480)
    eng.seek(1.0)                                     # while paused: lands exactly on 1.0
    st.pull(frames=1)
    assert eng.mixer.frame in (SR, SR + 1)
