import ast
import bisect
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import torquehero.ffb as ffb
from torquehero.config import Config
from torquehero.ffb import (
    BRAKE_STOP_DEG_S,
    BUDGET,
    DEADMAN_MS,
    ECHO_FULL_DEG,
    ECHO_LEAD_DEG,
    ECHO_MAX_DEG_S,
    EFFECT_TYPES,
    ENDED_FRACTION,
    ONESHOT_BUDGET,
    ONESHOTS,
    PATTERN_ECHO_DEG_S,
    RAMP_LIMIT,
    RAMP_WINDOW,
    RAMPED,
    RATE_FULL_HZ,
    RATE_OFF_HZ,
    SLOTS,
    SPRING_DEADMAN_MS,
    STEP_CAP,
    VEL_MAX,
    FfbEngine,
    FfbSettings,
    NullFfb,
    SdlHapticBackend,
    magnitude,
    open_backend,
    spin_active,
)
from torquehero.state import Beat, FfbCue, InputState, NoteView, RiserStatus, SectionChange, Snapshot


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def rig():
    """Factory: (engine, backend, clock) on a NullFfb, one fake clock for both.
    The engine is warmed up so the effective gain has reached its target.
    Closes every engine at the end."""
    made = []

    def make(supported=EFFECT_TYPES, settings=None, warm=True, max_effects=None, wheel_range_deg=None,
             backend=None, **cfg):
        clock = Clock()
        be = backend or NullFfb(supported, clock=clock, max_effects=max_effects, record_all=True)
        be.clock = clock
        if settings is None:              # the riser buzz is off by default; the tests exercise it
            settings = FfbSettings(strength={"riser": 1.0})
        eng = FfbEngine(be, Config(**cfg), settings, wheel_range_deg=wheel_range_deg, clock=clock)
        made.append(eng)
        if warm:
            run(eng, clock, 0.2)
        return eng, be, clock

    yield make
    for eng in made:
        eng.close()


def snap(t=0.0, phase="play", steer_deg=0.0, riser=None, **kw) -> Snapshot:
    return Snapshot(phase=phase, now=t, input=InputState(steer_deg=steer_deg), riser=riser or RiserStatus(), **kw)


def step(eng, clock, dt, events=(), **kw) -> dict:
    clock.t += dt
    return eng.update(list(events), snap(clock.t, **kw), dt)


def run(eng, clock, secs, dt=1 / 120, **kw) -> dict:
    out = {}
    for _ in range(round(secs / dt)):
        out = step(eng, clock, dt, **kw)
    return out


def updates(be, slot):
    return [c for c in be.calls if c.method == "update" and c.slot == slot]


def last(be, slot) -> dict:
    return updates(be, slot)[-1].params


def replay(be):
    """Replay the calls as a device plays them: `update` changes an effect's level
    at once (a playing effect keeps its timer), `run` (re)starts it for its
    length_ms, `stop`/`stop_all`/`close` end it, and it ends by itself when the
    length runs out. Returns [(t, {slot: output}, stop_all)] at every change,
    expiries included, in time order."""
    value, until, length, out = {}, {}, {}, []

    def expire(t):
        for slot, end in sorted(until.items(), key=lambda kv: kv[1]):
            if end <= t:
                del until[slot]
                out.append((end, {s: value[s] for s in until}, False))

    for c in be.calls:
        expire(c.t)
        cut = False
        if c.method in ("stop_all", "close"):
            until.clear()
            cut = True
        elif c.method == "update":
            value[c.slot] = math.copysign(magnitude(c.params), c.params.get("level", 1.0))
            length[c.slot] = c.params.get("length_ms", DEADMAN_MS) / 1000
        elif c.method == "run":
            until[c.slot] = c.t + length[c.slot]
        elif c.method == "stop":
            until.pop(c.slot, None)
        else:
            continue
        out.append((c.t, {s: value[s] for s in until}, cut))
    return out


def series(be, slot):
    """Output of one slot as [(t, value)] segments, split at stop_all (a safety stop
    may drop to zero at once). Starts from zero."""
    segs, cur = [], [(0.0, 0.0)]
    for t, outs, cut in replay(be):
        if cut:
            segs.append(cur)
            cur = [(t, 0.0)]
        elif not cur or cur[-1][1] != outs.get(slot, 0.0):
            cur.append((t, outs.get(slot, 0.0)))
    segs.append(cur)
    return segs


def assert_ramped(be, slot):
    """Rule 2: within any 10 ms the output of `slot` moves away from zero, or across
    it, by at most RAMP_LIMIT. A drop toward zero (the device's dead-man expiry, a
    budget cut) is always allowed; the gain ramp-down has its own test."""
    for seg in series(be, slot):
        for j in range(1, len(seg)):
            tj, vj = seg[j]
            for k in range(j - 1, -1, -1):
                if seg[k + 1][0] <= tj - RAMP_WINDOW:
                    break
                vk = seg[k][1]
                if abs(vj) <= abs(vk) and vj * vk >= 0:
                    continue
                assert abs(vj - vk) <= RAMP_LIMIT + 1e-9, (slot, seg[k], seg[j])


def device_load(be, per_call=False):
    """[(t, total, one-shot total)] of the device output at the end of each instant,
    or after every single call with `per_call`."""
    events = replay(be)
    res = []
    for i, (t, outs, _) in enumerate(events):
        if per_call or i + 1 == len(events) or events[i + 1][0] != t:
            shots = sum(abs(v) for s, v in outs.items() if s in ONESHOTS)
            res.append((t, sum(abs(v) for v in outs.values()), shots))
    return res


# --- module hygiene ---

def test_sdl2_imported_only_inside_functions():
    tree = ast.parse(Path(ffb.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
            assert not any(n.startswith("sdl2") for n in names)


def test_no_infinite_effects():
    assert "SDL_HAPTIC_INFINITY" not in Path(ffb.__file__).read_text()


def test_slot_caps_match_rule_2():
    for slot, spec in SLOTS.items():
        assert spec.type in EFFECT_TYPES
        if spec.length_ms and slot not in ("kick", "pulse"):
            assert spec.cap <= RAMP_LIMIT   # a one-shot steps, so it must stay within the ramp limit
        assert spec.cap <= (STEP_CAP if slot in ("kick", "pulse") else 1.0)


# --- NullFfb ---

def test_null_ffb_records_every_call_with_time():
    clock = Clock()
    be = NullFfb(clock=clock)
    assert isinstance(be, ffb.FfbBackend)
    clock.t = 1.5
    assert be.create("x", "spring")
    be.update("x", saturation=0.2)
    clock.t = 2.0
    be.run("x")
    be.set_gain(0.0)
    be.stop_all()
    be.close()
    assert [(c.t, c.method) for c in be.calls] == [(1.5, "create"), (1.5, "update"), (2.0, "run"),
                                                   (2.0, "set_gain"), (2.0, "stop_all"), (2.0, "close")]
    assert be.calls[1].params == {"saturation": 0.2} and be.closed and not be.running and be.gain == 0.0


# --- effects ---

def test_beat_pulse_from_strength_and_stale_beats_skipped(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    step(eng, clock, 0.01, [Beat(clock.t + 0.01, 1.0)])
    p = last(be, "pulse")
    assert p["magnitude"] == pytest.approx(0.5) and p["length_ms"] == 80
    step(eng, clock, 0.1, [Beat(clock.t + 0.1, 0.4)])
    assert last(be, "pulse")["magnitude"] == pytest.approx(0.2)
    n = len(updates(be, "pulse"))
    clock.t = 5.0
    step(eng, clock, 0.01, [Beat(1.0, 1.0)])    # crossed long ago (mid-song start)
    assert len(updates(be, "pulse")) == n


def test_section_weight_spring_and_damper_follow_weight(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    out = step(eng, clock, 0.01, [SectionChange(0.0, "chorus", 1.0)], weight=1.0)
    assert "WEIGHT 100%" in out["log"]
    run(eng, clock, 1.0, weight=1.0, steer_deg=-90.0)          # full spring level 90 deg off centre
    heavy = (last(be, "weight_spring")["level"], last(be, "weight_damper")["saturation"])
    run(eng, clock, 1.0, weight=0.0, steer_deg=-90.0)
    light = (last(be, "weight_spring")["level"], last(be, "weight_damper")["saturation"])
    assert heavy == (pytest.approx(0.3), pytest.approx(0.5))  # spring capped at 0.3 (sustained force)
    assert light == (pytest.approx(0.06), pytest.approx(0.1))
    run(eng, clock, 1.0, weight=1.0, steer_deg=45.0)           # right of centre: pushes left, half level
    assert last(be, "weight_spring")["level"] == pytest.approx(-0.15)
    assert_ramped(be, "weight_spring")


def test_riser_winds_up_alternates_and_drops_on_release(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    out = step(eng, clock, 0.01, [FfbCue(0.0, "riser_start")])
    assert "RISER wind-up" in out["log"]
    for i in range(240):
        step(eng, clock, 1 / 120, riser=RiserStatus(active=True, progress=i / 240, held=True))
    cycles = [(c.t, c.params) for c in be.calls if c.method == "update" and c.slot == "riser"]
    runs = [c.t for c in be.calls if c.method == "run" and c.slot == "riser"]
    assert all(p["period_ms"] == p["length_ms"] == ffb.RISER_CYCLE_MS for _, p in cycles)   # whole cycles
    assert all(b - a >= ffb.RISER_CYCLE_MS / 1000 - 1e-9 for a, b in zip(runs, runs[1:], strict=False))
    mags = [p["magnitude"] for _, p in cycles]
    assert mags[0] < mags[-1] <= SLOTS["riser"].cap + 1e-9                  # grows with the wind-up
    n = len(runs)
    for _ in range(60):                               # release: fades out over 2 cycles, whole cycles
        step(eng, clock, 1 / 120, riser=RiserStatus(active=True, progress=1.0, held=False))
    assert 1 <= len([c for c in be.calls if c.method == "run" and c.slot == "riser"]) - n <= 3
    assert not [c for c in be.calls if c.method == "stop" and c.slot == "riser"]   # never cut
    assert eng._wind == 0.0 and eng._wall >= eng._riser_end


def test_riser_inactive_gives_no_wind_up(rig):
    eng, be, clock = rig()
    run(eng, clock, 0.5, riser=RiserStatus(active=False, held=True))
    assert updates(be, "riser") == []


def test_echo_centre_moves_toward_target_at_rate_limit(rig):
    eng, be, clock = rig(ffb_gain=1.0, play_range_deg=90.0, wheel_range_deg=900.0)
    out = step(eng, clock, 0.01, echo="listen", echo_target=1.0)          # step to +90 deg opens the region
    assert "ECHO spring centre -> 90 deg" in out["log"]
    assert eng._centre == 0.0                                             # starts at the wheel
    for i in range(1, 60):                                                # the player lets the wheel follow
        step(eng, clock, 0.01, steer_deg=min(90.0, 1.8 * (i - 1)), echo="listen", echo_target=1.0)
        assert eng._centre == pytest.approx(min(90.0, 1.8 * i))           # 180 deg/s
    out = step(eng, clock, 0.01, steer_deg=90.0, echo="listen", echo_target=1.0)
    assert out["echo_centre"] == pytest.approx(1.0)
    p = last(be, "echo")
    assert set(p) == {"length_ms", "level"} and p["length_ms"] == SPRING_DEADMAN_MS   # a constant force now
    assert eng._out["weight_spring"] == 0.0 and eng._out["weight_damper"] > 0


def test_echo_centre_waits_for_a_held_wheel(rig):
    eng, be, clock = rig(play_range_deg=90.0)
    run(eng, clock, 1.0, steer_deg=0.0, echo="listen", echo_target=1.0)
    assert eng._centre == pytest.approx(ECHO_LEAD_DEG)
    run(eng, clock, 0.1, steer_deg=-30.0, echo="listen", echo_target=1.0)
    assert eng._centre == pytest.approx(-30.0 + ECHO_LEAD_DEG)


def test_echo_resyncs_after_pause_with_wheel_moved(rig):
    eng, be, clock = rig(play_range_deg=90.0)
    run(eng, clock, 0.3, steer_deg=0.0, echo="listen", echo_target=0.5)
    run(eng, clock, 0.2, phase="paused", steer_deg=200.0, echo="listen", echo_target=0.5)
    step(eng, clock, 0.01, steer_deg=200.0, echo="listen", echo_target=0.5)
    assert eng._centre == pytest.approx(200.0)
    for _ in range(50):
        step(eng, clock, 0.01, steer_deg=200.0, echo="listen", echo_target=0.5)
        assert abs(eng._centre - 200.0) <= ECHO_LEAD_DEG + 1e-9


def test_echo_rate_clamped_even_if_config_allows_more(rig):
    eng, be, clock = rig(echo_max_deg_s=5000.0)
    step(eng, clock, 0.01, echo="listen", echo_target=1.0)
    step(eng, clock, 0.01, echo="listen", echo_target=1.0)
    assert eng._centre == pytest.approx(ECHO_MAX_DEG_S * 0.01)


def test_wheel_range_minimum_turns_echo_off(rig):
    eng, be, clock = rig(wheel_range_deg=90.0)
    out = run(eng, clock, 0.3, echo="listen", echo_target=1.0)
    assert not eng.enabled["echo"] and updates(be, "echo") == []
    assert "NO ECHO - wheel range 90 deg < 180" in out["log"]


def test_perfect_kick_and_miss_rumble(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    out = step(eng, clock, 0.01, [FfbCue(0.0, "kick", dir=-1.0)])
    k = last(be, "kick")
    assert k == {"length_ms": 60, "level": pytest.approx(-STEP_CAP)}
    assert out["log"][0] == "KICK left" and out["torque"] < 0
    run(eng, clock, 0.1)
    step(eng, clock, 0.01, [FfbCue(0.0, "rumble")])
    r = last(be, "rumble")
    assert r["period_ms"] == 125 and r["length_ms"] == 250 and r["magnitude"] == pytest.approx(0.4)
    assert r["attack_ms"] > 0 and "rumble" in be.running


def test_ffb_sign_flips_constant_forces(rig):
    eng, be, clock = rig(settings=FfbSettings(ffb_sign=-1), ffb_gain=1.0)
    step(eng, clock, 0.01, [FfbCue(0.0, "kick", dir=1.0)])
    assert last(be, "kick")["level"] < 0


def test_log_is_short_newest_first_and_collapses_repeats(rig):
    eng, be, clock = rig()
    for _ in range(3):
        step(eng, clock, 0.01, [FfbCue(0.0, "rumble")])
    out = step(eng, clock, 0.01, [FfbCue(0.0, "kick", dir=1.0)])
    assert out["log"][:2] == ["KICK right", "MISS rumble x3"]
    for i in range(20):
        eng._note(f"m{i}")
    assert len(eng.report()["log"]) == eng.log_len


def test_echo_is_a_software_spring_on_the_angle_error(rig):
    eng, be, clock = rig(ffb_gain=1.0, play_range_deg=90.0)
    run(eng, clock, 1.0, steer_deg=0.0, echo="listen", echo_target=0.5)    # centre 45 deg right of the wheel
    assert eng._centre == pytest.approx(45.0)
    assert last(be, "echo")["level"] == pytest.approx(SLOTS["echo"].cap)    # full level at 45 deg of error
    run(eng, clock, 1.0, steer_deg=45.0 - 22.5, echo="listen", echo_target=0.5)
    assert last(be, "echo")["level"] == pytest.approx(SLOTS["echo"].cap / 2)

def test_velocity_term_opposes_wheel_motion(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    deg = -90.0
    for _ in range(60):                                   # wheel moving right at 120 deg/s
        deg += 1.0
        step(eng, clock, 1 / 120, weight=1.0, steer_deg=deg)
    spring = 0.3 * (0.0 - deg) / 90.0
    assert eng._pre["weight_spring"] == pytest.approx(spring - ffb.K_V * 120.0)


# --- rule 1: gain ---

def test_gain_is_clamped_to_0_1_and_read_only(rig):
    eng, be, clock = rig()
    assert eng.set_gain(3.0) == 1.0 and eng.set_gain(-1) == 0.0 and eng.set_gain(float("nan")) == 0.0
    with pytest.raises(AttributeError):
        eng.gain = 2.0
    cfg = Config()
    cfg.ffb_gain = 7.0                  # bypasses Config's own clamp
    e2 = FfbEngine(NullFfb(), cfg)
    assert e2.gain == 1.0
    e2.close()


def test_caps_apply_before_gain(rig):
    eng, be, clock = rig(ffb_gain=0.5)
    step(eng, clock, 0.01, [FfbCue(0.0, "kick", dir=1.0)])
    assert last(be, "kick")["level"] == pytest.approx(STEP_CAP * 0.5)   # not min(0.8 * 0.5, 0.6)


def test_lower_gain_ramps_down(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    run(eng, clock, 0.5, weight=1.0, steer_deg=-90.0)
    before = eng._out["weight_spring"]
    eng.set_gain(0.0)
    step(eng, clock, 1 / 120, weight=1.0, steer_deg=-90.0)
    assert 0 < eng._out["weight_spring"] < before
    run(eng, clock, 0.3, weight=1.0, steer_deg=-90.0)
    assert eng._out["weight_spring"] == 0.0 and eng.report()["gain_now"] == 0.0
    assert_ramped(be, "weight_spring")


def test_gain_rise_is_ramped(rig):
    eng, be, clock = rig(ffb_gain=0.0, play_range_deg=90.0)
    run(eng, clock, 0.2, echo="listen", echo_target=0.5)
    eng.set_gain(1.0)
    run(eng, clock, 0.5, echo="listen", echo_target=0.5)       # wheel held at 0, centre 45 deg ahead
    assert eng._out["echo"] == pytest.approx(SLOTS["echo"].cap)
    assert_ramped(be, "echo")


def test_settings_strength_scales_before_cap(rig, tmp_path):
    s = FfbSettings(strength={"kick": 0.5, "pulse": 9, "echo": "junk"}, ffb_sign=-3)
    assert s.strength["kick"] == 0.5 and s.strength["pulse"] == 1.0 and s.strength["echo"] == 1.0
    assert s.ffb_sign == -1
    path = s.save(tmp_path / "ffb.json")
    assert FfbSettings.load(path) == s
    (tmp_path / "old.json").write_text('{"echo_stiffness": 0.6, "ffb_sign": -1}')   # older files still load
    assert FfbSettings.load(tmp_path / "old.json").ffb_sign == -1
    assert FfbSettings.load(tmp_path / "missing.json") == FfbSettings()
    assert FfbSettings.load(tmp_path / "missing.json").sign_error is None
    eng, be, clock = rig(settings=FfbSettings(strength={"kick": 0.5}), ffb_gain=1.0)
    step(eng, clock, 0.01, [FfbCue(0.0, "kick", dir=1.0)])
    assert last(be, "kick")["level"] == pytest.approx(0.4)


@pytest.mark.parametrize("text", ["{nope", "[1]", '{"ffb_sign": "-1"}', '{"ffb_sign": true}',
                                  '{"ffb_sign": 0}', '{"ffb_sign": null}', "\xff\xfe"])
def test_settings_unreadable_sign_is_an_error_not_a_reset(tmp_path, text, caplog):
    path = tmp_path / "ffb.json"
    path.write_bytes(text.encode("latin-1"))
    s = FfbSettings.load(path)
    assert s.sign_error == ffb.SIGN_ERROR == "ffb.json: cannot read ffb_sign: fix the file or delete it"
    assert caplog.records
    FfbSettings(ffb_sign=-1).save(path)                          # the broken file is kept first
    assert (tmp_path / "ffb.json.bad").read_bytes() == text.encode("latin-1")


def test_settings_load_key_by_key_with_one_warning_per_bad_key(tmp_path, caplog):
    path = tmp_path / "ffb.json"
    path.write_text('{"strength": {"kick": 0.5, "pulse": "x", "echo": null, "nope": 1}, "ffb_sign": -1}')
    s = FfbSettings.load(path)
    assert s.sign_error is None and s.ffb_sign == -1
    assert s.strength["kick"] == 0.5 and s.strength["pulse"] == 1.0 and s.strength["echo"] == 1.0
    assert [r.getMessage().split(": ")[1] for r in caplog.records] == ["ignored strength.pulse='x'",
                                                                       "ignored strength.echo=None"]
    s.save(path)
    assert "pulse" in (tmp_path / "ffb.json.bad").read_text()
    path.write_text('{"strength": 3, "ffb_sign": -1.0}')
    s = FfbSettings.load(path)
    assert s.sign_error is None and s.ffb_sign == -1 and s.strength == ffb.DEFAULT_STRENGTH


def test_settings_accept_a_utf8_bom(tmp_path, caplog):
    path = tmp_path / "ffb.json"
    path.write_bytes(b"\xef\xbb\xbf" + b'{"ffb_sign": -1}')
    s = FfbSettings.load(path)
    assert s.ffb_sign == -1 and s.sign_error is None and not caplog.records


def test_report_shows_the_sign_in_use(rig):
    eng, be, clock = rig(settings=FfbSettings(ffb_sign=-1))
    assert eng.report()["sign"] == -1
    eng, be, clock = rig(settings=FfbSettings(sign_error=ffb.SIGN_ERROR))
    assert eng.report()["sign"] is None


def test_settings_default_path_is_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("TORQUEHERO_CONFIG_DIR", str(tmp_path))
    assert FfbSettings().save() == tmp_path / "ffb.json"


# --- rule 2: ramps and measured time ---

@pytest.mark.parametrize("dt", [0.001, 0.003, 1 / 60, 0.2])
def test_ramped_effects_never_step_more_than_limit(rig, dt):
    eng, be, clock = rig(ffb_gain=1.0)
    for i in range(round(1.0 / dt) + 6):
        on = (i // 3) % 2 == 0 if dt >= 0.1 else (i * dt) % 0.3 < 0.15
        step(eng, clock, dt, weight=1.0 if on else 0.0, echo="listen" if on else "none", echo_target=1.0,
             steer_deg=-60.0 if on else 30.0, riser=RiserStatus(active=True, progress=1.0, held=on))
    for slot in RAMPED:
        if dt > 1 / RATE_OFF_HZ and slot in ("echo", "weight_spring", "riser"):
            assert not updates(be, slot), slot          # rate floor: springs and buzz off below 50 Hz
        elif slot not in ("brake", "pattern"):          # after a stall / in test patterns only
            assert updates(be, slot), slot
        assert_ramped(be, slot)


def test_steps_use_measured_time_not_caller_dt(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    step(eng, clock, 0.001, echo="listen", echo_target=1.0, weight=1.0)
    pre, centre = dict(eng._pre), eng._centre
    for _ in range(3):
        eng.update([], snap(clock.t, echo="listen", echo_target=1.0, weight=1.0), 0.05)   # same instant
    assert eng._pre == pre and eng._centre == centre


def test_ramped_effects_refresh_at_half_length(rig):
    eng, be, clock = rig()
    n = len(be.calls)
    run(eng, clock, 1.0, weight=0.5)
    ups = [c for c in be.calls[n:] if c.method == "update" and c.slot == "weight_damper"]
    runs = [c.t for c in be.calls[n:] if c.method == "run" and c.slot == "weight_damper"]
    assert len(ups) == 120 and all(c.params["length_ms"] == DEADMAN_MS for c in ups)   # level every frame
    assert 8 <= len(runs) <= 11                                                        # re-run about every 100 ms
    assert all(b - a <= DEADMAN_MS / 1000 * 0.6 for a, b in zip(runs, runs[1:], strict=False))


def test_expired_slot_restarts_from_zero_below_the_gap_threshold(rig):
    # run at T, update at T+0.09 (no re-run yet), update at T+0.21: each gap is under
    # GAP_RESTART_S, but the device ended the effects at T+0.2
    eng, be, clock = rig(ffb_gain=1.0)
    kw = dict(steer_deg=0.0, echo="listen", echo_target=0.3, weight=1.0,
              riser=RiserStatus(active=True, progress=1.0, held=True))
    run(eng, clock, 1.0, **kw)
    for dt in [0.09, 0.12] * 6:
        out = step(eng, clock, dt, **kw)
        assert not any(m.startswith("GAP") for m in out["log"])
    for slot in RAMPED:
        assert_ramped(be, slot)


def _just_ran(eng, clock, slot, **kw):
    """Step at 120 Hz until `slot` was re-run on this frame."""
    for _ in range(40):
        step(eng, clock, 1 / 120, **kw)
        if eng._ran_at[slot] == eng._wall:
            return
    raise AssertionError("no re-run")


@pytest.mark.parametrize("gap", [0.02, 0.12, 0.149, 0.16, 0.19])
def test_live_slot_during_a_gap_is_stopped_and_counted(rig, gap):
    eng, be, clock = rig(ffb_gain=1.0)
    listen = dict(steer_deg=0.0, echo="listen", echo_target=0.5, weight=1.0)
    run(eng, clock, 1.0, **listen)
    _just_ran(eng, clock, "echo", **listen)
    clock.t += gap
    step(eng, clock, 1 / 120, [FfbCue(0.0, "kick", dir=1.0)], weight=1.0, steer_deg=0.0)   # listen over
    stopped = [c for c in be.calls if c.method == "stop" and c.slot == "echo"]
    if gap + 1 / 120 >= SLOTS["echo"].deadman_s * ENDED_FRACTION:
        assert stopped                                     # may have ended: stopped for certain
    else:
        assert not stopped and "echo" in eng._running      # live: keeps ramping down, still counted
    for t, total, _ in device_load(be):
        assert total <= BUDGET + 1e-9, (t, total)
    for slot in RAMPED:
        assert_ramped(be, slot)


def test_slot_counts_as_ended_150_ms_after_its_last_run(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    run(eng, clock, 1.0, weight=1.0)
    _just_ran(eng, clock, "weight_damper", weight=1.0)
    clock.t += 0.15
    step(eng, clock, 1 / 120, weight=1.0)
    stops = [c.t for c in be.calls if c.method == "stop" and c.slot == "weight_damper"]
    assert stops and eng._out["weight_damper"] <= 0.125 + 1e-9    # stopped, restarting from zero


def test_listen_edge_does_not_move_a_live_echo_centre(rig):
    eng, be, clock = rig(ffb_gain=1.0, play_range_deg=90.0)
    run(eng, clock, 1.0, steer_deg=0.0, echo="listen", echo_target=0.5)
    step(eng, clock, 1 / 120, steer_deg=30.0, echo="none")                 # listen ends, spring ramps down
    before = eng._centre
    step(eng, clock, 1 / 120, steer_deg=30.0, echo="listen", echo_target=0.0)    # next region opens
    assert eng._out["echo"] > 0 and abs(eng._centre - before) <= ECHO_MAX_DEG_S / 120 + 1e-9


def test_opposite_kick_is_skipped_while_one_is_active(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    step(eng, clock, 1 / 120, [FfbCue(0.0, "kick", dir=1.0)])
    step(eng, clock, 1 / 120, [FfbCue(0.0, "kick", dir=-1.0)])
    assert [c.params["level"] > 0 for c in updates(be, "kick")] == [True]
    run(eng, clock, 0.1)
    step(eng, clock, 1 / 120, [FfbCue(0.0, "kick", dir=-1.0)])
    assert last(be, "kick")["level"] < 0


def test_gap_past_the_dead_man_restarts_from_zero(rig):
    eng, be, clock = rig(ffb_gain=1.0, play_range_deg=90.0)
    run(eng, clock, 1.0, steer_deg=0.0, echo="listen", echo_target=0.5)
    assert eng._out["echo"] > 0.2
    clock.t += 0.3                                     # song load, GC pause, alt-tab hitch
    out = step(eng, clock, 1 / 120, steer_deg=60.0, echo="listen", echo_target=0.5)
    assert any(m.startswith("GAP 3") for m in out["log"])
    assert abs(eng._out["echo"]) <= 0.125 + 1e-9
    for slot in RAMPED:
        assert_ramped(be, slot)


# --- turn offset (SPEC 9 rule 10) ---

def test_centres_follow_the_turn_offset(rig):
    eng, be, clock = rig(ffb_gain=1.0, play_range_deg=90.0, wheel_range_deg=1080.0)
    run(eng, clock, 0.5, steer_deg=360.0 - 90.0, steer_offset_deg=360.0)
    assert last(be, "weight_spring")["level"] == pytest.approx(0.3 * 0.2)   # w 0: pulls toward 360, not 0
    run(eng, clock, 1.0, steer_deg=380.0, steer_offset_deg=360.0, echo="listen", echo_target=0.5)
    assert eng._centre == pytest.approx(360.0 + 45.0)          # goal 405 deg = 0.5 lane + one turn
    assert eng.report()["echo_centre"] == pytest.approx(0.5)


def test_unwind_holds_springs_off_and_keeps_the_damper(rig):
    eng, be, clock = rig(ffb_gain=1.0, wheel_range_deg=1080.0)
    run(eng, clock, 0.5, weight=1.0, steer_deg=200.0, steer_unwind=True, echo="listen", echo_target=0.0)
    assert eng._out["weight_spring"] == 0.0 and eng._out["echo"] == 0.0 and eng._out["weight_damper"] > 0.4
    run(eng, clock, 0.5, weight=1.0, steer_deg=-90.0, steer_unwind=False)
    assert 0 < eng._out["weight_spring"] < 0.1           # springs back: the centre returns gently
    run(eng, clock, 3.5, weight=1.0, steer_deg=-90.0, steer_unwind=False)
    assert eng._out["weight_spring"] > 0.25
    assert_ramped(be, "weight_spring")


def test_offset_change_resyncs_without_a_centre_jump(rig):
    eng, be, clock = rig(ffb_gain=1.0, wheel_range_deg=1080.0)
    run(eng, clock, 0.5, weight=1.0, steer_deg=-90.0)
    assert eng._out["weight_spring"] > 0.25
    step(eng, clock, 1 / 120, weight=1.0, steer_deg=270.0, steer_offset_deg=360.0, steer_offset_changed=True)
    assert eng._out["weight_spring"] == 0.0 and "weight_spring" not in be.running   # stopped, not moved
    run(eng, clock, 0.5, weight=1.0, steer_deg=270.0, steer_offset_deg=360.0)
    assert last(be, "weight_spring")["level"] == pytest.approx(0.3)                  # toward the new centre
    assert_ramped(be, "weight_spring")


def test_offset_change_detected_by_value_and_repeated_flag_is_harmless(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    run(eng, clock, 0.5, weight=1.0, steer_deg=-90.0)
    step(eng, clock, 1 / 120, weight=1.0, steer_deg=270.0, steer_offset_deg=360.0)   # flag missed
    assert eng._out["weight_spring"] == 0.0
    run(eng, clock, 0.5, weight=1.0, steer_deg=270.0, steer_offset_deg=360.0, steer_offset_changed=True)
    assert eng._out["weight_spring"] > 0.25


def test_active_spin_releases_the_weight_spring(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    run(eng, clock, 0.5, weight=1.0, steer_deg=-90.0)
    for _ in range(60):
        clock.t += 1 / 120
        eng.update([], snap(clock.t, weight=1.0, steer_deg=-90.0), 1 / 120, spinning=True)
    assert eng._out["weight_spring"] == 0.0 and eng._out["weight_damper"] > 0.4
    run(eng, clock, 0.5, weight=1.0, steer_deg=-90.0)
    assert eng._out["weight_spring"] > 0.25
    assert_ramped(be, "weight_spring")


def test_spin_active_reads_the_note_kind_from_the_chart():
    chart = SimpleNamespace(notes=[SimpleNamespace(kind="spin", t=1.0), SimpleNamespace(kind="riser", t=1.0)])
    assert spin_active(snap(1.2, notes=[NoteView(0, progress=0.3)]), chart)
    assert not spin_active(snap(0.5, notes=[NoteView(0, progress=0.0)]), chart)    # not started yet
    assert not spin_active(snap(1.2, notes=[NoteView(0, done=True, progress=1.0)]), chart)
    assert not spin_active(snap(1.2, notes=[NoteView(1, progress=0.5)]), chart)
    assert not spin_active(snap(1.2, notes=[NoteView(7, progress=0.5)]), chart)


def test_spin_active_from_a_real_game_snapshot():
    from conftest import game_for, inp

    g = game_for([{"t": 2.0, "kind": "spin", "dur": 1.0}])
    g.update(1.0, 0.01, inp(steer_deg=0.0, bound={"steer"}))
    s = g.snapshot()
    assert [v.progress for v in s.notes] == [0.0]          # in view with a progress, but ahead
    assert not spin_active(s, g.chart)
    g.update(2.05, 0.01, inp(steer_deg=20.0, bound={"steer"}))
    assert spin_active(g.snapshot(), g.chart)
    for k in range(1, 100):
        g.update(2.05 + k * 0.01, 0.01, inp(steer_deg=20.0 + k * 4.0, bound={"steer"}))
    assert not spin_active(g.snapshot(), g.chart)            # judged: done


def test_offset_change_keeps_damper_and_riser(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    kw = dict(weight=1.0, riser=RiserStatus(active=True, progress=1.0, held=True))
    run(eng, clock, 1.0, steer_deg=0.0, **kw)
    damper, riser = eng._out["weight_damper"], abs(eng._out["riser"])
    step(eng, clock, 1 / 120, steer_deg=360.0, steer_offset_deg=360.0, **kw)
    assert eng._out["weight_spring"] == 0.0
    assert eng._out["weight_damper"] >= damper and "weight_damper" in be.running   # not cut (freed budget)
    assert abs(eng._out["riser"]) > riser - 0.2 and "riser" in be.running


def test_springs_off_when_wheel_is_half_a_turn_from_the_offset(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    run(eng, clock, 0.5, weight=1.0, steer_deg=200.0, steer_unwind=False, echo="listen", echo_target=0.0)
    assert eng._out["weight_spring"] == 0.0 and eng._out["echo"] == 0.0 and eng._out["weight_damper"] > 0.4


def test_non_finite_steer_sample_releases_the_springs(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    run(eng, clock, 0.5, weight=1.0, steer_deg=-90.0)
    assert eng._out["weight_spring"] > 0.25
    run(eng, clock, 0.3, weight=1.0, steer_deg=float("nan"))
    assert eng._out["weight_spring"] == 0.0 and eng._out["weight_damper"] > 0.4


def test_non_finite_offset_holds_the_last_one(rig):
    eng, be, clock = rig(ffb_gain=1.0, wheel_range_deg=1080.0)
    run(eng, clock, 0.5, weight=1.0, steer_deg=270.0, steer_offset_deg=360.0)
    before = eng._out["weight_spring"]
    step(eng, clock, 1 / 120, weight=1.0, steer_deg=270.0, steer_offset_deg=float("nan"))
    assert before > 0.25 and eng._out["weight_spring"] == pytest.approx(before)


def test_stop_inside_a_frame_cuts_the_rest_of_it():
    class Stopper(NullFfb):
        engine = None

        def update(self, slot, **params):
            super().update(slot, **params)
            if slot == "weight_damper":
                self.engine.stop_all()      # e.g. a focus-loss handler running mid-frame

    clock = Clock()
    be = Stopper(clock=clock, record_all=True)
    eng = be.engine = FfbEngine(be, Config(ffb_gain=1.0), clock=clock)
    run(eng, clock, 0.2, weight=1.0, steer_deg=-90.0)
    i = next(i for i, c in enumerate(be.calls) if c.method == "stop_all")
    after = [c for c in be.calls[i:] if c.method in ("update", "run")]
    assert after == [] and eng.latched
    eng.close()


def test_default_clock_is_perf_counter():
    eng = FfbEngine(NullFfb(), Config())
    assert eng.clock is time.perf_counter and eng.backend.clock is time.perf_counter
    eng.close()


# --- steer, sign, clock ---

def test_non_finite_steer_holds_the_last_value(rig):
    eng, be, clock = rig(play_range_deg=90.0)
    run(eng, clock, 0.2, steer_deg=200.0, echo="listen", echo_target=1.0)
    run(eng, clock, 0.3, steer_deg=float("nan"), echo="listen", echo_target=1.0)
    assert abs(eng._centre - 200.0) <= ECHO_LEAD_DEG + 1e-9


def test_no_finite_steer_yet_keeps_echo_off():
    clock = Clock()
    be = NullFfb(clock=clock, record_all=True)
    eng = FfbEngine(be, Config(), clock=clock)
    run(eng, clock, 0.5, steer_deg=float("inf"), echo="listen", echo_target=1.0, weight=1.0)
    assert updates(be, "echo") == [] and "weight_damper" in be.running
    eng.close()


def test_ffb_sign_change_applies_at_the_next_start(rig):
    settings = FfbSettings()
    eng, be, clock = rig(settings=settings, ffb_gain=1.0)
    settings.ffb_sign = -1
    step(eng, clock, 0.01, [FfbCue(0.0, "kick", dir=1.0)])
    assert last(be, "kick")["level"] > 0                 # running: unchanged
    eng.stop_all()
    eng.resume()
    run(eng, clock, 0.2)
    step(eng, clock, 0.01, [FfbCue(0.0, "kick", dir=1.0)])
    assert last(be, "kick")["level"] < 0


def test_one_shot_holds_budget_one_frame_past_its_length(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    step(eng, clock, 0.01, [FfbCue(0.0, "kick", dir=1.0)])   # 0.6 for 60 ms
    step(eng, clock, 0.03)
    step(eng, clock, 0.0305, [Beat(clock.t + 0.0305, 1.0)])   # 60.5 ms after the kick
    assert not updates(be, "pulse")                          # the device may still play the kick
    step(eng, clock, 0.03, [Beat(clock.t + 0.03, 1.0)])
    assert updates(be, "pulse")


def test_clock_jump_does_not_free_a_running_rumble(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    step(eng, clock, 0.01, [FfbCue(0.0, "rumble")])          # 0.4 for 250 ms
    clock.t += 5.0
    step(eng, clock, 0.01, [FfbCue(0.0, "kick", dir=1.0)])
    assert last(be, "kick")["level"] == pytest.approx(0.2)   # only 0.6 - 0.4 left


# --- record cap and device loss ---

def test_null_ffb_record_is_capped_unless_record_all():
    be = NullFfb()
    for _ in range(10_000):
        be.stop_all()
    assert len(be.calls) <= 2 * NullFfb.RECORD_LIMIT
    full = NullFfb(record_all=True)
    for _ in range(5000):
        full.stop_all()
    assert len(full.calls) == 5000


def test_device_lost_closes_the_haptic_device_and_latches(rig):
    eng, be, clock = rig()
    run(eng, clock, 0.2, weight=1.0)
    eng.device_lost()
    assert be.closed and eng.latched and "DEVICE LOST" in eng.report()["log"]
    n = len(be.calls)
    eng.resume()
    run(eng, clock, 0.2, weight=1.0)
    assert be.calls[n:] == []


# --- rule 6: budget ---

def test_kick_on_a_beat_stays_in_budget(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    run(eng, clock, 1.0, weight=1.0, riser=RiserStatus(active=True, progress=1.0, held=True))
    out = step(eng, clock, 0.01, [FfbCue(0.0, "kick", dir=1.0), Beat(clock.t + 0.01, 1.0), FfbCue(0.0, "rumble")],
               weight=1.0, riser=RiserStatus(active=True, progress=1.0, held=True))
    assert updates(be, "kick") and not updates(be, "pulse") and not updates(be, "rumble")  # one-shots share 0.6
    assert out["load"] <= BUDGET + 1e-9
    for _, total, shots in device_load(be, per_call=True):
        assert total <= BUDGET + 1e-9 and shots <= ONESHOT_BUDGET + 1e-9


def test_reported_torque_is_not_clamped(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    eng._out["weight_spring"] = 3.0   # an impossible excess must show, not hide
    assert eng._estimate(0.0) == pytest.approx(3.0)


def test_budget_holds_after_every_call_when_a_kick_arrives(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    kw = dict(steer_deg=0.0, echo="listen", echo_target=0.0, weight=1.0,
              riser=RiserStatus(active=True, progress=1.0, held=True))
    run(eng, clock, 1.0, **kw)
    for i in range(20):
        step(eng, clock, 1 / 120, [FfbCue(0.0, "kick", dir=1.0)] if i % 10 == 0 else [], **kw)
    for t, total, shots in device_load(be, per_call=True):
        assert total <= BUDGET + 1e-9 and shots <= ONESHOT_BUDGET + 1e-9, (t, total)


def test_failed_send_and_failed_stop_keep_the_slot_in_the_budget(rig):
    class Stuck(NullFfb):
        broken = False

        def update(self, slot, **params):
            if self.broken and slot == "weight_damper":
                raise OSError("device refused")
            super().update(slot, **params)

        def stop(self, slot):
            if self.broken and slot == "weight_damper":
                raise OSError("device refused")
            super().stop(slot)

    eng, be, clock = rig(backend=Stuck(), ffb_gain=1.0)
    run(eng, clock, 1.0, weight=1.0)
    level = eng._pre["weight_damper"]
    be.broken = True
    step(eng, clock, 1 / 120, weight=1.0)
    assert eng._ghosts["weight_damper"][0] == pytest.approx(level)
    out = step(eng, clock, 1 / 120, weight=1.0, echo="listen", echo_target=0.0)
    assert sum(abs(eng._pre[s]) for s in RAMPED) + level <= BUDGET + 1e-9
    assert out["load"] >= level - 1e-9
    run(eng, clock, 0.3, weight=1.0)
    assert "weight_damper" not in eng._ghosts                        # device dead-man has ended it


# --- rule 1 on springs: gain scales the force near centre too ---

def test_gain_scales_springs_and_the_damper(rig):
    peaks = {}
    for gain in (1.0, 0.2):
        eng, be, clock = rig(ffb_gain=gain, play_range_deg=90.0)
        run(eng, clock, 1.0, weight=1.0, steer_deg=-90.0)
        run(eng, clock, 1.0, weight=0.0, steer_deg=0.0, echo="listen", echo_target=0.5)
        peaks[gain] = {"damper": max(c.params["coefficient"] for c in updates(be, "weight_damper")),
                       "echo": max(abs(c.params["level"]) for c in updates(be, "echo")),
                       "weight": max(abs(c.params["level"]) for c in updates(be, "weight_spring"))}
    assert peaks[1.0]["echo"] == pytest.approx(SLOTS["echo"].cap)
    for k, v in peaks[0.2].items():
        assert v <= 0.2 * peaks[1.0][k] + 1e-9, k


# --- first-run test patterns ---

def pattern(eng, clock, name, secs, steer=lambda t: 0.0, dt=1 / 120):
    outs = []
    for i in range(round(secs / dt)):
        clock.t += dt
        outs.append(eng.test_pattern(name, (i + 1) * dt, dt, steer((i + 1) * dt), "play"))
    return outs


def hold(eng, clock, name, secs, deg, dt=1 / 120, t0=0.0, phase="play"):
    """Run a pattern with the wheel held at `deg`; returns the last report."""
    out = {}
    for i in range(round(secs / dt)):
        clock.t += dt
        out = eng.test_pattern(name, t0 + (i + 1) * dt, dt, deg, phase)
    return out


@pytest.mark.parametrize("sign", [1, -1])
def test_centre_pattern_pushes_toward_centre(rig, sign):
    eng, be, clock = rig(settings=FfbSettings(ffb_sign=sign), ffb_gain=0.2, warm=False)
    hold(eng, clock, "centre", 1.0, 30.0)
    assert last(be, "pattern")["level"] == pytest.approx(-sign * SLOTS["pattern"].cap * 0.2 * 30 / 45)   # left
    hold(eng, clock, "centre", 1.0, -10.0, t0=1.0)
    assert last(be, "pattern")["level"] == pytest.approx(sign * SLOTS["pattern"].cap * 0.2 * 10 / 45)  # right
    assert eng._out["weight_spring"] == 0.0 and eng._out["weight_damper"] > 0
    assert not updates(be, "pulse") and not updates(be, "kick") and not updates(be, "echo")
    assert_ramped(be, "pattern")


def test_pattern_phase_pauses_like_play(rig):
    eng, be, clock = rig(ffb_gain=0.2, warm=False)
    hold(eng, clock, "centre", 0.5, 30.0)
    assert "pattern" in be.running
    hold(eng, clock, "centre", 0.5, 30.0, t0=0.5, phase="paused")
    assert not be.running and be.gain == 0.0 and not eng.latched
    hold(eng, clock, "centre", 0.5, 30.0, t0=1.0)
    assert "pattern" in be.running


def test_kick_pattern_repeats_right_then_left_every_3_s(rig):
    eng, be, clock = rig(ffb_gain=0.2, warm=False)
    pattern(eng, clock, "kick", 7.0)
    kicks = updates(be, "kick")
    assert [k.params["level"] > 0 for k in kicks] == [True, False, True, False, True]
    assert [round(k.t, 1) for k in kicks] == [0.5, 1.5, 3.5, 4.5, 6.5]
    assert all(k.params["length_ms"] == 150 for k in kicks)
    assert all(abs(k.params["level"]) <= STEP_CAP * 0.2 + 1e-9 for k in kicks)
    for _, total, _ in device_load(be, per_call=True):
        assert total <= 0.2 + 1e-9


def test_echo_pattern_waits_for_a_centred_wheel(rig):
    eng, be, clock = rig(ffb_gain=0.2, warm=False, echo_max_deg_s=180.0)
    out = hold(eng, clock, "echo", 3.0, 20.0)
    assert "CENTRE the wheel" in out["log"] and not updates(be, "echo")
    hold(eng, clock, "echo", 1.2, 0.0, t0=3.0)
    assert updates(be, "echo")


def test_echo_pattern_turns_to_45_degrees_and_back(rig):
    eng, be, clock = rig(ffb_gain=0.2, warm=False, play_range_deg=90.0)
    wheel = {"deg": 0.0}

    def follow(t):              # the wheel follows the spring centre
        wheel["deg"] = eng._centre
        return wheel["deg"]

    centres = []
    for _ in range(round(3.0 * 120)):
        clock.t += 1 / 120
        eng.test_pattern("echo", clock.t, 1 / 120, follow(clock.t), "play")
        centres.append(eng._centre)
    assert max(centres) == pytest.approx(45.0, abs=1.0) and centres[-1] == pytest.approx(0.0, abs=1.0)
    steps = [abs(b - a) for a, b in zip(centres, centres[1:], strict=False)]
    assert max(steps) <= 90.0 / 120 + 1e-6                      # the pattern moves at 90 deg/s
    assert_ramped(be, "echo")


def test_echo_pattern_centre_moves_at_most_90_deg_s(rig):
    eng, be, clock = rig(ffb_gain=0.2, warm=False, echo_max_deg_s=180.0, play_range_deg=90.0)
    centres = []
    for i in range(round(2.0 * 120)):
        t = (i + 1) / 120
        wheel = -40.0 if 1.0 <= t < 1.45 else 0.0      # held back while the target rises, then let go
        clock.t += 1 / 120
        eng.test_pattern("echo", t, 1 / 120, wheel, "play")
        centres.append((t, eng._centre))
    after = [c for t, c in centres if t >= 1.45]
    steps = [abs(b - a) for a, b in zip(after, after[1:], strict=False)]
    assert max(steps) <= PATTERN_ECHO_DEG_S / 120 + 1e-9


def test_echo_pattern_waits_again_after_a_pause_and_nothing_moves_while_waiting(rig):
    eng, be, clock = rig(ffb_gain=0.2, warm=False)
    hold(eng, clock, "echo", 1.5, 0.0)
    assert updates(be, "echo")
    hold(eng, clock, "echo", 0.5, 0.0, t0=1.5, phase="paused")
    n = len(be.calls)
    out = hold(eng, clock, "echo", 2.0, 30.0, t0=2.0)                 # back in play, wheel off centre
    assert "CENTRE the wheel" in out["log"]
    assert [c for c in be.calls[n:] if c.method in ("update", "run")] == []   # no spring, no damper
    hold(eng, clock, "echo", 1.5, 0.0, t0=4.0)
    assert [c for c in be.calls[n:] if c.method == "update" and c.slot == "echo"]


def test_kick_pattern_has_no_spring_or_damper(rig):
    eng, be, clock = rig(ffb_gain=0.2, warm=False)
    pattern(eng, clock, "kick", 3.0, steer=lambda t: 40.0)
    assert {c.slot for c in be.calls if c.method in ("update", "run")} == {"kick"}


def test_test_pattern_needs_the_phase():
    eng = FfbEngine(NullFfb(), Config())
    with pytest.raises(TypeError):
        eng.test_pattern("centre", 0.0, 0.01, 0.0)   # type: ignore[call-arg]
    eng.close()


# --- stability: software springs on a simulated wheel ---

class Wheel:
    """A hands-off wheel: inertia `j` (kg m2), coulomb friction, torque = level x `tmax` (Nm).
    The level is held for the whole frame; callers pass the engine the angle one frame late."""

    def __init__(self, j, tmax, deg=0.0, friction=0.1):
        self.j, self.tmax, self.friction = j, tmax, friction
        self.deg, self.speed = deg, 0.0

    def run(self, level, dt, sub=20):
        for _ in range(sub):
            torque = level * self.tmax - (math.copysign(self.friction, self.speed) if self.speed else 0.0)
            self.speed += math.degrees(torque / self.j) * dt / sub
            self.deg += self.speed * dt / sub


def device_level(be):
    return sum(be.params[s]["level"] for s in be.running if SLOTS[s].type == "constant")


def settles(trace, lo, hi):
    """Peaks of |angle| in [lo, hi) never grow."""
    xs = [d for t, d in trace if lo <= t < hi]
    peaks = [abs(xs[i]) for i in range(1, len(xs) - 1)
             if abs(xs[i]) >= abs(xs[i - 1]) and abs(xs[i]) > abs(xs[i + 1]) and abs(xs[i]) > 2.0]
    return all(b <= a + 0.5 for a, b in zip(peaks, peaks[1:], strict=False))


WHEELS = [(0.05, 25.0), (0.04, 25.0), (0.08, 20.0)]   # (inertia kg m2, full torque Nm); nominal first


@pytest.mark.parametrize("j,tmax", WHEELS)
@pytest.mark.parametrize("hz", [60, 144])
@pytest.mark.parametrize("gain", [1.0, 0.5, 0.2])
def test_echo_pattern_is_stable_on_a_simulated_wheel(j, tmax, hz, gain):
    clock = Clock()
    be = NullFfb(clock=clock)
    eng = FfbEngine(be, Config(ffb_gain=gain, play_range_deg=90.0), clock=clock)
    wheel, dt, trace = Wheel(j, tmax), 1 / hz, []
    seen = wheel.deg
    for i in range(round(6.0 * hz)):
        clock.t += dt
        eng.test_pattern("echo", (i + 1) * dt, dt, seen, "play")
        seen = wheel.deg                              # one frame of latency: next update sees this angle
        wheel.run(device_level(be), dt)
        trace.append(((i + 1) * dt, wheel.deg))
    eng.close()
    top = max(d for _, d in trace)
    assert top - 45.0 < 10.0 and -min(d for _, d in trace) < 10.0      # overshoot < 10 deg
    for k in range(2):                                                  # settle windows of each cycle
        assert settles(trace, 3 * k + 2 + dt, 3 * k + 3 + dt), k
    rf = min(1.0, hz / RATE_FULL_HZ)                                    # rate floor softening
    full = SLOTS["echo"].cap * gain * rf * rf * tmax                    # Nm at 45 deg of error
    dead = 0.1 / full * ECHO_FULL_DEG                                   # friction holds this far off
    assert max(abs(d) for t, d in trace if 5.5 < t) < 10.0 + dead


@pytest.mark.parametrize("j,tmax", WHEELS)
@pytest.mark.parametrize("hz", [60, 144])
@pytest.mark.parametrize("gain", [1.0, 0.5, 0.2])
def test_weight_spring_step_is_stable_on_a_simulated_wheel(j, tmax, hz, gain):
    clock = Clock()
    be = NullFfb(clock=clock)
    eng = FfbEngine(be, Config(ffb_gain=gain), clock=clock)
    wheel, dt, trace = Wheel(j, tmax, deg=60.0), 1 / hz, []
    seen = wheel.deg
    for i in range(round(4.0 * hz)):
        clock.t += dt
        eng.update([], snap((i + 1) * dt, weight=1.0, steer_deg=seen), dt)
        seen = wheel.deg                              # one frame of latency
        wheel.run(device_level(be), dt)
        trace.append(((i + 1) * dt, wheel.deg))
    eng.close()
    assert -min(d for _, d in trace) < 10.0                             # overshoot past centre < 10 deg
    assert settles(trace, 0.0, 4.0)
    assert abs(trace[-1][1]) < 60.0                                      # it did pull toward centre


class DeviceForce:
    """Follows a NullFfb's calls incrementally and gives the exact average force the device plays
    over [t1, t2]: constant slots at their level, sine slots integrated from their phase, each run
    clipped to its own start and length (holding a sample past an effect's end adds a false push)."""

    def __init__(self, be):
        self.be, self.seen, self.params, self.runs = be, 0, {}, {}

    def _catch_up(self):
        calls = self.be.calls
        for c in calls[self.seen:]:
            if c.method == "update":
                self.params[c.slot] = c.params
            elif c.method == "run":
                self.runs[c.slot] = (c.t, dict(self.params.get(c.slot, {})))
            elif c.method == "stop":
                if c.slot in self.runs:                   # stopped now: cut the run here
                    t0, p = self.runs[c.slot]
                    p = dict(p, length_ms=max(0.0, (c.t - t0) * 1000))
                    self.runs[c.slot] = (t0, p)
            elif c.method in ("stop_all", "close"):
                self.runs = {k: (t0, dict(p, length_ms=max(0.0, (c.t - t0) * 1000)))
                             for k, (t0, p) in self.runs.items()}
        self.seen = len(calls)

    def average(self, t1, t2, types=("constant", "sine")):
        self._catch_up()
        total = 0.0
        for slot, (t0, p) in self.runs.items():
            if SLOTS[slot].type not in types:
                continue
            a, b = max(t1, t0), min(t2, t0 + p.get("length_ms", 0) / 1000)
            if b <= a:
                continue
            if SLOTS[slot].type == "constant":
                total += p["level"] * (b - a)
            elif SLOTS[slot].type == "sine":
                w = 2 * math.pi * 1000 / p["period_ms"]
                ph = math.radians(p.get("phase_deg", 0.0))
                total += p["magnitude"] * (math.cos(w * (a - t0) + ph) - math.cos(w * (b - t0) + ph)) / w
        return total / (t2 - t1)


def device_level_at(be, t):
    """Force the device plays at time t: last run of each slot for its length; constant slots at
    their level, sine slots (riser, pulse, rumble) as a sine from their phase."""
    runs, params = {}, {}
    for c in be.calls:
        if c.method == "update":
            params[c.slot] = c.params
        elif c.method == "run":
            runs[c.slot] = (c.t, dict(params.get(c.slot, {})))
        elif c.method == "stop":
            runs.pop(c.slot, None)
        elif c.method in ("stop_all", "close"):
            runs.clear()
    total = 0.0
    for slot, (t0, p) in runs.items():
        if not t0 <= t < t0 + p.get("length_ms", 0) / 1000:
            continue
        if SLOTS[slot].type == "constant":
            total += p["level"]
        elif SLOTS[slot].type == "sine":
            phase = math.radians(p.get("phase_deg", 0.0))
            total += p["magnitude"] * math.sin(2 * math.pi * (t - t0) * 1000 / p["period_ms"] + phase)
    return total


STALL_START = {"echo": 0.0, "weight": -90.0, "riser": 0.0, "centre": 45.0,   # where the driver holds the wheel
               "echo+riser": 0.0, "weight+riser": -90.0,
               "riser-held": 0.0, "echo>riser": 0.0, "weight>riser": -90.0}
BUZZ_BEFORE = ("riser", "echo+riser", "weight+riser", "riser-held")   # riser at peak buzz when the stall hits
BUZZ_AFTER = ("riser-held", "echo>riser", "weight>riser")             # the buzz plays after the stall


def stall_frame(eng, kind, t, dt, deg, before):
    """One frame of a stall scenario. `before`: the driver still holds the wheel (and the handbrake)."""
    if kind == "centre":
        return eng.test_pattern("centre", t, dt, deg, "play")
    if kind == "echo":
        return eng.update([], snap(t, echo="listen", echo_target=0.5, steer_deg=deg), dt)
    if kind == "weight":
        return eng.update([], snap(t, weight=1.0, steer_deg=deg), dt)
    riser = RiserStatus(active=True, progress=1.0, held=before)     # the handbrake is let go with the wheel
    if kind in BUZZ_AFTER:                            # held through the stall / starts right after it
        riser = RiserStatus(active=True, progress=1.0, held=kind == "riser-held" or not before)
        if kind == "echo>riser":
            return eng.update([], snap(t, riser=riser, echo="listen", echo_target=0.5, steer_deg=deg), dt)
        if kind == "weight>riser":
            return eng.update([], snap(t, riser=riser, weight=1.0, steer_deg=deg), dt)
        return eng.update([], snap(t, riser=riser, steer_deg=deg), dt)
    if kind == "echo+riser":
        return eng.update([], snap(t, riser=riser, echo="listen", echo_target=0.5, steer_deg=deg), dt)
    if kind == "weight+riser":
        return eng.update([], snap(t, riser=riser, weight=1.0, steer_deg=deg), dt)
    return eng.update([], snap(t, riser=riser, steer_deg=deg), dt)


def stall_run(stall, damping=0.0, hz=144, j=0.05, tmax=25.0, hz_after=None, second=None, kind="echo"):
    """The driver holds the wheel against a sustained force at its maximum (`kind`: echo in listen,
    weight 1.0 with the wheel 90 deg off, riser at peak wind-up, centre pattern at 30 deg) and lets
    go as the game stalls for `stall` s; then the game runs 3 s at `hz_after` (default `hz`).
    `second` = (after_s, stall_s): a second stall that many seconds into the post-stall motion.
    Closed loop through the engine. Returns peak |speed| deg/s, travel from the stall start, final
    speed, push past the device length while stalled, and the largest speed against the first
    direction of travel (a reversal)."""
    clock = Clock()
    be = NullFfb(clock=clock, record_all=True)
    eng = FfbEngine(be, Config(ffb_gain=1.0, play_range_deg=90.0), FfbSettings(strength={"riser": 1.0}), clock=clock)
    dt, start = 1 / hz, STALL_START[kind]
    for _ in range(hz):
        clock.t += dt
        stall_frame(eng, kind, clock.t, dt, start, before=True)
    if kind in BUZZ_BEFORE:                           # whole cycles: the stall may hit any point of one
        for _ in range(hz):
            clock.t += dt
            stall_frame(eng, kind, clock.t, dt, start, before=True)
    else:                                             # full level (the rate floor softens or stops it)
        rf = min(1.0, hz / RATE_FULL_HZ)
        full = SLOTS["echo"].cap * rf * rf if hz >= RATE_OFF_HZ else 0.0
        assert abs(device_level_at(be, clock.t)) == pytest.approx(full)
    st = {"deg": start, "speed": 0.0, "peak": 0.0, "late": 0.0, "dir": 0.0, "rev": 0.0}
    trace = []                                        # (time, angle) after every physics step
    force = DeviceForce(be)

    def physics(t1, h, since=None):                   # exact average device force over [t1, t1 + h)
        level = force.average(t1, t1 + h)
        if since is not None and t1 >= since + SPRING_DEADMAN_MS / 1000 - 1e-9:
            st["late"] = max(st["late"], abs(force.average(t1, t1 + h, types=("constant",))))
        v = st["speed"]
        friction = math.copysign(0.1, v) if v else 0.0
        v += math.degrees((level * tmax - damping * math.radians(v) - friction) / j) * h
        st["deg"] += v * h
        st["speed"] = v
        st["peak"] = max(st["peak"], abs(v))
        if not st["dir"] and abs(v) > 1.0:
            st["dir"] = math.copysign(1.0, v)
        st["rev"] = max(st["rev"], -st["dir"] * v)
        trace.append((t1 + h, st["deg"]))

    def stalled(secs):                                # no updates: the device plays what it has
        since = clock.t
        while clock.t < since + secs - 1e-12:
            physics(clock.t, 0.001, since)
            clock.t += 0.001

    stalled(stall)
    hz2 = hz_after or hz
    dt2, seen, t_run = 1 / hz2, st["deg"], 0.0
    for _ in range((5 if kind in BUZZ_AFTER else 3) * hz2):   # 5 s: lets a gentle return finish
        if second and t_run <= second[0] < t_run + dt2:
            stalled(second[1])
        stall_frame(eng, kind, clock.t, dt2, seen, before=False)
        seen = st["deg"]                              # the next update sees this angle
        for k in range(10):                           # until the next update, dt2 later
            physics(clock.t + k * dt2 / 10, dt2 / 10)
        clock.t += dt2
        t_run += dt2
    eng.close()
    if kind in BUZZ_AFTER or kind in BUZZ_BEFORE:
        # the buzz moves the wheel back and forth by design: judge the drift. A cosine cycle returns
        # the wheel exactly to where it started, so sample the angle at each cycle start and whenever
        # no cycle plays; reversal and final speed come from those samples (the raw peak still counts)
        length = ffb.RISER_CYCLE_MS / 1000
        starts = [c.t for c in be.calls if c.method == "run" and c.slot == "riser"]
        times = [t for t, _ in trace]

        def playing(t):
            i = bisect.bisect_right(starts, t) - 1
            return i >= 0 and t < starts[i] + length

        samples = [(t, d) for t, d in trace if not playing(t)]
        for r in starts:
            if times[0] <= r <= times[-1]:
                samples.append(trace[min(bisect.bisect_left(times, r), len(trace) - 1)])
        spaced = []
        for t, d in sorted(samples):                  # at least one cycle apart: speeds over whole cycles
            if not spaced or t - spaced[-1][0] >= length:
                spaced.append((t, d))
        avg = [(d1 - d0) / (t1 - t0) for (t0, d0), (t1, d1) in zip(spaced, spaced[1:], strict=False)]
        first = next((v for v in avg if abs(v) > 1.0), 0.0)
        st["rev"] = max([0.0] + [-math.copysign(1.0, first) * v for v in avg]) if first else 0.0
        st["speed"] = avg[-1] if avg else 0.0
    return st["peak"], st["deg"] - start, st["speed"], st["late"], st["rev"]


def assert_stall_bounded(res):
    peak, travel, final, late, rev = res
    assert late <= 1e-9                               # no push past the device length
    assert peak < 360.0, res
    assert abs(travel) <= 360.0 and abs(final) < BRAKE_STOP_DEG_S + 1.0, res
    assert rev <= 60.0, res                           # no reversal faster than 60 deg/s


@pytest.mark.parametrize("kind", ["echo", "weight", "riser", "centre", "echo+riser", "weight+riser",
                                  "riser-held", "echo>riser", "weight>riser"])
@pytest.mark.parametrize("j,tmax", WHEELS)
@pytest.mark.parametrize("hz", [30, 60, 144])
@pytest.mark.parametrize("stall", [0.05, 0.1, 0.2])
def test_stall_is_bounded_without_base_damping(stall, hz, j, tmax, kind):
    """SPEC 10: no base damping at all, every sustained force at its maximum. Peak speed < 360 deg/s,
    rest within one turn of where the stall began, no reversal faster than 60 deg/s."""
    assert_stall_bounded(stall_run(stall, hz=hz, j=j, tmax=tmax, kind=kind))


@pytest.mark.parametrize("kind", ["echo", "weight", "riser", "centre", "echo+riser", "weight+riser",
                                  "riser-held", "echo>riser", "weight>riser"])
@pytest.mark.parametrize("j,tmax", WHEELS)
def test_stall_at_144_hz_then_60_hz_frames(j, tmax, kind):
    assert_stall_bounded(stall_run(0.2, hz=144, hz_after=60, j=j, tmax=tmax, kind=kind))


@pytest.mark.parametrize("kind", ["echo", "weight", "riser", "centre", "echo+riser", "weight+riser",
                                  "riser-held", "echo>riser", "weight>riser"])
@pytest.mark.parametrize("j,tmax", WHEELS)
@pytest.mark.parametrize("hz", [30, 60, 144])
def test_second_stall_during_the_post_stall_motion(hz, j, tmax, kind):
    assert_stall_bounded(stall_run(0.1, hz=hz, j=j, tmax=tmax, second=(0.03, 0.1), kind=kind))


@pytest.mark.parametrize("stall", [0.05, 0.1, 0.2])
def test_stall_with_base_damping_stays_within_45_deg_past_the_target(stall):
    peak, travel, final, late, rev = stall_run(stall, damping=0.5)     # for information: a damped base
    assert peak < 360.0 and abs(travel) <= 90.0, (peak, travel)


def test_stall_brake_opposes_motion_and_disarms(rig):
    eng, be, clock = rig(ffb_gain=0.5)
    run(eng, clock, 0.5, weight=0.0, steer_deg=0.0)
    clock.t += 0.2                                    # a stall: rate floor off, brake armed
    deg = 0.0
    levels = []
    for _ in range(30):                               # the wheel keeps turning right at 240 deg/s
        deg += 2.0
        step(eng, clock, 1 / 120, weight=0.0, steer_deg=deg)
        levels.append(eng._out["brake"])
    assert eng._brake_armed and min(levels) < -0.1
    assert all(v <= 0.0 for v in levels)              # only ever against the motion (to the left)
    assert all(abs(v) <= 0.3 * 0.5 + 1e-9 for v in levels)
    for _ in range(10):                               # the wheel stops: disarmed once the median is slow
        step(eng, clock, 1 / 120, weight=0.0, steer_deg=deg)
    assert not eng._brake_armed
    run(eng, clock, 0.2, weight=0.0, steer_deg=deg)
    assert eng._out["brake"] == 0.0


@pytest.mark.parametrize("j,tmax", WHEELS)
@pytest.mark.parametrize("hz", [30, 60, 144, 240])
def test_brake_loop_gain_per_frame_is_under_half_the_limit(hz, j, tmax):
    dt = 1 / hz
    k = ffb.K_BRAKE * min(1.0, ffb.BRAKE_FRAME_S / dt)
    gain = k * math.degrees(1.0) * tmax / j * dt       # level per deg/s -> Nm per rad/s, x dt / J
    assert gain <= 0.52 / 2, gain


def test_brake_disarms_only_when_all_three_samples_are_slow(rig):
    eng, be, clock = rig()
    run(eng, clock, 0.5, steer_deg=0.0)
    clock.t += 0.2
    deg = 0.0
    step(eng, clock, 1 / 120, steer_deg=deg)           # first frame after the gap: no sample yet
    for d in (2.0, 0.1, 0.1):                          # 240 deg/s then two slow samples: median 12
        deg += d
        step(eng, clock, 1 / 120, steer_deg=deg)
    assert len(eng._vels) == 3 and eng._brake_armed    # one fast sample left: still armed
    step(eng, clock, 1 / 120, steer_deg=deg + 0.1)
    assert not eng._brake_armed


def test_arming_the_brake_clears_the_speed_history(rig):
    eng, be, clock = rig()
    for i in range(10):
        step(eng, clock, 1 / 120, weight=1.0, steer_deg=-90.0 + i)
    assert len(eng._vels) == 3 and "weight_spring" in eng._running
    step(eng, clock, 0.03, weight=1.0, steer_deg=-80.0)  # 30 ms frame: spring ended, brake armed, no gap
    assert eng._brake_armed and len(eng._vels) <= 1      # no pre-stall samples left


def test_two_samples_use_the_smaller_magnitude(rig):
    eng, be, clock = rig()
    step(eng, clock, 0.01, steer_deg=float("nan"))     # history reset
    for deg in (0.0, 3.6, 3.2):                        # samples +360 then -40 deg/s
        step(eng, clock, 0.01, steer_deg=deg)
    assert len(eng._vels) == 2 and eng._vel == pytest.approx(-40.0)


def test_brake_is_zero_when_newest_and_median_disagree(rig):
    eng, be, clock = rig()
    run(eng, clock, 0.5, steer_deg=0.0)
    clock.t += 0.2
    deg = 0.0
    for d in (2.0, 2.0, 2.0, -0.5):                    # fast right, then the newest sample turns left
        deg += d
        step(eng, clock, 1 / 120, steer_deg=deg)
    assert eng._vels[-1] < 0 < eng._vel and eng._pre["brake"] * eng._out["brake"] >= 0
    assert eng._out["brake"] <= 0.0 and abs(eng._out["brake"]) < abs(ffb.K_BRAKE * eng._vel)


def test_brake_stays_armed_when_the_spring_cannot_output(rig):
    eng, be, clock = rig(settings=FfbSettings(strength={"echo": 0.0}), play_range_deg=90.0)
    run(eng, clock, 0.5, steer_deg=0.0, echo="listen", echo_target=0.5)
    clock.t += 0.2
    deg = 0.0
    for _ in range(round(0.8 * 120)):                  # springs allowed again after 0.5 s, echo strength 0
        deg += 1.0
        step(eng, clock, 1 / 120, steer_deg=deg, echo="listen", echo_target=0.5)
    assert eng._springs_ok and eng._brake_armed


def test_stall_brake_stops_when_springs_are_back(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    run(eng, clock, 0.5, steer_deg=0.0)
    clock.t += 0.2
    deg = 0.0
    for _ in range(round(0.8 * 120)):                 # slow drift above 30 deg/s for 0.8 s
        deg += 0.5
        step(eng, clock, 1 / 120, steer_deg=deg)
    assert eng._springs_ok and not eng._brake_armed    # springs back after the 0.5 s window


def test_every_sustained_force_is_short_and_capped():
    assert set(ffb.SUSTAINED) == {"echo", "weight_spring", "pattern", "brake"}
    for slot in ffb.SUSTAINED:
        spec = SLOTS[slot]
        assert spec.deadman_ms == SPRING_DEADMAN_MS == 32 and spec.cap == ffb.SUSTAINED_CAP == 0.3
        assert spec.refresh_s == 0.0                                   # re-run every frame
        bound = math.degrees(spec.cap * 25.0 * spec.deadman_s / 0.04)  # cap x Tmax x length / J_min
        assert bound < 360.0, bound
    riser = SLOTS["riser"]                            # whole sine cycles: no net push when held
    assert riser.type == "sine" and riser.length_ms == ffb.RISER_CYCLE_MS and riser.cap == 0.3


def test_stall_never_cuts_a_riser_cycle(rig):
    eng, be, clock = rig()
    held = RiserStatus(active=True, progress=1.0, held=True)
    run(eng, clock, 1.0, riser=held)
    clock.t += 0.2                                    # a stall: the device plays the cycle whole
    run(eng, clock, 0.5, riser=held)
    assert not [c for c in be.calls if c.slot == "riser" and c.method == "stop"]
    runs = [c.t for c in be.calls if c.method == "run" and c.slot == "riser"]
    assert all(b - a >= ffb.RISER_CYCLE_MS / 1000 - 1e-9 for a, b in zip(runs, runs[1:], strict=False))


def _echo_return_after(eng, clock, trigger, frames):
    listen = dict(echo="listen", echo_target=0.5)
    run(eng, clock, 1.5, steer_deg=45.0, **listen)                     # tracking: centre at the goal
    assert eng._centre == pytest.approx(45.0)
    for _ in range(frames):
        trigger()
    centres = []
    for _ in range(round(0.8 * 144)):
        step(eng, clock, 1 / 144, steer_deg=0.0, **listen)
        centres.append(eng._centre)
    return centres


@pytest.mark.parametrize("trigger", ["unwind", "nan", "half_turn", "rate_floor"])
def test_echo_returns_gently_after_the_springs_drop_mid_phrase(rig, trigger):
    eng, be, clock = rig(play_range_deg=90.0)
    listen = dict(echo="listen", echo_target=0.5)
    fire = {
        "unwind": lambda: step(eng, clock, 1 / 144, steer_deg=0.0, steer_unwind=True, **listen),
        "nan": lambda: step(eng, clock, 1 / 144, steer_deg=float("nan"), **listen),
        "half_turn": lambda: step(eng, clock, 1 / 144, steer_deg=200.0, **listen),
        "rate_floor": lambda: step(eng, clock, 1 / 40, steer_deg=0.0, **listen),
    }[trigger]
    centres = _echo_return_after(eng, clock, fire, 12)
    moving = [b - a for a, b in zip(centres, centres[1:], strict=False) if b != a]
    assert moving and max(moving) <= ffb.ECHO_RECOVER_DEG_S / 144 + 1e-9   # at most 30 deg/s
    assert centres[0] == pytest.approx(0.0, abs=1.0)                        # restarted at the wheel


def test_sustained_sum_is_capped(rig):
    eng, be, clock = rig(ffb_gain=1.0, play_range_deg=90.0)
    run(eng, clock, 2.0, steer_deg=0.0, echo="listen", echo_target=0.5,
        riser=RiserStatus(active=True, progress=1.0, held=True))
    sums = [sum(abs(v) for s, v in replay_at(be, i).items() if s in (*ffb.SUSTAINED, "riser"))
            for i, c in enumerate(be.calls) if c.method in ("update", "run", "stop")]
    assert max(sums) <= ffb.SUSTAINED_CAP + 1e-9
    assert max(sums) > 0.29


def replay_at(be, i):
    """Magnitudes the device plays right after call i, each run lasting its length."""
    level, runs = {}, {}
    now = be.calls[i].t
    for c in be.calls[: i + 1]:
        if c.method == "update":
            level[c.slot] = (magnitude(c.params), c.params.get("length_ms", DEADMAN_MS))
        elif c.method == "run":
            runs[c.slot] = (c.t, level.get(c.slot, (0.0, 0))[1])
        elif c.method == "stop":
            runs.pop(c.slot, None)
        elif c.method in ("stop_all", "close"):
            runs.clear()
    return {s: level[s][0] for s, (t0, n) in runs.items() if now < t0 + n / 1000}


@pytest.mark.parametrize("fraction", [0.5, 0.1])
def test_buzz_follows_wall_clock_not_the_callers_dt(rig, fraction):
    eng, be, clock = rig(ffb_gain=1.0)
    for _ in range(144):                              # 1 s of wall time, caller claims less
        clock.t += 1 / 144
        eng.update([], snap(clock.t, riser=RiserStatus(active=True, progress=1.0, held=True)),
                   fraction / 144)
    runs = [c for c in be.calls if c.method == "run" and c.slot == "riser"]
    assert 10 <= len(runs) <= 12


def test_riser_start_does_not_disarm_the_brake(rig):
    eng, be, clock = rig()
    run(eng, clock, 0.5, steer_deg=0.0)
    clock.t += 0.2                                    # a stall arms the brake
    step(eng, clock, 1 / 120, steer_deg=0.0, riser=RiserStatus(active=True, progress=0.1, held=True))
    assert eng._brake_armed                           # an (auto) riser starting in the hitch leaves it on
    deg = 0.0
    for _ in range(12):                               # the wheel coasts: the brake acts
        deg += 2.0
        step(eng, clock, 1 / 120, steer_deg=deg, riser=RiserStatus(active=True, progress=0.2, held=True))
    assert eng._brake_armed and eng._out["brake"] < 0


@pytest.mark.parametrize("bad_dt", [0.0, float("nan")])
def test_a_bad_caller_dt_cannot_freeze_a_level(rig, bad_dt):
    eng, be, clock = rig(ffb_gain=1.0)
    run(eng, clock, 1.5, weight=1.0, steer_deg=-90.0, riser=RiserStatus(active=True, progress=1.0, held=True))
    assert eng._out["weight_spring"] > 0.1 and eng._wall < eng._riser_end
    for _ in range(round(0.5 * 144)):                 # wall time moves, the caller passes dt 0 / NaN
        clock.t += 1 / 144
        eng.update([], snap(clock.t, weight=1.0, steer_deg=-90.0, echo="listen", echo_target=0.0,
                            riser=RiserStatus(active=True, progress=1.0, held=False)), bad_dt)
    assert eng._wall >= eng._riser_end and eng._wind == 0.0             # released: the buzz is gone
    assert eng._out["weight_spring"] == 0.0


def test_a_stuck_sustained_slot_counts_in_the_sustained_sum(rig):
    class Stuck(NullFfb):
        broken = False

        def update(self, slot, **params):
            if self.broken and slot == "echo":
                raise OSError("device refused")
            super().update(slot, **params)

        def stop(self, slot):
            if self.broken and slot == "echo":
                raise OSError("device refused")
            super().stop(slot)

    eng, be, clock = rig(backend=Stuck(), ffb_gain=1.0, play_range_deg=90.0)
    listen = dict(steer_deg=0.0, echo="listen", echo_target=0.5)
    run(eng, clock, 1.0, **listen)                    # echo at its cap, 0.3
    be.broken = True
    held = RiserStatus(active=True, progress=1.0, held=True)
    for _ in range(3):                                # inside the ghost's 32 ms: the riser gets no room
        step(eng, clock, 1 / 144, riser=held, **listen)
        ghost = eng._ghost_load(ffb.SUSTAINED)
        assert ghost == pytest.approx(0.3)
        assert sum(abs(eng._pre[s]) for s in ffb.SUSTAINED) + ghost <= ffb.SUSTAINED_CAP + 1e-12


def _flip(eng, clock, caller_dt, frame, before, after):
    """Hold `before` 1 s at normal dt, then `after` with a tiny caller dt; returns the levels."""
    for _ in range(144):
        clock.t += 1 / 144
        frame(eng, clock.t, 1 / 144, **before)
    levels = []
    for _ in range(round(0.1 * 144)):
        clock.t += 1 / 144
        frame(eng, clock.t, caller_dt(1 / 144), **after)
        levels.append(eng._pre)
    return levels


def _update(eng, t, dt, **kw):
    eng.update([], snap(t, **kw), dt)


@pytest.mark.parametrize("caller_dt", [lambda wall: 1e-6, lambda wall: wall / 10])
@pytest.mark.parametrize("slot", ["echo", "weight_spring", "pattern", "brake"])
def test_a_level_crosses_zero_even_with_a_tiny_caller_dt(rig, slot, caller_dt):
    eng, be, clock = rig(ffb_gain=1.0, play_range_deg=90.0)
    if slot == "echo":            # centre 45 deg right of the wheel, then the wheel 45 deg past it
        _flip(eng, clock, caller_dt, _update, dict(steer_deg=0.0, echo="listen", echo_target=0.5),
              dict(steer_deg=90.0, echo="listen", echo_target=0.5))
    elif slot == "weight_spring":
        _flip(eng, clock, caller_dt, _update, dict(steer_deg=-90.0, weight=1.0), dict(steer_deg=90.0, weight=1.0))
    elif slot == "pattern":
        def centre(eng, t, dt, steer_deg):
            eng.test_pattern("centre", t, dt, steer_deg, "play")
        _flip(eng, clock, caller_dt, centre, dict(steer_deg=-45.0), dict(steer_deg=45.0))
    else:
        run(eng, clock, 0.5, steer_deg=0.0)
        clock.t += 0.2                                 # a stall arms the brake
        deg = 0.0
        for _ in range(30):                            # turning right: brake pushes left
            deg += 2.0
            step(eng, clock, 1 / 144, steer_deg=deg)
        assert eng._pre["brake"] < 0
        for _ in range(round(0.1 * 144)):              # now turning left fast: brake must flip
            deg -= 2.0
            clock.t += 1 / 144
            eng.update([], snap(clock.t, steer_deg=deg), caller_dt(1 / 144))
        assert eng._pre["brake"] > 0, eng._pre["brake"]
        return
    assert eng._pre[slot] < 0, (slot, eng._pre[slot])   # crossed zero, not frozen on the old side


def test_riser_pattern_winds_up_and_releases_every_4_s(rig):
    eng, be, clock = rig(ffb_gain=0.2, warm=False)
    for i in range(round(8.0 * 144)):
        clock.t += 1 / 144
        eng.test_pattern("riser", (i + 1) / 144, 1 / 144, 0.0, "play")
    runs = [c.t % 4.0 for c in be.calls if c.method == "run" and c.slot == "riser"]
    assert runs and max(runs) < 3.3 and min(runs) > 0.0                   # cycles in the first 3 s of 4
    mags = [c.params["magnitude"] for c in be.calls if c.method == "update" and c.slot == "riser"]
    assert max(mags) <= SLOTS["riser"].cap * 0.2 + 1e-9
    assert {c.slot for c in be.calls if c.method == "update"} == {"riser"}


def riser_frames(kind, hz, n):
    """Frame times: 'equal' at hz; 'locked': a long frame every 12th slot, locked to the buzz
    (11 x 6 ms + 1 x 17 ms = one 83 ms cycle); 'random': 7 to 19 ms, seeded."""
    if kind == "equal":
        return [1 / hz] * n
    if kind == "locked":
        return [0.017 if i % 12 == 11 else 0.006 for i in range(n)]
    rnd = random.Random(hz)
    return [rnd.uniform(0.007, 0.019) for _ in range(n)]


def riser_run(hz, j, tmax, secs=3.0, phase=0.0, frames="equal", pulse=False, riser=True, friction=0.1):
    """A hands-off free wheel through a full riser: wind-up over `secs` (plus `phase` of one buzz
    cycle, to release at any point), release, 1 s after. No weight spring (the strict case).
    `pulse`: a strong beat every 0.25 s and a heavy damper, so the budget scales the riser.
    Returns (max |displacement| deg, max |speed| deg/s, final displacement deg)."""
    secs += phase / ffb.RISER_HZ
    clock = Clock()
    be = NullFfb(clock=clock, record_all=True)
    strength = {"weight_spring": 0.0, "riser": 1.0 if riser else 0.0}
    eng = FfbEngine(be, Config(ffb_gain=1.0), FfbSettings(strength=strength), clock=clock)
    for _ in range(hz):                               # settle, wheel at rest
        clock.t += 1 / hz
        eng.update([], snap(clock.t, steer_deg=0.0), 1 / hz)
    wheel, seen, far, fast, t = Wheel(j, tmax, friction=friction), 0.0, 0.0, 0.0, 0.0
    beat, force = 0.0, DeviceForce(be)
    for dt in riser_frames(frames, hz, 10 * round((secs + 1.0) * hz)):
        if t >= secs + 1.0:
            break
        for k in range(10):   # the device plays what the last update set, until this frame's update
            wheel.run(force.average(clock.t + k * dt / 10, clock.t + (k + 1) * dt / 10), dt / 10, sub=1)
            far, fast = max(far, abs(wheel.deg)), max(fast, abs(wheel.speed))
        riser = RiserStatus(active=t < secs, progress=min(1.0, t / secs), held=t < secs)
        events = []
        if pulse and t >= beat:
            events, beat = [Beat(clock.t + dt, 1.0)], beat + 0.25
        clock.t += dt
        t += dt
        eng.update(events, snap(clock.t, riser=riser, steer_deg=seen, weight=1.0 if pulse else 0.0), dt)
        seen = wheel.deg                              # one frame of latency
    eng.close()
    return far, fast, wheel.deg


@pytest.mark.parametrize("frames", ["equal", "locked", "random"])
@pytest.mark.parametrize("j,tmax", WHEELS)
@pytest.mark.parametrize("hz", [60, 144])
def test_riser_buzz_barely_moves_a_free_wheel(hz, j, tmax, frames):
    for k in range(8):                                # release at 8 phases across one buzz cycle
        far, fast, _ = riser_run(hz, j, tmax, phase=k / 8, frames=frames)
        assert far <= 10.0 and fast < 360.0, (k, far, fast)


@pytest.mark.parametrize("j,tmax", WHEELS)
def test_riser_buzz_with_budget_scaling_by_beat_pulses(j, tmax):
    # the phase-0 beat pulses push a free wheel by themselves (SPEC decision). Judge what the riser
    # adds, riser on against riser off under the same pulses (pulses fire first and are never scaled
    # by the riser). No friction: the wheel is then linear and the difference is the riser's share.
    for k in range(4):
        far, fast, end = riser_run(144, j, tmax, phase=k / 4, pulse=True, friction=0.0)
        _, _, end_off = riser_run(144, j, tmax, phase=k / 4, pulse=True, riser=False, friction=0.0)
        assert abs(end - end_off) <= 10.0 and fast < 360.0, (k, end, end_off, fast)


# --- rate floor ---

def run_at(eng, clock, hz, secs, **kw):
    out = {}
    for _ in range(round(secs * hz)):
        out = step(eng, clock, 1 / hz, **kw)
    return out


def test_rate_below_100_hz_softens_the_springs(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    out = run_at(eng, clock, 45, 1.0, weight=1.0, steer_deg=-90.0)        # 45 Hz: off
    assert eng._out["weight_spring"] == 0.0 and eng._out["weight_damper"] > 0.4
    assert out["log"].count("SPRINGS off - 45 Hz") == 1
    run_at(eng, clock, 80, 4.0, weight=1.0, steer_deg=-90.0)               # 80 Hz: on (after a 3 s return), x 0.8^2
    assert eng._out["weight_spring"] == pytest.approx(0.3 * 0.8 ** 2, rel=1e-3)
    run_at(eng, clock, 120, 2.0, weight=1.0, steer_deg=-90.0)              # >= 100 Hz: full
    assert eng._out["weight_spring"] == pytest.approx(0.3)


def test_rate_at_30_hz_keeps_echo_off_and_logs_once(rig):
    eng, be, clock = rig(ffb_gain=1.0, play_range_deg=90.0)
    n = len(updates(be, "echo"))
    out = run_at(eng, clock, 30, 2.0, echo="listen", echo_target=0.5, steer_deg=0.0)
    assert len(updates(be, "echo")) == n and eng._out["weight_damper"] > 0
    assert sum(m.startswith("SPRINGS off") for m in out["log"]) == 1


def test_rate_floor_has_hysteresis(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    run_at(eng, clock, 45, 1.0, weight=1.0, steer_deg=-90.0)
    run_at(eng, clock, 52, 2.0, weight=1.0, steer_deg=-90.0)               # between 50 and 55: stays off
    assert eng._out["weight_spring"] == 0.0
    out = run_at(eng, clock, 60, 2.0, weight=1.0, steer_deg=-90.0)
    assert eng._out["weight_spring"] > 0 and any(m.startswith("SPRINGS on") for m in out["log"])


def test_report_shows_frame_rate_and_rate_floor(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    out = run_at(eng, clock, 120, 1.0, weight=1.0)
    assert out["fps"] == pytest.approx(120, abs=1) and out["springs"] is None
    out = run_at(eng, clock, 80, 1.0, weight=1.0)
    assert out["fps"] == pytest.approx(80, abs=1) and out["springs"] == "reduced"
    out = run_at(eng, clock, 45, 1.0, weight=1.0)
    assert out["springs"] == "off" and ffb.rate_text(out) == "FPS 45 (worst 45)   SPRINGS off"
    eng.stop_all()                                          # latched: the rate still shows
    out = run_at(eng, clock, 60, 1.0, weight=1.0)
    assert out["fps"] == pytest.approx(60, abs=1)
    assert "LATCHED" in ffb.rate_text(out)
    assert ffb.rate_text({}) == "FPS measuring"


def test_frame_rate_shows_the_worst_frame_and_skips_gaps(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    for _ in range(40):                                     # 144 Hz with every other frame at 72 Hz
        step(eng, clock, 1 / 144, weight=1.0)
        out = step(eng, clock, 1 / 72, weight=1.0)
    assert out["fps_worst"] == pytest.approx(72, abs=0.5) and out["fps"] == pytest.approx(96, abs=1)
    assert ffb.rate_text(out) == f"FPS {out['fps']:.0f} (worst 72)   SPRINGS reduced"
    out = run_at(eng, clock, 144, 1.0)
    eng.stop_all()                                          # a hitch pauses the game
    step(eng, clock, 0.4)
    out = step(eng, clock, 1 / 144)
    assert out["fps_worst"] == pytest.approx(144, abs=0.5) and out["fps"] == pytest.approx(144, abs=0.5)


def test_springs_show_measuring_before_any_frame_time(rig):
    eng, be, clock = rig(warm=False)
    assert eng.report()["springs"] is None and eng.report()["fps"] is None
    out = eng.update([], snap(0.0), 0.0)                     # first play frame: no frame time yet
    assert out["springs"] == "measuring" and ffb.rate_text(out) == "FPS measuring   SPRINGS measuring"


def test_strength_lookup_falls_back_to_the_default_strength(rig):
    eng, be, clock = rig()
    eng.settings.strength = {}
    assert eng._strength("riser") == ffb.DEFAULT_STRENGTH["riser"] == 0.0
    assert eng._cap("riser", 0.5) == 0.0 and eng._cap("kick", 0.1) == pytest.approx(0.1)


# --- echo centre re-sync after a restart ---

def test_echo_centre_resyncs_after_a_gap(rig):
    eng, be, clock = rig(ffb_gain=1.0, play_range_deg=90.0)
    run(eng, clock, 1.0, steer_deg=0.0, echo="listen", echo_target=0.5)
    assert eng._centre == pytest.approx(45.0)
    clock.t += 0.3
    step(eng, clock, 1 / 120, steer_deg=-20.0, echo="listen", echo_target=0.5)
    assert eng._centre == pytest.approx(-20.0)


def test_echo_centre_resyncs_after_a_dead_man_end(rig):
    eng, be, clock = rig(ffb_gain=1.0, play_range_deg=90.0)
    run(eng, clock, 1.0, steer_deg=0.0, echo="listen", echo_target=0.5)
    clock.t += 0.04                                   # past the spring's end threshold, below the gap log
    step(eng, clock, 1 / 120, steer_deg=-20.0, echo="listen", echo_target=0.5)
    assert [c for c in be.calls if c.method == "stop" and c.slot == "echo"]
    assert eng._centre == pytest.approx(-20.0)


def test_echo_centre_resyncs_after_a_failed_call(rig):
    class Flaky(NullFfb):
        fail = False

        def update(self, slot, **params):
            if self.fail and slot == "echo":
                self.fail = False
                raise OSError("device busy")
            super().update(slot, **params)

    eng, be, clock = rig(backend=Flaky(), ffb_gain=1.0, play_range_deg=90.0)
    run(eng, clock, 1.0, steer_deg=0.0, echo="listen", echo_target=0.5)
    be.fail = True
    step(eng, clock, 1 / 120, steer_deg=0.0, echo="listen", echo_target=0.5)
    run(eng, clock, 1.2, steer_deg=-20.0, echo="listen", echo_target=0.5)   # retry after 1 s
    assert abs(eng._centre - -20.0) <= ECHO_LEAD_DEG + 1e-9 and eng.enabled["echo"]
    assert "echo" in be.running


# --- velocity term ---

def test_velocity_term_is_clamped_to_its_spring(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    deg = 0.0
    for _ in range(30):                               # 600 deg/s at weight 0: spring full level 0.1
        deg -= 5.0
        step(eng, clock, 1 / 120, weight=0.0, steer_deg=deg)
    assert -150.0 <= deg and eng._pre["weight_spring"] == pytest.approx(0.12)  # spring 0.06 + damping 0.06


def test_wheel_speed_is_clamped_median_filtered_and_reset(rig):
    eng, be, clock = rig()
    for d in (0.0, 1.0, 2.0, 3.0):
        step(eng, clock, 0.01, steer_deg=d)
    assert eng._vel == pytest.approx(100.0)
    step(eng, clock, 0.01, steer_deg=50.0)            # one spike: the median ignores it
    assert eng._vel == pytest.approx(100.0)
    for d in (100.0, 200.0, 300.0):
        step(eng, clock, 0.01, steer_deg=d)
    assert eng._vel == pytest.approx(VEL_MAX)         # 10000 deg/s clamped
    step(eng, clock, 0.01, steer_deg=float("nan"))
    step(eng, clock, 0.01, steer_deg=0.0)             # history reset: no speed from 300 -> 0
    assert eng._vel == 0.0


def test_spring_slots_rerun_every_frame_with_a_short_length(rig):
    eng, be, clock = rig(play_range_deg=90.0)
    n = len(be.calls)
    run(eng, clock, 0.5, weight=1.0, steer_deg=-60.0)
    ups = [c for c in be.calls[n:] if c.method == "update" and c.slot == "weight_spring"]
    runs = [c for c in be.calls[n:] if c.method == "run" and c.slot == "weight_spring"]
    assert len(ups) == len(runs) == 60 and all(c.params["length_ms"] == SPRING_DEADMAN_MS for c in ups)


def test_torque_estimate_is_the_constant_levels_sent(rig):
    eng, be, clock = rig(ffb_gain=1.0, play_range_deg=90.0)
    run(eng, clock, 1.0, steer_deg=0.0, echo="listen", echo_target=0.5)
    sent = sum(c.params["level"] for s in ("echo", "riser", "weight_spring") if s in be.running
               for c in updates(be, s)[-1:])
    assert eng.report()["torque"] == pytest.approx(sent, abs=1e-4)
    assert eng.report()["torque"] == pytest.approx(SLOTS["echo"].cap, abs=1e-4)


def test_failed_stop_of_an_ended_slot_keeps_it_in_the_budget(rig):
    class NoStop(NullFfb):
        broken = False

        def stop(self, slot):
            if self.broken:
                raise OSError("device refused")
            super().stop(slot)

    eng, be, clock = rig(backend=NoStop(), ffb_gain=1.0)
    run(eng, clock, 1.0, weight=1.0)
    level, ran = abs(eng._pre_sent["weight_damper"]), eng._ran_at["weight_damper"]
    be.broken = True
    clock.t = ran + 0.16                                  # past ENDED_S, device may still play it
    eng.update([], snap(clock.t, weight=1.0), 0.16)
    assert eng._ghosts["weight_damper"] == (pytest.approx(level), pytest.approx(ran + DEADMAN_MS / 1000))


def test_test_pattern_obeys_the_latch_and_rejects_unknown_names(rig):
    eng, be, clock = rig(warm=False)
    with pytest.raises(ValueError):
        eng.test_pattern("spin", 0.0, 0.01, 0.0, "play")
    eng.stop_all()
    n = len(be.calls)
    pattern(eng, clock, "kick", 2.0)
    assert be.calls[n:] == []


# --- rule 4: stop ---

def test_pause_stops_and_play_brings_effects_back(rig):
    eng, be, clock = rig()
    run(eng, clock, 0.2, weight=1.0, echo="listen", echo_target=0.5)
    assert be.running
    run(eng, clock, 0.2, phase="paused", weight=1.0)
    assert not be.running and be.gain == 0.0
    assert [c.method for c in be.calls].count("stop_all") == 1
    assert eng.report()["torque"] == 0.0 and not eng.latched
    n = len(be.calls)
    step(eng, clock, 0.01, phase="paused", events=[FfbCue(0.0, "kick", dir=1.0)])
    assert be.calls[n:] == []                                      # nothing plays while paused
    run(eng, clock, 0.2, weight=1.0, steer_deg=-90.0)              # resume ramps up from zero
    assert "weight_spring" in be.running and be.gain == 1.0
    assert_ramped(be, "weight_spring")


def test_stop_all_latches_until_resume(rig):
    eng, be, clock = rig()
    run(eng, clock, 0.1, weight=1.0, steer_deg=-90.0)
    eng.stop_all()
    assert not be.running and be.gain == 0.0 and eng.latched
    n = len(be.calls)
    for _ in range(100):
        out = step(eng, clock, 0.01, [FfbCue(0.0, "kick", dir=1.0)], weight=1.0, steer_deg=-90.0)
    assert be.calls[n:] == [] and out["latched"]
    eng.resume()
    run(eng, clock, 0.2, weight=1.0, steer_deg=-90.0)
    assert "weight_spring" in be.running and be.gain == 1.0 and "RESUME" in eng.report()["log"]
    assert_ramped(be, "weight_spring")


def test_context_manager_stops_and_closes_on_exit_and_exception():
    be = NullFfb()
    with FfbEngine(be, Config()) as eng:
        eng.update([], snap(weight=1.0), 0.05)
    assert be.closed and "stop_all" in [c.method for c in be.calls]
    be = NullFfb()
    with pytest.raises(ZeroDivisionError), FfbEngine(be, Config()) as eng:
        eng.update([], snap(weight=1.0), 0.05)
        raise ZeroDivisionError
    assert be.closed and not be.running


def test_exception_inside_update_stops_all_and_latches(rig):
    eng, be, clock = rig()
    run(eng, clock, 0.1, weight=1.0)
    bad = snap(clock.t)
    bad.input = None
    with pytest.raises(AttributeError):
        eng.update([], bad, 0.01)
    assert be.calls[-1].method == "set_gain" and be.gain == 0.0 and not be.running and eng.latched
    n = len(be.calls)
    run(eng, clock, 0.1, weight=1.0)
    assert be.calls[n:] == []


def test_close_registered_with_atexit(monkeypatch):
    registered, unregistered = [], []
    monkeypatch.setattr(ffb.atexit, "register", registered.append)
    monkeypatch.setattr(ffb.atexit, "unregister", unregistered.append)
    be = NullFfb()
    eng = FfbEngine(be, Config())
    assert registered == [eng.close]
    registered[0]()                      # what atexit would run
    assert be.closed and unregistered == [eng.close]
    eng.close()                          # second close is a no-op
    assert [c.method for c in be.calls].count("close") == 1


def test_failed_stop_all_stops_each_effect_and_zeroes_gain(rig):
    class Broken(NullFfb):
        def stop_all(self):
            raise OSError("usb gone")

    eng, be, clock = rig(backend=Broken())
    run(eng, clock, 0.1, weight=1.0)
    eng.stop_all()
    assert {c.slot for c in be.calls if c.method == "stop"} >= set(SLOTS)
    assert be.gain == 0.0
    assert eng.report()["log"][0] == "STOP FAILED - stopped each effect, device gain 0"


# --- rule 5: degradation ---

def test_missing_effect_type_is_a_noop_for_that_effect_only(rig):
    eng, be, clock = rig(supported={"sine", "damper"}, ffb_gain=1.0)
    assert not any(eng.enabled[s] for s in ("echo", "weight_spring", "brake", "kick"))
    out = run(eng, clock, 0.3, weight=1.0, echo="listen", echo_target=1.0, steer_deg=-30.0,
              riser=RiserStatus(active=True, progress=0.5, held=True))
    step(eng, clock, 0.01, [FfbCue(0.0, "kick", dir=1.0)])
    run(eng, clock, 0.1)
    step(eng, clock, 0.01, [Beat(clock.t + 0.01, 1.0)])
    touched = {c.slot for c in be.calls if c.method in ("update", "run")}
    assert touched == {"weight_damper", "pulse", "riser"}
    assert "NO CONSTANT - echo off" in out["log"] and out["echo_centre"] is None


def test_transient_failure_is_retried(rig):
    class Flaky(NullFfb):
        fails = 2

        def update(self, slot, **params):
            if slot == "weight_damper" and self.fails:
                self.fails -= 1
                raise OSError("device busy")
            super().update(slot, **params)

    eng, be, clock = rig(backend=Flaky(), warm=False)
    out = run(eng, clock, 0.5, weight=1.0, steer_deg=-90.0)
    assert "FAIL weight_damper - retry in 1 s" in out["log"] and "weight_damper" not in be.running
    assert "weight_spring" in be.running
    run(eng, clock, 2.5, weight=1.0, steer_deg=-90.0)
    assert eng.enabled["weight_damper"] and "weight_damper" in be.running


def test_persistent_failure_turns_the_effect_off(rig):
    class Dead(NullFfb):
        def update(self, slot, **params):
            if slot == "weight_damper":
                raise OSError("device refused")
            super().update(slot, **params)

    eng, be, clock = rig(backend=Dead(), warm=False)
    out = run(eng, clock, 7.0, weight=1.0, steer_deg=-90.0)
    assert not eng.enabled["weight_damper"] and "FAIL weight_damper - off" in out["log"]
    assert eng._fails["weight_damper"] == 5 and "weight_spring" in be.running


def test_device_with_few_effects_drops_the_pattern_first(rig):
    eng, be, clock = rig(max_effects=len(SLOTS) - 1)   # 8 effects: the test pattern goes, rumble stays
    assert not eng.enabled["pattern"] and all(eng.enabled[s] for s in SLOTS if s != "pattern")
    eng, be, clock = rig(max_effects=len(SLOTS) - 3)
    assert [s for s in SLOTS if not eng.enabled[s]] == ["rumble", "pulse", "pattern"]
    log = eng.report()["log"]
    assert "DROP pattern - device plays 6 effects" in log and "DROP rumble - device plays 6 effects" in log


def test_riser_buzz_is_off_by_default(tmp_path):
    assert FfbSettings().strength["riser"] == 0.0
    assert json.loads(FfbSettings().save(tmp_path / "ffb.json").read_text())["strength"]["riser"] == 0.0
    assert FfbSettings.load(tmp_path / "missing.json").strength["riser"] == 0.0
    clock = Clock()
    be = NullFfb(clock=clock, record_all=True)
    eng = FfbEngine(be, Config(), clock=clock)
    run(eng, clock, 1.0, riser=RiserStatus(active=True, progress=1.0, held=True))
    assert not [c for c in be.calls if c.slot == "riser" and c.method in ("update", "run")]
    assert eng.report()["log"].count("riser buzz off (experimental)") == 1
    eng.close()


def test_riser_cycles_wait_3_ms_and_keep_clear_of_pulses(rig):
    eng, be, clock = rig(ffb_gain=1.0)
    held = RiserStatus(active=True, progress=1.0, held=True)
    beats = []
    for i in range(round(3.0 * 144)):
        events = [Beat(clock.t + 1 / 144, 1.0)] if i % 36 == 0 else []
        step(eng, clock, 1 / 144, events, riser=held)
        if events:
            beats.append(clock.t)
    runs = [c.t for c in be.calls if c.method == "run" and c.slot == "riser"]
    spacing = 0.083 + 0.003 - 1e-9                    # SPEC: whole 83 ms cycle, then 3 ms
    assert len(runs) > 20 and all(b - a >= spacing for a, b in zip(runs, runs[1:], strict=False))
    assert all(abs(r - b) >= 0.010 - 1e-9 for r in runs for b in beats)   # SPEC: 10 ms from a pulse


def test_device_without_sine_leaves_the_riser_silent(rig):
    eng, be, clock = rig(supported={"constant", "spring", "damper"})
    run(eng, clock, 1.0, riser=RiserStatus(active=True, progress=1.0, held=True))
    assert not eng.enabled["riser"] and not [c for c in be.calls if c.slot == "riser"]
    assert eng.report()["log"].count("NO SINE - riser off") == 1


def test_open_backend_without_joystick_is_null():
    be = open_backend(None)
    assert isinstance(be, NullFfb) and be.reason
    eng = FfbEngine(be, Config())
    assert "NO FFB - no steer binding" in eng.report()["log"]
    eng.close()


# --- property: random streams never exceed caps, budget, ramps or the echo rate ---

@pytest.mark.parametrize("seed", range(25))
def test_random_streams_never_exceed_limits(seed):
    rnd = random.Random(seed)
    clock = Clock()
    be = NullFfb(clock=clock, record_all=True)
    eng = FfbEngine(be, Config(ffb_gain=rnd.random()), FfbSettings(strength={"riser": 1.0}), clock=clock)
    frames = []                                   # (first call index, time, target gain, steer)
    weight, echo, target, riser, steer = 0.5, "none", 0.0, RiserStatus(), 0.0
    offset, unwind = 0.0, False
    centres = []                                  # (time, echo centre, echo active, steer, offset) per frame
    for _ in range(600):
        if rnd.random() < 0.02:
            eng.set_gain(rnd.choice([0.0, 0.3, 1.0, 2.5, -1.0, rnd.random()]))
        if rnd.random() < 0.01:
            eng.stop_all()
        if rnd.random() < 0.05:
            eng.resume()
        dt = rnd.choice([0.0, 0.001, 0.004, 1 / 120, 1 / 60, 0.033, 0.09, 0.11, 0.12, 0.13, 0.149, 0.15, 0.16, 0.18,
                         0.199, 0.2, -0.01,
                         float("nan")])
        clock.t += dt if math.isfinite(dt) and dt > 0 else 0.0
        now = clock.t
        if rnd.random() < 0.05:
            weight = rnd.choice([0.0, 1.0, rnd.random(), 5.0, float("nan")])
        if rnd.random() < 0.05:
            echo = rnd.choice(["none", "listen", "repeat"])
        if rnd.random() < 0.1:
            target = rnd.choice([rnd.uniform(-1, 1), -1.0, 1.0, 3.0, None, float("nan")])
        if rnd.random() < 0.05:
            riser = RiserStatus(active=rnd.random() < 0.7, progress=rnd.random(), held=rnd.random() < 0.6)
        if rnd.random() < 0.2:
            steer = rnd.choice([steer + rnd.uniform(-5, 5), rnd.uniform(-540, 540)])
        events = []
        for _ in range(rnd.randint(0, 3)):
            events.append(rnd.choice([
                Beat(now - rnd.random() * 0.3, rnd.random()),
                SectionChange(now, "s", rnd.random()),
                FfbCue(now, "kick", dir=rnd.choice([-1.0, 1.0, 0.0, 7.0])),
                FfbCue(now, "rumble"),
                FfbCue(now, "riser_start"),
                FfbCue(now, "riser_release"),
            ]))
        changed = rnd.random() < 0.02
        if changed:
            offset = rnd.choice([-360.0, 0.0, 360.0])
        unwind = rnd.random() < 0.3 if rnd.random() < 0.05 else unwind
        phase = "paused" if rnd.random() < 0.02 else "play"
        start = len(be.calls)
        g_before = eng.report()["gain_now"]
        out = eng.update(events, snap(now, phase=phase, steer_deg=steer, weight=weight, echo=echo,
                                      echo_target=target, riser=riser, steer_offset_deg=offset,
                                      steer_unwind=unwind, steer_offset_changed=changed), dt,
                         spinning=rnd.random() < 0.1)
        # the effective gain only rises toward the target, never above it
        assert out["gain_now"] <= max(g_before, eng.gain) + 1e-9 and out["gain_now"] <= 1.0
        frames.append((start, now, out["gain_now"], steer))
        centres.append((now, eng._centre, eng._echo_on, steer, eng._offset))
    eng.close()

    def gain_ceiling(t):   # one-shots fired up to their longest length ago may hold an older gain
        return max([g for _, ft, g, _ in frames if t - 0.25 <= ft <= t] + [0.0])

    ends = [f[0] for f in frames[1:]] + [len(be.calls)]
    for (start, _, g, _), end in zip(frames, ends, strict=True):
        for c in be.calls[start:end]:
            if c.method != "update":
                continue
            assert magnitude(c.params) <= SLOTS[c.slot].cap * g + 1e-9, (seed, c)
            assert 0.0 <= c.params.get("coefficient", 0.0) <= 1.0
            assert "centre" not in c.params                      # no device springs: software springs only
            assert 0 < c.params["length_ms"] <= max(DEADMAN_MS, max(SLOTS[s].length_ms for s in ONESHOTS))
    # rule 6: the SUM across slots, as the device holds it over time
    for t, total, shots in device_load(be, per_call=True):   # after every call, not only per frame
        assert total <= BUDGET * gain_ceiling(t) + 1e-9, (seed, t, total)
        assert shots <= ONESHOT_BUDGET * gain_ceiling(t) + 1e-9, (seed, t, shots)
    for slot in RAMPED:
        assert_ramped(be, slot)
    # rule 3 and the lead limit on the echo centre, in wheel degrees, frame by frame
    for (ta, ca, on_a, _, off_a), (tb, cb, on_b, steer_b, off_b) in zip(centres, centres[1:], strict=False):
        if not on_b:
            continue
        assert abs(cb - steer_b) <= ECHO_LEAD_DEG + 1e-6, (seed, tb, cb, steer_b)
        lead = abs(abs(cb - steer_b) - ECHO_LEAD_DEG) <= 1e-6   # waiting for the wheel: moves with it
        if on_a and off_a == off_b and not lead:
            assert abs(cb - ca) <= ECHO_MAX_DEG_S * (tb - ta) + 1e-6, (seed, ta, ca, tb, cb)


# --- SdlHapticBackend against a fake SDL (never a real device) ---

@pytest.fixture
def fake_sdl(monkeypatch):
    sdl2 = pytest.importorskip("sdl2")
    log = []
    effects = {}
    mask = sdl2.SDL_HAPTIC_SINE | sdl2.SDL_HAPTIC_CONSTANT | sdl2.SDL_HAPTIC_GAIN | sdl2.SDL_HAPTIC_AUTOCENTER

    def rec(name, ret=0):
        def f(*a):
            log.append((name, a))
            return ret
        return f

    def new_effect(h, ref):
        eid = len(effects)
        effects[eid] = ref._obj
        log.append(("new", eid))
        return eid

    for name in [n for n in dir(sdl2) if n.startswith("SDL_Haptic") and callable(getattr(sdl2, n))
                 and not isinstance(getattr(sdl2, n), type)]:
        monkeypatch.setattr(sdl2, name, rec(name))   # nothing reaches the real library
    monkeypatch.setattr(sdl2, "SDL_WasInit", rec("SDL_WasInit", 1))
    monkeypatch.setattr(sdl2, "SDL_InitSubSystem", rec("SDL_InitSubSystem", 0))
    monkeypatch.setattr(sdl2, "SDL_GetError", rec("SDL_GetError", b"fake"))
    monkeypatch.setattr(sdl2, "SDL_HapticOpenFromJoystick", rec("open", "HANDLE"))
    monkeypatch.setattr(sdl2, "SDL_HapticQuery", rec("query", mask))
    monkeypatch.setattr(sdl2, "SDL_HapticNewEffect", new_effect)
    return sdl2, log, effects


def test_sdl_backend_builds_effects_and_degrades(fake_sdl):
    sdl2, log, effects = fake_sdl
    be = SdlHapticBackend(object())
    assert ("SDL_HapticSetGain", ("HANDLE", 100)) in log
    assert ("SDL_HapticSetAutocenter", ("HANDLE", 0)) in log
    assert be.warnings == [] and be.max_effects is None
    assert be.supports("sine") and not be.supports("spring")
    assert not be.create("echo", "spring")
    assert be.create("kick", "constant") and be.create("pulse", "sine")
    be.update("kick", level=-0.5, length_ms=60)
    c = effects[0].constant
    assert effects[0].type == sdl2.SDL_HAPTIC_CONSTANT and c.level == -16384 and c.length == 60
    assert c.direction.type == sdl2.SDL_HAPTIC_CARTESIAN and c.direction.dir[0] == 1
    be.update("pulse", magnitude=2.0, period_ms=40)
    p = effects[1].periodic
    assert p.magnitude == 32767 and p.period == 40 and p.length == DEADMAN_MS
    assert p.direction.type == sdl2.SDL_HAPTIC_CARTESIAN and p.direction.dir[0] == 1
    be.update("echo", saturation=1.0)                   # never created: ignored
    be.run("kick")
    assert log[-1][0] == "SDL_HapticRunEffect"
    be.set_gain(0.25)
    assert log[-1] == ("SDL_HapticSetGain", ("HANDLE", 25))
    be.close()
    names = [n for n, _ in log]
    assert names[-1] == "SDL_HapticClose" and "SDL_HapticStopAll" in names
    assert names.count("SDL_HapticDestroyEffect") == 2


def test_sdl_backend_condition_effects(fake_sdl, monkeypatch):
    sdl2, log, effects = fake_sdl
    monkeypatch.setattr(sdl2, "SDL_HapticQuery", lambda h: sdl2.SDL_HAPTIC_SPRING | sdl2.SDL_HAPTIC_DAMPER)
    be = SdlHapticBackend(object())
    assert be.create("echo", "spring") and be.create("weight_damper", "damper")
    be.update("echo", coefficient=1.0, saturation=0.5, centre=-0.2, length_ms=DEADMAN_MS)
    c = effects[0].condition
    assert effects[0].type == sdl2.SDL_HAPTIC_SPRING and c.length == DEADMAN_MS
    assert c.right_sat[0] == c.left_sat[0] == 32768 and c.right_coeff[0] == 32767 and c.center[0] == -6553
    assert c.direction.type == sdl2.SDL_HAPTIC_CARTESIAN and c.direction.dir[0] == 1
    assert effects[1].type == sdl2.SDL_HAPTIC_DAMPER


def test_sdl_backend_checks_return_codes_and_effect_count(fake_sdl, monkeypatch):
    sdl2, log, effects = fake_sdl
    monkeypatch.setattr(sdl2, "SDL_HapticSetGain", lambda *a: -1)
    monkeypatch.setattr(sdl2, "SDL_HapticSetAutocenter", lambda *a: -1)
    monkeypatch.setattr(sdl2, "SDL_HapticQuery", lambda h: sdl2.SDL_HAPTIC_SINE | sdl2.SDL_HAPTIC_CONSTANT
                        | sdl2.SDL_HAPTIC_SPRING | sdl2.SDL_HAPTIC_DAMPER | sdl2.SDL_HAPTIC_GAIN
                        | sdl2.SDL_HAPTIC_AUTOCENTER)
    monkeypatch.setattr(sdl2, "SDL_HapticNumEffects", lambda h: 16)
    monkeypatch.setattr(sdl2, "SDL_HapticNumEffectsPlaying", lambda h: 4)
    be = SdlHapticBackend(object())
    assert be.warnings == ["device gain not set", "autocentre still on"] and be.max_effects == 4
    assert be.create("kick", "constant")
    monkeypatch.setattr(sdl2, "SDL_HapticStopEffect", lambda *a: -1)
    monkeypatch.setattr(sdl2, "SDL_HapticStopAll", lambda *a: -1)
    with pytest.raises(RuntimeError):
        be.stop("kick")
    with pytest.raises(RuntimeError):
        be.stop_all()
    with pytest.raises(RuntimeError):
        be.set_gain(0.0)
    eng = FfbEngine(be, Config())
    assert "WARN device gain not set" in eng.report()["log"]
    assert "DROP rumble - device plays 4 effects" in eng.report()["log"]
    eng.stop_all()
    assert eng.report()["log"][0].startswith("STOP FAILED")
    eng.close()


def test_sdl_backend_without_haptics_falls_back_to_null(fake_sdl, monkeypatch):
    sdl2, log, effects = fake_sdl
    monkeypatch.setattr(sdl2, "SDL_HapticOpenFromJoystick", lambda j: None)
    be = open_backend(object())
    assert isinstance(be, NullFfb) and "no haptic device" in be.reason


def test_sdl_update_failure_is_handled_by_the_engine(fake_sdl, monkeypatch):
    sdl2, log, effects = fake_sdl
    be = SdlHapticBackend(object())
    monkeypatch.setattr(sdl2, "SDL_HapticUpdateEffect", lambda *a: -1)
    clock = Clock()
    eng = FfbEngine(be, Config(ffb_gain=1.0), clock=clock)
    for _ in range(30):
        clock.t += 0.01
        eng.update([], snap(clock.t), 0.01)
    clock.t += 0.01
    eng.update([FfbCue(0.0, "kick", dir=1.0)], snap(clock.t), 0.01)
    assert eng._fails["kick"] == 1 and eng.enabled["pulse"]
    eng.close()
