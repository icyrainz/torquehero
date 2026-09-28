"""The judge: pure game logic (SPEC section 3). No I/O, no audio, no devices.

`Game.update(now, dt, inp)` advances to song time `now` and returns the
events of that frame; `Game.snapshot()` describes the frame for the views.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .chart import Chart, Note, phrase_x, road_at
from .config import Config
from .state import (
    LAYERS,
    Beat,
    FfbCue,
    GameEvent,
    InputState,
    Judgement,
    LayerStatus,
    NoteView,
    Phase,
    Popup,
    RiserStatus,
    SectionChange,
    Snapshot,
    SongInfo,
    Sound,
    spin_in_progress,
)

SPECIAL_KINDS = ("spin", "expr", "stab", "tom", "riser", "fader")
OFF_LANE = 1.35     # popup x offset from the road for notes that are not on the road
VIEW_BEHIND = 0.5   # seconds of past notes kept in Snapshot.notes
SPIN_LOCK_DEG = 540.0  # one turn plus half; the lock limit is min(this, wheel_range_deg / 2 - 5)

# Which layer a pressed control plays, for sounds and free play.
PRESS_LAYER = {
    "brake": "kick", "clutch": "hat", "paddle_l": "fills", "paddle_r": "fills", "handbrake": "riser",
    **{f"gate{i}": "pads" for i in range(1, 7)},
}
TAP_KINDS = ("kick", "hat", "stab", "tom")  # judged by a single press


def _matches(n: Note, ctrl: str) -> bool:
    """Whether a press of `ctrl` is the right input for tap note `n`."""
    match n.kind:
        case "kick":
            return ctrl == "brake"
        case "hat":
            return ctrl == "clutch"
        case "stab":
            return ctrl == f"gate{n.gate}"
        case "tom":
            return ctrl == ("paddle_l" if n.side == "L" else "paddle_r")
    return False


@dataclass
class _NoteState:
    done: bool = False
    result: str | None = None  # "perfect", "good", "miss", "auto", "listen"
    entered: bool = False      # long notes: first frame at or after n.t seen
    best: float = math.inf     # gate: best position error inside the good window
    acc: float = 0.0           # expr/fader: integrated absolute error
    tot: float = 0.0           # expr/fader: integrated time
    net: float = 0.0           # spin: signed degrees
    travel: float = 0.0        # spin: absolute degrees
    deg: float = 0.0           # spin: degrees counted by the active spin mode
    start_deg: float = 0.0     # spin: steer_deg on the entry frame
    held: float = 0.0          # riser: seconds held
    winding: bool = False      # riser: between a riser_start and its riser_release
    released: float | None = None  # riser: release time near the drop


@dataclass
class _Press:
    cause: str = "free"        # "hit", "miss", "free"
    index: int | None = None   # chart note the press was judged against


class Game:
    def __init__(self, chart: Chart, cfg: Config | None = None, layer_modes: dict[str, str] | None = None,
                 wheel_range_deg: float | None = None):
        """`wheel_range_deg`: the calibrated lock-to-lock range (the steer binding's
        range_deg); None or not positive uses Config.wheel_range_deg."""
        self.chart = chart
        self.cfg = cfg or Config()
        self.wheel_range_deg = wheel_range_deg if wheel_range_deg and wheel_range_deg > 0 else self.cfg.wheel_range_deg
        modes = layer_modes or {}
        self.layer_modes = {k: modes.get(k, "you" if k == "melody" else "auto") for k in LAYERS}
        self.phase: Phase = "play"
        self._ts = [n.t for n in chart.notes]
        self._max_dur = max((n.dur or 0.0 for n in chart.notes), default=0.0)
        self._echo_runs = self._build_echo_runs()
        self._zones = chart.blind_zones(self.cfg.good_window)
        self._drawn = chart.drawn_road(self.cfg.good_window)
        self.reset()

    def reset(self) -> None:
        self.now = 0.0
        self.score = self.combo = self.max_combo = 0
        self.counts = {"perfect": 0, "good": 0, "miss": 0}
        self.stats = {k: LayerStatus(mode=self.layer_modes[k]) for k in LAYERS}  # type: ignore[arg-type]
        self.popups: list[Popup] = []
        self._st = [_NoteState() for _ in self.chart.notes]
        self._head = 0
        self._beat_i = 0
        self._section_i = 0
        self._beat_strength = 0.0
        self._prev_deg: float | None = None
        self._offset = 0.0  # turn offset in degrees (SPEC 9 rule 10): 0, +360 or -360
        self._offset_due = True       # re-derive the offset from the wheel on the next update
        self._offset_changed = False  # the offset changed during the last update
        self._lane_jump = False  # Snapshot.steer_lane may jump this frame
        self._inp = InputState()
        self._riser_judged: tuple[int, str] | None = None  # riser judged by this frame's release

    # --- derived values ---

    @property
    def multiplier(self) -> int:
        return min(self.cfg.max_multiplier, 1 + self.combo // self.cfg.combo_step)

    @property
    def accuracy(self) -> float:
        judged = sum(self.counts.values())
        return (self.counts["perfect"] + 0.5 * self.counts["good"]) / judged if judged else 0.0

    @property
    def finished(self) -> bool:
        """Past the song length and every note finalized (waits for the last good
        window and any forced riser release)."""
        return self.now > self.chart.length and self._head >= len(self.chart.notes)

    def is_auto(self, layer: str) -> bool:
        return self.layer_modes.get(layer) == "auto"

    def spin_mode(self, inp: InputState) -> str:
        if self.cfg.spin_mode in ("net", "abs"):
            return self.cfg.spin_mode
        return "net" if "steer" in inp.bound else "abs"

    @property
    def steer_offset_deg(self) -> float:
        return self._offset

    @property
    def spin_lock_deg(self) -> float:
        """A spin finalized with |steer_deg| at or above this is a miss: one and a half
        turns, or 5 degrees inside the wheel's physical lock if that comes first."""
        return min(SPIN_LOCK_DEG, self.wheel_range_deg / 2 - 5.0)

    @property
    def spinning(self) -> bool:
        """A spin note is in progress (`state.spin_in_progress`): started and not yet finalized."""
        notes, t = self.chart.notes, self.now
        for i in range(self._head, len(notes)):
            if notes[i].t > t:
                break
            if spin_in_progress(notes[i], self._st[i].done, t):
                return True
        return False

    def _derive_offset(self, inp: InputState, deg: float | None = None) -> None:
        """Offset = the whole turn nearest to `deg` (default: the wheel), within one turn; 0 in
        abs mode. Called only after reset and when a spin is finalized, so steering never flips it."""
        deg = inp.steer_deg if deg is None else deg
        turns = math.floor(deg / 360.0 + 0.5) if self.spin_mode(inp) == "net" else 0
        offset = 360.0 * max(-1, min(1, turns))
        self._offset_changed = self._offset_changed or offset != self._offset
        self._offset = offset

    def steer_unwind(self, inp: InputState) -> bool:
        """The wheel is more than half a turn from the offset: the player must turn back.
        Never during a spin, where the wheel is meant to leave the offset."""
        return (not self.spinning and self.spin_mode(inp) == "net"
                and abs(inp.steer_deg - self._offset) > 180.0)

    def steer_lane(self, inp: InputState) -> float:
        """Lane position used for judging: `inp.steer` while the offset is 0, else
        `(steer_deg - offset) / play_range_deg` clamped to -1..1. `Snapshot.steer_lane`
        shows the drawn road instead while a spin is active."""
        if not self._offset:
            return inp.steer
        return max(-1.0, min(1.0, (inp.steer_deg - self._offset) / self.cfg.play_range_deg))

    def drawn_road_x(self, t: float) -> float:
        """The road as shown (`Chart.drawn_road_x`), from keyframes cached at construction."""
        return road_at(self._drawn, t)

    def in_blind_zone(self, t: float) -> bool:
        return any(a <= t <= b for a, b in self._zones)

    def song_info(self) -> SongInfo:
        c = self.chart
        used = {n.layer for n in c.notes}
        return SongInfo(
            title=c.title, artist=c.artist, bpm=c.bpm, length=c.length,
            sections=[{"t": s.t, "name": s.name, "weight": s.weight} for s in c.sections],
            layer_modes=dict(self.layer_modes), layers_used=[k for k in LAYERS if k in used],
        )

    # --- frame ---

    def update(self, now: float, dt: float, inp: InputState) -> list[GameEvent]:
        """Advance to `now` (song seconds, before audio offset) with `dt` seconds since
        the last frame. Returns this frame's events in order."""
        t = now - self.cfg.audio_offset
        was_spinning = self.spinning
        self.now, self._inp = t, inp
        ev: list[GameEvent] = []
        self._clock_events(t, ev)

        d_deg = 0.0 if self._prev_deg is None else inp.steer_deg - self._prev_deg
        self._prev_deg = inp.steer_deg
        self._riser_judged = None
        self._offset_changed = False
        if self._offset_due:  # first frame after reset: always reported as changed
            self._offset_due = False
            self._derive_offset(inp)
            self._offset_changed = True
        elif self._offset and self.spin_mode(inp) != "net":  # steer left `bound` mid-song
            self._offset = 0.0
            self._offset_changed = True
        presses = {c: _Press() for c in sorted(inp.pressed)
                   if c in PRESS_LAYER and not self.is_auto(PRESS_LAYER[c])}
        self._judge_presses(t, inp, presses, ev)

        gw = self.cfg.good_window
        notes = self.chart.notes
        for i in range(self._head, len(notes)):
            n, s = notes[i], self._st[i]
            if n.t - gw > t:
                break
            if s.done:
                continue
            if self.is_auto(n.layer):
                self._auto(i, n, s, t, ev)
            else:
                self._judge(i, n, s, t, dt, d_deg, inp, ev)
        while self._head < len(notes) and self._st[self._head].done:
            self._head += 1
        self._lane_jump = self._offset_changed or was_spinning != self.spinning

        self._press_sounds(t, inp, presses, ev)
        self.popups = [p for p in self.popups if t - p.t0 <= self.cfg.popup_life]
        return ev

    def _clock_events(self, t: float, ev: list[GameEvent]) -> None:
        beats, sections = self.chart.beats, self.chart.sections
        while self._beat_i < len(beats) and beats[self._beat_i].t <= t:
            b = beats[self._beat_i]
            self._beat_i += 1
            self._beat_strength = b.s
            ev.append(Beat(b.t, b.s))
        while self._section_i < len(sections) and sections[self._section_i].t <= t:
            s = sections[self._section_i]
            self._section_i += 1
            ev.append(SectionChange(s.t, s.name, s.weight))

    def _timing(self, n: Note, t: float) -> str:
        return "perfect" if abs(t - n.t) <= self.cfg.perfect_window else "good"

    def _judge_presses(self, t: float, inp: InputState, presses: dict[str, _Press], ev: list[GameEvent]) -> None:
        """A press judges the earliest unfinished note in the good window that it matches.
        Only when none matches does a gate or paddle press count as a wrong press on the
        nearest unfinished note of its layer. A handbrake pull during a riser is a hit."""
        gw = self.cfg.good_window
        notes = self.chart.notes
        lo, hi = bisect_left(self._ts, t - gw), bisect_right(self._ts, t + gw)
        for ctrl, p in presses.items():
            layer = PRESS_LAYER[ctrl]
            if layer == "riser":
                for i in range(bisect_left(self._ts, t - self._max_dur), hi):
                    n = notes[i]
                    if n.kind == "riser" and not self._st[i].done and n.t - gw <= t <= n.end:
                        p.cause, p.index = "hit", i
                        break
                continue
            cands = [i for i in range(lo, hi) if notes[i].layer == layer and notes[i].kind in TAP_KINDS
                     and not self._st[i].done]
            match = next((i for i in cands if _matches(notes[i], ctrl)), None)
            if match is not None:
                n = notes[match]
                vel = inp.velocity.get(ctrl, 1.0)
                self._finalize(match, n, self._timing(n, t), t, ev, error=t - n.t, vel=vel)
                p.cause, p.index = "hit", match
            elif cands and layer in ("pads", "fills"):
                i = min(cands, key=lambda i: abs(t - notes[i].t))
                self._finalize(i, notes[i], "miss", t, ev, error=t - notes[i].t)
                p.cause, p.index = "miss", i

    def _judge(self, i: int, n: Note, s: _NoteState, t: float, dt: float, d_deg: float,
               inp: InputState, ev: list[GameEvent]) -> None:
        cfg = self.cfg
        late = t > n.t + cfg.good_window
        if n.kind in TAP_KINDS:
            if late:
                self._finalize(i, n, "miss", t, ev)
            return
        if n.kind == "gate":
            if n.listen:
                if t >= n.t:
                    s.done, s.result = True, "listen"
                return
            if late:
                self._finalize(i, n, "good" if s.best <= cfg.good_pos else "miss", t, ev, error=s.best)
                return
            err = abs(self.steer_lane(inp) - (n.x or 0.0))
            s.best = min(s.best, err)
            if err <= cfg.perfect_pos and abs(t - n.t) <= cfg.perfect_window:
                self._finalize(i, n, "perfect", t, ev, error=err)
            return
        if t < n.t:
            return
        # Long notes: the entry frame counts no motion and only the time since n.t.
        first = not s.entered
        s.entered = True
        match n.kind:
            case "spin":
                if first:
                    s.start_deg = inp.steer_deg
                else:
                    s.net += d_deg
                    s.travel += abs(d_deg)
                net = self.spin_mode(inp) == "net"
                if not net:
                    s.deg = s.travel  # abs mode ignores dir
                elif n.dir == "cw":
                    s.deg = max(0.0, s.net)
                elif n.dir == "ccw":
                    s.deg = max(0.0, -s.net)
                else:
                    s.deg = abs(s.net)
                res = None
                if s.deg >= cfg.spin_perfect_deg - cfg.spin_tolerance_deg:
                    res = "perfect"
                elif t > n.end:
                    res = "good" if s.deg >= cfg.spin_good_deg else "miss"
                if res is None:
                    return
                if net and abs(inp.steer_deg) >= self.spin_lock_deg:
                    res = "miss"  # beyond one and a half turns, or at the lock
                if res == "perfect" and net:  # finished early: one turn on from where it started
                    turn = 1.0 if n.dir == "cw" else -1.0 if n.dir == "ccw" else math.copysign(1.0, s.net)
                    self._derive_offset(inp, s.start_deg + 360.0 * turn)
                else:
                    self._derive_offset(inp)
                self._finalize(i, n, res, t, ev, error=0.0 if res == "perfect" else cfg.spin_perfect_deg - s.deg)
            case "expr" | "fader":
                if t <= n.end:
                    v = inp.throttle if n.kind == "expr" else (inp.lever1 if n.lever == 1 else inp.lever0)
                    step = min(dt, t - n.t) if first else dt
                    s.acc += abs(v - n.value_at(t)) * step
                    s.tot += step
                else:
                    e = s.acc / s.tot if s.tot else 1.0
                    res = "perfect" if e < cfg.expr_perfect else "good" if e < cfg.expr_good else "miss"
                    self._finalize(i, n, res, t, ev, error=e)
            case "riser":
                self._judge_riser(i, n, s, t, min(dt, t - n.t) if first else dt, inp, ev)

    def _judge_riser(self, i: int, n: Note, s: _NoteState, t: float, dt: float,
                     inp: InputState, ev: list[GameEvent]) -> None:
        """Every riser_start cue gets exactly one riser_release, on every path."""
        cfg = self.cfg
        pulled = "handbrake" in inp.down
        if t <= n.end and pulled:
            s.held += dt
            if not s.winding:
                s.winding = True
                ev.append(FfbCue(t, "riser_start"))
        if s.winding and not pulled:
            s.winding = False
            ev.append(FfbCue(t, "riser_release"))
            if t >= n.end - cfg.good_window:
                s.released = t
        if s.released is not None or t > n.end + cfg.good_window:
            if s.winding:
                s.winding = False
                ev.append(FfbCue(t, "riser_release"))
            frac = min(1.0, s.held / n.dur) if n.dur else 0.0
            rel = s.released if s.released is not None else t
            terr = abs(rel - n.end)
            if frac >= cfg.riser_perfect_hold and terr <= cfg.perfect_window:
                res = "perfect"
            elif frac >= cfg.riser_good_hold and terr <= cfg.good_window:
                res = "good"
            else:
                res = "miss"
            if s.released == t:
                self._riser_judged = (i, res)
            self._finalize(i, n, res, t, ev, error=rel - n.end)

    def _auto(self, i: int, n: Note, s: _NoteState, t: float, ev: list[GameEvent]) -> None:
        """Auto layers are played by the song: never judged. Sounds carry cause="auto"
        (the audio engine schedules those from the chart and ignores these events)."""
        if t < n.t:
            return
        audio = self.chart.audio
        mode = audio.mode(n.layer)
        tag = {"cause": "auto", "note_index": i, "note_t": n.t}
        if n.kind == "riser":
            loop = audio.oneshot_for(n.layer, n) or "riser"
            if not s.winding:
                s.winding = True
                ev.append(FfbCue(t, "riser_start"))
                if mode == "riser":
                    ev.append(Sound(t, n.layer, loop, "start", **tag))  # type: ignore[arg-type]
            if t < n.end:
                return
            s.winding = False
            ev.append(FfbCue(t, "riser_release"))
            if mode == "riser":
                ev.append(Sound(t, n.layer, loop, "stop", **tag))  # type: ignore[arg-type]
                ev.append(Sound(t, n.layer, "impact", **tag))  # type: ignore[arg-type]
        elif n.kind == "gate" and n.listen:
            pass
        elif mode == "trigger" and (name := audio.oneshot_for(n.layer, n)):
            ev.append(Sound(t, n.layer, name, vel=n.vel if n.vel is not None else 1.0, **tag))  # type: ignore[arg-type]
        s.done, s.result = True, "auto"

    def _press_sounds(self, t: float, inp: InputState, presses: dict[str, _Press], ev: list[GameEvent]) -> None:
        """The instrument: player input makes its sound, hit or not (SPEC 3 free play, 8.2 modes)."""
        audio = self.chart.audio
        notes = self.chart.notes
        for ctrl, p in presses.items():
            layer = PRESS_LAYER[ctrl]
            mode = audio.mode(layer)
            note = notes[p.index] if p.index is not None else None
            tag = {"cause": p.cause, "note_index": p.index, "note_t": note.t if note else None}
            if mode == "riser":
                ev.append(Sound(t, layer, audio.oneshot_for(layer) or "riser", "start", **tag))  # type: ignore[arg-type]
                continue
            if not (mode == "trigger" or (mode == "gate" and p.cause == "hit")):
                continue
            if p.cause == "miss":
                name = "dud"
            else:
                gate = int(ctrl[4:]) if ctrl.startswith("gate") else 0
                side = "L" if ctrl == "paddle_l" else "R" if ctrl == "paddle_r" else ""
                name = audio.oneshot_for(layer, note, gate=gate, side=side)
            if name:
                ev.append(Sound(t, layer, name, vel=inp.velocity.get(ctrl, 1.0), **tag))  # type: ignore[arg-type]
        if "handbrake" in inp.released and not self.is_auto("riser") and audio.mode("riser") == "riser":
            tag = {"cause": "free", "note_index": None, "note_t": None}
            if self._riser_judged is not None:
                i, res = self._riser_judged
                tag = {"cause": "miss" if res == "miss" else "hit", "note_index": i, "note_t": notes[i].t}
            ev.append(Sound(t, "riser", audio.oneshot_for("riser") or "riser", "stop", **tag))  # type: ignore[arg-type]
            ev.append(Sound(t, "riser", "impact", **tag))  # type: ignore[arg-type]

    def _finalize(self, i: int, n: Note, result: str, t: float, ev: list[GameEvent],
                  error: float = 0.0, vel: float = 1.0) -> None:
        cfg = self.cfg
        s = self._st[i]
        s.done, s.result = True, result
        self.counts[result] += 1
        st = self.stats[n.layer]
        st.total += 1
        if result == "miss":
            st.alive = False
            self.combo = 0
            ev.append(FfbCue(t, "rumble"))
        else:
            st.hit += 1
            st.alive = True
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self.score += (cfg.score_perfect if result == "perfect" else cfg.score_good) * self.multiplier
        if result == "perfect" and n.kind in ("gate", "spin"):
            side = self.steer_lane(self._inp) - self.drawn_road_x(t)
            ev.append(FfbCue(t, "kick", dir=math.copysign(1.0, side or 1.0)))
        err = error if math.isfinite(error) else 1.0
        ev.append(Judgement(t, n.layer, n.kind, i, n.t, result, err, vel))  # type: ignore[arg-type]
        self.popups.append(Popup(result.upper(), n.layer, self._popup_x(n, t), t))

    def _popup_x(self, n: Note, t: float) -> float:
        if n.kind == "gate":
            return n.x or 0.0
        road = self.drawn_road_x(t)
        if n.kind == "spin":
            return road
        return road - OFF_LANE if n.layer in ("kick", "hat", "expr") else road + OFF_LANE

    # --- echo ---

    def _build_echo_runs(self) -> list[tuple[float, float, str, list[tuple[float, float]]]]:
        """Runs of consecutive listen (or blind) gates, as (open, close, kind, [(t, x), ...]).
        A run opens one beat before its first note and closes one good window after its
        last, so the echo state does not flicker between notes."""
        lead = 60.0 / self.chart.bpm
        runs = []
        run: list[tuple[float, float]] = []
        run_kind = ""

        def close() -> None:
            if run and run_kind:
                runs.append((run[0][0] - lead, run[-1][0] + self.cfg.good_window, run_kind, run))

        for n in (n for n in self.chart.notes if n.kind == "gate"):
            k = "listen" if n.listen else "repeat" if n.blind else ""
            if k != run_kind:
                close()
                run, run_kind = [], k
            run.append((n.t, n.x or 0.0))
        close()
        return runs

    def _echo_run(self, t: float):
        for run in self._echo_runs:
            if run[0] <= t <= run[1]:
                return run
        return None

    def echo_at(self, t: float) -> str:
        run = self._echo_run(t)
        return run[2] if run else "none"

    def echo_target_at(self, t: float) -> float | None:
        """Lane position the base steers to during listen: phrase_x through the run's gates."""
        run = self._echo_run(t)
        return phrase_x(run[3], t) if run and run[2] == "listen" else None

    # --- snapshot ---

    def _progress(self, n: Note, s: _NoteState, t: float) -> float | None:
        if n.kind == "spin":
            return 1.0 if s.result == "perfect" else min(1.0, s.deg / self.cfg.spin_perfect_deg)
        if n.kind in ("riser", "expr", "fader") and n.dur:
            return min(1.0, max(0.0, (t - n.t) / n.dur))
        return None

    def snapshot(self) -> Snapshot:
        t, inp, cfg, chart = self.now, self._inp, self.cfg, self.chart
        notes = chart.notes
        road = self.drawn_road_x(t)
        echo = self.echo_at(t)
        sec = chart.section_at(t)
        lane = road if self.spinning else self.steer_lane(inp)

        riser = RiserStatus()
        expr_target: float | None = None
        faders: list[float | None] = [None, None]
        upcoming: list[dict] = []
        views: list[NoteView] = []
        start = bisect_left(self._ts, t - VIEW_BEHIND - self._max_dur)
        stop = bisect_right(self._ts, t + max(cfg.lookahead, cfg.upcoming_horizon))
        for i in range(start, stop):
            n, s = notes[i], self._st[i]
            if n.end >= t - VIEW_BEHIND and n.t <= t + cfg.lookahead:
                result = s.result if s.result in ("perfect", "good", "miss", "auto") else None
                views.append(NoteView(i, s.done, result, self._progress(n, s, t)))
            if n.kind in SPECIAL_KINDS and t <= n.t <= t + cfg.upcoming_horizon and not s.done:
                upcoming.append({**{k: v for k, v in n.to_dict().items() if k != "curve"}, "index": i})
            if n.kind == "riser" and n.dur and n.t <= t:
                auto = self.is_auto("riser")
                if (t <= n.end) if auto else not s.done:  # player riser: active until judged
                    riser = RiserStatus(True, min(1.0, (t - n.t) / n.dur),
                                        auto or "handbrake" in inp.down,
                                        1.0 if auto else min(1.0, s.held / n.dur))
                continue
            if n.dur is None or not n.t <= t <= n.end:
                continue
            if n.kind == "expr":
                expr_target = n.value_at(t)
            elif n.kind == "fader" and n.lever in (0, 1):
                faders[n.lever] = n.value_at(t)

        beats = chart.beats
        beat_phase = 0.0
        if 0 < self._beat_i < len(beats):
            a, b = beats[self._beat_i - 1].t, beats[self._beat_i].t
            beat_phase = (t - a) / (b - a) if b > a else 0.0

        layers = {k: LayerStatus(v.mode, v.alive or v.mode == "auto", v.hit, v.total) for k, v in self.stats.items()}
        return Snapshot(
            phase=self.phase, now=t, length=chart.length, score=self.score, combo=self.combo,
            max_combo=self.max_combo, multiplier=self.multiplier, accuracy=self.accuracy,
            counts=dict(self.counts), layers=layers, input=inp,
            section=sec.name if sec else "", weight=sec.weight if sec else 0.0,
            beat_phase=beat_phase, beat_strength=self._beat_strength, road_x=road,
            on_road=echo == "listen" or self.in_blind_zone(t) or abs(lane - road) <= cfg.road_tolerance,
            echo=echo, echo_target=self.echo_target_at(t),  # type: ignore[arg-type]
            riser=riser, expr_target=expr_target, fader_targets=faders, upcoming=upcoming,
            notes=views, popups=list(self.popups), steer_offset_deg=self._offset, steer_lane=lane,
            steer_unwind=self.steer_unwind(inp), steer_offset_changed=self._offset_changed,
            steer_lane_jump=self._lane_jump,
        )
