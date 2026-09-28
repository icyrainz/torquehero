"""Audio engine (SPEC 4, 8.2): stem mixer whose played-frame count is the song clock.

`Mixer` is the pure core: loaded arrays + control targets -> `render(n)` samples.
`AudioEngine` runs a Mixer in a sounddevice callback and derives the song clock
from it. `NullAudio` has the same interface on a monotonic clock and records calls.

Per-layer behaviour follows the manifest mode (SPEC 8.2, allowed pairs in 8.9):
trigger and riser layers are one-shots and loops; gate, filter, level and levers
layers shape the gain and low-pass of their stem. Auto-layer sounds are scheduled
here from the chart, sample-accurately; `Sound` events with cause "auto" are ignored.

Times are song seconds, gains 0..1, audio is float32 stereo at the device rate.
scipy is imported only by the Mixer and the loaders, so NullAudio stays light.
"""
from __future__ import annotations

import json
import logging
import math
import time as _time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .chart import ALLOWED_MODES
from .config import backup_bad, config_dir, finite, load_settings
from .state import Sound

if TYPE_CHECKING:
    from .chart import AudioManifest, Chart, Note
    from .state import GameEvent, Snapshot

log = logging.getLogger(__name__)

SETTINGS_FILE = "audio.json"

STEM_MODES = ("gate", "filter", "level", "levers")

# SPEC 8.2 mix values and ramps.
GATE_FLOOR = 0.2
GATE_DOWN_S = 0.060          # full duck 1 -> 0.2
GATE_UP_S = 0.030            # full recovery 0.2 -> 1
FILTER_OFF_GAIN = 0.35
FILTER_OFF_HZ = 450.0
OPEN_HZ = 20000.0            # "open" low-pass, clamped below Nyquist; bypassed at rest
LEVEL_FLOOR = 0.15
LEVEL_REST_V = (0.55 - LEVEL_FLOOR) / (1 - LEVEL_FLOOR)   # SPEC 10: gain 0.55 before the first expr note
LEVER_FLOOR = 0.2
LEVER_HZ = (200.0, 8000.0)
TAU_FILTER = 0.030           # smoothing time constants (seconds)
TAU_LEVEL = 0.050
TAU_LEVERS = 0.030
TAU_MASTER = 0.030
FILTER_SUB = 64              # frames per coefficient update while a cutoff moves
CUTOFF_STEPS = 48            # coefficient cache resolution, steps per octave
LOOP_FADE_S = 0.020          # riser loop fade-out on stop
STEAL_FADE_S = 0.005         # fade of a stolen one-shot
TRANSPORT_FADE_S = 0.010     # pause, resume and seek fades
MAX_VOICES = 32
RESERVE_VOICES = 8           # slots where stolen voices fade out
KEEP_RECENT = 4              # the newest voices are never stolen

_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


# --- settings ---

@dataclass
class AudioSettings:
    """Audio module settings, persisted in `config_dir()/audio.json` (not in Config).
    The timing trim is `Config.audio_offset`, applied by the Game, not here."""

    device: str | int | None = None     # sounddevice output device; None = system default
    samplerate: int | None = None       # None = the device's default rate
    blocksize: int = 256                # frames per callback; raise on crackle
    latency: str | float = "low"        # "low", "high" or seconds; raise on crackle

    @classmethod
    def load(cls, path: str | Path | None = None) -> AudioSettings:
        """Defaults when the file is missing; key by key, a bad key keeps its default (`load_settings`)."""
        path = Path(path) if path else config_dir() / SETTINGS_FILE
        return load_settings(cls, path, AUDIO_CHECKS)

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else config_dir() / SETTINGS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        backup_bad(path)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


AUDIO_CHECKS = {
    "device": lambda v: v is None or isinstance(v, str) or (_int(v) and 0 <= v < 10_000),
    "samplerate": lambda v: v is None or (_int(v) and 0 < v <= 768_000),
    "blocksize": lambda v: _int(v) and 0 < v <= 65_536,
    "latency": lambda v: v in ("low", "high") or (isinstance(v, int | float) and not isinstance(v, bool)
                                                  and finite(v) and 0 <= v <= 10.0),
}


# --- loading ---

@dataclass
class SongAudio:
    """Decoded audio of one chart at rate `sr`. Arrays may be mono (n,) or (n, ch)
    and any dtype; `normalised` makes them float32 (n, 2) at the mixer rate."""

    sr: int
    backing: list[np.ndarray]
    stems: dict[str, np.ndarray]
    oneshots: dict[str, np.ndarray]

    def normalised(self, sr: int) -> SongAudio:
        def f(a):
            return resample(stereo(a), self.sr, sr)
        return SongAudio(sr, [f(a) for a in self.backing], {k: f(a) for k, a in self.stems.items()},
                         {k: f(a) for k, a in self.oneshots.items()})


def stereo(a: np.ndarray) -> np.ndarray:
    """Any (n,) or (n, ch) array as contiguous float32 (n, 2). Integer PCM is scaled to -1..1."""
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / np.float32(np.iinfo(a.dtype).max + 1)
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    a = np.repeat(a, 2, axis=1) if a.shape[1] == 1 else a[:, :2]
    return np.ascontiguousarray(a)


def resample(a: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Polyphase resample of a (n, 2) array; done once at load."""
    if sr_from == sr_to or len(a) == 0:
        return a
    from scipy.signal import resample_poly

    g = math.gcd(int(sr_from), int(sr_to))
    return np.ascontiguousarray(resample_poly(a, sr_to // g, sr_from // g, axis=0), dtype=np.float32)


def read_audio(path: Path, sr: int) -> np.ndarray | None:
    """wav/flac as (n, 2) float32 at `sr`, or None (logged once) when unreadable."""
    import soundfile as sf

    try:
        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError, sf.LibsndfileError) as e:
        _warn_once(f"file:{path}", "audio: cannot read %s (%s); it stays silent", path, e)
        return None
    return resample(stereo(data), file_sr, sr)


def load_song_audio(chart: Chart, sr: int) -> SongAudio:
    """Read every file the manifest names, resampled to `sr`. Missing files are skipped."""
    a = chart.audio
    backing = [x for p in a.backing if (x := read_audio(chart.path(p), sr)) is not None]
    stems = {k: x for k, p in a.stems.items() if (x := read_audio(chart.path(p), sr)) is not None}
    shots = {k: x for k, p in a.oneshots.items() if (x := read_audio(chart.path(p), sr)) is not None}
    return SongAudio(sr, backing, stems, shots)


# --- mixer core ---

class Ramp:
    """A control value moving to `target`: exponential with time constant `tau`, or
    linear at `up`/`down` units per second. `fill` writes one block, sample by sample."""

    def __init__(self, value: float, tau: float | None = None, up: float = 0.0, down: float = 0.0):
        self.value = self.target = float(value)
        self.tau, self.up, self.down = tau, up, down

    def fill(self, out: np.ndarray, sr: int, idx: np.ndarray) -> None:
        """`idx` is 1..n as float; writes n = len(out) values and advances."""
        n, tgt, v = len(out), self.target, self.value
        if v == tgt:
            out.fill(v)
            return
        if self.tau is not None:
            np.multiply(idx[:n], -1.0 / (self.tau * sr), out=out)
            np.exp(out, out=out)
            out *= v - tgt
            out += tgt
            end = float(out[-1])
            self.value = tgt if abs(end - tgt) < 1e-6 else end
            return
        step = (self.up if tgt > v else -self.down) / sr
        np.multiply(idx[:n], step, out=out)
        out += v
        (np.minimum if tgt > v else np.maximum)(out, tgt, out=out)
        self.value = float(out[-1])

    def step(self, n: int, sr: int) -> float:
        """Advance an exponential ramp by n samples in the log domain (filter cutoffs)."""
        tgt = self.target
        if self.value != tgt:
            k = math.exp(-n / ((self.tau or TAU_FILTER) * sr))
            v = tgt * (self.value / tgt) ** k
            self.value = tgt if abs(v / tgt - 1) < 1e-3 else v
        return self.value


def lowpass_coeffs(fc: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """RBJ biquad low-pass, Q = 1/sqrt(2)."""
    fc = min(max(fc, 20.0), 0.45 * sr)
    w0 = 2 * math.pi * fc / sr
    alpha, c = math.sin(w0) / math.sqrt(2), math.cos(w0)
    a0 = 1 + alpha
    b = np.array([(1 - c) / 2, 1 - c, (1 - c) / 2]) / a0
    a = np.array([1.0, -2 * c / a0, (1 - alpha) / a0])
    return b, a


@dataclass
class _LayerCtl:
    """Mix controls of one stem layer, set from the main thread."""

    mode: str
    stem: str
    alive: bool = True
    on_road: bool = True
    v: float = 1.0          # level value, or lever0
    v1: float = 1.0         # lever1


class _Track:
    """One continuous audio stream (backing or stem) and its gain/filter state."""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.gate: Ramp | None = None
        self.gains: dict[str, Ramp] = {}          # layer -> gain ramp
        self.ramps: list[Ramp] = []               # gate + gains, fixed at setup
        self.cutoff: Ramp | None = None
        self.hist = np.zeros((4, 2))              # filter history x[n-1], x[n-2], y[n-1], y[n-2]
        self.filtering = False                    # hist holds live filter state
        self.muted = False


class _Voice:
    __slots__ = ("data", "pos", "delay", "gain", "loop", "fade", "fade_len", "stop_in", "active", "seq", "layer")

    def __init__(self):
        self.data = np.zeros((0, 2), np.float32)
        self.pos = self.delay = self.fade = self.seq = 0
        self.fade_len = 1
        self.stop_in = -1       # block offset where the fade-out begins, -1 = none
        self.gain = 0.0
        self.loop = self.active = False
        self.layer = ""         # loop owner, "" for one-shots

    def copy_from(self, o: _Voice) -> None:
        for k in self.__slots__:
            setattr(self, k, getattr(o, k))


class Mixer:
    """Pure mixer: `render(n)` returns the next n stereo frames of the song.

    Main-thread calls (`play`, `start_loop`, `stop_loop`, `seek`, `set_paused`) are
    queued and take effect at the start of the next block; pause, resume and seek
    fade over TRANSPORT_FADE_S. Stem controls (`set_layer`) are plain target values
    read by the next block. `frame` is the song frame of the next block's first
    sample; it keeps advancing past the end of the audio. Build a Mixer, `setup` it,
    then hand it to the audio thread; never `setup` one the audio thread renders."""

    def __init__(self, sr: int, max_block: int = 8192, volume: float = 1.0):
        from scipy.signal import lfilter

        self._lfilter = lfilter
        self.sr, self.max_block = int(sr), int(max_block)
        self.frame = 0
        self.paused = True
        self.render_errors = 0
        self.master = Ramp(volume, tau=TAU_MASTER)
        self.transport = Ramp(0.0, up=1 / TRANSPORT_FADE_S, down=1 / TRANSPORT_FADE_S)
        self.clock_seq = 0        # bumped by each queued seek/pause/resume
        self.applied_seq = 0      # clock_seq the next block's frame belongs to
        self._drained_seq = 0     # newest clock_seq drained; applied once no seek is pending
        self.disabled: set[str] = set()   # layers with a mode SPEC 8.9 does not allow
        self._pausing = False
        self._pending_seek: tuple[int, int] | None = None   # (frame, seq) after the fade-out
        self._seek_elapsed = 0    # frames played while the seek waited; the jump lands that much later
        self._pending_resume: int | None = None   # rewind frame of a resume that came mid-fade
        self._cmds: deque = deque()
        self._tracks: list[_Track] = []
        self._stem_tracks: dict[str, _Track] = {}
        self._layers: dict[str, _LayerCtl] = {}
        self._shots: dict[str, np.ndarray] = {}
        self._voices = [_Voice() for _ in range(MAX_VOICES)]
        self._reserve = [_Voice() for _ in range(RESERVE_VOICES)]
        self._loops: dict[str, _Voice] = {}
        self._vseq = 0
        self._coef: dict[int, tuple] = {}
        self._zi = np.zeros((2, 2))
        self._sched_f = np.zeros(0, np.int64)
        self._sched: list[tuple] = []
        self._si = 0
        self._idx = np.arange(1, self.max_block + 1, dtype=np.float32)
        self._out = np.zeros((self.max_block, 2), np.float32)
        self._tmp = np.zeros((self.max_block, 2), np.float32)
        self._flt = np.zeros((self.max_block, 2), np.float32)
        self._g = np.zeros(self.max_block, np.float32)
        self._g2 = np.zeros(self.max_block, np.float32)
        self._open_hz = min(OPEN_HZ, 0.45 * self.sr)

    # --- setup (not real-time) ---

    def setup(self, manifest: AudioManifest, audio: SongAudio, notes: Sequence[Note] = (),
              layer_modes: dict[str, str] | None = None) -> None:
        """Wire loaded arrays to the manifest's layers and schedule the auto layers."""
        audio = audio.normalised(self.sr)
        modes = layer_modes or {}
        self._shots = dict(audio.oneshots)
        self._tracks = [_Track(a) for a in audio.backing]
        self._stem_tracks = {k: _Track(a) for k, a in audio.stems.items()}
        self._tracks += self._stem_tracks.values()
        self._layers = {}
        self.disabled = set()
        claimed: set[str] = set()
        for layer, la in manifest.layers.items():
            if la.mode not in ALLOWED_MODES.get(layer, ()):
                _warn_once(f"mode:{layer}:{la.mode}", "audio: layer %s cannot use mode %r; it stays silent",
                           layer, la.mode)
                self.disabled.add(layer)
                if la.stem:
                    claimed.add(la.stem)
                continue
            if la.mode not in STEM_MODES:
                continue
            tr = self._stem_tracks.get(la.stem or "")
            if tr is None:
                _warn_once(f"stem:{la.stem}", "audio: layer %s names stem %r with no audio", layer, la.stem)
                continue
            self._layers[layer] = _LayerCtl(la.mode, la.stem)  # type: ignore[arg-type]
            if la.mode == "gate":
                tr.gate = tr.gate or Ramp(1.0, up=(1 - GATE_FLOOR) / GATE_UP_S, down=(1 - GATE_FLOOR) / GATE_DOWN_S)
            else:
                tr.gains[layer] = Ramp(1.0, tau={"filter": TAU_FILTER, "level": TAU_LEVEL}.get(la.mode, TAU_LEVERS))
                if la.mode in ("filter", "levers"):
                    tr.cutoff = tr.cutoff or Ramp(self._open_hz, tau=TAU_FILTER)
        for tr in self._tracks:
            tr.ramps = [r for r in (tr.gate, *tr.gains.values()) if r is not None]
        # a stem named only by disallowed layers is silent
        used = {c.stem for c in self._layers.values()}
        for name in claimed - used:
            if name in self._stem_tracks:
                self._stem_tracks[name].muted = True
        for ctl in self._layers.values():
            if ctl.mode == "level":
                ctl.v = LEVEL_REST_V    # SPEC 10: before the first note
        self._retarget(snap=True)
        self._build_schedule(manifest, notes, modes)
        self._cmds.clear()
        for v in (*self._voices, *self._reserve):
            v.active = False
        self._loops.clear()
        self.frame, self._si = 0, 0

    def _build_schedule(self, manifest: AudioManifest, notes, modes: dict[str, str]) -> None:
        """Auto-layer sounds as (frame, action, layer, name, gain), like Game._auto plays them."""
        ev: list[tuple[int, int, str, str, str, float]] = []   # (frame, order, action, layer, name, gain)
        for n in notes:
            if modes.get(n.layer) != "auto" or n.layer in self.disabled:
                continue
            mode = manifest.mode(n.layer)
            f0 = round(n.t * self.sr)
            if n.kind == "riser" and mode == "riser":
                loop = manifest.oneshot_for(n.layer, n) or "riser"
                f1 = round(n.end * self.sr)
                ev += [(f0, 1, "start", n.layer, loop, 1.0), (f1, 0, "stop", n.layer, loop, 1.0),
                       (f1, 1, "play", n.layer, "impact", 1.0)]
            elif mode == "trigger" and not (n.kind == "gate" and n.listen):
                if name := manifest.oneshot_for(n.layer, n):
                    ev.append((f0, 1, "play", n.layer, name, n.vel if n.vel is not None else 1.0))
        ev.sort(key=lambda e: (e[0], e[1]))
        self._sched = [(a, layer, name, g) for _, _, a, layer, name, g in ev]
        self._sched_f = np.array([e[0] for e in ev], np.int64)

    # --- main-thread controls ---

    def set_layer(self, layer: str, *, alive: bool | None = None, on_road: bool | None = None,
                  v: float | None = None, v1: float | None = None) -> None:
        """Mix inputs of a stem layer: gate `alive`, filter `on_road`, level `v`, levers `v`/`v1`."""
        ctl = self._layers.get(layer)
        if ctl is None:
            return
        if alive is not None:
            ctl.alive = alive
        if on_road is not None:
            ctl.on_road = on_road
        if v is not None:
            ctl.v = min(1.0, max(0.0, float(v)))
        if v1 is not None:
            ctl.v1 = min(1.0, max(0.0, float(v1)))
        self._retarget()

    @property
    def layer_modes(self) -> dict[str, str]:
        """Stem layers wired in this mixer -> their manifest mode."""
        return {k: c.mode for k, c in self._layers.items()}

    def _retarget(self, snap: bool = False) -> None:
        by_stem: dict[str, list[tuple[str, _LayerCtl]]] = {}
        for layer, c in self._layers.items():
            by_stem.setdefault(c.stem, []).append((layer, c))
        lo, hi = LEVER_HZ
        for stem, ctls in by_stem.items():
            tr = self._stem_tracks[stem]
            gates = [c.alive for _, c in ctls if c.mode == "gate"]
            cut = self._open_hz
            for layer, c in ctls:
                if c.mode == "filter":
                    g, cut = (1.0, cut) if c.on_road else (FILTER_OFF_GAIN, min(cut, FILTER_OFF_HZ))
                elif c.mode == "level":
                    g = LEVEL_FLOOR + (1 - LEVEL_FLOOR) * c.v
                elif c.mode == "levers":
                    g, cut = LEVER_FLOOR + (1 - LEVER_FLOOR) * c.v, min(cut, lo * (hi / lo) ** c.v1)
                else:
                    continue
                tr.gains[layer].target = g
            if tr.gate is not None:
                tr.gate.target = GATE_FLOOR + (1 - GATE_FLOOR) * sum(gates) / len(gates)
            if tr.cutoff is not None:
                tr.cutoff.target = cut
            if snap:
                for r in (tr.gate, tr.cutoff, *tr.gains.values()):
                    if r is not None:
                        r.value = r.target

    def set_volume(self, v: float) -> None:
        self.master.target = min(1.0, max(0.0, float(v)))

    def play(self, name: str, vel: float = 1.0) -> bool:
        """Start one-shot `name` at the next block. False when there is no such sample."""
        if name not in self._shots:
            return False
        self._cmds.append(("play", None, name, float(vel)))
        return True

    def start_loop(self, layer: str, name: str, vel: float = 1.0) -> bool:
        if name not in self._shots:
            return False
        self._cmds.append(("start", layer, name, float(vel)))
        return True

    def stop_loop(self, layer: str) -> None:
        self._cmds.append(("stop", layer, "", 0.0))

    def seek(self, frame: int) -> None:
        self.clock_seq += 1
        self._cmds.append(("seek", self.clock_seq, int(frame), 0.0))

    def set_paused(self, paused: bool, frame: int | None = None) -> None:
        """Pause or resume with a fade. A resume with `frame` restarts there when the
        mixer is silent (the engine passes the frame of its frozen clock)."""
        self.clock_seq += 1
        self._cmds.append(("pause" if paused else "resume", self.clock_seq, frame, 0.0))

    # --- real-time ---

    def render(self, n: int) -> np.ndarray:
        """Next n frames as a new (n, 2) array (tests; the callback uses render_into)."""
        out = np.zeros((n, 2), np.float32)
        self.render_into(out)
        return out

    def render_into(self, out: np.ndarray) -> tuple[int, int, bool]:
        """Fill `out` (n, 2) with the next frames, in blocks of at most max_block.
        Returns (song frame of out[0], applied clock_seq, whether it was playing),
        read after the block's own queue drain so the three always agree."""
        i, n = 0, len(out)
        first = (self.frame, self.applied_seq, False)
        while i < n:
            m = min(self.max_block, n - i)
            self._drain()
            if i == 0:
                first = (self.frame, self.applied_seq, not self.paused)
            out[i:i + m] = self._block(m)
            i += m
        return first

    def _block(self, n: int) -> np.ndarray:
        out = self._out[:n]
        out.fill(0.0)
        if self.paused:
            return out
        f0 = self.frame
        try:
            for tr in self._tracks:
                self._mix_track(tr, out, f0, n)
            self._fire_schedule(f0, n)
            for v in self._voices:
                if v.active:
                    self._mix_voice(v, out, n)
            for v in self._reserve:
                if v.active:
                    self._mix_voice(v, out, n)
            g = self._g[:n]
            self.master.fill(g, self.sr, self._idx)
            np.multiply(out, g[:, None], out=out)
            ok = True
        except Exception:  # never stop the clock, the transport or the stream
            ok = False
            self._si = int(np.searchsorted(self._sched_f, f0 + n))   # drop, never play late
            self.render_errors += 1
            if "render" not in _warned:
                _warned.add("render")
                log.exception("audio: render failed; block silenced")
        g = self._g[:n]
        self.transport.fill(g, self.sr, self._idx)
        if ok:
            np.multiply(out, g[:, None], out=out)
            np.clip(out, -1.0, 1.0, out=out)
        else:
            out.fill(0.0)
        self.frame = f0 + n
        if self._pending_seek is not None or self._pending_resume is not None:
            self._seek_elapsed += n
        if self.transport.value == 0.0 and self.transport.target == 0.0:
            if self._pending_seek is not None:
                self._apply_seek(*self._pending_seek)
            elif self._pending_resume is not None:
                self._rewind(self._pending_resume + self._seek_elapsed)
                self._seek_elapsed = 0
                self._pending_resume = None
                self.applied_seq = self._drained_seq
                self.transport.target = 1.0
            if self._pausing:
                self._pausing, self.paused = False, True
        return out

    def _drain(self) -> None:
        while self._cmds:
            cmd, a, b, c = self._cmds.popleft()
            if cmd == "stop":
                self._stop(a)
            elif cmd in ("play", "start"):
                if self.paused or self._pausing:
                    continue        # sounds queued during pause are dropped
                if cmd == "play":
                    self._start_voice(self._shots[b], c, loop=False)
                else:
                    self._stop(a)
                    self._start_voice(self._shots[b], c, loop=True, layer=a)
            elif cmd == "seek":
                self._drained_seq = a
                self._pending_resume = None
                if self.paused or self.transport.value == 0.0:
                    self._apply_seek(b, a)
                else:
                    if self._pending_seek is None:
                        self._seek_elapsed = 0
                    self._pending_seek = (b, a)
                    self.transport.target = 0.0
            elif cmd == "pause":
                self._drained_seq = a
                self._pending_resume = None
                if self._pending_seek is None:
                    self.applied_seq = a
                    self._seek_elapsed = 0  # frames counted for the dropped resume
                if not self.paused:
                    self._pausing = True
                    self.transport.target = 0.0
                    if self.transport.value == 0.0 and self._pending_seek is None:
                        self._pausing, self.paused = False, True
            else:
                self._drained_seq = a
                if self._pending_seek is None:
                    if b is not None and not (self.paused or self.transport.value == 0.0):
                        self._pending_resume = b    # mid-fade: finish the fade, rewind, fade in
                        self._seek_elapsed = 0
                    else:
                        self.applied_seq = a
                        if b is not None:
                            self._rewind(b)         # resume where the clock froze, not after the fade
                        self.transport.target = 1.0
                self.paused = self._pausing = False

    def _rewind(self, frame: int) -> None:
        """Move a silent (faded) mixer to `frame`, keeping its voices."""
        self.frame = frame
        self._si = int(np.searchsorted(self._sched_f, frame))
        for tr in self._tracks:
            tr.filtering = False

    def _apply_seek(self, frame: int, seq: int) -> None:
        frame += self._seek_elapsed     # the song moved on while the old audio faded out
        self._seek_elapsed = 0
        self.frame = frame
        self._si = int(np.searchsorted(self._sched_f, frame))
        for v in (*self._voices, *self._reserve):
            v.active = False
        self._loops.clear()
        for tr in self._tracks:
            tr.filtering = False
        self._pending_seek = None
        self.applied_seq = max(seq, self._drained_seq)
        if not self._pausing:
            self.transport.target = 1.0

    def _start_voice(self, data: np.ndarray, gain: float, loop: bool, delay: int = 0,
                     layer: str = "") -> _Voice | None:
        v = None
        for x in self._voices:
            if not x.active:
                v = x
                break
        if v is None:
            v = self._victim()
            if v is None:
                _warn_once("voices", "audio: all voices busy; a sound was dropped")
                return None
            self._fade_out_copy(v)
        if v.layer and self._loops.get(v.layer) is v:
            del self._loops[v.layer]
        self._vseq += 1
        v.data, v.pos, v.delay, v.gain, v.loop, v.fade, v.stop_in, v.active, v.seq, v.layer = (
            data, 0, delay, max(0.0, min(1.0, gain)), loop, 0, -1, True, self._vseq, layer)
        if layer:
            self._loops[layer] = v
        return v

    def _victim(self) -> _Voice | None:
        """Oldest one-shot that is not a loop and not among the KEEP_RECENT newest."""
        best = None
        newest_kept = self._vseq - KEEP_RECENT
        for x in self._voices:
            if not x.loop and x.seq <= newest_kept and (best is None or x.seq < best.seq):
                best = x
        return best

    def _fade_out_copy(self, v: _Voice) -> None:
        """Move a stolen voice to a reserve slot where it fades over STEAL_FADE_S."""
        for r in self._reserve:
            if not r.active:
                r.copy_from(v)
                n = max(1, round(STEAL_FADE_S * self.sr))
                if not r.fade or r.fade > n:
                    r.fade = r.fade_len = n
                r.stop_in, r.layer = -1, ""
                return

    def _stop(self, layer: str, at: int = 0) -> None:
        """Fade out the layer's loop from offset `at` of the current block."""
        v = self._loops.pop(layer, None)
        if v is not None and v.active and not v.fade:
            v.stop_in = at

    def _fire_schedule(self, f0: int, n: int) -> None:
        end = f0 + n
        while self._si < len(self._sched) and self._sched_f[self._si] < end:
            f = int(self._sched_f[self._si])
            action, layer, name, gain = self._sched[self._si]
            self._si += 1
            at = max(0, f - f0)
            if action == "stop":
                self._stop(layer, at)
                continue
            data = self._shots.get(name)
            if data is None:
                _warn_once(f"shot:{name}", "audio: no sample named %r; skipped", name)
                continue
            if action == "start":
                self._stop(layer, at)
                self._start_voice(data, gain, loop=True, delay=at, layer=layer)
            else:
                self._start_voice(data, gain, loop=False, delay=at)

    def _coeffs(self, fc: float) -> tuple[np.ndarray, np.ndarray]:
        """Biquad (b, a) for a cutoff quantised to 1/CUTOFF_STEPS octave."""
        q = round(math.log2(max(fc, 1.0)) * CUTOFF_STEPS)
        c = self._coef.get(q)
        if c is None:
            c = self._coef[q] = lowpass_coeffs(2.0 ** (q / CUTOFF_STEPS), self.sr)
        return c

    def _filter(self, tr: _Track, b: np.ndarray, a: np.ndarray, seg: np.ndarray, dst: np.ndarray) -> None:
        """Low-pass `seg` into `dst`. The state is rebuilt from the last two inputs and
        outputs for the current coefficients (direct form I), so a cutoff change between
        sub-blocks does not click."""
        h, zi = tr.hist, self._zi
        np.multiply(h[0], b[1], out=zi[0])
        zi[0] += b[2] * h[1] - a[1] * h[2] - a[2] * h[3]
        np.multiply(h[0], b[2], out=zi[1])
        zi[1] -= a[2] * h[2]
        dst[:], _ = self._lfilter(b, a, seg, axis=0, zi=zi)
        if len(seg) >= 2:
            h[0], h[1], h[2], h[3] = seg[-1], seg[-2], dst[-1], dst[-2]
        else:
            h[1], h[0], h[3], h[2] = h[0], seg[-1], h[2], dst[-1]

    def _mix_track(self, tr: _Track, out: np.ndarray, f0: int, n: int) -> None:
        sr, idx, g, g2 = self.sr, self._idx, self._g[:n], self._g2[:n]
        for k, r in enumerate(tr.ramps):   # advance ramps even while the stem is silent
            r.fill(g if k == 0 else g2, sr, idx)
            if k:
                g *= g2
        a = max(0, -f0)
        p = max(0, f0)
        m = min(n - a, len(tr.data) - p)
        audible = m > 0 and not tr.muted
        cut = tr.cutoff
        if cut is None or cut.value == cut.target == self._open_hz:
            tr.filtering = False           # open and at rest: bypass
            if not audible:
                return
            src = tr.data[p:p + m]
        else:
            if not audible:
                tr.filtering = False
                cut.step(n, sr)
                return
            src = self._flt[a:a + m]
            j = 0
            while j < n:
                k = n - j if cut.value == cut.target else min(FILTER_SUB, n - j)
                b_, a_ = self._coeffs(cut.value)
                cut.step(k, sr)
                s, e = max(j, a), min(j + k, a + m)
                if s < e:
                    i0 = p + s - a
                    if not tr.filtering:   # enter from bypass: history = the raw signal, no click
                        tr.hist[0] = tr.hist[2] = tr.data[i0 - 1] if i0 >= 1 else 0.0
                        tr.hist[1] = tr.hist[3] = tr.data[i0 - 2] if i0 >= 2 else 0.0
                        tr.filtering = True
                    self._filter(tr, b_, a_, tr.data[i0:i0 + e - s], src[s - a:e - a])
                j += k
        dst = self._tmp[a:a + m]
        if tr.ramps:
            np.multiply(src, g[a:a + m, None], out=dst)
            out[a:a + m] += dst
        else:
            out[a:a + m] += src

    def _mix_voice(self, v: _Voice, out: np.ndarray, n: int) -> None:
        i = v.delay
        v.delay = 0
        data, L = v.data, len(v.data)
        while i < n and v.active:
            if 0 <= v.stop_in <= i:
                v.stop_in = -1
                v.fade = v.fade_len = max(1, round(LOOP_FADE_S * self.sr))
            if v.pos >= L:
                if v.loop and L:
                    v.pos = 0
                else:
                    v.active = False
                    break
            m = min(n - i, L - v.pos)
            if v.stop_in > i:
                m = min(m, v.stop_in - i)
            if v.fade:
                m = min(m, v.fade)
            dst = self._tmp[:m]
            np.multiply(data[v.pos:v.pos + m], v.gain, out=dst)
            if v.fade:
                env = self._g2[:m]
                np.subtract(v.fade + 1, self._idx[:m], out=env)
                env *= 1.0 / v.fade_len
                dst *= env[:, None]
                v.fade -= m
                if v.fade <= 0:
                    v.active = False
            out[i:i + m] += dst
            v.pos += m
            i += m
        if v.stop_in >= 0:
            v.stop_in = max(0, v.stop_in - n)


# --- clock ---

def clock_time(frame0: int, frames: int, sr: int, dac_time: float, now: float) -> float:
    """Song seconds at the DAC now: the block starting at song frame `frame0` reaches
    the DAC at `dac_time`; interpolate with the same clock's `now`, clamped to that
    block (never ahead of frames rendered)."""
    return frame0 / sr + min(now - dac_time, frames / sr)


class AudioEngine:
    """Mixer on a sounddevice output stream. The song clock `time()` is frames played
    minus output latency, monotonic from the first call; the Game applies the user's
    Config.audio_offset.

    Use: `load(chart, layer_modes)`, `start(at)`, then each frame `handle(events)`,
    `update(snapshot)` and read `time()`. `stream_factory` replaces sd.OutputStream
    (tests pass a fake). `load` builds a new Mixer and swaps it in, so it is safe
    while the stream runs."""

    def __init__(self, settings: AudioSettings | None = None, volume: float = 0.8,
                 stream_factory=None, samplerate: int | None = None, perf_counter=_time.perf_counter):
        self.settings = settings or AudioSettings()
        self._factory = stream_factory
        self._perf = perf_counter
        self.sr = int(samplerate or self.settings.samplerate or self._device_rate())
        self._max_block = max(8192, self.settings.blocksize)
        self.mixer = Mixer(self.sr, self._max_block, volume=volume)
        self.modes: dict[str, str] = {}
        self.underruns = 0
        self._stream = None
        self._running = False
        # (mixer, seq, frame0, frames, dac_time, uses_perf_counter) of the last played block
        self._clock: tuple | None = None
        self._frozen = 0.0     # song seconds while no valid clock
        self._last = 0.0       # monotonic floor of time()
        self._paused = False   # paused by pause(): resume rewinds to the frozen clock
        self._prev_stamp = (-math.inf, 0)   # (perf stamp, frames) of the last block
        self._lead: float | None = None     # measured DAC lead of the last played block (seconds)
        # (mixer, clock_seq, stream time) of the last seek/start/resume: time() runs on from the
        # frozen value until a block of the new position plays, as the mixer lands the jump later
        self._hold: tuple | None = None
        self._expr = LEVEL_REST_V
        self._levers = [1.0, 1.0]

    def _device_rate(self) -> int:
        if self._factory is not None:
            return 48000
        import sounddevice as sd

        return int(sd.query_devices(self.settings.device, "output")["default_samplerate"])

    @property
    def render_errors(self) -> int:
        return self.mixer.render_errors

    # --- song ---

    def load(self, chart: Chart, layer_modes: dict[str, str], audio: SongAudio | None = None) -> None:
        """Load the chart's audio (files, or arrays from `audio` at any rate and channel
        count) into a new Mixer, then swap it in. A playing mixer fades out first.
        The song waits for `start`."""
        old = self.mixer
        fading = self._stream is not None and not old.paused
        if fading:
            old.set_paused(True)
            deadline = self._perf() + TRANSPORT_FADE_S + 2 * self.settings.blocksize / self.sr + self._latency()
        mixer = Mixer(self.sr, self._max_block, volume=old.master.target)
        mixer.setup(chart.audio, audio or load_song_audio(chart, self.sr), chart.notes, layer_modes)
        while fading and not old.paused and self._perf() < deadline:
            _time.sleep(0.002)
        self._running, self._paused, self._clock = False, False, None
        self._freeze(-self._latency())
        self.modes = dict(layer_modes)
        self.mixer = mixer
        self._reset_holds()

    def start(self, at: float = 0.0) -> None:
        """Play from song time `at` (negative = lead-in silence). Opens the stream once.
        The only call that plays again after `stop()`."""
        if self._stream is None:
            self._stream = self._open()
            self._stream.start()
        self.seek(at)
        self._running, self._paused = True, False
        self.mixer.set_paused(False)
        self._hold_from_now()

    def _open(self):
        s = self.settings
        kw = dict(samplerate=self.sr, blocksize=s.blocksize, channels=2, dtype="float32",
                  latency=s.latency, callback=self._callback)
        if self._factory is not None:
            return self._factory(**kw)
        import sounddevice as sd

        return sd.OutputStream(device=s.device, **kw)

    def _stream_latency(self) -> float:
        return float(self._stream.latency) if self._stream is not None else 0.0

    def _latency(self) -> float:
        """Device lead: measured from the last played block, else the stream's latency."""
        return self._lead if self._lead is not None else self._stream_latency()

    def _freeze(self, t: float) -> None:
        self._frozen = self._last = t

    def _now(self) -> float:
        c = self._clock
        if (c is not None and c[5]) or self._stream is None:
            return self._perf()
        return self._stream.time

    def _hold_from_now(self) -> None:
        self._hold = (self.mixer, self.mixer.clock_seq, self._now())

    def _reset_holds(self) -> None:
        """SPEC 10 values before the first note: level gain 0.55, levers open."""
        self._expr, self._levers = LEVEL_REST_V, [1.0, 1.0]
        for layer, mode in self.mixer.layer_modes.items():
            if mode == "level":
                self.mixer.set_layer(layer, v=self._expr)
            elif mode == "levers":
                self.mixer.set_layer(layer, v=1.0, v1=1.0)

    def pause(self) -> None:
        """Freeze the clock and fade out. No-op unless playing."""
        if not self._running:
            return
        self._freeze(self.time())
        self._running, self._paused = False, True
        self.mixer.set_paused(True)

    def resume(self) -> None:
        """Continue after `pause()` from the frozen clock: the audio restarts at the frame the
        clock froze on (the fade-out is replayed), so time() does not step. No-op otherwise."""
        if not self._paused:
            return
        frame = round((self._frozen + self._latency()) * self.sr)
        self._running, self._paused = True, False
        self.mixer.set_paused(False, frame)
        self._hold_from_now()

    def seek(self, t: float = 0.0) -> None:
        """Jump to song time `t`; time() holds at t - lead until that audio plays. Resets the
        held level/lever targets. Keeps the play/pause state."""
        self._freeze(t - self._latency())
        self._reset_holds()
        self.mixer.seek(round(t * self.sr))
        self._hold_from_now()

    def stop(self) -> None:
        if self._running:
            self._freeze(self.time())
        self._running = self._paused = False
        self.mixer.set_paused(True)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        lvl = logging.WARNING if self.underruns or self.render_errors else logging.INFO
        log.log(lvl, "audio: %d underruns, %d render errors this song (blocksize %d, latency %s)",
                self.underruns, self.render_errors, self.settings.blocksize, self.settings.latency)

    @property
    def volume(self) -> float:
        return self.mixer.master.target

    @volume.setter
    def volume(self, v: float) -> None:
        self.mixer.set_volume(v)

    # --- clock ---

    def time(self) -> float:
        """Song seconds at the speaker now: frames played minus output latency. Never decreases,
        except on seek."""
        c, m = self._clock, self.mixer
        if not self._running or self._stream is None:
            return self._frozen
        if c is not None and c[0] is m and c[1] == m.clock_seq:
            now = self._perf() if c[5] else self._stream.time
            t = clock_time(c[2], c[3], self.sr, c[4], now)
        elif (h := self._hold) is not None and h[0] is m and h[1] == m.clock_seq:
            cap = TRANSPORT_FADE_S + 2 * self.settings.blocksize / self.sr + self._latency()
            t = self._frozen + min(max(self._now() - h[2], 0.0), cap)
        else:
            return self._frozen
        if t > self._last:
            self._last = t
        return self._last

    def _callback(self, outdata, frames, time_info, status) -> None:
        stamp = self._perf()
        if status and status.output_underflow:
            self.underruns += 1
        m = self.mixer
        frame0, seq, played = m.render_into(outdata)
        if not played:
            return
        dac, cur, perf = time_info.outputBufferDacTime, time_info.currentTime, False
        if not cur:
            _warn_once("zero-time", "audio: host reports no stream timestamps; using perf_counter")
            prev, prev_n = self._prev_stamp
            if stamp - prev < 0.001:    # a burst of callbacks: space them by block length
                stamp = prev + prev_n / self.sr
            self._prev_stamp = (stamp, frames)
            cur, dac, perf = stamp, stamp + self._stream_latency(), True
        elif not dac:
            dac = cur + self._stream_latency()
        self._lead = dac - cur
        self._clock = (m, seq, frame0, frames, dac, perf)

    # --- per frame ---

    def handle(self, events: list[GameEvent]) -> None:
        """Play the game's Sound events now; cause "auto" is scheduled from the chart instead."""
        mixer = self.mixer
        for e in events:
            if not isinstance(e, Sound) or e.cause == "auto" or e.layer in mixer.disabled:
                continue
            if e.action == "stop":
                mixer.stop_loop(e.layer)
                continue
            ok = mixer.start_loop(e.layer, e.name, e.vel) if e.action == "start" else mixer.play(e.name, e.vel)
            if not ok:
                _warn_once(f"shot:{e.name}", "audio: no sample named %r; skipped", e.name)

    def update(self, snap: Snapshot) -> None:
        """Stem controls from this frame's snapshot, per SPEC 8.2 and 10. `level` and
        `levers` stems follow the control (player) or target (auto) only during a note of
        their layer; between notes they hold the last target of the note that ended
        (before the first note: gain 0.55, levers open)."""
        inp, mixer = snap.input, self.mixer
        in_expr = snap.expr_target is not None
        if in_expr:
            self._expr = snap.expr_target  # type: ignore[assignment]
        in_fader = [ft is not None for ft in snap.fader_targets[:2]]
        for i in (0, 1):
            if in_fader[i]:
                self._levers[i] = snap.fader_targets[i]  # type: ignore[assignment]
        for layer, mode in mixer.layer_modes.items():
            auto = self.modes.get(layer) == "auto"
            st = snap.layers.get(layer)
            if mode == "gate":
                mixer.set_layer(layer, alive=auto or st is None or st.alive)
            elif mode == "filter":
                mixer.set_layer(layer, on_road=auto or snap.on_road)
            elif mode == "level":
                mixer.set_layer(layer, v=inp.throttle if in_expr and not auto else self._expr)
            elif mode == "levers":
                v0 = inp.lever0 if in_fader[0] and not auto else self._levers[0]
                v1 = inp.lever1 if in_fader[1] and not auto else self._levers[1]
                mixer.set_layer(layer, v=v0, v1=v1)


class NullAudio:
    """Same interface and transport rules as AudioEngine, no device: a monotonic song clock,
    every call recorded in `calls` and every played Sound in `played`. For tests and --no-audio.
    Rules: `resume` acts only after `pause`; `pause` only while playing; only `start` plays
    after `stop` or `load`; `seek` keeps the play/pause state."""

    def __init__(self, settings: AudioSettings | None = None, volume: float = 0.8, clock=_time.monotonic):
        self.settings = settings or AudioSettings()
        self.volume = volume
        self.underruns = self.render_errors = 0
        self.calls: list[tuple] = []
        self.played: list = []
        self._clock = clock
        self._t0: float | None = None   # clock value at song time 0 while playing
        self._frozen = 0.0
        self._paused = False

    def load(self, chart: Chart, layer_modes: dict[str, str], audio: SongAudio | None = None) -> None:
        self.calls.append(("load", chart.title, dict(layer_modes)))
        self._t0, self._frozen, self._paused = None, 0.0, False

    def start(self, at: float = 0.0) -> None:
        self.calls.append(("start", at))
        self._t0, self._paused = self._clock() - at, False

    def pause(self) -> None:
        self.calls.append(("pause",))
        if self._t0 is not None:
            self._frozen, self._t0, self._paused = self.time(), None, True

    def resume(self) -> None:
        self.calls.append(("resume",))
        if self._paused:
            self._t0, self._paused = self._clock() - self._frozen, False

    def seek(self, t: float = 0.0) -> None:
        self.calls.append(("seek", t))
        self._frozen = t
        if self._t0 is not None:
            self._t0 = self._clock() - t

    def stop(self) -> None:
        self.calls.append(("stop",))
        self._frozen, self._t0, self._paused = self.time(), None, False

    def time(self) -> float:
        return self._clock() - self._t0 if self._t0 is not None else self._frozen

    def handle(self, events: list[GameEvent]) -> None:
        self.played += [e for e in events if isinstance(e, Sound) and e.cause != "auto"]

    def update(self, snap: Snapshot) -> None:
        self.calls.append(("update", snap.now))
