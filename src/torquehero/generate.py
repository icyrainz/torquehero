"""Chart generator: any song file -> chart.json (format v1), stems/ and a one-shot kit.

`torquehero gen SONG [-o DIR] [--difficulty easy|normal|hard] [--stems] [--bpm N] [--meter 3|4] [--force]`

Pipeline: `analyse()` turns audio into beats, sections, a melody pitch track and
percussion onsets; `build_chart()` turns that into a chart dict (pure, no I/O);
`generate()` loads the song, runs demucs when asked, writes every file.

Tempo is tracked in 8 s windows, so steady songs and slow drift work; free tempo
(rubato, no pulse) does not, and `--bpm` fixes half- or double-time errors.

Every note goes through one placement pass with a model of two hands and two feet
(SPEC 9): a note that does not fit the limbs is dropped, never forced. Difficulty
decides which layers are in the chart (SPEC 10) and how dense they are.

Without --stems the whole song is the backing stem and a copy of it is the melody
stem; percussion is charted only when the analysis is confident, and its one-shots
stay quiet. With --stems (demucs) the drums stem carries kick and hat (gate, kit
one-shots as reinforcement), melody follows vocals or other, expr follows the other
stem, faders follow the bass.

librosa, soundfile, demucs and torch are imported inside the functions that need them.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .chart import playability, validate_dict
from .config import Config, config_dir
from .state import DIFFICULTIES, DIFFICULTY_LAYERS, LAYERS

SR_OUT = 48000   # sample rate of every file written
SR_AN = 22050    # analysis sample rate
HOP = 512
DEMUCS_MODELS = ("htdemucs", "htdemucs_ft")  # four-stem family only
DEMUCS_STEMS = ("drums", "bass", "vocals", "other")
KIT = ("kick", "hat", "tom_l", "tom_r", *(f"stab{i}" for i in range(1, 7)), "riser", "impact", "dud")

TRAVEL = 0.4          # seconds a hand or foot needs between two controls (SPEC 9 rules 3, 7)
NOTE_MIN_GAP = 0.2    # seconds between two notes of one layer
REST_BARS = 8         # each foot layer rests at least one bar in every REST_BARS bars
SILENCE_DB = -50.0    # frames this far under the loudest frame are silence
MIN_MUSIC = 2.0       # seconds of non-silent audio needed to chart anything
PEAK_SUM = 10 ** (-3 / 20)   # sum of all stems peaks at or below -3 dBFS
PEAK_STEM = 10 ** (-1 / 20)  # no single stem above -1 dBFS
KICK_FLOOR = 0.02     # kick-band percussive power at an onset / mean mix power
HAT_FLOOR = 0.002
TRANSIENT = 3.0       # a drum onset: percussive band power / tonal band power at that instant
HAT_SHARE = 0.0015    # a hat onset: high-band percussive power / whole-frame mix power (a hat is noise;
                      # the click of a plucked lead is a sliver of its own spectrum)
CONFIDENT = 0.35      # drum onsets' 16th-grid fit needed to chart the layer; 0 = random phases, 1 = exact
                      # (see _lock). Random onsets score about 0 +- 0.08 at any tempo
LOCK_REFUSE = 0.35    # tempo lock under this (whole song) stops the generator unless --bpm is given
LOCK_WARN = 0.35      # an 8 s window under this is reported
ECHO_MARGIN = 0.95    # headroom under the echo rate limit

# notes per second, averaged over a section; 0 = layer not charted
DENSITY = {
    "easy": {"melody": 1.0, "kick": 0.5, "hat": 0.0},
    "normal": {"melody": 1.5, "kick": 1.0, "hat": 1.0},
    "hard": {"melody": 2.5, "kick": 2.0, "hat": 2.0},
}


@dataclass(frozen=True)
class Level:
    grid: int            # grid points per beat for gates and percussion
    risers: int
    spins: int           # even, so the turn offset returns to zero
    stab_every: int      # stabs on section downbeats, plus every N bars in loud sections (0: no stabs)
    toms: int            # toms before each section change (0, 2 or 4)
    echo: bool
    expr: bool
    faders: bool


LEVELS = {
    "easy": Level(1, 1, 0, 0, 0, False, False, False),
    "normal": Level(2, 2, 2, 8, 2, True, False, False),
    "hard": Level(4, 3, 4, 4, 4, True, True, True),
}


class GenError(Exception):
    """An expected failure, reported as one clear message and exit code 2."""


class StemsUnavailable(GenError):
    pass


STEMS_INSTALL = (
    "--stems needs the optional 'stems' extra (demucs + torch). Install it with:\n"
    "  uv sync --extra stems\n"
    "On Windows this pulls torch 2.11.0 with CUDA 12.8 from https://download.pytorch.org/whl/cu128 "
    "(set in pyproject). Without uv:\n"
    "  pip install demucs torch==2.11.0 torchaudio==2.11.0 --extra-index-url https://download.pytorch.org/whl/cu128"
)


@dataclass
class GenSettings:
    """Generator settings, persisted in `config_dir()/generate.json`."""
    lane_span: float = 0.85      # widest lane position a gate uses
    lead_in: float = 1.0         # no note before this many seconds
    demucs_model: str = "htdemucs"
    device: str = "auto"         # "auto", "cuda", "cpu"

    FILE = "generate.json"

    @classmethod
    def load(cls, path: str | Path | None = None) -> GenSettings:
        path = Path(path) if path else config_dir() / cls.FILE
        if not path.exists():
            return cls()
        try:
            d = json.loads(path.read_text())
            known = {f.name for f in fields(cls)}
            s = cls(**{k: v for k, v in d.items() if k in known})
        except (ValueError, TypeError, AttributeError) as e:
            raise GenError(f"{path} is not valid generator settings: {e}") from e
        if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in (s.lane_span, s.lead_in)) \
                or not 0 < s.lane_span <= 1 or s.lead_in < 0:
            raise GenError(f"{path}: lane_span must be in (0, 1] and lead_in >= 0")
        if s.demucs_model not in DEMUCS_MODELS:
            raise GenError(f"{path}: demucs_model must be one of {', '.join(DEMUCS_MODELS)}")
        if s.device not in ("auto", "cuda", "cpu"):
            raise GenError(f"{path}: device must be auto, cuda or cpu")
        return s

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else config_dir() / self.FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path


# --- analysis ---

@dataclass
class Analysis:
    length: float
    meter: int
    start: float                 # non-silent span
    end: float
    beat_times: np.ndarray       # every beat from before 0 to past the end
    beat0: int                   # index in beat_times of beat position 0
    beat_s: np.ndarray           # strength 0..1 per beat_times entry, 0 in silence
    phase: int                   # beat positions with (pos - phase) % meter == 0 are downbeats
    sections: list[dict]         # {t, weight, pc, music}; first at 0.0
    bar_times: np.ndarray        # downbeats inside the non-silent span
    bar_pc: np.ndarray           # dominant pitch class per bar
    f0_t: np.ndarray             # melody pitch frames
    midi: np.ndarray             # NaN where unvoiced
    mel_onsets: np.ndarray
    kicks: list[tuple[float, float]]   # (t, strength 0..1), after the energy floor
    hats: list[tuple[float, float]]
    kick_conf: float             # 0..1, share of onsets on the 16th grid (0 when too few)
    hat_conf: float
    env_t: np.ndarray            # analysis frame times
    loud: np.ndarray             # per frame: not silence
    db: np.ndarray | None = None  # per frame, dB under the loudest frame
    lock: float | None = 1.0     # tempo lock 0..1 over the music (None: too few onsets to judge)
    lock_bad: list[float] = field(default_factory=list)  # starts of 8 s windows with a poor lock
    expr_env: np.ndarray | None = None  # other-stem loudness 0..1
    bass_level: np.ndarray | None = None
    bass_bright: np.ndarray | None = None

    @property
    def period(self) -> float:
        return float(np.polyfit(np.arange(len(self.beat_times)), self.beat_times, 1)[0])

    @property
    def bpm(self) -> float:
        return 60.0 / self.period

    def b2t(self, pos):
        idx = np.arange(len(self.beat_times)) - self.beat0
        return np.interp(pos, idx, self.beat_times)

    def t2b(self, t):
        idx = np.arange(len(self.beat_times)) - self.beat0
        return np.interp(t, self.beat_times, idx)

    def pitch_at(self, t: float) -> float:
        m = self.midi[(self.f0_t >= t + 0.03) & (self.f0_t <= t + 0.12)]
        m = m[~np.isnan(m)]
        return float(np.median(m)) if m.size else math.nan

    def voiced(self, a: float, b: float) -> float:
        """Share of pitch frames in [a, b] with a melody note."""
        m = self.midi[(self.f0_t >= a) & (self.f0_t <= b)]
        return float(np.mean(~np.isnan(m))) if m.size else 0.0

    def level(self, a: float, b: float) -> float:
        """Mean dB (under the loudest frame) of [a, b]."""
        w = self.db[(self.env_t >= a) & (self.env_t <= b)] if self.db is not None else np.array([])
        return float(w.mean()) if w.size else 0.0

    def loud_span(self, a: float, b: float, frac: float = 0.8) -> bool:
        """True when at least `frac` of the frames in [a, b] are not silence."""
        i = int(np.searchsorted(self.env_t, a - 0.03))
        j = max(int(np.searchsorted(self.env_t, b + 0.03)), i + 1)
        w = self.loud[i:j]
        return bool(w.size) and float(w.mean()) >= frac


def _norm(v, lo_pct: float = 5, hi_pct: float = 95) -> np.ndarray:
    v = np.asarray(v, float)
    if v.size == 0:
        return v
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    if hi - lo < 1e-9:
        return np.full_like(v, 0.5)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def _mono_an(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    y = librosa.to_mono(np.asarray(y, np.float32))
    return librosa.resample(y, orig_sr=sr, target_sr=SR_AN) if sr != SR_AN else y


def _band_onsets(S2: np.ndarray, T2: np.ndarray, M: np.ndarray, freqs: np.ndarray, lo: float, hi: float,
                 floor: float, share: float = 0.0) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Drum onsets of one band as (t, strength 0..1), plus the band's onset envelope.
    `S2` is the power spectrogram of the percussive part, `T2` of the tonal part. An
    onset counts only when its band power reaches `floor` (absolute, set from the
    full mix) and is TRANSIENT times the tonal power in that band at that instant:
    the attack of a sustained note (a lead, a plucked bass) has its tone under it.
    With `share`, the band must also carry that share of the frame's mix power `M`."""
    import librosa

    sel = (freqs >= lo) & (freqs < hi)
    power, tonal = S2[sel].sum(axis=0), T2[sel].sum(axis=0)
    env = librosa.onset.onset_strength(S=librosa.power_to_db(S2[sel] + 1e-12), sr=SR_AN, hop_length=HOP)
    frames = [int(f) for f in librosa.onset.onset_detect(onset_envelope=env, sr=SR_AN, hop_length=HOP)
              if power[f:f + 3].max(initial=0.0) >= max(floor, TRANSIENT * tonal[max(0, f - 1):f + 3].mean(),
                                                         share * M[f:f + 3].max(initial=0.0))]
    if not frames:
        return [], env
    pk = np.array([power[f:f + 3].max() for f in frames])
    s = np.sqrt(pk / max(float(np.percentile(pk, 95)), 1e-12))
    ts = librosa.frames_to_time(np.array(frames), sr=SR_AN, hop_length=HOP)
    return [(float(t), float(min(v, 1.0))) for t, v in zip(ts, s, strict=True)], env


def _refine(x: np.ndarray, lo: float, hi: float,
            onsets: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Onset times sharpened with a 23 ms window: each moves to the steepest rise of the
    band's power within 60 ms (the 93 ms analysis window fires early on a strong attack),
    and onsets that land within 60 ms of each other merge, keeping the stronger."""
    import librosa

    S = np.abs(librosa.stft(x, n_fft=512, hop_length=64)) ** 2
    freqs = librosa.fft_frequencies(sr=SR_AN, n_fft=512)
    power = S[(freqs >= lo) & (freqs < hi)].sum(axis=0)
    rise = np.diff(power, prepend=power[:1])
    tt = librosa.frames_to_time(np.arange(len(power)), sr=SR_AN, hop_length=64)
    out: list[tuple[float, float]] = []
    for t, s in onsets:
        w = (tt >= t - 0.06) & (tt <= t + 0.06)
        t = float(tt[w][np.argmax(rise[w])]) if w.any() else t
        if out and t - out[-1][0] < 0.06:
            if s > out[-1][1]:
                out[-1] = (t, s)
        else:
            out.append((t, s))
    return out


def _fit(t_env: np.ndarray, env: np.ndarray, a: float, b: float, periods: np.ndarray) -> tuple[float, float, float]:
    """(score, period, beat time) of the grid in [a, b] that lands on the most onset energy."""
    best = (-1.0, float(periods[0]), a)
    for p in periods:
        phases = np.arange(0.0, p, 0.004)
        grid = a + phases[:, None] + np.arange(max(int((b - a) / p), 1) + 1)[None, :] * p
        score = np.interp(grid, t_env, env, left=0.0, right=0.0).mean(axis=1)
        i = int(np.argmax(score))
        if score[i] > best[0]:
            best = (float(score[i]), float(p), float(a + phases[i]))
    return best


def _track_beats(env: np.ndarray, start: float, end: float, length: float, p0: float,
                 span: float) -> np.ndarray:
    """Beat times from before 0 to past `length`. One global fit of period (within
    `span` of `p0`) and phase, re-fitted in 8 s windows every 4 s (within 4 %); beats
    step by the local period and are pulled halfway toward the local grid, then
    nudged onto nearby peaks. `_lock` says afterwards whether the grid fits."""
    fps = SR_AN / HOP
    t_env = (np.arange(len(env)) - 1) / fps  # flux peaks one frame after the attack
    _, p, anchor = _fit(t_env, env, start, end, p0 * np.linspace(1 - span, 1 + span, 201))
    centers = np.arange(start + 4.0, end - 4.0 + 1e-9, 4.0)
    if len(centers) == 0:
        centers = np.array([(start + end) / 2])
    wins = []
    for c in centers:
        a, b = max(start, c - 4.0), min(end, c + 4.0)
        if b - a > 4 * p:
            _, pw, aw = _fit(t_env, env, a, b, p * np.linspace(1 - min(span, 0.04), 1 + min(span, 0.04), 81))
        else:
            pw, aw = p, anchor
        wins.append((pw, aw))
    pws = np.array([w[0] for w in wins])

    def local_grid(t: float) -> float:
        pw, aw = wins[int(np.argmin(np.abs(centers - t)))]
        return aw + round((t - aw) / pw) * pw

    def step(t: float, sign: int) -> float:
        pw = float(np.interp(t, centers, pws))
        pred = t + sign * pw
        d = local_grid(pred) - pred
        return pred + 0.5 * d if abs(d) < 0.3 * pw else pred

    first = local_grid(start)
    fwd, back = [first], [first]
    while fwd[-1] < length + 2 * p:
        fwd.append(step(fwd[-1], 1))
    while back[-1] > -2 * p:
        back.append(step(back[-1], -1))
    grid = np.array(back[:0:-1] + fwd)

    off = np.full(len(grid), np.nan)
    floor = float(np.median(env[env > 0])) if (env > 0).any() else 0.0
    for i, g in enumerate(grid):
        w = (t_env >= g - 0.07) & (t_env <= g + 0.07)
        if w.any() and env[w].max() > floor:
            off[i] = t_env[w][np.argmax(env[w])] - g
    smooth = np.array([np.nanmedian(off[max(0, i - 4): i + 5]) if not np.isnan(off[max(0, i - 4): i + 5]).all()
                       else 0.0 for i in range(len(grid))])
    return grid + smooth


def _lock(onsets: np.ndarray, beats: np.ndarray, a: float = -math.inf, b: float = math.inf) -> float | None:
    """How well onsets in [a, b] sit on the 16th grid of `beats`: 1 - mean error /
    mean error of the same onsets with their phases shuffled (uniform over a beat).
    1 is exact, 0 is no better than chance. None with fewer than 8 onsets."""
    ts = onsets[(onsets >= a) & (onsets <= b)]
    if len(ts) < 8:
        return None
    pos = np.interp(ts, beats, np.arange(len(beats)))

    def err(p: np.ndarray) -> float:
        q = p * 4
        return float(np.mean(np.abs(q - np.round(q))))

    rng = np.random.default_rng(0)
    base = np.mean([err(pos + rng.uniform(0, 1, len(pos))) for _ in range(20)])
    return float(np.clip(1 - err(pos) / base, 0.0, 1.0)) if base > 0 else None


def analyse(mix: np.ndarray, sr: int, stems: dict[str, np.ndarray] | None = None, stems_sr: int | None = None,
            melody_stem: str = "other", bpm: float | None = None, meter: int = 4) -> Analysis:
    """Beats, sections, melody and percussion of a song. `mix` is an array at `sr`,
    `stems` arrays at `stems_sr` (mono or channels-first). Raises GenError when
    there is too little non-silent audio."""
    import librosa

    y = _mono_an(mix, sr)
    length = len(y) / SR_AN
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    ref_rms = max(float(rms.max()), 1e-10)
    db = librosa.amplitude_to_db(rms + 1e-10, ref=ref_rms)
    loud = db > SILENCE_DB
    env_t = librosa.times_like(rms, sr=SR_AN, hop_length=HOP)
    music_s = float(env_t[loud][-1] - env_t[loud][0]) if loud.any() else 0.0
    if music_s < MIN_MUSIC:
        raise GenError(f"only {music_s:.1f} s of non-silent audio; need at least {MIN_MUSIC:.0f} s")
    start, end = float(env_t[loud][0]), float(env_t[loud][-1])

    if stems:
        mel_src, drum_src = _mono_an(stems[melody_stem], stems_sr), _mono_an(stems["drums"], stems_sr)
        m = min(len(y), len(drum_src))
        tonal_src = y[:m] - drum_src[:m]
    else:
        tonal_src, drum_src = librosa.effects.hpss(y)
        mel_src = tonal_src

    # drums: band onsets above an absolute floor that stand out from the tonal content
    S2 = np.abs(librosa.stft(drum_src, n_fft=2048, hop_length=HOP)) ** 2
    T2 = np.abs(librosa.stft(tonal_src, n_fft=2048, hop_length=HOP)) ** 2
    mix_power = (np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)) ** 2).sum(axis=0)
    n = min(S2.shape[1], T2.shape[1], len(mix_power), len(loud))
    S2, T2 = S2[:, :n], T2[:, :n]
    ref = float(mix_power[:n][loud[:n]].mean())
    freqs = librosa.fft_frequencies(sr=SR_AN, n_fft=2048)
    kicks, kick_env = _band_onsets(S2, T2, mix_power[:n], freqs, 30, 150, KICK_FLOOR * ref)
    hats, _ = _band_onsets(S2, T2, mix_power[:n], freqs, 6000, 11000, HAT_FLOOR * ref, HAT_SHARE)
    kicks, hats = _refine(drum_src, 30, 150, kicks), _refine(drum_src, 6000, 11000, hats)

    # beats: onset energy with the kick band weighted in, silence masked out
    onset_env = librosa.onset.onset_strength(y=y, sr=SR_AN, hop_length=HOP)
    m = min(len(onset_env), len(kick_env), len(loud))
    env = (_norm(onset_env[:m], 0, 99) + _norm(kick_env[:m], 0, 99)) * loud[:m]
    if bpm:
        p0, span = 60.0 / bpm, 0.01
    else:
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env[:m] * loud[:m], sr=SR_AN, hop_length=HOP)
        p0, span = 60.0 / (float(np.atleast_1d(tempo)[0]) or 120.0), 0.05
    beat_times = _track_beats(env, start, end, length, p0, span)
    strong = librosa.onset.onset_detect(onset_envelope=onset_env * loud[: len(onset_env)], sr=SR_AN,
                                        hop_length=HOP)
    if len(strong):
        strong = strong[onset_env[strong] >= np.median(onset_env[strong])]
    onset_ts = (np.asarray(strong, float) - 1) * HOP / SR_AN  # the same one-frame flux latency as the fit
    lock = _lock(onset_ts, beat_times)
    lock_bad = [float(c) for c in np.arange(start, max(start, end - 8.0) + 1e-9, 4.0)
                if (w := _lock(onset_ts, beat_times, c, c + 8.0)) is not None and w < LOCK_WARN]
    beat0 = int(np.searchsorted(beat_times, 0.0))
    beat_frames = librosa.time_to_frames(beat_times, sr=SR_AN, hop_length=HOP).clip(0, m - 1)
    beat_s = np.clip(0.3 + 0.7 * _norm(onset_env[beat_frames]), 0.0, 1.0)
    pos = np.arange(len(beat_times)) - beat0
    low = S2[(freqs >= 30) & (freqs < 150)].sum(axis=0)  # the accent shows in power, not in dB flux
    lf = beat_frames.clip(0, len(low) - 1)
    peak = np.array([low[f:f + 3].max(initial=0.0) for f in lf])
    phase = int(np.argmax([peak[(pos - p) % meter == 0].sum() for p in range(meter)]))
    down = (pos - phase) % meter == 0
    beat_s[down] = np.maximum(beat_s[down], 0.8)
    beat_s[~loud[beat_frames]] = 0.0
    downbeats = beat_times[down]

    def confidence(onsets: list[tuple[float, float]]) -> float:
        inside = [t for t, _ in onsets if start <= t <= end]
        if len(inside) < 0.3 * (end - start):
            return 0.0
        return _lock(np.array(inside), beat_times) or 0.0  # how much better than random phases

    # melody pitch track
    with np.errstate(invalid="ignore"):  # librosa's autocorrelation on silent frames
        f0, voiced, _ = librosa.pyin(mel_src, fmin=100.0, fmax=1100.0, sr=SR_AN, frame_length=2048,
                                     hop_length=HOP)
    midi = np.where(voiced & ~np.isnan(f0), librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    f0_t = librosa.times_like(f0, sr=SR_AN, hop_length=HOP)
    mel_onsets = librosa.onset.onset_detect(y=mel_src, sr=SR_AN, hop_length=HOP, units="time")

    # sections: agglomerative segmentation of bar-level chroma, timbre and loudness in the music
    chroma = librosa.feature.chroma_stft(y=y, sr=SR_AN, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR_AN, n_mfcc=13, hop_length=HOP)
    n = min(chroma.shape[1], mfcc.shape[1], len(rms))
    z = lambda a: (a - a.mean(axis=-1, keepdims=True)) / (a.std(axis=-1, keepdims=True) + 1e-9)  # noqa: E731
    f_start = int(np.searchsorted(env_t, start))
    f_end = max(min(int(np.searchsorted(env_t, end)) + 1, n), f_start + 1)
    feats = np.zeros((12 + 13 + 1, n))  # standardised over the music only, so silence cannot flatten it
    fs = slice(f_start, f_end)
    feats[:, fs] = np.vstack([chroma[:, fs], z(mfcc[:, fs]), 2 * z(db[None, fs])])
    half_bar = meter * p0 / 2  # no sliver columns at either end of the music
    bar_times = downbeats[(downbeats > start + half_bar) & (downbeats < end - half_bar)]
    bar_frames = librosa.time_to_frames(bar_times, sr=SR_AN, hop_length=HOP)
    keep = (bar_frames > f_start) & (bar_frames < f_end)
    bar_times, bar_frames = bar_times[keep], bar_frames[keep]
    edges = np.concatenate([[f_start], bar_frames, [f_end]]).astype(int)
    spans = list(zip(edges, edges[1:], strict=False))
    cols = np.stack([feats[:, a:b].mean(axis=1) for a, b in spans], axis=1)
    col_t = np.concatenate([[0.0 if start < 1.0 else start], bar_times])
    bar_pc = np.array([int(np.argmax(chroma[:, a:b].mean(axis=1))) for a, b in spans])
    k = int(np.clip(round((end - start) / 20), 2, 8))
    bounds = (sorted(set(librosa.segment.agglomerative(cols, min(k, cols.shape[1])).tolist()) | {0})
              if cols.shape[1] >= 2 else [0])
    sec_edges = [0]
    for b in [*bounds[1:], cols.shape[1]]:
        if b - sec_edges[-1] >= 2:
            sec_edges.append(b)
    if len(sec_edges) == 1:
        sec_edges.append(cols.shape[1])
    sec_edges[-1] = cols.shape[1]
    music_db = db[f_start:f_end][loud[f_start:f_end]]
    lo, hi = np.percentile(music_db, [5, 95])
    sections = [{"t": 0.0, "weight": 0.0, "pc": 0, "music": False}] if start >= 1.0 else []
    for a, b in zip(sec_edges, sec_edges[1:], strict=False):
        fa, fb = edges[a], edges[b]
        w = loud[fa:fb]
        sec_db = 10 * np.log10(np.mean(rms[fa:fb][w] ** 2) / ref_rms ** 2 + 1e-20) if w.any() else lo
        weight = float(0.15 + 0.85 * np.clip((sec_db - lo) / (hi - lo), 0, 1)) if hi - lo > 1e-6 else 0.6
        sections.append({"t": float(col_t[a]), "weight": weight, "music": True,
                         "pc": int(np.argmax(chroma[:, fa:fb].mean(axis=1)))})
    if length - end >= 1.0:
        sections.append({"t": end, "weight": 0.0, "pc": 0, "music": False})

    an = Analysis(length=length, meter=meter, start=start, end=end, beat_times=beat_times, beat0=beat0,
                  beat_s=beat_s, phase=phase, sections=sections, bar_times=bar_times, bar_pc=bar_pc[1:],
                  f0_t=f0_t, midi=midi, mel_onsets=np.asarray(mel_onsets, float), kicks=kicks, hats=hats,
                  kick_conf=confidence(kicks), hat_conf=confidence(hats), env_t=env_t, loud=loud, db=db, lock=lock,
                  lock_bad=lock_bad)
    if stems:
        def loudness(name):
            src = _mono_an(stems[name], stems_sr)
            r = librosa.feature.rms(y=src, hop_length=HOP)[0]
            return np.resize(_norm(librosa.amplitude_to_db(r + 1e-9, ref=1.0)), len(env_t)), src
        if melody_stem != "other":
            an.expr_env, _ = loudness("other")
        an.bass_level, bass = loudness("bass")
        cent = librosa.feature.spectral_centroid(y=bass, sr=SR_AN, hop_length=HOP)[0]
        an.bass_bright = np.resize(_norm(np.log(cent + 1.0)), len(env_t))
    return an


# --- chart building (pure) ---

class Limbs:
    """Who is on which control, and when (SPEC 9). Hands "L", "R", feet "LF", "RF".
    A need is (limb, control, a, b). One limb may stay on one control across
    spans, but needs `travel` seconds between two different controls. A spin holds
    the wheel as control "spin", so toms and one-hand gates keep their distance."""

    def __init__(self, travel: float = TRAVEL) -> None:
        self.travel = travel
        self.use: list[tuple[str, str, float, float, object]] = []

    def fits(self, needs: list[tuple[str, str, float, float]]) -> bool:
        return all(c == ctrl or b + self.travel <= x + 1e-9 or y + self.travel <= a + 1e-9
                   for limb, ctrl, a, b in needs for lb, c, x, y, _ in self.use if lb == limb)

    def place(self, needs: list[tuple[str, str, float, float]], tag: object = None) -> bool:
        if not self.fits(needs):
            return False
        self.use += [(*n, tag) for n in needs]
        return True

    def remove(self, tag: object) -> None:
        self.use = [u for u in self.use if u[4] is not tag]

    def expr_at(self, t: float) -> bool:
        return any(c == "throttle" and x <= t <= y for _, c, x, y, _ in self.use)


def _r(v: float) -> float:
    return round(float(v), 4)


def manifest(lead: str | None, expr: bool, faders: bool) -> dict:
    """Audio manifest. `lead` None: no stems, the whole song is backing plus a lead copy.
    With stems, a stem that no charted layer drives goes to backing."""
    layers: dict[str, dict] = {
        "pads": {"oneshot": "stab{gate}", "mode": "trigger"},
        "fills": {"oneshot": "tom_{side}", "mode": "trigger"},
        "riser": {"oneshot": "riser", "mode": "riser"},
    }
    if lead is None:
        backing, stems = ["stems/backing.wav"], {"lead": "stems/lead.wav"}
        layers |= {"melody": {"stem": "lead", "mode": "filter"},
                   "kick": {"oneshot": "kick", "mode": "trigger"},
                   "hat": {"oneshot": "hat", "mode": "trigger"}}
    else:
        used = {lead, "drums"} | ({"other"} if expr else set()) | ({"bass"} if faders else set())
        backing = [f"stems/{s}.wav" for s in DEMUCS_STEMS if s not in used]
        stems = {s: f"stems/{s}.wav" for s in DEMUCS_STEMS if s in used}
        layers |= {"melody": {"stem": lead, "mode": "filter"},
                   "kick": {"stem": "drums", "mode": "gate", "oneshot": "kick"},
                   "hat": {"stem": "drums", "mode": "gate", "oneshot": "hat"}}
        if expr:
            layers["expr"] = {"stem": "other", "mode": "level"}
        if faders:
            layers["faders"] = {"stem": "bass", "mode": "levers"}
    return {"sr": SR_OUT, "backing": backing, "stems": stems,
            "oneshots": {k: f"oneshots/{k}.wav" for k in KIT}, "layers": layers}


def layer_modes(difficulty: str) -> dict[str, str]:
    """Default modes (defaults.layers): `you` for the layers the difficulty gives the player, `auto` for the rest."""
    return {layer: "you" if layer in DIFFICULTY_LAYERS[difficulty] else "auto" for layer in LAYERS}


def build_chart(an: Analysis, difficulty: str = "normal", *, title: str, artist: str = "",
                lead: str | None = None, settings: GenSettings | None = None,
                cfg: Config | None = None, report: list[str] | None = None) -> dict:
    """Chart dict (format v1) from an analysis. `lead` names the melody stem when the
    song was separated (None: no stems). `report` receives one line per featured note
    type: where it went, or why there is none. Raises GenError if the result is
    invalid or not playable (chart.playability)."""
    lv, dens, st, cfg = LEVELS[difficulty], DENSITY[difficulty], settings or GenSettings(), cfg or Config()
    steer = cfg.max_lane_rate[difficulty]
    report = report if report is not None else []
    length = math.floor(an.length * 1000) / 1000
    period, meter = an.period, an.meter
    b2t, t2b = an.b2t, an.t2b
    lo_t, hi_t = max(st.lead_in, an.start), min(length - 0.5, an.end)
    limbs = Limbs(cfg.limb_travel)
    notes: list[dict] = []
    busy: list[tuple[float, float]] = []      # echo, risers and spins do not overlap each other
    no_gates: list[tuple[float, float]] = []  # echo guard and spin margins (SPEC 9 rule 11)

    def free(a: float, b: float, spans: list) -> bool:
        return all(b <= x or a >= y for x, y in spans)

    def downbeat_at_or_after(t: float) -> int:
        p = math.ceil(float(t2b(t)) - 1e-6)
        return p + (an.phase - p) % meter

    def bar_of(t: float) -> int:
        return math.floor((float(t2b(t)) - an.phase + 1e-6) / meter)

    secs = [{**s, "t": _r(s["t"])} for s in an.sections]
    ends = [s["t"] for s in secs[1:]] + [length]
    music = [i for i, s in enumerate(secs) if s["music"]]

    def section_of(t: float) -> int:
        return max(i for i, s in enumerate(secs) if s["t"] <= t + 1e-9)

    # echo: two bars demonstrated, two played back; the quietest section with four bars of
    # melody, else the quietest four bars of melody anywhere
    echo = None
    if lv.echo:
        def echo_fits(p0: int) -> bool:
            a, b = float(b2t(p0)), float(b2t(p0 + 4 * meter))
            return lo_t <= a and a >= 2.0 and b <= hi_t and an.loud_span(a, b, 0.95) \
                and an.voiced(a, float(b2t(p0 + 2 * meter))) >= 0.5 and an.voiced(float(b2t(p0 + 2 * meter)), b) >= 0.5
        for i in sorted(music, key=lambda i: secs[i]["weight"]):
            p0 = downbeat_at_or_after(max(secs[i]["t"], lo_t, 2.0))
            if float(b2t(p0 + 4 * meter)) <= ends[i] and echo_fits(p0):
                echo = p0
                break
        if echo is None:
            first, last = downbeat_at_or_after(max(lo_t, 2.0)), math.floor(float(t2b(hi_t))) - 4 * meter
            fits = [p for p in range(first, last + 1, meter) if echo_fits(p)]
            if fits:
                echo = min(fits, key=lambda p: an.level(float(b2t(p)), float(b2t(p + 4 * meter))))
        if echo is None:
            report.append("echo: none, no four loud bars with a melody in both halves")
        else:
            busy.append((float(b2t(echo - 1)), float(b2t(echo + 4 * meter + 1))))
            no_gates.append(busy[-1])
    order = sorted(secs[i]["weight"] for i in music)
    for k, i in enumerate(music):
        secs[i]["name"] = ("intro" if k == 0 and len(music) > 2 else "outro" if k == len(music) - 1
                           and len(music) > 2 else "chorus" if secs[i]["weight"] >= order[len(order) // 2]
                           else "verse")
    for s in secs:
        s.setdefault("name", "silence")
    if echo is not None:
        secs[section_of(float(b2t(echo)))]["name"] = "echo"

    # risers before the biggest energy jumps, the drop on the section downbeat; if there is
    # no clear jump, one riser before the loudest section
    def place_riser(i: int) -> float | None:
        drop = _r(secs[i]["t"])
        for beats in (2 * meter, meter):
            t0 = _r(b2t(t2b(drop) - beats))
            dur = _r(drop - t0)
            if t0 >= lo_t and drop <= hi_t + 0.5 and free(t0, drop, busy) and an.loud_span(t0, drop) \
                    and limbs.place([("R", "handbrake", t0, t0 + dur), ("L", "wheel", t0, t0 + dur)]):
                notes.append({"t": t0, "kind": "riser", "layer": "riser", "dur": dur})
                busy.append((t0, drop))
                return drop
        return None

    jumps = sorted(((secs[i]["weight"] - secs[i - 1]["weight"], i) for i in music if i > 0 and secs[i - 1]["music"]),
                   reverse=True)
    drops = [d for jump, i in jumps[: lv.risers] if jump >= 0.15 and (d := place_riser(i)) is not None]
    if not drops:
        for i in sorted((i for i in music if i > 0), key=lambda i: -secs[i]["weight"]):
            if (d := place_riser(i)) is not None:
                drops.append(d)
                break
    report.append(f"risers: drops at {', '.join(f'{d:.1f} s' for d in drops)}" if drops else
                  "risers: none, no section start with a loud bar before it that the hands can reach")

    # spins on the longest held melody notes (3 beats and 1.5 s, then 2 beats, then 1.5 beats),
    # else a pair on the quietest bars of melody; an even count, alternating direction. A spin
    # holds both hands on the wheel as control "spin"; gates keep a beat away (no_gates)
    runs, run_start, ref = [], None, None
    for t, m in zip(an.f0_t, an.midi, strict=True):
        if run_start is not None and (math.isnan(m) or abs(m - ref) > 0.6):
            runs.append((run_start, t))
            run_start = None
        if run_start is None and not math.isnan(m):
            run_start, ref = t, m
    spins: list[dict] = []

    def place_spin(t0: float, longest: float, min_dur: float) -> bool:
        t0 = _r(t0)
        dur = min(longest, 4.0, 8 * period)
        while dur >= min_dur and not (an.loud_span(t0, t0 + dur) and t0 + dur <= hi_t):
            dur -= period / 2  # a held note fades: spin over its loud part only
        dur = _r(dur)
        if t0 < lo_t or dur < min_dur or not free(t0 - period, t0 + dur + period, busy):
            return False
        n = {"t": t0, "kind": "spin", "layer": "melody", "dur": dur}
        if not limbs.place([("L", "spin", t0, t0 + dur), ("R", "spin", t0, t0 + dur)], tag=n):
            return False
        spins.append(n)
        busy.append((t0 - period, t0 + dur + period))
        return True

    held_tiers = ((max(1.5, 3 * period), 1.2), (max(1.0, 2 * period), 1.0), (1.5 * period, 1.5 * period))
    for min_run, min_dur in held_tiers if lv.spins else ():
        if len(spins) >= 2:
            break
        for a, b in sorted((r for r in runs if r[1] - r[0] >= min_run), key=lambda r: r[0] - r[1]):
            if len(spins) >= lv.spins:
                break
            place_spin(float(b2t(round(float(t2b(a)) * lv.grid) / lv.grid)), b - a, min_dur)
    fallback = lv.spins and len(spins) < 2
    if fallback:
        bars = [p for p in range(downbeat_at_or_after(lo_t), math.floor(float(t2b(hi_t))) - meter + 1, meter)
                if an.voiced(float(b2t(p)), float(b2t(p + meter))) >= 0.5]
        for p in sorted(bars, key=lambda p: an.level(float(b2t(p)), float(b2t(p + meter)))):
            if len(spins) >= 2:
                break
            place_spin(float(b2t(p)), float(b2t(p + meter - 1)) - float(b2t(p)), max(1.2, 1.5 * period))
    if len(spins) % 2:
        last = min(spins, key=lambda n: n["dur"])
        spins.remove(last)
        limbs.remove(last)
        busy.remove((last["t"] - period, last["t"] + last["dur"] + period))
    spins.sort(key=lambda n: n["t"])
    for k, n in enumerate(spins):
        n["dir"] = "cw" if k % 2 == 0 else "ccw"
        no_gates.append((n["t"] - period, n["t"] + n["dur"] + period))
    notes += spins
    if lv.spins:
        report.append(f"spins: {len(spins)} at {', '.join(f'{n['t']:.1f} s' for n in spins)}"
                      + (" (not enough held notes: placed on the quietest bars of melody)" if fallback else "")
                      if spins else "spins: none, no two bars of melody clear of the echo, risers and stabs")

    # toms before section changes (a run needs both hands on the wheel), then stabs
    pattern = {0: [], 2: [(-1.0, "L"), (-0.5, "R")], 4: [(-1.0, "L"), (-0.75, "R"), (-0.5, "L"), (-0.25, "R")]}
    for i in music[1:]:
        run = [(_r(b2t(t2b(secs[i]["t"]) + off)), side) for off, side in pattern[lv.toms]]
        if run and run[0][0] >= max(lo_t, secs[i - 1]["t"] + period) and run[-1][0] <= hi_t \
                and an.loud_span(run[0][0], run[-1][0]) \
                and limbs.place([("L", "wheel", run[0][0], run[-1][0]), ("R", "wheel", run[0][0], run[-1][0])]):
            notes += [{"t": t, "kind": "tom", "layer": "fills", "side": s} for t, s in run]

    def place_stab(t: float, gate: int) -> bool:
        t = _r(t)
        if lo_t <= t <= hi_t and an.loud_span(t, t) and limbs.place([("R", "shifter", t, t), ("L", "wheel", t, t)]):
            notes.append({"t": t, "kind": "stab", "layer": "pads", "gate": gate})
            return True
        return False

    if lv.stab_every:
        bar_list = list(an.bar_times)
        for i in music:
            s = secs[i]
            bar = float(b2t(downbeat_at_or_after(s["t"] - 0.05)))
            if not place_stab(bar, s["pc"] % 6 + 1):
                place_stab(float(b2t(t2b(bar) + meter)), s["pc"] % 6 + 1)
            if s["weight"] >= 0.5:
                inner = [t for t in bar_list if bar < t < ends[i]]
                for j, bt in enumerate(inner):
                    if (j + 1) % lv.stab_every == 0:
                        place_stab(bt, int(an.bar_pc[bar_list.index(bt)]) % 6 + 1)

    # faders and expr: short featured passages, at most one per section, where the stem moves most
    def passages(env: np.ndarray, needs, make) -> None:
        for i in music:
            p_from = downbeat_at_or_after(max(secs[i]["t"], lo_t))
            cands = []
            for p in range(p_from, math.floor(float(t2b(min(ends[i], hi_t)))) - 2 * meter + 1, meter):
                t0, t1 = float(b2t(p)), float(b2t(p + 2 * meter - 0.5))
                w = env[(an.env_t >= t0) & (an.env_t <= t1)]
                if w.size and an.loud_span(t0, t1):
                    cands.append((float(np.std(w)), p))
            for _, p in sorted(cands, reverse=True):
                t0 = _r(b2t(p))
                dur = _r(float(b2t(p + 2 * meter - 0.5)) - t0)
                if limbs.place(needs(t0, t0 + dur)):
                    pts = sorted({min(_r(float(b2t(k)) - t0), dur) for k in np.arange(p, p + 2 * meter - 0.5)}
                                 | {dur})
                    notes.append(make(t0, dur, pts))
                    break

    def curve(env: np.ndarray, t0: float, pts: list[float]) -> list[list[float]]:
        return [[dt, _r(np.interp(t0 + dt, an.env_t, env))] for dt in pts]

    if lv.faders and an.bass_level is not None and an.bass_bright is not None:
        count = [0]

        def fader(t0, dur, pts):
            lever = count[0] % 2  # one lever per passage, alternating: the same hand moves both
            count[0] += 1
            env = an.bass_level if lever == 0 else an.bass_bright
            return {"t": t0, "kind": "fader", "layer": "faders", "dur": dur, "lever": lever,
                    "curve": curve(env, t0, pts)}

        passages(an.bass_level, lambda a, b: [("L", "levers", a, b), ("R", "wheel", a, b)], fader)
    if lv.expr and an.expr_env is not None:
        passages(an.expr_env, lambda a, b: [("RF", "throttle", a, b)],
                 lambda t0, dur, pts: {"t": t0, "kind": "expr", "layer": "expr", "dur": dur,
                                       "curve": curve(an.expr_env, t0, pts)})

    # kick and hat: confident band onsets snapped to the grid, a rest bar at most every
    # REST_BARS - 1 bars, then the strongest first up to the section's density budget
    def budget(i: int, per_s: float) -> int:
        a, b = max(secs[i]["t"], lo_t), min(ends[i], hi_t)
        if b <= a:
            return 0
        w = an.loud[(an.env_t >= a) & (an.env_t <= b)]
        return math.floor(per_s * (b - a) * (float(w.mean()) if w.size else 0.0) + 1e-9)

    def rests(cands: list[tuple[float, float]]) -> set[int]:
        if not cands:
            return set()
        strength: dict[int, float] = {}
        for t, s in cands:
            strength[bar_of(t)] = strength.get(bar_of(t), 0.0) + s
        first, last = min(strength), max(strength)
        out: set[int] = set()
        r = min(range(first, first + REST_BARS - 1), key=lambda b: (strength.get(b, 0.0), b))
        out.add(r)
        while last - r >= REST_BARS - 1:
            r = min(range(r + 4, r + REST_BARS), key=lambda b: (strength.get(b, 0.0), b))
            out.add(r)
        return out

    def percussion(onsets, per_s: float, place, make) -> None:
        if per_s <= 0:
            return
        cands: dict[float, float] = {}
        for t, s in onsets:
            g = _r(b2t(round(float(t2b(t)) * lv.grid) / lv.grid))
            if abs(g - t) <= 0.06 and lo_t <= g <= hi_t and an.loud_span(g, g):
                cands[g] = max(cands.get(g, 0.0), s)
        rest = rests(list(cands.items()))
        chosen: list[float] = []
        by_sec: dict[int, list[tuple[float, float]]] = {}
        for t, s in cands.items():
            if bar_of(t) not in rest:
                by_sec.setdefault(section_of(t), []).append((t, s))
        for i, cs in by_sec.items():
            left = budget(i, per_s)
            for t, s in sorted(cs, key=lambda c: (-c[1], c[0])):
                if left <= 0:
                    break
                if all(abs(t - u) >= NOTE_MIN_GAP - 1e-9 for u in chosen) and place(t):
                    chosen.append(t)
                    notes.append(make(t, s))
                    left -= 1

    def place_kick(t: float) -> bool:
        feet = ("LF",) if limbs.expr_at(t) else ("RF", "LF")  # SPEC 9 rule 6
        return any(limbs.place([(foot, "brake", t, t)]) for foot in feet)

    def kick_vel(s: float) -> float:
        return 0.3 + 0.7 * s if lead else 0.25 + 0.35 * s  # quiet reinforcement over the full song

    if an.kick_conf >= CONFIDENT:
        percussion(an.kicks, dens["kick"], place_kick,
                   lambda t, s: {"t": t, "kind": "kick", "layer": "kick", "vel": round(kick_vel(s), 3)})
    if an.hat_conf >= CONFIDENT:
        percussion(an.hats, dens["hat"], lambda t: limbs.place([("LF", "clutch", t, t)]),
                   lambda t, s: {"t": t, "kind": "hat", "layer": "hat"})

    # melody gates: strongest note starts per section, then thinned so no move beats `steer`
    voiced = an.midi[~np.isnan(an.midi)]
    lo, hi = (np.percentile(voiced, [5, 95]) if voiced.size else (60.0, 72.0))
    if hi - lo < 3:
        lo, hi = (lo + hi) / 2 - 1.5, (lo + hi) / 2 + 1.5

    def lane(m: float) -> float:
        return float(np.clip(st.lane_span * (2 * (m - lo) / (hi - lo) - 1), -st.lane_span, st.lane_span))

    def candidates(p_from: float, p_to: float) -> list[tuple[float, float, float]]:
        """(beat position, midi, score) of grid points where the melody starts a note."""
        out, prev, prev_t = [], math.nan, -math.inf
        for p in np.arange(math.ceil(p_from * lv.grid) / lv.grid, p_to, 1 / lv.grid):
            t = float(b2t(p))
            m = an.pitch_at(t)
            if math.isnan(m) or not an.loud_span(t, t):
                continue
            onset = bool(an.mel_onsets.size) and float(np.min(np.abs(an.mel_onsets - t))) <= 0.07
            rest = t - prev_t > 0.25 + period / lv.grid
            if onset or rest or abs(m - prev) >= 0.8:
                on_beat = abs(p - round(p)) < 1e-6
                score = 1.0 if math.isnan(prev) else onset + min(abs(m - prev), 12) / 12
                score += 0.5 * on_beat + 0.5 * (on_beat and (round(p) - an.phase) % meter == 0)
                out.append((float(p), m, score))
            prev, prev_t = m, t
        return out

    fixed: list[tuple[float, float, str]] = []
    if echo is not None:
        p0 = echo
        phrase_dur = float(b2t(p0 + 2 * meter)) - float(b2t(p0))
        cap = max(3, math.floor(dens["melody"] * phrase_dur))
        phrase = [(p, m) for p, m, _ in candidates(p0, p0 + 2 * meter - 0.5)]
        kept: list[tuple[float, float]] = []
        for p, m in phrase:
            if not kept or p - kept[-1][0] >= 0.5:
                kept.append((p, m))
        if len(kept) < 3:
            kept = [(float(p0 + k), math.nan) for k in range(2 * meter)]
            xs = [0.35 * math.sin(math.pi / 2 * k) for k in range(len(kept))]
        else:
            xs = [lane(m) for _, m in kept]
        step = max(1, math.ceil(len(kept) / cap))
        kept, xs = kept[::step][:cap], xs[::step][:cap]
        xs = [min(max(x, -0.5), 0.5) for x in xs]
        tl = [_r(b2t(p)) for p, _ in kept]
        tb = [_r(b2t(p + 2 * meter)) for p, _ in kept]
        echo_rate = ECHO_MARGIN * cfg.echo_max_deg_s / (1.5 * cfg.play_range_deg)

        def phrase_ok(xs: list[float]) -> bool:
            ts, xx = tl + tb, xs + xs
            return all(abs(xx[i] - xx[i - 1]) <= steer * (ts[i] - ts[i - 1]) * 0.999 for i in range(1, len(ts))) \
                and all(abs(xs[i] - xs[i - 1]) <= echo_rate * (tl[i] - tl[i - 1]) for i in range(1, len(tl)))

        for _ in range(40):
            if phrase_ok(xs):
                break
            xs = [x * 0.8 for x in xs]
        else:
            xs = [0.0] * len(xs)
        fixed = [(t, _r(x), "listen") for t, x in zip(tl, xs, strict=True)]
        fixed += [(t, _r(x), "blind") for t, x in zip(tb, xs, strict=True)]

    min_gap = max(max(1.0 / lv.grid, 0.5 if difficulty != "hard" else 0.25) * period, NOTE_MIN_GAP)
    picked: list[tuple[float, float]] = []
    by_sec: dict[int, list] = {}
    for p, m, score in candidates(float(t2b(lo_t)), float(t2b(hi_t))):
        t = _r(b2t(p))
        if free(t, t, no_gates):
            by_sec.setdefault(section_of(t), []).append((t, m, score))
    n_blind: dict[int, int] = {}
    for t, _, flag in fixed:
        n_blind[section_of(t)] = n_blind.get(section_of(t), 0) + (flag == "blind")
    for i, cs in by_sec.items():
        left = budget(i, dens["melody"]) - n_blind.get(i, 0)
        chosen: list[float] = []
        for t, m, _ in sorted(cs, key=lambda c: (-c[2], c[0])):
            if left <= 0:
                break
            if all(abs(t - u) >= min_gap - 1e-9 for u in chosen):
                chosen.append(t)
                picked.append((t, _r(lane(m))))
                left -= 1

    # thin, do not flatten: a gate that moves faster than `steer` from the one before is dropped
    keys = [(0.0, 0.0, "start")] + sorted([(t, x, None) for t, x in picked] + fixed)
    road: list[tuple[float, float, str | None]] = []
    for k in keys:
        def too_fast(a, b=k) -> bool:
            return abs(b[1] - a[1]) > steer * (b[0] - a[0]) + 1e-9
        if k[2] is not None:
            while road and road[-1][2] is None and too_fast(road[-1]):
                road.pop()
            road.append(k)
        elif not road or not too_fast(road[-1]):
            road.append(k)
    for t, x, flag in road[1:]:
        n = {"t": t, "kind": "gate", "layer": "melody", "x": x}
        if flag:
            n[flag] = True
        notes.append(n)

    notes.sort(key=lambda n: n["t"])
    beats = [{"t": _r(t), "s": round(float(s), 3)}
             for t, s in zip(an.beat_times, an.beat_s, strict=True) if 0 <= t <= length]
    kinds = {n["kind"] for n in notes}
    chart = {
        "format": 1, "title": title, "artist": artist, "bpm": round(an.bpm, 2), "length": length,
        "sections": [{"t": _r(s["t"]), "name": s["name"], "weight": round(s["weight"], 3)} for s in secs],
        "beats": beats, "road": [{"t": t, "x": x} for t, x, _ in road], "notes": notes,
        "audio": manifest(lead, "expr" in kinds, "fader" in kinds),
    }
    chart["defaults"] = {"layers": layer_modes(difficulty)}
    errors = validate_dict(chart, cfg.play_range_deg, cfg.echo_max_deg_s)
    if errors:
        raise GenError("generated chart is invalid (generator bug):\n  " + "\n  ".join(errors))
    unplayable = playability(chart, difficulty, cfg)
    if unplayable:
        raise GenError("generated chart breaks the limb budget (generator bug):\n  " + "\n  ".join(unplayable))
    counts = {k: sum(n["kind"] == k for n in notes) for k in ("gate", "kick", "hat", "stab", "tom", "expr", "fader")}
    report.append("notes: " + ", ".join(f"{v} {k}" for k, v in counts.items() if v))
    if lv.echo and echo is not None:
        report.append(f"echo: {float(b2t(echo)):.1f} s to {float(b2t(echo + 4 * meter)):.1f} s")
    if dens["kick"] and an.kick_conf < CONFIDENT:
        report.append(f"kick: none, no confident drum onsets in the low band (confidence {an.kick_conf:.2f})")
    if dens["hat"] and an.hat_conf < CONFIDENT:
        report.append(f"hat: none, no confident drum onsets in the high band (confidence {an.hat_conf:.2f})")
    return chart


# --- one-shot kit ---

def kit(sr: int = SR_OUT, percussion_gain: float = 1.0) -> dict[str, np.ndarray]:
    """The built-in one-shot kit, synthesised: kick, hat, toms, six stabs, riser loop, impact, dud.
    `percussion_gain` scales kick, hat and toms (quiet reinforcement over a full song)."""
    rng = np.random.default_rng(7)

    def tt(d: float) -> np.ndarray:
        return np.arange(int(d * sr)) / sr

    def sweep(f: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * np.cumsum(f) / sr)

    def tom(f0: float) -> np.ndarray:
        t = tt(0.35)
        return sweep(f0 * (1 + 0.5 * np.exp(-t / 0.05))) * np.exp(-t / 0.15)

    def stab(root: int) -> np.ndarray:
        t = tt(0.3)
        hz = [440.0 * 2 ** ((m - 69) / 12) for m in (root, root + 4, root + 7)]
        return sum(np.sin(2 * np.pi * f * k * t) / k for f in hz for k in range(1, 6)) * np.exp(-t / 0.1)

    t = tt(0.4)
    out = {"kick": sweep(45 + 110 * np.exp(-t / 0.04)) * np.exp(-t / 0.12)}
    t = tt(0.09)
    out["hat"] = np.diff(rng.standard_normal(len(t) + 1)) * np.exp(-t / 0.02)
    out["tom_l"], out["tom_r"] = tom(110.0), tom(150.0)
    for i, root in enumerate((60, 62, 64, 65, 67, 69), start=1):
        out[f"stab{i}"] = stab(root)
    t = tt(2.0)
    out["riser"] = 0.15 * np.diff(rng.standard_normal(len(t) + 1)) + sweep(300 * 2 ** (t / 2.0)) * 0.5
    t = tt(1.2)
    out["impact"] = (rng.standard_normal(len(t)) * np.exp(-t / 0.15) * 0.6
                     + np.sin(2 * np.pi * 50 * t) * np.exp(-t / 0.4))
    t = tt(0.12)
    out["dud"] = np.sign(np.sin(2 * np.pi * 110 * t)) * np.exp(-t / 0.04)
    fade = int(0.005 * sr)
    for k, v in out.items():
        gain = percussion_gain if k in ("kick", "hat", "tom_l", "tom_r") else 1.0
        v = v / (np.max(np.abs(v)) + 1e-9) * 0.8 * gain
        v[-fade:] *= np.linspace(1, 0, fade)
        if k == "riser":
            v[:fade] *= np.linspace(0, 1, fade)
        out[k] = v.astype(np.float32)
    return out


def headroom(parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One gain for every part so their sum peaks at or below -3 dBFS and no part
    above -1 dBFS (loud material is turned down, quiet material is left alone)."""
    total = float(np.max(np.abs(sum(parts.values())))) if parts else 0.0
    single = max((float(np.max(np.abs(v))) for v in parts.values()), default=0.0)
    g = min(1.0, PEAK_SUM / total if total > 0 else 1.0, PEAK_STEM / single if single > 0 else 1.0)
    return {k: (v * g).astype(np.float32) for k, v in parts.items()}


# --- stems ---

def demucs_separator(settings: GenSettings | None = None):
    """A separator (y, sr) -> ({source: channels-first array}, model sample rate).
    Imports torch and demucs here; raises StemsUnavailable when the extra is missing."""
    st = settings or GenSettings()
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as e:
        raise StemsUnavailable(f"{STEMS_INSTALL}\n(import failed: {e})") from e
    except OSError as e:  # a torch DLL that does not load (Windows)
        raise StemsUnavailable(f"torch is installed but failed to load: {e}. On Windows install the Microsoft "
                               f"Visual C++ Redistributable (x64) and check the CUDA wheel.\n{STEMS_INSTALL}") from e
    if st.device == "cuda" and not torch.cuda.is_available():
        raise GenError("generate.json asks for device 'cuda' but torch sees no CUDA GPU; set device to auto or cpu, "
                       "or install the CUDA wheel (uv sync --extra stems on Windows pulls it from the cu128 index)")

    def separate(y: np.ndarray, sr: int) -> tuple[dict[str, np.ndarray], int]:
        import librosa

        try:
            model = get_model(st.demucs_model)
        except Exception as e:  # noqa: BLE001 - network, cache and checkpoint errors all end here
            raise GenError(f"could not load demucs model {st.demucs_model!r} ({e.__class__.__name__}: {e}); "
                           f"the first run downloads it, so check the network") from e
        missing = set(DEMUCS_STEMS) - set(model.sources)
        if missing:
            raise GenError(f"demucs model {st.demucs_model!r} has sources {list(model.sources)}, "
                           f"missing {', '.join(sorted(missing))}")
        msr = model.samplerate
        x = np.asarray(y, np.float32)
        x = np.stack([x, x]) if x.ndim == 1 else x[:2]
        x = librosa.resample(x, orig_sr=sr, target_sr=msr) if sr != msr else x  # the one resample in
        wav = torch.from_numpy(np.ascontiguousarray(x))
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8

        def run(device: str):
            model.to(device).eval()
            with torch.no_grad():
                return apply_model(model, ((wav - mean) / std)[None], device=device, shifts=0, split=True,
                                   overlap=0.25, progress=True)[0]

        device = st.device if st.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            out = run(device)
        except RuntimeError as e:  # torch.cuda.OutOfMemoryError is a RuntimeError
            if device != "cuda" or "out of memory" not in str(e).lower():
                raise
            _log("CUDA out of memory; retrying on the CPU (much slower)")
            torch.cuda.empty_cache()
            out = run("cpu")
        out = (out * std + mean).cpu().numpy()
        return dict(zip(model.sources, out, strict=True)), msr

    return separate


# --- driver ---

def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load(song: Path) -> tuple[np.ndarray, int]:
    import librosa

    try:
        y, sr = librosa.load(song, sr=None, mono=False)
    except Exception as e:  # noqa: BLE001 - soundfile, audioread and ffmpeg each raise their own
        first = str(e).splitlines()[0] if str(e) else ""
        raise GenError(f"cannot decode {song.name}: {e.__class__.__name__}: {first}") from e
    y = np.stack([y, y]) if y.ndim == 1 else y[:2]
    if y.shape[-1] == 0:
        raise GenError(f"{song.name} has no audio")
    return y, int(sr)


GENERATOR = "torquehero gen"  # marks gen.json, so --force only ever replaces our own output


def _check_out(out: Path, song: Path, force: bool) -> None:
    """Refuse an output path that is a file, or an existing directory unless it is a
    previous output of this generator (gen.json from us, source file not inside) and
    --force is given. Nothing else is ever deleted."""
    if out.is_file():
        raise GenError(f"-o {out} is a file; give a directory")
    if not out.exists():
        return
    try:
        ours = json.loads((out / "gen.json").read_text()).get("generator") == GENERATOR
    except (OSError, ValueError, AttributeError):
        ours = False
    if not ours:
        raise GenError(f"{out} exists and is not a torquehero gen output; choose a new -o directory")
    if song.resolve().is_relative_to(out.resolve()):
        raise GenError(f"{out} contains the source file {song.name}; choose another -o directory")
    if not force:
        raise GenError(f"{out} holds an earlier chart; pass --force to replace it")


def generate(song: str | Path, out: str | Path | None = None, difficulty: str = "normal", stems: bool = False, *,
             separator=None, settings: GenSettings | None = None, cfg: Config | None = None,
             bpm: float | None = None, meter: int = 4, force: bool = False) -> Path:
    """Analyse `song`, write chart.json, gen.json, stems/ and oneshots/ into `out`
    (built in a temp dir, then renamed into place). Returns the chart path.
    `separator` replaces demucs (tests); `stems` without one uses demucs.
    `cfg` supplies the play range and echo limit the chart is validated against.
    Raises GenError for every expected failure."""
    import librosa
    import soundfile as sf

    st = settings or GenSettings.load()
    cfg = cfg or Config()
    song = Path(song)
    out = Path(out) if out else song.with_name(f"{song.stem}-torquehero")
    _check_out(out, song, force)
    _log(f"loading {song.name}")
    y, sr = _load(song)
    parts, psr, lead = None, None, None
    if stems:
        _log("separating stems (demucs)")
        parts, psr = (separator or demucs_separator(st))(y, sr)
        missing = set(DEMUCS_STEMS) - set(parts)
        if missing:
            raise GenError(f"separator returned {sorted(parts)}, missing {', '.join(sorted(missing))}")
        rms = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in parts.items()}
        lead = "vocals" if rms["vocals"] >= 0.25 * rms["other"] else "other"
    _log("analysing")
    an = analyse(y, sr, parts, psr, lead or "other", bpm, meter)
    if an.lock is not None and an.lock < LOCK_REFUSE:
        msg = (f"no steady beat found (tempo lock {an.lock:.2f}, need {LOCK_REFUSE}); free tempo and big tempo "
               f"changes are not supported")
        if bpm is None:
            raise GenError(f"{msg}. If the song has a steady beat, pass --bpm N (and --meter 3 for waltz time)")
        _log(f"warning: {msg}; charting anyway at --bpm {bpm:g}")
    elif an.lock_bad:
        _log("warning: the beat grid fits poorly around " + ", ".join(f"{t:.0f} s" for t in an.lock_bad[:6])
             + "; notes there may feel off the beat (--bpm may help)")
    report: list[str] = []
    chart = build_chart(an, difficulty, title=song.stem, lead=lead, settings=st, cfg=cfg, report=report)
    for line in report:
        _log(f"  {line}")

    def to_out(v: np.ndarray, from_sr: int) -> np.ndarray:
        v = np.asarray(v, np.float32)
        return librosa.resample(v, orig_sr=from_sr, target_sr=SR_OUT) if from_sr != SR_OUT else v

    if parts is None:
        mix = to_out(y, sr)
        audio = headroom({"backing": 0.5 * mix, "lead": 0.5 * mix})
    else:
        audio = headroom({k: to_out(v, psr) for k, v in parts.items()})  # the one resample out

    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        (tmp / "stems").mkdir(parents=True)
        (tmp / "oneshots").mkdir()
        named = set(chart["audio"]["backing"]) | set(chart["audio"]["stems"].values())
        for name, v in audio.items():
            rel = f"stems/{name}.wav"
            if rel not in named:  # an extra model source nothing names: mix it into the backing
                chart["audio"]["backing"].append(rel)
            sf.write(tmp / rel, v.T, SR_OUT, subtype="PCM_24")
        for name, v in kit(percussion_gain=1.0 if parts else 0.4).items():
            sf.write(tmp / "oneshots" / f"{name}.wav", v, SR_OUT, subtype="PCM_24")
        (tmp / "chart.json").write_text(json.dumps(chart, indent=1) + "\n")
        (tmp / "gen.json").write_text(json.dumps(
            {"generator": GENERATOR, "source": song.name, "difficulty": difficulty, "stems": stems, "melody_stem": lead,
             "bpm": chart["bpm"], "meter": meter, "layer_modes": layer_modes(difficulty)}, indent=1) + "\n")
        if out.exists():
            _check_out(out, song, force)  # again: the directory may have changed while we worked
            shutil.rmtree(out)
        tmp.rename(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out / "chart.json"


def _layers_flag(modes: dict[str, str]) -> str:
    you = ",".join(k for k, v in modes.items() if v == "you")
    auto = ",".join(k for k, v in modes.items() if v == "auto")
    return f"you:{you}" + (f" auto:{auto}" if auto else "")


def _run(args) -> int:
    song = Path(args.song)
    try:
        if not song.is_file():
            raise GenError(f"no such file: {song}")
        if args.bpm is not None and not 30 <= args.bpm <= 300:
            raise GenError("--bpm must be between 30 and 300")
        settings = GenSettings.load()
        try:
            cfg = Config.load()
        except (ValueError, TypeError) as e:
            raise GenError(f"cannot read the game config: {e}") from e
        if args.stems:
            demucs_separator(settings)
        path = generate(song, args.output, args.difficulty, args.stems, settings=settings, cfg=cfg,
                        bpm=args.bpm, meter=args.meter, force=args.force)
    except GenError as e:
        _log(f"gen: {e}")
        return 2
    chart = json.loads(path.read_text())
    print(f"wrote {path}: {len(chart['notes'])} notes, {chart['bpm']} bpm, {len(chart['sections'])} sections")
    print(f"play: torquehero play {path} --layers {_layers_flag(layer_modes(args.difficulty))}")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("gen", help="generate a chart and stems from a song file")
    p.add_argument("song", help="audio file (anything librosa can read)")
    p.add_argument("-o", "--output", help="output directory (default: SONG-torquehero next to the song)")
    p.add_argument("--difficulty", choices=DIFFICULTIES, default="normal")
    p.add_argument("--stems", action="store_true", help="separate stems with demucs (optional 'stems' extra)")
    p.add_argument("--bpm", type=float, help="tempo override, fixes half- or double-time detection")
    p.add_argument("--meter", type=int, choices=(3, 4), default=4, help="beats per bar")
    p.add_argument("--force", action="store_true", help="replace an existing output directory")
    p.set_defaults(func=_run)
