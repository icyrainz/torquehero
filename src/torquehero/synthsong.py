"""Built-in demo song (SPEC 3, 8.1, 8.2, 9).

A port of the composition in `docs/design/browser-prototype.html` (compose(), roadX(), exprTarget()
and the synth object), rendered offline with numpy: stems (backing, lead, pad,
bass) and one-shots as 16-bit mono wav files at 48 kHz, plus the chart that
plays them. The render is deterministic on the same machine and numpy build:
fixed seeds, no time-dependent values.

Differences from the web demo, on purpose:
- The chart obeys the limb budget (SPEC 9): chorus is kick + hats with no
  swell, verse and outro are swell + kick; stabs never meet a riser, spin, tom
  run or fader; gates move at most 1.5 lane units per second.
- The lead line is voice-led: each bar takes the octave that starts nearest the
  previous note, and lane x is 0.05 per semitone, so bar lines do not jump.
- Echo phrases are quarter notes, not eighths, so the listen phrase stays inside
  the 180 deg/s echo limit.
- Each shifter gate owns one chord (stab1..4 follow the progression, stab5/6
  are Am and F an octave up), so free play on any gate sounds in key.
- A `faders` layer (not in the demo) rides the bass stem in its own passages:
  lever 1 opens the filter through the intro, lever 0 dips the level through
  the echo listen phrase, lever 1 closes the filter through the outro.
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import time
import wave
import zlib
from pathlib import Path

import numpy as np

from .chart import DEFAULT_LAYER_AUDIO, FORMAT, Chart, playability, validate_dict

VERSION = 3  # bump when the render or the chart changes; names the cache dir
SR = 48000
BPM = 118.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
BARS = 40
LENGTH = BARS * BAR + 1.5
TITLE = "Torque Hero Demo"
ARTIST = "torquehero"

STRUCT = [("intro", 0, 4), ("verse", 4, 12), ("riser", 12, 14), ("chorus", 14, 22), ("echo", 22, 26),
          ("spin", 26, 27), ("riser", 27, 28), ("chorus", 28, 36), ("outro", 36, 40)]
WEIGHT = {"intro": 0.2, "verse": 0.5, "riser": 0.7, "chorus": 0.9, "echo": 0.3, "spin": 0.6, "outro": 0.3}
CHORDS = [(45, True), (41, False), (48, False), (43, False)]  # (root midi, minor): Am F C G
# Spin bar -> turn direction, alternating so the turn offset returns to zero (SPEC 9.10).
SPIN_DIR = {10: "cw", 26: "ccw"}
SPIN_DUR = 2.5 * BEAT  # ends 1.5 beats before the bar line: clear of the gate or riser after it
FILL_BARS = {11, 21, 25, 35}  # the bar before a riser, the echo, the spin, the outro
FILL_BEATS = (1.0, 1.5, 2.0, 2.5)  # ends 1.5 beats before the next bar (riser at bar 12)
ARP_V = [0, 1, 2, 3]
ARP_C = [0, 2, 3, 4, 3, 2, 1, 2]
ARP_E = [1, 2, 4, 2]
LEAD_CENTER, LEAD_STEP = 78, 0.05  # lane x = (midi - centre) * step
LEAD_RANGE = (61, 95)  # midi notes inside lane -0.85..0.85
# Shifter gate -> (chord index, octave shift) of its stab.
STAB_CHORD = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0), 5: (0, 12), 6: (1, 12)}
KICK_VEL = (1.0, 0.7, 0.85, 0.7)  # per beat: accented downbeat

STEMS = ("lead", "pad", "bass")
ONESHOTS = ("kick", "hat", "tom_l", "tom_r", *(f"stab{g}" for g in STAB_CHORD), "riser", "impact", "dud")
PEAK_CEILING = 10 ** (-1 / 20)  # -1 dBFS

# Peak level of each file in dBFS; the mix balance. stab, tom: every file of the family.
LEVEL = {"backing": -16, "lead": -15, "pad": -16, "bass": -15,
         "kick": -10, "hat": -19, "tom": -13, "stab": -16, "riser": -16, "impact": -12, "dud": -19}


# --- composition ---

def section(bar: int) -> str:
    for name, a, z in STRUCT:
        if a <= bar < z:
            return name
    return "outro"


def chord_tones(minor: bool) -> list[int]:
    return [0, 3 if minor else 4, 7, 12, 15 if minor else 16, 19]


def lead_x(midi: int) -> float:
    return min(0.85, max(-0.85, (midi - LEAD_CENTER) * LEAD_STEP))


def voice(base: int, offsets: list[int], prev: int | None) -> list[int]:
    """`base + offsets` voiced inside LEAD_RANGE with the smallest largest step,
    counting the step from `prev`: an octave shift of the bar (-1, 0, +1), and if
    that is not enough, the first note moved to the octave nearest `prev`."""
    best = None
    for o in (0, -12, 12):
        midis = [base + o + k for k in offsets]
        variants = [midis]
        if prev is not None:
            first = midis[0] + 12 * round((prev - midis[0]) / 12)
            variants.append([first, *midis[1:]])
        for v in variants:
            if min(v) < LEAD_RANGE[0] or max(v) > LEAD_RANGE[1]:
                continue
            path = v if prev is None else [prev, *v]
            cost = max((abs(y - x) for x, y in zip(path, path[1:], strict=False)), default=0)
            if best is None or cost < best[0]:
                best = (cost, v)
    return best[1] if best else [base + k for k in offsets]


def expr_target(t: float) -> float:
    return 0.55 + 0.4 * math.sin(2 * math.pi * t / (8 * BEAT))


def stab_gate(bar: int, ci: int) -> int:
    return ci + 5 if bar >= 32 and ci < 2 else ci + 1


def stab_midis(gate: int) -> list[int]:
    ci, octave = STAB_CHORD[gate]
    root, minor = CHORDS[ci]
    base = root + 24 + octave
    return [base, base + chord_tones(minor)[1], base + 7]


def _r(v: float) -> float:
    return round(float(v), 6)


def _curve(fn, dur: float, steps: int) -> list[list[float]]:
    """`fn` sampled at `steps` equal intervals of `dur`, from the integer step index."""
    return [[_r(dur * i / steps), _r(min(1.0, max(0.0, fn(i / steps))))] for i in range(steps + 1)]


# Curve parts as functions of u = 0..1 through the note.
def _ramp(a: float, b: float):
    return lambda u: a + (b - a) * u


def _wobble(amp: float, cycles: int):
    return lambda u: amp * math.sin(2 * math.pi * cycles * u)


def _dip(depth: float):
    return lambda u: -depth * math.sin(math.pi * u)


def _faders() -> list[dict]:
    """Own passages, one lever at a time. Each lever starts where it last stopped and
    ends on the level the stem holds until its next note (SPEC 10): open, 1.0."""
    def note(bar, bars, lever, *parts):
        return {"t": _r(bar * BAR), "kind": "fader", "layer": "faders", "dur": _r(bars * BAR), "lever": lever,
                "curve": _curve(lambda u: sum(p(u) for p in parts), bars * BAR, bars * 8)}
    return [
        note(0, 4, 1, _ramp(0.1, 1.0), _wobble(0.06, 4)),
        note(7, 2, 0, _ramp(1.0, 1.0), _dip(0.5), _wobble(0.08, 4)),
        note(36, 4, 1, _ramp(1.0, 0.1), _wobble(0.05, 4)),
    ]


def compose() -> dict:
    """Chart parts (sections, beats, road, notes) and the synth events of every stem."""
    notes, beats = [], []
    road = [{"t": 0.0, "x": 0.0}]
    ev = {"lead": [], "bass": [], "pad": [], "clap": []}
    last_x, prev, echo_voicing = 0.0, None, {}
    for bar in range(BARS):
        sec, t0 = section(bar), bar * BAR
        ci = (bar - 22) % 2 if sec == "echo" else bar % 4
        root, minor = CHORDS[ci]
        tones = chord_tones(minor)
        chorus_start = sec == "chorus" and section(bar - 1) != "chorus"
        listen, blind = sec == "echo" and bar < 24, sec == "echo" and bar >= 24
        for beat in range(4):
            tb = t0 + beat * BEAT
            beats.append({"t": _r(tb), "s": 1.0 if beat == 0 else 0.8 if sec == "chorus" else 0.5})
            if sec == "chorus" or (sec in ("intro", "verse", "outro") and beat in (0, 2)):
                notes.append({"t": _r(tb), "kind": "kick", "layer": "kick", "vel": KICK_VEL[beat]})
            if sec in ("chorus", "verse") and beat in (1, 3):
                ev["clap"].append(tb)
            # No hat right before the outro: its first kick is the left foot's (SPEC 9.7).
            if sec == "chorus" and not (beat == 3 and section(bar + 1) == "outro"):
                notes.append({"t": _r(tb + BEAT / 2), "kind": "hat", "layer": "hat"})
        if sec != "riser":
            for s in range(8):
                up = 12 if s in (3, 6) and sec == "chorus" else 0
                ev["bass"].append((t0 + s * BEAT / 2, root + up, sec in ("intro", "echo")))
        ev["pad"].append((t0, [root + 12, root + 12 + tones[1], root + 19], BAR, sec == "chorus"))
        if sec in ("verse", "outro"):
            notes.append({"t": _r(t0), "kind": "expr", "layer": "expr", "dur": _r(BAR),
                          "curve": _curve(lambda u, t0=t0: expr_target(t0 + u * BAR), BAR, 16)})

        pat = {"verse": ARP_V, "outro": ARP_V, "chorus": ARP_C, "echo": ARP_E}.get(sec)
        if bar in SPIN_DIR:
            midi = voice(root + 24, [tones[2]], prev)[0]
            ev["lead"].append((t0, midi, BAR * 0.95))
            road.append({"t": _r(t0), "x": _r(last_x)})
            notes.append({"t": _r(t0), "kind": "spin", "layer": "melody", "dur": _r(SPIN_DUR),
                          "dir": SPIN_DIR[bar]})
        elif pat:
            if blind:
                midis = echo_voicing[bar - 2]
            else:
                midis = voice(root + 24, [tones[p] for p in pat], prev)
                echo_voicing[bar] = midis
            div = len(pat)
            for i, midi in enumerate(midis):
                tn = t0 + i * BAR / div
                last_x, prev = lead_x(midi), midi
                ev["lead"].append((tn, midi, BAR / div * 0.9))
                road.append({"t": _r(tn), "x": _r(last_x)})
                if not chorus_start or i > 0:
                    n = {"t": _r(tn), "kind": "gate", "layer": "melody", "x": _r(last_x)}
                    if listen:
                        n["listen"] = True
                    if blind:
                        n["blind"] = True
                    notes.append(n)

        if sec == "chorus":
            # On the riser's drop the right hand is on the handbrake: the stab waits a beat.
            ts = t0 + BEAT if chorus_start else t0
            notes.append({"t": _r(ts), "kind": "stab", "layer": "pads", "gate": stab_gate(bar, ci)})
        if bar in FILL_BARS:
            for b, side in zip(FILL_BEATS, "LRLR", strict=True):
                notes.append({"t": _r(t0 + b * BEAT), "kind": "tom", "layer": "fills", "side": side})
        if sec == "riser" and section(bar - 1) != "riser":
            dur = (2 if bar == 12 else 1) * BAR
            notes.append({"t": _r(t0), "kind": "riser", "layer": "riser", "dur": _r(dur)})
    notes += _faders()
    notes.sort(key=lambda n: n["t"])

    sections = []
    for name, a, _ in STRUCT:
        sections.append({"t": _r(a * BAR), "name": name, "weight": WEIGHT[name]})
    return {"sections": sections, "beats": beats, "road": road, "notes": notes, "events": ev}


def audio_manifest() -> dict:
    return {
        "sr": SR,
        "backing": ["stems/backing.wav"],
        "stems": {k: f"stems/{k}.wav" for k in STEMS},
        "oneshots": {k: f"oneshots/{k}.wav" for k in ONESHOTS},
        "layers": {k: dict(v) for k, v in DEFAULT_LAYER_AUDIO.items()},
    }


def chart_dict(song: dict | None = None) -> dict:
    song = song or compose()
    return {
        "format": FORMAT, "title": TITLE, "artist": ARTIST, "bpm": BPM, "length": _r(LENGTH),
        "sections": song["sections"], "beats": song["beats"], "road": song["road"], "notes": song["notes"],
        "audio": audio_manifest(),
        "defaults": {"layers": dict.fromkeys(DEFAULT_LAYER_AUDIO, "you")},
    }


# --- synthesis ---

def _n(sec: float) -> int:
    return int(round(sec * SR))


def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(tag.encode()))


def _midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _t(n: int) -> np.ndarray:
    return np.arange(n) / SR


def _phase(freq, n: int) -> np.ndarray:
    """Phase in cycles for a constant or per-sample frequency."""
    if np.ndim(freq) == 0:
        return freq * _t(n)
    return np.concatenate(([0.0], np.cumsum(freq[:-1]))) / SR


def _blep(p: np.ndarray, dt) -> np.ndarray:
    out = np.zeros_like(p)
    dt = np.broadcast_to(dt, p.shape)
    lo, hi = p < dt, p > 1 - dt
    x = p[lo] / dt[lo]
    out[lo] = x + x - x * x - 1
    x = (p[hi] - 1) / dt[hi]
    out[hi] = x * x + x + x + 1
    return out


def _saw(freq, n: int) -> np.ndarray:
    p = _phase(freq, n) % 1.0
    return 2 * p - 1 - _blep(p, freq / SR)


def _square(freq, n: int) -> np.ndarray:
    p = _phase(freq, n) % 1.0
    return np.where(p < 0.5, 1.0, -1.0) + _blep(p, freq / SR) - _blep((p + 0.5) % 1.0, freq / SR)


def _sine(freq, n: int) -> np.ndarray:
    return np.sin(2 * np.pi * _phase(freq, n))


def _sweep(a: float, b: float, dur: float, n: int) -> np.ndarray:
    """Exponential glide from a to b over dur, then held (Web Audio exponentialRamp)."""
    return a * (b / a) ** np.minimum(_t(n) / dur, 1.0)


def _perc(dur: float, peak: float, n: int | None = None, attack: float = 0.004) -> np.ndarray:
    """Web demo envelope: exponential rise to peak in `attack`, exponential fall
    to -80 dB at `dur`, shifted so it starts and ends at exactly 0."""
    na, nd = _n(attack), _n(dur) - _n(attack)
    e = np.concatenate((np.geomspace(1e-4, 1, na, endpoint=False), np.geomspace(1, 1e-4, nd)))
    e = (e - 1e-4) / (1 - 1e-4) * peak
    return np.pad(e, (0, max(0, (n or 0) - len(e))))


def _lp(fc: float, order: int = 2):
    return lambda f: 1 / np.sqrt(1 + (f / fc) ** (2 * order))


def _hp(fc: float, order: int = 2):
    return lambda f: np.sqrt((r := (f / fc) ** (2 * order)) / (1 + r))


def _bp(fc: float, q: float = 1.0):
    def resp(f):
        w = f / (q * fc)
        return w / np.sqrt((1 - (f / fc) ** 2) ** 2 + w * w)
    return resp


def _filter(x: np.ndarray, resp, pad: float = 0.05) -> np.ndarray:
    """Zero-phase filter by FFT; `pad` seconds of zeros absorb the ringing."""
    m = len(x) + _n(pad)
    f = np.fft.rfftfreq(m, 1 / SR)
    return np.fft.irfft(np.fft.rfft(x, m) * resp(f), m)[: len(x)]


def _noise(tag: str, n: int) -> np.ndarray:
    return _rng(tag).uniform(-1, 1, n)


def _fade(x: np.ndarray, sec: float = 0.003) -> np.ndarray:
    """Short linear fade in and out so no sound starts or stops on a step."""
    k = min(_n(sec), len(x) // 2)
    ramp = np.linspace(0, 1, k, endpoint=False)
    x = x.copy()
    x[:k] *= ramp
    x[len(x) - k:] *= ramp[::-1]
    return x


# one-shots

def kick(vel: float = 1.0) -> np.ndarray:
    n = _n(0.35)
    return _sine(_sweep(160, 42, 0.22, n), n) * _perc(0.32, 0.5 + 0.5 * vel, n)


def clap(tag: str) -> np.ndarray:
    n = _n(0.14)
    return _filter(_noise(tag, n), _bp(1600)) * _perc(0.14, 0.35, n)


def hat() -> np.ndarray:
    n = _n(0.045)
    return _filter(_noise("hat", n), _hp(7000)) * _perc(0.045, 0.22, n)


def tom(side: str) -> np.ndarray:
    n, hi = _n(0.35), 210 if side == "L" else 150
    return _sine(_sweep(hi, hi * 0.55, 0.25, n), n) * _perc(0.3, 0.6, n)


def stab(gate: int) -> np.ndarray:
    n = _n(0.4)
    tone = sum((_saw(_midi_hz(m), n) for m in stab_midis(gate)), np.zeros(n))
    tone = _filter(tone, _lp(3000)) * _perc(0.35, 0.12, n)
    click = _filter(_noise(f"stab{gate}", n), _hp(4000)) * _perc(0.08, 0.15, n)
    return tone + click


def dud() -> np.ndarray:
    n = _n(0.06)
    return _filter(_noise("dud", n), _lp(500)) * _perc(0.06, 0.3, n)


def impact() -> np.ndarray:
    n = _n(0.65)
    boom = np.pad(kick(1.0), (0, n - _n(0.35)))
    return (boom + _filter(_noise("impact", n), _lp(900)) * _perc(0.6, 0.6, n)) * 0.8


def _below_nyquist(f: np.ndarray) -> np.ndarray:
    """Gain that fades a partial out between 16 and 20 kHz, well before Nyquist."""
    u = np.clip((f - 16000) / 4000, 0, 1)
    return 0.5 + 0.5 * np.cos(np.pi * u)


def shepard(n: int) -> np.ndarray:
    """Shepard tone rising one octave in `n` samples, loopable: every partial ends
    on a whole number of cycles, so partial j at the end continues as partial j+1
    at the start."""
    L = n / SR
    f0 = round(55 * L / math.log(2)) * math.log(2) / L  # f0 * L / ln2 is an integer
    t = _t(n)
    shep = np.zeros(n)
    for j in range(8):
        pos = j + t / L  # octaves above f0
        w = np.exp(-0.5 * ((pos - 4.0) / 1.3) ** 2)
        ph = 2 * np.pi * f0 * 2**j * L / math.log(2) * (2 ** (t / L) - 1)
        f = f0 * 2 ** pos
        shep += w * sum(a * _below_nyquist(h * f) * np.sin(h * ph) for h, a in ((1, 1.0), (2, 0.35), (3, 0.15)))
    return shep


def riser() -> np.ndarray:
    """One bar that loops seamlessly: a Shepard tone plus sixteenth-gated hiss."""
    n = _n(BAR)
    t = _t(n)
    shep = shepard(n)
    spec = np.fft.rfft(_noise("riser", n)) * _bp(3000, 0.7)(np.fft.rfftfreq(n, 1 / SR))
    hiss = np.fft.irfft(spec, n)  # circular filter: loops without a seam
    gate = 0.5 - 0.5 * np.cos(2 * np.pi * (t / (BEAT / 4)))
    return shep / np.abs(shep).max() * 0.3 + hiss / np.abs(hiss).max() * 0.25 * gate


# stems

def _place(buf: np.ndarray, t: float, sig: np.ndarray) -> None:
    i = _n(t)
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i]


def _lead(events, n: int) -> np.ndarray:
    dry = np.zeros(n)
    for t, midi, dur in events:
        m = _n(dur) + _n(0.01)
        _place(dry, t, _square(_midi_hz(midi), m) * _perc(dur, 0.16, m))
    # Lowpass 2600 Hz, then the demo's feedback delay (0.75 beat, feedback 0.32
    # through a 2800 Hz lowpass, send 0.35) as one closed-form frequency response.
    m = n + _n(3.0)
    f = np.fft.rfftfreq(m, 1 / SR)
    z = np.exp(-2j * np.pi * f * (0.75 * BEAT)) * _lp(2800)(f)
    resp = _lp(2600)(f) * (1 + 0.35 * z / (1 - 0.32 * z))
    return np.fft.irfft(np.fft.rfft(dry, m) * resp, m)[:n]


def _bass(events, n: int) -> np.ndarray:
    out = np.zeros(n)
    dur = BEAT * 0.45
    m = _n(dur) + _n(0.01)
    # Filter envelope 900 -> 220 Hz over 0.2 s: crossfade a bright and a dark render.
    cutoff = _sweep(900, 220, 0.2, m)
    mix = (cutoff - 220) / (900 - 220)
    for t, midi, soft in events:
        saw = _saw(_midi_hz(midi), m)
        tone = _filter(saw, _lp(900)) * mix + _filter(saw, _lp(220)) * (1 - mix)
        _place(out, t, tone * _perc(dur, 0.18 if soft else 0.34, m))
    return out


def _pad(events, n: int) -> np.ndarray:
    out = np.zeros(n)
    for t, midis, dur, bright in events:
        m = _n(dur)
        tone = sum((_saw(_midi_hz(mi) * 2 ** (det / 1200), m) for mi in midis for det in (-7, 7)), np.zeros(m))
        e = np.interp(_t(m), [0, 0.4, dur - 0.3, dur], [0, 0.09, 0.09, 0])
        _place(out, t, _filter(tone, _lp(1400 if bright else 800)) * e)
    return out


def _backing(events, n: int) -> np.ndarray:
    out = np.zeros(n)
    for i, t in enumerate(events):
        _place(out, t, clap(f"clap{i % 4}"))
    return out


def _level(x: np.ndarray, name: str) -> np.ndarray:
    return x * (10 ** (LEVEL[name.rstrip("0123456789").removesuffix("_l").removesuffix("_r")] / 20) / np.abs(x).max())


def render(song: dict | None = None) -> dict[str, np.ndarray]:
    """Every audio file of the song, keyed by its manifest path."""
    song = song or compose()
    ev, n = song["events"], _n(LENGTH)
    stems = {"backing": _backing(ev["clap"], n), "lead": _lead(ev["lead"], n),
             "pad": _pad(ev["pad"], n), "bass": _bass(ev["bass"], n)}
    shots = {"kick": kick(), "hat": hat(), "tom_l": tom("L"), "tom_r": tom("R"), "riser": riser(),
             "impact": impact(), "dud": dud(), **{f"stab{g}": stab(g) for g in STAB_CHORD}}
    out = {f"stems/{k}.wav": _level(_fade(v), k) for k, v in stems.items()}
    out |= {f"oneshots/{k}.wav": _level(_fade(v), k) for k, v in shots.items() if k != "riser"}
    out["oneshots/riser.wav"] = _level(shots["riser"], "riser")  # a loop: its ends already meet
    for k, v in out.items():
        peak = float(np.abs(v).max())
        if peak > PEAK_CEILING:
            raise AssertionError(f"{k} peaks at {20 * math.log10(peak):.2f} dBFS, over -1 dBFS")
    return out


def write_wav(path: Path, x: np.ndarray) -> None:
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2") / 32767


# --- output ---

def build(out_dir: str | Path) -> Path:
    """Render the song into `out_dir` and return the path of its chart.json."""
    out_dir = Path(out_dir)
    song = compose()
    d = chart_dict(song)
    errors = validate_dict(d) or playability(d, "normal")
    if errors:
        raise AssertionError("demo chart is invalid or unplayable:\n  " + "\n  ".join(errors))
    for rel, x in render(song).items():
        write_wav(out_dir / rel, x)
    chart = Chart.from_dict(d, out_dir, validate=False)
    path = out_dir / "chart.json"
    chart.save(path)
    return path


def cache_dir() -> Path:
    """Platform cache dir. `TORQUEHERO_CACHE_DIR` overrides it."""
    if env := os.environ.get("TORQUEHERO_CACHE_DIR"):
        return Path(env)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "torquehero" / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "torquehero"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "torquehero"


STALE_TMP = 3600  # seconds before a leftover temp render dir counts as abandoned


def demo_chart_path(cache: str | Path | None = None) -> Path:
    """Chart of the built-in song, rendered into the cache on first use (`play --demo`).
    Renders into a temp dir and renames it into place, so a reader only ever sees a
    complete render; when two first runs race, the first complete one stays."""
    root = Path(cache or cache_dir())
    target = root / f"demo-song-v{VERSION}"
    chart = target / "chart.json"
    if chart.exists():
        return chart
    root.mkdir(parents=True, exist_ok=True)
    _remove_stale(root)
    tmp = root / f"{target.name}.tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    build(tmp)
    try:
        os.replace(tmp, target)
    except OSError as e:
        if chart.exists():  # another process finished first: keep its render
            shutil.rmtree(tmp, ignore_errors=True)
            return chart
        # A partial target without chart.json (a crash in an older version): move it aside.
        old = root / f"{target.name}.tmp{os.getpid()}-old"
        try:
            os.replace(target, old)
            os.replace(tmp, target)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            raise OSError(f"cannot move the demo song render into {target}: {e}") from e
        shutil.rmtree(old, ignore_errors=True)
    return chart


def _remove_stale(root: Path) -> None:
    now = time.time()
    for p in root.glob("demo-song-v*.tmp*"):
        try:
            if now - p.stat().st_mtime > STALE_TMP:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def _cmd(args) -> int:
    path = build(args.out) if args.out else demo_chart_path()
    print(path)
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("demo-song", help="write the built-in song's stems and chart")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output directory (default: the cache that play --demo uses)")
    p.set_defaults(func=_cmd)
