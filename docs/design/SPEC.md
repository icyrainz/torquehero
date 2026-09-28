# Torque Hero — design spec (v3, "the rig is an instrument")

Status: design notes kept from the build. The code is authoritative where the two differ;
the notes record why things are the way they are.
Reference implementation of the feel: `docs/design/browser-prototype.html` (browser demo, no FFB).

## 1. Vision

A rhythm game played on a sim-racing rig. The rig is an instrument, not a car.
Each control owns one layer of the song. Play the layer in time and it sounds;
miss and it drops out of the mix. The road is scenery that bends with the
melody and tells the player where to steer.

Non-goals: car physics, speed, grip, gears-as-gearbox, lap times. If a mechanic
only makes sense because "that is how driving works", it does not belong.

## 2. Hardware and how each capability is used

Reference rig: a Windows 11 PC with an NVIDIA GPU and the hardware below. Development
and tests run on macOS with no rig attached.

| Hardware | Capability | Use in game |
|---|---|---|
| Fanatec direct-drive base | Force feedback: periodic, constant, spring with movable centre, damper, friction | Beat pulse, section weight, riser wind-up, echo (base turns the wheel), perfect kick, miss rumble |
| Porsche GT3 rim | Steering axis, paddles, buttons, funky switch / rotaries | Melody (steer), fills (paddles), menu navigation, live trim of audio offset |
| 3-pedal set, load-cell brake | Analog pressure on brake, analog throttle and clutch | Kick with velocity (brake), swell (throttle), hats (clutch) |
| H-pattern shifter | 6+ discrete gates | Stab sampler, one one-shot per gate |
| Handbrake | Analog pull | Riser hold and drop release |
| VKB STECS throttle | Two lever axes, mini-stick, many buttons | "Faders" layer: two lever lanes that track filter curves. Optional, auto when not bound |
| Button box with rotary | Encoder + buttons | Master volume, FFB gain, pause |
| Triple monitors | ~5760x1080 surround | One borderless window spanning all three. Road centre screen, shoulders and scenery on the sides |
| Top monitor | Extra 1080p above | Companion page `/top`: song timeline, layer mixer, fader lanes |
| Dash display | Small screen by the wheel | Companion page `/dash`: combo, multiplier, layer lamps, next special note |
| NVIDIA GPU | CUDA | Stem separation (demucs) in the chart generator |

Stretch, documented only, not built in this batch: Fanatec rev LEDs and pedal
vibration motors (need the Fanatec SDK or SimHub), VR, bass shakers.

## 3. Layers, notes, judgement

Layers: `melody` (wheel), `kick` (brake), `hat` (clutch), `expr` (throttle),
`pads` (shifter gates), `fills` (paddles), `riser` (handbrake), `faders`
(STECS levers). Each layer is `you` or `auto`. `auto` layers are played by the
song and never judged. A layer whose control is not bound defaults to `auto`,
except `melody`, which always falls back to mouse/keyboard.

Lane units: steering position is -1..1 across the play range (default ±90° of
wheel rotation). Times are seconds from song start.

| Note kind | Layer | Player action | Judged on |
|---|---|---|---|
| `gate` | melody | Be at lane x when it crosses the line | Position error within the timing window |
| `spin` | melody | One full 360° turn within `dur`, either direction | Net signed rotation with a wheel (`spin_mode=net`), summed absolute travel on keyboard/mouse (`spin_mode=abs`). 360° perfect, 240° good |
| `kick` | kick | Brake rises past 0.5 | Timing. Pressure at the crossing is the hit velocity |
| `hat` | hat | Clutch rises past 0.5 | Timing |
| `expr` | expr | Track the target curve for `dur` | Mean absolute error |
| `stab` | pads | Enter gate `n` | Timing, and the right gate |
| `tom` | fills | Left or right paddle | Timing, and the right side |
| `riser` | riser | Pull at start, hold, release on the drop | Held fraction and release timing |
| `fader` | faders | Track a per-lever target curve for `dur` | Mean absolute error |

Echo sections: `gate` notes flagged `listen` are demonstrated (the base turns
the wheel, no judgement). The same phrase then repeats flagged `blind`: the
road is hidden and the player plays it back.

Windows: perfect ±50 ms and ≤0.08 lane units. Good ±120 ms and ≤0.20. For a
`gate`, position is the best error seen inside the good window; perfect needs
an error ≤0.08 at some instant inside ±50 ms. Road tolerance (on-road check
that drives the melody filter) is 0.20. Expression and fader: mean error <0.15 perfect, <0.30 good.
Score: perfect 100, good 50, times a 1x–4x multiplier that steps every 10
combo. A miss resets combo and marks that layer not alive until its next hit.
All thresholds live in `Config`.

Free play: a control pressed outside any window still makes its sound. It is
an instrument.

## 4. Architecture

Python 3.12, `uv`. Package `torquehero` under `src/`.

Dependencies are fixed by the foundation task: `raylib` (pyray) for the
window, `pysdl2` + `pysdl2-dll` for joystick and haptics, `sounddevice` for
audio output, `numpy`, `librosa` + `soundfile` for analysis. Optional extra
`stems` adds `demucs`/`torch`. Dev: `pytest`, `ruff`. Companion pages use the
standard library only. Only the foundation task edits `pyproject.toml` and
`uv.lock`. A later task that needs another dependency does not add it; it
names the package in its handoff line and the supervisor adds it.

| Module | Responsibility | Depends on |
|---|---|---|
| `config.py` | `Config` dataclass, load/save (TOML or JSON) in the platform config dir | - |
| `state.py` | Shared dataclasses: `InputState`, `GameEvent` kinds, `Snapshot` (everything a view needs to draw one frame) | - |
| `chart.py` | `Note`, `Chart`, JSON format v1, validation | - |
| `game.py` | `Game(chart, cfg, layer_modes)`; `update(now, dt, inp) -> list[GameEvent]`; `snapshot()`; pure, no I/O | state, chart |
| `input.py` | `InputSource` protocol, `SdlInput` (many devices), `KeyboardMouseInput` | state, bindings |
| `bindings.py` | Binding model keyed by device GUID + name, learn mode, persistence, `probe`/`bind` CLI | - |
| `ffb.py` | `FfbBackend` protocol (`SdlHapticBackend`, `NullFfb` that records calls), `FfbEngine` mapping events and snapshot to effects, safety limits | state |
| `audio.py` | `AudioEngine` on sounddevice: stem mixing, per-layer gain and low-pass, one-shots, sample-accurate clock. `NullAudio` for tests | state |
| `synthsong.py` | Deterministic built-in demo song: renders stems and one-shots with numpy and builds its chart (port of the composition in the web demo) | chart |
| `generate.py` | Chart from any audio file: beats, onsets, optional demucs stems, difficulty levels | chart |
| `render.py`, `hud.py` | raylib scene and HUD drawn from a `Snapshot`. Window spanning, hidden-window mode for smoke tests | state |
| `companion/` | Local HTTP server with server-sent events publishing `Snapshot` JSON, pages `/dash` and `/top` | state |
| `app.py` | Main loop: input -> game -> audio/ffb/render/companion. Pause, calibration, results | all |
| `__main__.py` | CLI. Subcommands are discovered: for each known module it imports `add_cli(subparsers)` if the module exists. Feature tasks never edit this file | - |

Clock: the audio engine's played-frame count is the song clock. With
`NullAudio` the clock is a monotonic timer.

### CLI

```
torquehero play [CHART] [--demo] [--layers you:melody,kick auto:hat ...]
                [--span WxH+X+Y] [--fullscreen] [--hidden] [--frames N]
                [--no-ffb] [--ffb-gain 0..1] [--no-audio] [--kb] [--companion PORT]
torquehero probe                 # list devices, live axes and buttons
torquehero bind [CONTROL ...]    # learn mode in the terminal, saves bindings
torquehero gen SONG [-o DIR] [--difficulty easy|normal|hard] [--stems]
torquehero demo-song [-o DIR]    # write the built-in song's stems and chart
```

## 5. Force feedback design and safety

Effects (SDL haptic types in brackets):

- Beat pulse [sine, 80 ms]. Magnitude from beat strength.
- Section weight [spring + damper]. Coefficients follow section dynamics so
  loud passages feel heavy.
- Riser wind-up [sine, 12 Hz, whole cycles]. A buzz that grows while the
  handbrake is held, zero on release. Each cycle is a complete device sine
  period, so the buzz has no net push. It does not sway the wheel (section 10).
- Echo playback [spring with moving centre, high coefficient]. The centre
  follows the phrase so the base turns the wheel. Rate-limited.
- Perfect kick [constant, 60 ms]. Miss rumble [sine 8 Hz, 250 ms].

Safety rules, all mandatory:

1. Global gain from config, default 0.5, applied by the engine on top of
   every effect. Per-effect caps in rule 2 apply before this gain. `--ffb-gain` can lower or raise it, never above 1.0.
2. Every effect magnitude ramps; no step from 0 to more than 0.5 inside 10 ms
   except the 60 ms kick and the beat pulse, which are capped at 0.6.
3. Echo centre moves at most 180°/s of wheel rotation.
4. All effects stop on pause, on window focus loss, on exit, and on any
   unhandled exception (context manager plus `atexit`).
5. A device that lacks an effect type degrades to no-op for that effect only.
6. Total budget. Before gain, the sum of the magnitudes of all active effects
   never exceeds 1.0, and the sum of one-shot effects alone never exceeds 0.6.
   Ramped effects are scaled down first.
7. A stop latches. After `stop_all()` nothing is sent until `resume()`.
8. Dead-man. No effect has infinite length. Ramped effects last about 200 ms
   and are refreshed each frame, so a hung game lets the motor go quiet.
9. The echo spring centre is re-synced to the wheel whenever output starts,
   and stays within 45° of the real wheel position.
10. Limits are bounded by measured time inside the engine, not only by the
    frame time the caller reports.
11. Device gain is full while effects run and zero on every stop. The engine
    gain is applied to each effect by the engine; mirroring it to the device
    would apply it twice.
13. A gap between updates longer than the dead-man counts as a fresh start:
    ramps begin from zero and the echo centre re-syncs.
12. First run on hardware starts at `--ffb-gain 0.2`.

## 6. Testing policy

- `uv run pytest -q` and `uv run ruff check src tests` must pass on macOS with
  no joystick, no audio device and no display.
- Hardware is always behind a protocol with a null or fake implementation.
  Tests never open a real haptic device and never play audio out loud.
- Window code is exercised by a smoke run: `uv run torquehero play --demo
  --hidden --frames 300 --no-audio --no-ffb --kb` exits 0.
- Never drive the user's physical hardware from a build task.

## 7. Delivery

The player tries the game on their own rig. The repo must contain a
Windows quickstart (`README.md`) and a setup script that goes from a fresh
checkout to `torquehero probe`, `torquehero bind`, `torquehero play --demo`
spanning the triples, with companion pages for the top monitor and the dash.

## 8. Contracts (the foundation task turns these into code; after it merges, the code is authoritative)

### 8.1 Chart format v1

One JSON file. Paths inside it are relative to the chart file.

```
{
  "format": 1, "title": str, "artist": str, "bpm": float, "length": seconds,
  "sections": [{"t": 0.0, "name": "intro", "weight": 0.2}, ...],   # weight 0..1 drives FFB section weight
  "beats":    [{"t": 0.0, "s": 1.0}, ...],                          # s = strength 0..1
  "road":     [{"t": 0.0, "x": 0.0}, ...],                          # melody centreline keyframes
  "notes":    [{"t": 1.0, "kind": "gate", "layer": "melody", ...}], # sorted by t
  "audio":    { ...manifest, 8.2... }
}
```

`road_x(t)`: between keyframes a and b, hold `a.x` for the first 55% of the
interval, then smoothstep to `b.x`. Before the first keyframe `x = 0`.

Fields per note kind, besides `t`, `kind`, `layer`:

| kind | fields |
|---|---|
| gate | `x`; optional `listen: true` or `blind: true` |
| spin | `dur` |
| kick | optional `vel` 0..1 |
| hat | optional `open: true` |
| expr | `dur`, `curve: [[dt, v], ...]` with v 0..1, linear between points |
| stab | `gate` 1..6, optional `sample` |
| tom | `side`: "L" or "R" |
| riser | `dur`; the drop is at `t + dur` |
| fader | `dur`, `lever` 0 or 1, `curve` as for expr |

Generators must keep `listen` phrases inside the echo rate limit (180°/s of
wheel rotation at the default play range).

### 8.2 Audio manifest

```
"audio": {
  "sr": 48000,
  "backing":  ["stems/backing.wav"],                     # always on
  "stems":    {"lead": "stems/lead.wav", "pad": "...", "bass": "...", "drums": "..."},
  "oneshots": {"kick": "oneshots/kick.wav", "hat": "...", "tom_l": "...", "tom_r": "...",
               "stab1": "...", "riser": "...", "impact": "...", "dud": "..."},
  "layers": {
    "melody": {"stem": "lead", "mode": "filter"},
    "expr":   {"stem": "pad",  "mode": "level"},
    "faders": {"stem": "bass", "mode": "levers"},
    "kick":   {"oneshot": "kick", "mode": "trigger"},
    "hat":    {"oneshot": "hat",  "mode": "trigger"},
    "pads":   {"oneshot": "stab{gate}", "mode": "trigger"},
    "fills":  {"oneshot": "tom_{side}", "mode": "trigger"},
    "riser":  {"oneshot": "riser", "mode": "riser"}
  }
}
```

Modes:

- `trigger`: the sound is a one-shot. Layer `you`: fired by the player's
  input, at the moment of input, hit or not. Layer `auto`: scheduled by the
  engine at each note time.
- `gate`: the sound is inside a stem. The stem plays continuously; its gain
  ducks to 0.2 over 60 ms when the layer is not alive and recovers over 30 ms.
  Several layers may name one stem (demucs gives one `drums` stem for `kick`
  and `hat`): stem gain is 0.2 + 0.8 × fraction of those layers alive. An
  `auto` layer counts as alive. A `gate` entry may also carry `oneshot`, fired
  on player hits as reinforcement.
- `filter`: stem is open when the player is on the road, low-passed at 450 Hz
  and ducked to 0.35 when off it.
- `level`: stem gain is 0.15 + 0.85 × control value (throttle, or the target
  curve when the layer is `auto`).
- `levers`: lever 0 sets stem gain 0.2..1, lever 1 sets a low-pass cutoff
  200 Hz..8 kHz on a log scale.
- `riser`: looped riser sound starts on the pull, stops on release, then the
  `impact` one-shot fires.

The built-in song uses `trigger` for the percussive layers. Generated songs
with demucs stems use `gate` plus reinforcement one-shots from a built-in kit.

### 8.3 Control names

Performance: `steer`, `brake`, `clutch`, `throttle`, `handbrake`, `gate1`..`gate6`,
`paddle_l`, `paddle_r`, `lever0`, `lever1`.
System: `menu_up`, `menu_down`, `menu_ok`, `menu_back`, `pause`, `trim_minus`,
`trim_plus` (audio offset), `vol_up`, `vol_down`, `ffb_up`, `ffb_down`.

Kinds: `steer` is a bipolar axis. `brake`, `clutch`, `throttle`, `handbrake`,
`lever0`, `lever1` are unipolar 0..1 and accept an axis (rest value and
direction learned) or a button. The rest are buttons and accept a button or an
axis past 0.5.

`InputState` carries every performance control as a value, rising edges for
the button-like ones, `steer` in lane units, `steer_deg` in raw wheel degrees,
the set of system controls pressed this frame, and which controls are bound.

Keyboard and mouse are read through a `KeySource` protocol defined in
`state.py` (`is_down`, `pressed`, `mouse_x_norm`). The renderer provides the
raylib implementation; tests use a fake. `input.py` never imports raylib.

### 8.4 Snapshot

`Snapshot` is a plain dataclass with `to_json()`. It must carry at least:
`phase` (attract, countdown, play, paused, calibrate, results), `now`,
`length`, `score`, `combo`, `max_combo`, `multiplier`, `accuracy`, `counts`,
per-layer `{mode, alive, hit, total}`, the `InputState`, `section` and
`weight`, `beat_phase` and last `beat_strength`, `echo` (none, listen, repeat)
and `echo_target`, `riser` `{active, progress, held}`, `expr_target`,
`fader_targets`, `upcoming` (special notes in the next 8 s), `popups`, and
`ffb` `{torque, log}` as reported by the FFB engine. The foundation task ships
a fixture chart and a sample snapshot under `tests/fixtures/` for the tasks
that cannot wait for the built-in song.

### 8.5 CLI discovery

`__main__.py` holds a fixed list of module names (`app`, `bindings`,
`generate`, `synthsong`, `companion`, `render`). For each it imports the
module's `add_cli(subparsers)` lazily. A module that does not exist yet is
skipped. An `ImportError` raised from inside an existing module is an error
and is shown. Heavy imports (librosa, sdl2, torch, pyray) happen inside the
command handler, not at module import.

### 8.6 Other decisions

- The companion server binds `0.0.0.0` by default so a tablet or a second
  machine can show a page. `--companion-host` overrides.
- FFB gets its haptic device from the joystick that the `steer` binding names
  (device GUID). No `steer` binding, no FFB.
- Pause, results and calibration are drawn by the renderer from
  `Snapshot.phase`. Their behaviour lives in `app.py`. The focus-loss FFB stop
  is wired in `app.py` using `FfbEngine.stop_all()`.
- A miss ducks or silences a layer as the mode says. There is no other mute.

### 8.7 Decisions from the foundation review (2026-09-27)

- **Auto-layer audio.** The audio engine schedules auto-layer sounds from the
  chart, sample-accurately. It ignores `Sound` events whose cause is `auto`;
  those exist for views and FFB. `Sound` and `Judgement` carry the note's
  chart time and index.
- **Wrong presses.** A gate or paddle press first matches the earliest
  unfinished note in its window that it satisfies. Only a press that matches
  nothing counts as a wrong press on the nearest note.
- **Snapshot carries per-note view state** (index, done, result, progress) for
  the visible window. The renderer combines it with the chart. Static song
  data for the companion pages comes from `Game.song_info()`.
- **Echo.** `echo_target` follows the demonstrated phrase and reaches each
  listen gate's x at its time. Validation checks the peak rate of that same
  interpolation against 180°/s. The FFB engine still clamps.
- **Riser cues pair.** Every `riser_start` has exactly one `riser_release`.
- **`bound` means a real device.** Controls served by keyboard or mouse are in
  `fallback`. Spin is judged on net rotation only when `steer` is in `bound`.
- **One press threshold**, `Config.press_threshold`, for edges and holds.
- **Where the spec and the web demo differ, the spec wins.** A riser release
  is accepted until the good window after the drop. A free-play gate press
  plays its stab, not the dud.
- **Settings of feature modules** (audio device, companion port, bindings)
  live in their own files under the config dir, not in `Config`.
- **Manifest validation** requires a layer entry for every layer with notes
  and every one-shot name the game can emit for that chart. The audio engine
  still tolerates a missing sample by logging once and staying silent.

### 8.8 Decisions from the second foundation review (2026-09-27)

- **End of song.** The game is finished when the clock is past the chart
  length and every note is finalized. The app loop runs until then.
- **Riser.** FFB wind-up reads `Snapshot.riser` (active and held), valid until
  the riser is finalized. The `riser_start` and `riser_release` cues are
  one-off markers.
- **Hit velocity** comes from the input layer, per pressed control, from how
  fast the control rose. Pressure at the threshold crossing is not used.
  Keyboard presses have velocity 1.0.
- **Layer defaults.** A layer is `you` only when every control the chart needs
  on that layer is available.
- **Companion pages** get the chart and the song info from the app when a
  song starts (`publish_song`), and the `Snapshot` every frame.
- **During echo** the road is drawn from `road_x` and the ghost marker from
  `echo_target`. Generators put road keyframes on the listen and blind gates
  so the two agree at each gate. The FFB engine rate-limits toward
  `echo_target`, including the step when a listen region opens.

### 8.9 Decisions at the foundation merge (2026-09-27)

- **The song clock keeps advancing after the last stem frame.** The audio
  engine never stops the clock at the end of the audio; the game needs about
  `length + audio_offset + good_window` to finalize the last notes.
- **Allowed audio modes per layer.** `melody`: filter or gate. `expr`: level.
  `faders`: levers. `kick`, `hat`, `pads`, `fills`: trigger or gate. `riser`:
  riser. The audio engine treats any other pairing as silent and logs it once.
- **Code against demucs 4.1.0.**
- Open follow-ups on the foundation, owned by a later task: `validate_dict`
  must not raise on non-string `kind` or `stem` or on oversized integers, and
  must enforce the allowed modes above.

## 9. Playability: the limb budget (2026-09-27)

A chart must be playable by one person with two hands and two feet on the
real rig. Keyboard play hid this in the web demo. Every chart author (the
built-in song, the generator) obeys these rules, and chart validation reports
violations through `chart.playability()`.

Where the controls are: shifter and handbrake are on the right, the STECS
levers on the left, paddles are on the wheel. Pedals left to right are clutch,
brake, throttle.

Hands:

1. Each hand is on one control at a time. Right hand: wheel, shifter or
   handbrake. Left hand: wheel or one lever.
2. One hand is enough for gates. `spin` and `tom` need both hands on the wheel.
3. A hand needs 0.4 s to move between controls. So there is no `stab` within
   0.4 s of a riser's start, hold or drop; no `stab` or riser and no active
   `fader` from 0.4 s before a `spin` or a `tom` run until 0.4 s after it.
4. The two levers are moved by the same hand. `fader` notes for lever 0 and
   lever 1 do not overlap, and the target of one lever does not jump at a
   note boundary.
5. A `fader` note and a right-hand note (`stab`, riser) may overlap only if no
   gate in that span moves more than 0.4 lane units from the previous gate,
   because the wheel is then steered with no hand firmly on it. Simplest: do
   not overlap them.

Feet:

6. Right foot: throttle, or brake when no `expr` note is active. Left foot:
   clutch, or brake.
7. A foot needs 0.4 s between two different pedals. While an `expr` note is
   active the left foot plays both kick and hat, so a kick and a hat are then
   at least 0.4 s apart. With no `expr` active, kicks (right foot) and hats
   (left foot) are independent.

Steering:

8. Consecutive gates are at most 1.5 lane units per second apart on normal
   difficulty (2.5 on hard), measured gate to gate.
9. The road has a keyframe at `t = 0` with `x = 0` and never steps.

Spins and the turn offset:

10. A `spin` note has a direction, `dir`: "cw" or "ccw". A completed spin
    leaves the wheel one full turn from where it was. The game keeps a turn
    offset, plus or minus 360°, and judges steering relative to it, so the
    player does not unwind. Spins alternate direction so the offset returns to
    zero, and it never exceeds one turn. `Snapshot` reports the offset, and
    the FFB engine adds it to every centre it commands. The offset is
    re-derived from the wheel position when any spin is finalized, hit or
    miss (see section 10).
11. No gate within one beat before a spin or one beat after it.
12. Recommended wheel rotation in the driver is 1080°, so one turn plus the
    play range stays clear of the lock.

## 10. Decisions from the wave-2 reviews (2026-09-27)

- **Blind echo phase.** Nothing on screen reveals the melody path. The road
  is drawn straight and dim, blind gates are hidden, and everything anchored
  to the road is anchored to the straight centre line. This replaces the
  sentence in 8.8 about the road during echo for the blind phase; the listen
  phase still draws the road from `road_x` and the ghost from `echo_target`.
- **Continuous layers follow the player only during their notes.** In the
  `level` and `levers` modes the stem follows the control while an `expr` or
  `fader` note of that layer is active. Between notes the stem holds the
  last target of the note that ended (0.55 for level and open for levers
  before the first note), for `you` and `auto` alike. A foot that leaves the
  throttle for the brake in a chorus does not thin the pad.
- **Difficulty decides what is in a chart.** Easy: melody, kick, riser.
  Normal adds hat, pads, fills, spins and echo. Hard adds expr and faders.
- **Companion results** are kept on the server per song and sent on connect.
- **Press and release.** A control is pressed when it rises to the press
  threshold and released when it falls below the threshold minus a margin of
  0.1. `InputState.down` is the latched state. The game reads `down`, never
  the raw level, to decide whether a control is held. Inside one frame,
  `pressed` comes before `released`.
- **Who owns the wheel.** The input layer owns the joystick. The FFB engine
  borrows it. Before the input layer closes the wheel's joystick (unplug,
  e-stop, shutdown) it calls the handlers registered with `on_steer_lost`,
  and the app registers the FFB engine's device-lost handler there.
- **Pause and resume do not move the song clock.**
- **The turn offset comes from the wheel.** It is `360 × round(steer_deg /
  360)`, limited to one turn, and is re-derived only at reset and when a spin
  is finalized. A spin that would take the wheel past one and a half turns
  from zero is a miss. `Snapshot.steer_unwind` is true while the wheel and
  the offset disagree by more than half a turn; views show a hint and the FFB
  engine holds its springs off. Default wheel range is 1080°.
- **The app constructs or resets the game at the end of the countdown**, not
  before it, so the turn offset is derived from where the wheel is when play
  starts.
- **For a bound wheel, `steer` equals `clamp(steer_deg / play_range_deg)`.**
  The input layer applies no separate centre or deadzone to `steer`.
- **The FFB engine detects an offset change by comparing `steer_offset_deg`
  with its last value**, not only by the one-frame flag.
- **One drawn road.** `Chart.drawn_road_x(t)` is the road as it is shown:
  every blind run is removed and pinned to the centre line, blended in and
  out, and a run never opens before the previous listen run closes.
  `Snapshot.road_x` is this drawn road. A blind zone runs from the last road
  keyframe before a blind run opens to the first keyframe after it closes.
  Inside a blind zone `on_road` is true, so the marker colour and the melody
  filter reveal nothing and punish nothing. Blind gates are still judged at
  their real x.
- **A restart on a playing audio engine uses a lead-in of at least 15 ms**
  (`start(-0.015)` or more), so auto sounds at t = 0 are not skipped by the
  seek fade.
- **Springs are computed by the engine.** A device condition spring on a
  1080° axis reaches full force only half a turn from centre, so the echo and
  the section-weight spring are software springs: the engine computes a
  constant force from the angle error each frame (echo: full level at 45° of
  error; weight: full level at 90°), ramped and budgeted like every other
  effect. The device damper condition effect stays on for stability, and the
  engine adds a velocity term. A simulated wheel updated at 60 Hz must not
  oscillate.
- **A stall is bounded by the engine alone.** Echo phases ask for hands off,
  so the engine cannot rely on the base's own damping. After any stall a
  software velocity brake runs until the springs are back, and the echo level
  is capped at 0.3 before gain. A simulated hands-off wheel with no base
  damping must stay under 360°/s and stop within one turn after a stall.
- **Sustained constant forces are short and capped.** Every sustained
  constant-force effect (echo, section weight, riser, the centre test pattern,
  the stall brake) lasts 32 ms on the device, is re-run every frame, and is
  capped at 0.3 before gain, so a stall of any length cannot launch a
  hands-off wheel past 360°/s. One-shot effects keep their fixed lengths.
- **The riser is a vibration, not a sway.** During a riser the wheel may be
  hands-off, so the riser is a 12 Hz buzz that grows with the wind-up, capped
  like every sustained force. A free wheel must move less than 10° during a
  full riser.
- **The sum of sustained forces is capped.** Together, all sustained
  constant forces stay at or under 0.3 before gain. They scale down together
  when over.
- **The riser is built from whole device sine cycles.** Each 12 Hz cycle is a
  one-shot of exactly one period, fired back to back, with its amplitude and
  budget scale fixed for the whole cycle. Frame timing cannot bias it.
- **The riser buzz is experimental and off by default.** Its whole-cycle
  implementation stays, but `strength.riser` defaults to 0 until it has been
  tried on the rig. Pulse and rumble start at phase 0 with their attack and
  fade.
