# Force feedback: first run on the rig

Do this once on your rig before a full song, and again after any change to
`src/torquehero/ffb.py` or the wheel driver settings. The direct-drive base can
hurt a wrist.

**Two people: the driver at the wheel and a helper at the keyboard.** The
helper types every command and does every stop. The driver keeps both hands on
the rim unless a step says otherwise.

The first checks use test patterns, not a song. `torquehero play --ffb-test
kick|centre|echo` plays one controlled effect with every safety rule in force.
Every pattern command below says `--ffb-gain 0.2` (SPEC 5 rule 12); without
it, `--ffb-test` also uses 0.2, or your saved gain if that is lower. **A
pattern starts paused**: nothing moves until the helper presses **P** or
**Enter** in the game window. **Backspace** quits, but only while the pattern
is stopped: press **Escape** or **P** first. Do the
patterns in the order below: the sign check comes first, because with a wrong
sign a spring pushes away from centre and pushes harder the further the wheel
goes.

FFB settings live in `ffb.json` in the config dir (`%APPDATA%\torquehero` on
Windows): `{"strength": {effect: 0..1}, "ffb_sign": 1}`. The test window shows
the sign in use on the `WHEEL` line (`ffb_sign +1`). If the game cannot read
`ffb_sign`, it runs without force feedback and says so in the FFB log.

## 1. Before you start

1. In Fanatec's driver panel, set the wheel rotation to **1080°** as a fixed
   value, not "Auto" (SPEC 9 rule 12).
2. Set the driver's FFB strength to **30 to 50%** for the first run, not the
   usual level.
3. In the base's tuning menu, set the damper (NDP, natural damper) to **20**.
   It must not be 0. After a hitch the game brakes a free wheel by itself
   (checked in simulation with no base damping at all); NDP adds margin on
   top. Fanatec does not give NDP in physical units, so 20 is a starting
   value, not a measured one: report the value you used.
4. Close every other Python program: sections 6 and 7 suspend and kill every
   `python.exe`.
5. Helper: get Sysinternals PsTools, put `pssuspend.exe` on the PATH, and run
   `pssuspend -accepteula python.exe` once in PowerShell to accept the licence
   dialog (it may report that no process was found; that is fine).
6. Both: read section 2.

## 2. How to stop

| Stop | Who | What happens | How effects come back |
|---|---|---|---|
| Pause: keyboard **P** or **Escape**, or the pause button bound on the wheel or button box | Helper | All effects stop at once, device gain goes to 0 | Unpause (P or Enter): effects ramp up from zero; the echo pattern waits for a centred wheel again |
| Focus loss: click another window or alt-tab | Helper | All effects stop and stay stopped (latched) | Only when the game window has focus again AND you unpause |
| Quit the game normally | Helper | Effects stop on exit | - |
| Game crashes with a Python error | - | Effects stop and latch before the error is shown | - |
| Base e-stop button (if fitted), or pull the base's power | Driver (left hand) or helper | Motor off | - |

A killed or frozen process skips all of the above; only the dead-man then
applies (sections 6 and 7).

## 3. Sign check (kick)

Helper: `uv run torquehero play --ffb-test kick --ffb-gain 0.2`

The pattern runs no spring and no damper: only the shoves.

1. Driver: hold the rim lightly at centre. Helper: press **P** to start.
2. Every three seconds the pattern gives a shove to the RIGHT (150 ms), and one
   second later a shove to the LEFT. It repeats until you quit.
3. If the first shove of each pair goes left: helper presses **Escape**, then
   **Backspace** to quit, sets
   `"ffb_sign": -1` in `ffb.json`, and runs the pattern again. It must now go
   right first. **Do not go on to section 4 until it does.**
4. Helper: press **Escape** (or **P**) to stop, then **Backspace** to quit.

## 4. Centre check

Helper: `uv run torquehero play --ffb-test centre --ffb-gain 0.2`

The pattern is a spring toward centre computed by the game: the pull grows with
the angle and reaches its full strength (0.06 of full torque at gain 0.2) at
45°, plus light damping. **Never release the rim in this pattern.**

1. Driver: hands on the rim. Helper: press **P** to start.
2. Driver: turn the wheel 90° to the right. The game must show the same angle
   (HUD steering readout). If it does not, stop: the wheel range in the driver
   and in the game do not match.
3. Driver: turn about 30° to the right and hold lightly. The wheel must pull to
   the LEFT, toward centre. Then 30° to the left: it must pull to the RIGHT.
4. **If it pushes away from centre, helper presses P at once**, then
   **Backspace** to quit.
   The sign in section 3 is wrong or the driver inverts forces: stop and
   report.
5. Driver: hold 30° off centre for five seconds. The pull must feel smooth
   and steady. The game refreshes the spring every frame. A driver that
   restarts the effect at each refresh gives a fine buzz at the frame rate
   (about 144 a second), or a pull that is clearly weaker than in step 3:
   stop and report.
6. If you feel no pull at all: press **Escape**, then **Backspace** to quit,
   and run once more with `--ffb-gain 0.3`. Still
   nothing: stop and report (with the HUD FFB log). Do not go higher.
7. Helper: press **Escape** (or **P**) to stop, then **Backspace** to quit.

## 5. Echo check

Helper: `uv run torquehero play --ffb-test echo --ffb-gain 0.2`

1. Helper: press **P** to start, read the `FPS` line in the test window,
   then press **P** again to stop. The line shows the mean rate and the
   slowest frame of the last half second, for example `FPS 144 (worst 72)`.
   The game judges its springs by the worst number: it softens them below
   100 frames a second (`SPRINGS reduced`) and switches them off below 50
   (`SPRINGS off`). **If the worst number is below 100**, press **Backspace**
   to quit, then either:
   - set `"vsync": false` in `render.json` (same folder as `ffb.json`) and
     run this step again: the game then caps itself at 144 frames a second
     instead of following the monitor; or
   - accept reduced springs: go on, and report the frame rate. The echo and
     the centring pull will feel lighter than on a faster PC.

   Below 50 (`SPRINGS off`), do not go on: the checks below would test
   nothing. Report the frame rate.
2. Driver: bring the wheel to centre (within 5°). Helper: press **P** to start.
   Until the wheel is centred nothing runs, no spring and no damper, and the
   HUD FFB log shows `CENTRE the wheel`.
3. Driver: hold the rim with a light grip, fingers open, thumbs off the spokes.
   The base turns the wheel gently; it is a guide, not a fight.
4. Every three seconds the base turns the wheel by itself to 45° right at
   90° per second, then back to centre. It never gets more than 45° ahead of
   where you hold the wheel. Between moves it holds the wheel at centre.
5. **Driver: if the wheel shakes, grip firmly, do not let go.** Helper presses
   **P** at once, then **Backspace** to quit. Report what you felt (too
   fast, wrong direction, a jolt, a shake that grows).
6. Release check: driver turns the wheel 45° to the LEFT and holds it there
   (the base pulls it right), then lets go at once, hands off but ready. The
   wheel must come back and settle, stopping within about 45° past the point
   the base is turning it to, not swing on. If it swings further or keeps
   oscillating: helper presses P; stop and report.
7. Helper: press **Escape** (or **P**) to stop, then **Backspace** to quit.

In a song the echo moves at up to 180° per second, twice the pattern speed.

## 6. Dead-man check: suspend

The game refreshes every sustained force (springs, brake) every frame, and each
lasts 32 ms on the device; the damper is refreshed every 100 ms
and lasts 200 ms. This test freezes the game
while its window keeps focus, so nothing but the effect length can stop the
force. Do not touch the mouse or keyboard after step 4: a click outside the
game window would stop the effects through the focus-loss stop, and the test
would pass on any driver.

`uv run` may start two `python.exe` processes. `pssuspend python.exe` suspends
every match, which is what this test needs.

1. Helper, FIRST, in a second PowerShell window, type this whole line and
   press Enter:
   `Start-Sleep 60; pssuspend python.exe; Start-Sleep 5; pssuspend -r python.exe`
2. Helper, at once, in the first window:
   `uv run torquehero play --ffb-test echo --ffb-gain 0.2`
3. Helper clicks into the game window.
4. Driver: centre the wheel. Helper: press **P** to start the pattern. From now
   on nobody touches the mouse or the keyboard.
5. Driver: hold the wheel **45° to the LEFT**, firmly. At that angle the pull
   to the right stays at its limit (0.06 of full torque at gain 0.2).
6. When the 60 s run out, the game freezes: **the HUD must stop moving**. Only
   then judge the pull: **it must disappear within a quarter second.**
7. Five seconds later the timer line resumes the game. The window moves again
   and shows `STOPPED`, and the FFB log shows `STOP all` and `CENTRE the
   wheel`: the game saw the hitch, paused the pattern and latched it. The
   pull stays off. Helper: press **P** to resume; the log shows `RESUME`.
   Driver: centre the wheel. The pattern starts again and the pull ramps up
   from zero, not as a jolt.
8. Helper: press **Escape** (or **P**) to stop, then **Backspace** to quit.

How to tell a driver centring spring from a failed dead-man: the Fanatec
driver may add its own light centring spring when no game controls the base.
That spring is weak and even, and it pulls toward centre (to the right here,
but much lighter). A failed dead-man keeps the game's pull at the same strength
as before the freeze. If the pull stays as strong as it was for more than a
quarter second, stop and report: the driver does not honour effect lengths,
and the game must not be played on this base.

## 7. Dead-man check: kill

1. Helper, FIRST, in a second PowerShell window:
   `Start-Sleep 60; taskkill /f /im python.exe`
2. Helper, at once, in the first window:
   `uv run torquehero play --ffb-test echo --ffb-gain 0.2`
3. Helper clicks into the game window.
4. Driver: centre the wheel. Helper: press **P** to start the pattern; then
   nobody touches the mouse or keyboard.
5. Driver: hold the wheel 45° to the left, firmly.
6. When the 60 s run out, the game window closes. The wheel must go limp at
   once. This proves less than section 6: when a process ends, the driver may
   stop its effects by itself.

## 8. Raising the gain

Only after sections 3 to 7 pass.

1. Start the echo pattern again at `--ffb-gain 0.2` and press P.
2. Helper: press **ffb_up** (or `]`) once: +0.05 per press. The `GAIN` line
   shows the new gain. The game saves it because the command has
   `--ffb-gain`; a test started without it never saves the gain.
3. Repeat section 5 steps 3 to 6 at this gain.
4. Continue in steps of 0.05. **Stop at the first sign of shake** and go back
   one step (ffb_down or `[`). That is the highest gain for this rig; report it.
5. Helper: press **Escape** (or **P**) to stop, then **Backspace** to quit.

What the simulation says (a hands-off wheel, the game running at 60 or 144
frames a second): the springs settle without growing shake at every gain for a
normal rim on a base up to 25 Nm, and for a heavier rim up to 30 Nm. A very
light rim on a strong base, at full gain and a low frame rate, can shake. The
game softens its springs below 100 frames a second and switches them off below
50. Run the game at 120 frames a second or more.

## 9. Expected feel in a song

After section 8, start the demo song:
`uv run torquehero play --demo --ffb-gain 0.2` (or the gain from section 8).
Effects run only while the song is in play; the menu, countdown, pause and
results screens send no force.

Driver: during the song, read the `FPS` line in the HUD (under the wheel
readout) a few times. If the worst number drops below 100 or the line shows
`SPRINGS reduced` or `SPRINGS off`, the springs feel lighter than in the
checks; report it with the frame rate.

| Effect | When | Feel |
|---|---|---|
| Beat pulse | Every beat | Short 25 Hz buzz (80 ms), stronger on downbeats |
| Section weight | Always in play | Centring pull that grows over the first 90° from centre, plus damping; heavier in loud sections |
| Riser wind-up | Handbrake held during a riser | Nothing by default: the riser buzz is experimental and off (see the last section) |
| Echo (listen) | Echo phrase demonstration | The base turns the wheel through the phrase at up to 180°/s; it waits if you hold the wheel and never gets more than 45° ahead |
| Perfect kick | Perfect melody gate or spin | Short sideways shove (60 ms) |
| Miss rumble | Any miss | 8 Hz rumble for 250 ms |

All effects together never exceed the set gain. A kick on a beat replaces the
pulse instead of adding to it.

## 10. Turn offset check

1. Play to a `spin` note and complete it (one full turn). During the spin the
   centring pull lets go (damping stays), so it does not fight the turn. Do not unwind.
2. The centring pull must now centre on the new position (one turn from
   where you started), not pull the wheel back a full turn.
3. Turn the wheel more than half a turn away from that position: the pulls
   let go (only damping stays) and the HUD shows the unwind hint. Come back:
   they ramp back in, with no jolt.
4. The next spin in the other direction brings the offset back to zero.

## 11. Device-loss check

1. During play, helper pulls the base's USB cable.
2. Expect `DEVICE LOST` in the HUD FFB log and no crash; the game goes on
   without force feedback.
3. Plug the cable back in and restart the game before playing again.

## 12. What to report back

- Result of each check above (pass, or what happened instead), and whether
  `ffb_sign` had to be `-1`.
- The frame rate (the `FPS` line), the NDP value used, and the highest gain from section 8.
- Whether the wheel ever kept turning by itself (coasted) after a pause, a
  stutter or a freeze, and how far.
- Whether a pause or a stutter ever left a push on the wheel for longer than a
  blink. The dead-man test in section 6 cannot tell a 32 ms end from a 200 ms
  end by feel, so this report is the only check of the short lengths on the
  rig.
- Whether the echo is strong enough (lower `strength.echo` in `ffb.json` if too
  strong; it cannot go above the built-in level).
- Whether section weight feels sensible, too light or too heavy.
- Any `WARN`, `DROP`, `NO`, `FAIL`, `GAP`, `SPRINGS` or `DEVICE LOST` lines in
  the HUD FFB log: they name the effects the Fanatec driver refused or the
  hitches the game saw.

## Experimental: riser buzz

Optional, and off by default. During a riser the game can play a 12 Hz buzz in
the rim that grows while the handbrake is held. It is built from whole 83 ms
device sine cycles, each starting as a cosine so that a cycle does not push the
wheel one way. Do this only after every section above passes.

1. Enable it: in `ffb.json` set `"strength": {"riser": 1.0}` (keep the other
   entries). `0` switches it off again.
2. Helper: `uv run torquehero play --ffb-test riser --ffb-gain 0.2`, then press
   **P** to start. The pattern plays a full riser every 4 seconds.
3. Driver: **hands on the rim** throughout. The buzz must grow over 3 seconds,
   stop, and start again from nothing. It must feel like a vibration, not a push
   to one side.
4. **Stop rule:** if the wheel moves more than 5° by itself during a riser,
   helper presses P, then **Backspace** to quit, sets `"riser": 0` in
   `ffb.json` again, and reports
   it. (Simulation, hands off, gain 1.0, a light 0.04 kg·m² rim on a 25 Nm base,
   no base damping, frame times up to 19 ms: at most 8.1° over a full 3 s
   riser. The rig may differ: the driver's call latency and its handling of
   effect phase are not modelled.)
5. Helper: press **Escape** (or **P**) to stop, then **Backspace** to quit.
6. Report whether the buzz felt right and whether the wheel moved.
