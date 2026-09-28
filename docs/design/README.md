# Design notes

These files were written while Torque Hero was being built. They record the
decisions and the reasons behind them, including the force feedback safety
rules. Some wording refers to the build process (tasks, reviews), because the
game was built in stages with each stage reviewed before it was merged.

| File | What it is |
|---|---|
| [SPEC.md](SPEC.md) | The design spec: vision, layers and notes, force feedback safety rules, playability rules, and every decision made along the way |
| [CONTRACT.md](CONTRACT.md) | Notes on the shared data types between modules |
| [browser-prototype.html](browser-prototype.html) | The first prototype: a single HTML page that plays in a browser with mouse and keyboard (a gamepad works in Chrome). No force feedback. |

Where these notes and the code disagree, the code is right.
