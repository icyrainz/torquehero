"""HUD and phase screens, drawn over the scene from a `Snapshot` (SPEC 8.4, 8.6).

Everything sits in the centre region of the window (`Layout.cx0`, `Layout.cw`),
so on a triple-monitor surround it stays on the middle screen. Only the
progress bar spans the full width. Called by `render.Renderer.draw`.
"""
from __future__ import annotations

import math

from .ffb import rate_text
from .render import COL, LAYER_STYLE, LEVER_COL, RESULT_COL, Color, Layout, fade, turn_back_dir, turn_offset
from .state import LAYERS, Snapshot

TRACK = fade(COL["ink"], 0.12)
PANEL = fade(COL["bg"], 0.82)
ECHO_TEXT = {"listen": ("ECHO · HANDS OFF, THE BASE PLAYS", COL["amber"]),
             "repeat": ("ECHO · YOUR TURN, THE ROAD IS DARK", COL["violet"])}
NEXT_LABEL = {"spin": "SPIN", "expr": "SWELL", "stab": "STAB", "tom": "FILL", "riser": "RISER", "fader": "FADER"}


def layer_accuracy(hit: int, total: int) -> float | None:
    """Per-layer accuracy for the results screen: hits / judged notes, None if none."""
    return hit / total if total else None


def grade(accuracy: float) -> str:
    for g, lo in (("S", 0.95), ("A", 0.85), ("B", 0.7), ("C", 0.5)):
        if accuracy >= lo:
            return g
    return "D"


def ffb_log_lines(log) -> list[str]:
    """FFB engine log entries as text, newest first (strings or {"s"|"text": ...})."""
    out = []
    for e in log or []:
        if isinstance(e, dict):
            e = e.get("s") or e.get("text") or e.get("msg") or ""
        out.append(str(e))
    return out


def turn_tag(turns: int) -> str:
    """Wheel gauge tag for the spin turn offset: "+1 TURN", "-1 TURN", or ""."""
    return f"{turns:+d} TURN" if turns else ""


def next_special(snap: Snapshot) -> tuple[str, float] | None:
    """Label and seconds to go of the next special note, from `Snapshot.upcoming`."""
    for u in snap.upcoming:
        dt = u.get("t", 0.0) - snap.now
        if dt >= 0:
            label = NEXT_LABEL.get(u.get("kind", ""), str(u.get("kind", "")).upper())
            if u.get("kind") == "stab" and u.get("gate"):
                label += f" {u['gate']}"
            elif u.get("kind") == "tom" and u.get("side"):
                label += f" {u['side']}"
            return label, dt
    return None


def draw(r, snap: Snapshot, lay: Layout) -> None:
    """HUD for play-like phases, then the overlay of the current phase."""
    if snap.phase == "attract":
        attract(r, snap, lay)
        return
    stats(r, snap, lay)
    chips(r, snap, lay)
    ffb_meter(r, snap, lay)
    gauges(r, snap, lay)
    shifter(r, snap, lay)
    progress(r, snap, lay)
    banners(r, snap, lay)
    turn_back(r, snap, lay)
    match snap.phase:
        case "countdown":
            countdown(r, snap, lay)
        case "paused":
            paused(r, snap, lay)
        case "calibrate":
            calibrate(r, snap, lay)
        case "results":
            results(r, snap, lay)


# --- play HUD ---

def stats(r, snap: Snapshot, lay: Layout) -> None:
    p, u = r.paint, lay.unit
    x, y = lay.cx0 + u * 0.9, u * 0.9
    p.text("SCORE", x, y, u * 0.8, COL["dim"])
    p.text(f"{snap.score:06d}", x, y + u * 0.8, u * 2.4, COL["ink"])
    p.text(f"{snap.combo} COMBO", x, y + u * 3.3, u * 1.05, COL["cyan"] if snap.combo >= 10 else COL["ink"])
    mult = f"×{snap.multiplier}"
    mx = x + p.text_width(f"{snap.combo} COMBO", u * 1.05) + u * 0.6
    p.text(mult, mx, y + u * 3.1, u * 1.3, COL["sun"] if snap.multiplier > 1 else COL["dim"])
    judged = sum(snap.counts.values())
    acc = f"{round(100 * snap.accuracy)}% ACCURACY" if judged else "—% ACCURACY"
    p.text(acc, x, y + u * 4.6, u * 0.85, COL["dim"])
    if snap.section:
        p.text(snap.section.upper(), x, y + u * 5.7, u * 0.8, fade(COL["violet"], 0.9))


def chips(r, snap: Snapshot, lay: Layout) -> None:
    """Layer chips, top centre: colour when alive, red when dropped, grey when auto."""
    p, u = r.paint, lay.unit
    cw = u * 3.6
    x0 = lay.cx - len(LAYERS) * cw / 2
    y = u * 0.9
    for i, k in enumerate(LAYERS):
        st = snap.layers.get(k)
        if st is None:
            continue
        name, col = LAYER_STYLE[k]
        x = x0 + i * cw
        auto = st.mode == "auto"
        bar = fade(COL["ink"], 0.08) if auto else col if st.alive else fade(COL["red"], 0.25)
        ink = COL["dim"] if auto else COL["ink"] if st.alive else COL["red"]
        p.rect(x, y, cw - u * 0.3, u * 0.35, bar)
        p.text(name, x, y + u * 0.55, u * 0.8, ink)
        p.text("AUTO" if auto else "YOU", x, y + u * 1.45, u * 0.6, fade(ink, 0.7))
    nxt = next_special(snap)
    if nxt:
        label, dt = nxt
        a = 1.0 if dt < 2 else 0.6
        p.text(f"NEXT  {label}  {dt:.1f}s", lay.cx, y + u * 2.5, u * 0.8, fade(COL["ink"], a), "center")


def ffb_meter(r, snap: Snapshot, lay: Layout) -> None:
    """Torque bar, section weight, wheel angle, frame rate and the FFB event log (top right)."""
    rl, p, u = r.rl, r.paint, lay.unit
    right = lay.cx0 + lay.cw - u * 0.9
    mw = u * 8
    mx, my = right - mw, u * 0.9 + u * 1.2
    ffb = snap.ffb if isinstance(snap.ffb, dict) else {}
    torque = max(-1.0, min(1.0, float(ffb.get("torque") or 0.0)))
    p.text("FFB TORQUE", right, u * 0.9, u * 0.8, COL["dim"], "right")
    p.rect(mx, my, mw, u * 0.7, TRACK)
    tw = torque * mw / 2
    p.rect(mx + mw / 2 + min(0.0, tw), my, abs(tw), u * 0.7, COL["sun"] if abs(torque) > 0.6 else COL["cyan"])
    p.rect(mx + mw / 2 - 1, my - 2, 2, u * 0.7 + 4, COL["ink"])
    p.rect(mx, my + u * 0.9, mw, u * 0.25, TRACK)
    p.rect(mx, my + u * 0.9, mw * max(0.0, min(1.0, snap.weight)), u * 0.25, COL["violet"])
    p.text(f"WEIGHT {round(snap.weight * 100)}%", right, my + u * 1.3, u * 0.7, COL["dim"], "right")
    wr = u * 1.5
    wx, wy = right - wr, my + u * 2.4 + wr + torque * 3 * lay.s
    rl.draw_ring((wx, wy), wr - 1.5 * lay.s, wr + 1.5 * lay.s, 0, 360, 48, COL["ink"])
    ang = math.radians(snap.input.steer_deg)
    c, s = math.cos(ang), math.sin(ang)
    th = 3 * lay.s
    rl.draw_line_ex((wx, wy), (wx + wr * s, wy - wr * c), th, COL["amber"])
    rl.draw_line_ex((wx - wr * c, wy - wr * s), (wx + wr * c, wy + wr * s), th, COL["amber"])
    p.text(f"{round(snap.input.steer_deg)}°", wx - wr - u * 0.5, wy - u * 0.45, u * 0.85, COL["dim"], "right")
    tag = turn_tag(turn_offset(snap))
    if tag:
        p.text(tag, wx - wr - u * 0.5, wy + u * 0.5, u * 0.7, COL["amber"], "right")
    y = wy + wr + u * 0.6
    warn = ffb.get("springs") in ("reduced", "off") or ffb.get("latched")
    p.text(rate_text(ffb), right, y, u * 0.72, COL["amber"] if warn else COL["dim"], "right")
    for i, line in enumerate(ffb_log_lines(ffb.get("log"))[:5]):
        p.text(line, right, y + (i + 1) * u, u * 0.72, fade(COL["ink"], max(0.15, 1 - i * 0.22)), "right")


def _gauge(p, x: float, y: float, w: float, h: float, v: float, col: Color, label: str, u: float,
           target: float | None = None) -> None:
    v = max(0.0, min(1.0, v))
    p.rect(x, y, w, h, TRACK)
    p.rect(x, y + h * (1 - v), w, h * v, col)
    if target is not None:
        t = max(0.0, min(1.0, target))
        p.rect(x - 3, y + h * (1 - t) - 1.5, w + 6, 3, COL["ink"])
    p.text(label, x + w / 2, y - u * 1.0, u * 0.8, COL["dim"], "center")


def gauges(r, snap: Snapshot, lay: Layout) -> None:
    """Pedals and handbrake with the expr target; levers with the fader targets (bottom left)."""
    p, u, inp = r.paint, lay.unit, snap.input
    bh, bw, gap = u * 3.2, u * 0.9, u * 0.5
    x, y = lay.cx0 + u * 0.9, lay.h - u * 0.9 - bh - u * 0.4
    riser = snap.riser
    items = [
        ("T", inp.throttle, COL["green"], snap.expr_target),
        ("B", inp.brake, COL["red"], None),
        ("C", inp.clutch, COL["amber"], None),
        ("H", inp.handbrake, COL["violet"], None),
    ]
    for i, (label, v, col, target) in enumerate(items):
        _gauge(p, x + i * (bw + gap), y, bw, bh, v, col, label, u, target)
    if riser.active:
        hx = x + 3 * (bw + gap)
        p.rect(hx - 3, y + bh + u * 0.2, (bw + 6) * riser.progress, u * 0.2, COL["violet"])
    x += 4 * (bw + gap) + u * 0.6
    targets = list(snap.fader_targets) + [None, None]
    for i, (label, v) in enumerate((("L0", inp.lever0), ("L1", inp.lever1))):
        _gauge(p, x + i * (bw + gap), y, bw, bh, v, LEVER_COL[i], label, u,
               targets[i])


def shifter(r, snap: Snapshot, lay: Layout) -> None:
    """Gate indicator and paddle lamps (bottom right)."""
    p, u, inp = r.paint, lay.unit, snap.input
    right = lay.cx0 + lay.cw - u * 0.9
    base = lay.h - u * 0.9 - u * 0.4
    g = inp.gate
    p.text(f"GATE {g}" if g else "SHIFTER", right, base - u * 1.5, u * 1.3, COL["magenta"] if g else COL["dim"],
           "right")
    for i, (ctrl, label) in enumerate((("paddle_r", "R"), ("paddle_l", "L"))):
        on = ctrl in inp.down
        x = right - u * 0.9 - i * u * 1.3
        p.rect(x, base - u * 2.7, u * 0.9, u * 0.9, COL["sun"] if on else TRACK)
        p.text(label, x + u * 0.45, base - u * 2.65, u * 0.8, COL["bg"] if on else COL["dim"], "center")


def progress(r, snap: Snapshot, lay: Layout) -> None:
    if snap.length <= 0:
        return
    h = max(3.0, 3 * lay.s)
    pr = max(0.0, min(1.0, snap.now / snap.length))
    r.paint.rect(0, lay.h - h, lay.w, h, fade(COL["ink"], 0.15))
    r.paint.rect(0, lay.h - h, lay.w * pr, h, COL["sun"])


def turn_back(r, snap: Snapshot, lay: Layout) -> None:
    """While the wheel is half a turn or more from the turn offset (and no spin runs):
    a clear hint with the direction to turn, toward the offset."""
    d = turn_back_dir(snap, r.chart)
    if not d:
        return
    p, u = r.paint, lay.unit
    size, y = u * 2.2, lay.h * 0.6
    text = "TURN BACK"
    w = p.text_width(text, size)
    x = lay.cx - (w + size) / 2
    pulse = 0.65 + 0.35 * abs(math.sin(snap.now * 6))
    p.rect(x - u, y - u * 0.5, w + size + u * 2.2, size + u, fade(COL["bg"], 0.7))
    p.text(text, x, y, size, fade(COL["amber"], pulse))
    p.turn_arrow(x + w + size * 0.6, y + size * 0.48, size * 0.36, d, max(2.0, size * 0.1),
                 fade(COL["amber"], pulse))


def banners(r, snap: Snapshot, lay: Layout) -> None:
    if snap.phase != "play" or snap.echo not in ECHO_TEXT:
        return
    text, col = ECHO_TEXT[snap.echo]
    y, size = lay.h * 0.2, lay.unit * 1.5
    r.paint.text(text, lay.cx + 2 * lay.s, y + 2 * lay.s, size, fade(COL["bg"], 0.8), "center")
    r.paint.text(text, lay.cx, y, size, col, "center")


# --- phase screens ---

def _veil(r, lay: Layout, a: float = 0.72) -> None:
    r.paint.rect(0, 0, lay.w, lay.h, fade(COL["bg"], a))


def attract(r, snap: Snapshot, lay: Layout) -> None:
    p, u = r.paint, lay.unit
    chart = r.chart
    p.text("TORQUE HERO", lay.cx, lay.h * 0.12, u * 4.2, COL["ink"], "center")
    p.text("THE RIG IS AN INSTRUMENT", lay.cx, lay.h * 0.12 + u * 4.4, u * 0.9, COL["sun"], "center")
    song = chart.title + (f" — {chart.artist}" if chart.artist else "")
    p.text(song.upper(), lay.cx, lay.h * 0.27, u * 1.2, COL["dim"], "center")
    blink = 0.45 + 0.55 * abs(math.sin(snap.now * 2.2))
    p.text("PRESS OK TO START", lay.cx, lay.h * 0.93, u * 1.1, fade(COL["ink"], blink), "center")
    progress(r, snap, lay)


def countdown(r, snap: Snapshot, lay: Layout) -> None:
    n = math.ceil(-snap.now)
    text = str(n) if n > 0 else "GO"
    frac = (-snap.now) % 1.0 if n > 0 else 0.0
    size = lay.h * 0.16 * (1 + 0.15 * frac)
    r.paint.text(text, lay.cx, lay.h * 0.5 - size / 2, size, fade(COL["ink"], 0.5 + 0.5 * frac), "center")


def paused(r, snap: Snapshot, lay: Layout) -> None:
    p, u = r.paint, lay.unit
    _veil(r, lay)
    p.text("PAUSED", lay.cx, lay.h * 0.36, u * 4, COL["ink"], "center")
    p.text("OK  RESUME       BACK  QUIT TO MENU", lay.cx, lay.h * 0.36 + u * 5, u * 1.0, COL["dim"], "center")
    p.text("FORCE FEEDBACK STOPPED", lay.cx, lay.h * 0.36 + u * 6.6, u * 0.8, fade(COL["amber"], 0.8), "center")


def calibrate(r, snap: Snapshot, lay: Layout) -> None:
    """Audio offset calibration: a pulse on every beat to tap along with."""
    rl, p, u = r.rl, r.paint, lay.unit
    _veil(r, lay, 0.8)
    p.text("CALIBRATE", lay.cx, lay.h * 0.2, u * 3, COL["ink"], "center")
    p.text("TAP THE BRAKE ON THE BEAT.  TRIM - / +  MOVES THE AUDIO OFFSET.", lay.cx, lay.h * 0.2 + u * 3.6,
           u * 0.9, COL["dim"], "center")
    pulse = (1 - max(0.0, min(1.0, snap.beat_phase))) ** 2
    cy, rad = lay.h * 0.56, u * 4
    rl.draw_ring((lay.cx, cy), rad * 0.94, rad, 0, 360, 64, fade(COL["ink"], 0.3))
    rl.draw_circle(int(lay.cx), int(cy), rad * (0.35 + 0.6 * pulse), fade(COL["sun"], 0.25 + 0.75 * pulse))
    if "brake" in snap.input.down:
        rl.draw_ring((lay.cx, cy), rad * 1.05, rad * 1.15, 0, 360, 64, COL["cyan"])


def results(r, snap: Snapshot, lay: Layout) -> None:
    p, u = r.paint, lay.unit
    _veil(r, lay, 0.85)
    w = min(lay.cw * 0.6, u * 34)
    x0, y = lay.cx - w / 2, lay.h * 0.1
    p.text("RESULTS", lay.cx, y, u * 1.2, COL["dim"], "center")
    p.text(r.chart.title.upper(), lay.cx, y + u * 1.4, u * 1.6, COL["ink"], "center")
    y += u * 3.8
    p.text(f"{snap.score:,}", x0, y, u * 3.6, COL["ink"])
    p.text(grade(snap.accuracy), x0 + w, y - u * 0.4, u * 4.4, COL["sun"], "right")
    y += u * 4.2
    summary = (f"{round(snap.accuracy * 100)}% ACCURACY    MAX COMBO {snap.max_combo}    "
               f"PERFECT {snap.counts.get('perfect', 0)}   GOOD {snap.counts.get('good', 0)}   "
               f"MISS {snap.counts.get('miss', 0)}")
    p.text(summary, x0, y, u * 0.85, COL["dim"])
    y += u * 1.8
    p.rect(x0, y, w, 1, fade(COL["ink"], 0.2))
    y += u * 0.6
    row = u * 1.35
    for k in LAYERS:
        st = snap.layers.get(k)
        if st is None or (st.total == 0 and st.mode == "you"):
            continue
        name, col = LAYER_STYLE[k]
        p.text(name, x0, y, u * 0.9, COL["ink"])
        bx, bw = x0 + u * 6, w - u * 12
        if st.mode == "auto":
            p.text("AUTO", bx, y + u * 0.1, u * 0.75, COL["dim"])
        else:
            acc = layer_accuracy(st.hit, st.total) or 0.0
            p.rect(bx, y + u * 0.3, bw, u * 0.4, TRACK)
            p.rect(bx, y + u * 0.3, bw * acc, u * 0.4, col)
            p.text(f"{st.hit}/{st.total}  {round(acc * 100)}%", x0 + w, y, u * 0.9, RESULT_COL["perfect"]
                   if acc >= 0.95 else COL["ink"], "right")
        y += row
    p.text("OK  PLAY AGAIN       BACK  MENU", lay.cx, lay.h * 0.92, u * 0.9, COL["dim"], "center")
