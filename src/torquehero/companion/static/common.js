// Shared by the companion pages: live connection, waiting state, layer colours.
'use strict';
const TH = (() => {
  const LAYERS = {
    melody: { name: 'Melody', ctrl: 'Wheel', col: '#34e5ff' },
    kick: { name: 'Kick', ctrl: 'Brake', col: '#ff4d4d' },
    hat: { name: 'Hats', ctrl: 'Clutch', col: '#ffc844' },
    expr: { name: 'Swell', ctrl: 'Throttle', col: '#5cff8a' },
    pads: { name: 'Stabs', ctrl: 'Shifter', col: '#ff3fa4' },
    fills: { name: 'Fills', ctrl: 'Paddles', col: '#ff7a45' },
    riser: { name: 'Riser', ctrl: 'Handbrake', col: '#b48cff' },
    faders: { name: 'Faders', ctrl: 'Levers', col: '#7fa8ff' },
  };
  const ORDER = Object.keys(LAYERS);
  const STALE_MS = 3000;     // no snapshot for this long: show the waiting state
  const REOPEN_MS = 6000;    // ... and for this long: drop the stream and open a new one

  // Calls onSong({info, chart, results} or null) and onState(snapshot); snapshot is null while waiting.
  function connect({ onSong, onState }) {
    const waiting = document.getElementById('waiting');
    let es = null, lastState = 0, opened = 0, live = false;
    const setLive = (on, why) => {
      if (on === live && !why) return;
      live = on;
      waiting.classList.toggle('hide', on);
      if (why) waiting.querySelector('span').textContent = why;
      if (!on) onState(null);
    };
    const open = () => {
      if (es) es.close();
      opened = performance.now();
      es = new EventSource('/events');
      es.addEventListener('song', (e) => onSong(JSON.parse(e.data)));
      es.addEventListener('state', (e) => {
        const s = JSON.parse(e.data);
        if (!s) return;
        lastState = performance.now();
        s._recv = lastState;
        setLive(true);
        onState(s);
      });
      es.onerror = () => {
        setLive(false, 'No game running. This page reconnects when it starts.');
        if (es.readyState === EventSource.CLOSED) setTimeout(open, 1000);
      };
    };
    setInterval(() => {
      const now = performance.now(), since = now - Math.max(lastState, opened);
      if (live && now - lastState > STALE_MS) setLive(false, 'The game stopped sending. Waiting for it.');
      if (since > REOPEN_MS) open();
    }, 500);
    open();
  }

  // Song time now, extrapolated between snapshots while playing.
  function clock(s) {
    if (!s) return 0;
    if (s.phase !== 'play') return s.now;
    return s.now + Math.min(0.25, (performance.now() - s._recv) / 1000);
  }

  const NOTE_LABEL = {
    stab: (n) => ['Gate ' + n.gate, 'Shifter'],
    tom: (n) => ['Paddle ' + n.side, 'Fill'],
    riser: () => ['Riser', 'Pull, hold, drop'],
    spin: () => ['Spin 360°', 'Wheel'],
    expr: () => ['Swell', 'Throttle'],
    fader: (n) => ['Lever ' + (n.lever + 1), 'Fader'],
  };
  const noteLabel = (n) => (NOTE_LABEL[n.kind] || (() => [n.kind, '']))(n);
  const fmtTime = (t) => { t = Math.max(0, t || 0); return Math.floor(t / 60) + ':' + String(Math.floor(t % 60)).padStart(2, '0'); };

  // Write DOM only when the value changed: cheap on a weak tablet.
  function setText(el, v) { v = String(v); if (el._v !== v) { el._v = v; el.textContent = v; } }
  function setClass(el, v) { if (el._c !== v) { el._c = v; el.className = v; } }

  return { LAYERS, ORDER, connect, clock, noteLabel, fmtTime, setText, setClass };
})();
