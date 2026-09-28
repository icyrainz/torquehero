"""Companion pages: a local web server streaming the game Snapshot to extra screens.

The app calls `start_companion(host, port)` once, `publish_song(info, chart)`
when a song starts and `publish(snapshot)` every frame, then `close()`.
"""
from __future__ import annotations

import sys

from .server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    CompanionError,
    CompanionServer,
    CompanionSettings,
    NullCompanion,
    start_companion,
)

__all__ = [
    "DEFAULT_HOST", "DEFAULT_PORT", "CompanionError", "CompanionServer", "CompanionSettings",
    "NullCompanion", "add_cli", "start_companion",
]


def add_cli(sub) -> None:
    p = sub.add_parser("companion", help="serve the companion pages (/dash, /top) without the game")
    p.add_argument("--replay", metavar="FIXTURE", nargs="?", const="", default=None,
                   help="animate a sample snapshot JSON (default: the sample shipped in the package); "
                        "a chart.json next to it drives the notes")
    p.add_argument("--chart", metavar="CHART", help="chart for --replay (default: chart.json next to FIXTURE)")
    p.add_argument("--host", help=f"bind address (saved setting, default {DEFAULT_HOST})")
    p.add_argument("--port", type=int, help=f"port (saved setting, default {DEFAULT_PORT}); 0 picks a free one")
    p.add_argument("--seconds", type=float, help="stop after this many seconds")
    p.set_defaults(func=_cmd)


def _cmd(args) -> int:
    import time

    from . import replay

    try:
        s = CompanionSettings.load()
        server = CompanionServer(args.host or s.host, s.port if args.port is None else args.port,
                                 allowed_hosts=s.allowed_hosts, port_flag="--port").start()
    except CompanionError as e:
        print(e, file=sys.stderr)
        return 1
    print(server.start_line(), flush=True)
    try:
        if args.replay is not None:
            rp = replay.load(args.replay or replay.default_fixture(), args.chart)
            replay.run(server, rp, args.seconds)
        elif args.seconds is not None:
            time.sleep(args.seconds)
        else:
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as e:  # unreadable fixture or chart
        print(f"companion: {e}", file=sys.stderr)
        return 1
    finally:
        server.close()
    return 0
