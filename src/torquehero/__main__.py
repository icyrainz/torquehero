"""Command line entry point (SPEC 8.5).

Subcommands are discovered: each module in `MODULES` that exists provides
`add_cli(subparsers)`, which adds its parser and sets `func` as the handler
(`parser.set_defaults(func=handler)`, handler takes the parsed args and returns
an exit code or None). Feature modules register their own subcommands, so this file stays small. Heavy imports
(librosa, sdl2, torch, pyray) belong inside the handler.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from collections.abc import Sequence

from . import __version__

PACKAGE = "torquehero"
MODULES = ("app", "bindings", "generate", "synthsong", "companion", "render")


class DiscoveryError(RuntimeError):
    pass


def discover(subparsers, modules: Sequence[str] = MODULES, package: str = PACKAGE) -> list[str]:
    """Call `add_cli(subparsers)` of each existing module. A missing module is skipped;
    an ImportError from inside an existing module raises DiscoveryError. Returns the
    names of the modules that registered."""
    found = []
    for name in modules:
        full = f"{package}.{name}"
        if importlib.util.find_spec(full) is None:
            continue
        try:
            mod = importlib.import_module(full)
        except ImportError as e:
            raise DiscoveryError(f"module {full} exists but failed to import: {e}") from e
        add_cli = getattr(mod, "add_cli", None)
        if add_cli is not None:
            add_cli(subparsers)
            found.append(name)
    return found


def build_parser(modules: Sequence[str] = MODULES, package: str = PACKAGE) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torquehero", description="Rhythm game played on a sim-racing rig.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    discover(sub, modules, package)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        parser = build_parser()
    except DiscoveryError as e:
        print(f"torquehero: {e}", file=sys.stderr)
        raise
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return int(func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
