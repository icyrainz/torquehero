import argparse
import subprocess
import sys
import textwrap

import pytest

from torquehero.__main__ import MODULES, DiscoveryError, build_parser, discover, main


@pytest.fixture
def fake_pkg(tmp_path, monkeypatch):
    """A throwaway package on sys.path; returns a function that writes a module into it."""
    pkg = tmp_path / "fakehero"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield lambda name, src: (pkg / f"{name}.py").write_text(textwrap.dedent(src))
    for k in [k for k in sys.modules if k.startswith("fakehero")]:
        del sys.modules[k]


def test_module_list_is_fixed():
    assert MODULES == ("app", "bindings", "generate", "synthsong", "companion", "render")


def test_missing_modules_are_skipped(fake_pkg):
    fake_pkg("app", """
        def add_cli(sub):
            p = sub.add_parser("play")
            p.set_defaults(func=lambda args: 7)
    """)
    sub = argparse.ArgumentParser().add_subparsers(dest="command")
    assert discover(sub, MODULES, "fakehero") == ["app"]
    assert main.__module__ == "torquehero.__main__"


def test_handler_runs_and_returns_code(fake_pkg, monkeypatch):
    fake_pkg("app", """
        def add_cli(sub):
            p = sub.add_parser("play")
            p.add_argument("--frames", type=int)
            p.set_defaults(func=lambda args: args.frames)
    """)
    parser = build_parser(MODULES, "fakehero")
    args = parser.parse_args(["play", "--frames", "3"])
    assert args.func(args) == 3


def test_import_error_inside_module_is_surfaced(fake_pkg):
    fake_pkg("bindings", """
        import does_not_exist_anywhere
        def add_cli(sub): pass
    """)
    sub = argparse.ArgumentParser().add_subparsers()
    with pytest.raises(DiscoveryError, match="does_not_exist_anywhere"):
        discover(sub, MODULES, "fakehero")


def test_module_without_add_cli_is_ignored(fake_pkg):
    fake_pkg("render", "X = 1\n")
    sub = argparse.ArgumentParser().add_subparsers()
    assert discover(sub, MODULES, "fakehero") == []


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage: torquehero" in capsys.readouterr().out


def test_help_via_entry_point():
    r = subprocess.run([sys.executable, "-m", "torquehero", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "usage: torquehero" in r.stdout
