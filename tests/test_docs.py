"""README commands parse with the real CLI; the Windows scripts are portable."""
import importlib.util
import re
import shlex
from pathlib import Path

import pytest

from torquehero.__main__ import build_parser

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SCRIPTS = ROOT / "scripts"
SCRIPT_FILES = ("setup.ps1", "check.ps1", "check_env.py", "play-demo.cmd", "play-triples.cmd")

# Subcommand -> module that registers it; a command whose module is not on this branch is skipped.
OWNER = {"play": "app", "probe": "bindings", "bind": "bindings", "gen": "generate",
         "demo-song": "synthsong", "companion": "companion", "preview": "render"}

USER_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|/Users/|/home/", re.IGNORECASE)


def readme_commands() -> list[str]:
    """Every `torquehero ...` line inside a fenced code block, `uv run` removed."""
    cmds, fenced = [], False
    for line in README.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        s = line.strip()
        if fenced and s.startswith("uv run torquehero "):
            s = s.removeprefix("uv run ")
        if fenced and s.startswith("torquehero "):
            cmds.append(s)
    return cmds


def cmd_scripts() -> list[str]:
    """The `torquehero ...` command of each .cmd launcher, with its `set` variables filled in."""
    out = []
    for f in sorted(SCRIPTS.glob("*.cmd")):
        text = f.read_text(encoding="utf-8")
        env = dict(re.findall(r"(?im)^set (\w+)=(.*?)\s*$", text))
        for m in re.finditer(r"(?m)^uv run (torquehero .*?)\s*$", text):
            out.append(re.sub(r"%(\w+)%", lambda v, env=env: env.get(v.group(1), v.group(0)), m.group(1)))
    return out


def split(cmd: str) -> list[str]:
    """PowerShell-ish split: quotes group words and are removed, backslashes are kept."""
    return [w[1:-1] if len(w) > 1 and w[0] == w[-1] and w[0] in "\"'" else w
            for w in shlex.split(cmd, posix=False)]


def test_readme_has_the_quickstart_commands():
    cmds = " \n".join(readme_commands())
    for needed in ("torquehero probe --live", "torquehero bind", "torquehero play --ffb-test centre",
                   "torquehero play --demo --ffb-gain 0.2", "--span 5760x1080", "torquehero gen ", "--stems",
                   "--bpm", "torquehero companion --replay", "torquehero play --no-ffb --kb"):
        assert needed in cmds, needed


@pytest.mark.parametrize("cmd", readme_commands() + cmd_scripts())
def test_command_parses(cmd):
    argv = split(cmd)[1:]
    module = OWNER.get(argv[0])
    assert module, f"unknown subcommand in {cmd!r}"
    if importlib.util.find_spec(f"torquehero.{module}") is None:
        pytest.skip(f"torquehero.{module} is not on this branch yet")
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        pytest.fail(f"{cmd!r} does not parse (exit {e.code})")
    assert args.command == argv[0]
    if getattr(args, "span", None):
        from torquehero.render import parse_span

        parse_span(args.span)


def script_play_commands() -> list[tuple[str, str]]:
    """(file, command) for every non-comment `torquehero play` line of every .cmd and .ps1 file,
    with the .cmd `set` variables filled in."""
    out = []
    for f in sorted([*SCRIPTS.glob("*.cmd"), *SCRIPTS.glob("*.ps1")]):
        text = f.read_text(encoding="utf-8")
        env = dict(re.findall(r"(?im)^set (\w+)=(.*?)\s*$", text))
        for line in text.splitlines():
            s = line.strip()
            if s.lower().startswith(("rem", "#", "::")) or "torquehero play" not in s:
                continue
            out.append((f.name, re.sub(r"%(\w+)%", lambda v, env=env: env.get(v.group(1), v.group(0)), s)))
    return out


def test_launchers_start_without_ffb():
    found = script_play_commands()
    assert {name for name, _ in found} >= {"play-demo.cmd", "play-triples.cmd"}
    for name, cmd in found:
        assert "--no-ffb" in split(cmd), f"{name}: {cmd}"


def readme_play_blocks() -> list[tuple[str, str]]:
    """(command, paragraph just above its code block) for every README `torquehero play` line."""
    out, fenced, para, prev, last = [], False, [], [], ""
    for line in README.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            if not fenced:
                last = " ".join(para or prev)
            fenced, para, prev = not fenced, [], []
            continue
        s = line.strip()
        if fenced:
            if s.startswith("uv run torquehero play"):
                out.append((s, last))
        elif s:
            para.append(s)
        elif para:
            prev, para = para, []
    return out


def test_readme_play_lines_guard_ffb():
    blocks = readme_play_blocks()
    assert blocks
    for cmd, above in blocks:
        argv = split(cmd)
        assert {"--no-ffb", "--ffb-gain", "--ffb-test"} & set(argv), cmd
        if "--no-ffb" not in argv and "--ffb-test" not in argv:
            assert "checklist" in above.lower(), f"no checklist reminder above: {cmd}"


def test_readme_ffb_patterns_follow_the_checklist_order():
    order = [m.group(1) for c in readme_commands() if (m := re.search(r"--ffb-test (\w+)", c))]
    assert order == ["kick", "centre", "echo"]


def test_ffb_checklist_exists():
    path = ROOT / "docs" / "FFB-CHECKLIST.md"
    if not path.exists():
        pytest.skip("docs/FFB-CHECKLIST.md arrives with the force feedback branch")
    text = path.read_text(encoding="utf-8")
    assert "--ffb-test kick" in text


def test_scripts_exist():
    for name in SCRIPT_FILES:
        assert (SCRIPTS / name).is_file(), name


@pytest.mark.parametrize("name", SCRIPT_FILES)
def test_scripts_have_no_user_paths(name):
    text = (SCRIPTS / name).read_text(encoding="utf-8")
    assert not USER_PATH.search(text), USER_PATH.search(text).group(0)


def test_readme_has_no_user_paths():
    assert not USER_PATH.search(README.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ("setup.ps1", "check.ps1", "play-demo.cmd", "play-triples.cmd"))
def test_windows_scripts_are_ascii(name):
    # Windows PowerShell 5.1 reads a .ps1 without a BOM as the ANSI code page; cmd.exe likewise.
    (SCRIPTS / name).read_bytes().decode("ascii")


@pytest.mark.parametrize("name", ("play-demo.cmd", "play-triples.cmd"))
def test_cmd_files_use_crlf(name):
    data = (SCRIPTS / name).read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n")


def test_setup_never_runs_the_game_or_touches_ffb():
    text = (SCRIPTS / "setup.ps1").read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("#", "Write-Host")))
    assert not re.search(r"torquehero (play|bind|probe)", code)
    assert "ffb.json" not in code
