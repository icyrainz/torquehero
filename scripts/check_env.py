"""Checks for check.ps1 that need the game's Python. One line per check:
RESULT|name|detail|fix, where RESULT is PASS, WARN, FAIL or SKIP.
Opens no joystick, no window and no sound stream."""
from __future__ import annotations

import importlib
import sys
import uuid

PACKAGES = ("torquehero", "numpy", "scipy", "soundfile", "sounddevice", "librosa", "sdl2", "pyray")


def out(result: str, name: str, detail: str, fix: str = "") -> None:
    print("|".join((result, name, detail.replace("|", "/"), fix.replace("|", "/"))), flush=True)


def packages() -> None:
    bad = []
    for name in PACKAGES:
        try:
            importlib.import_module(name)
        except Exception as e:  # a broken DLL raises OSError, not ImportError
            bad.append(f"{name} ({type(e).__name__}: {e})")
    if bad:
        out("FAIL", "Packages import", "; ".join(bad), "run .\\scripts\\setup.ps1 again")
    else:
        out("PASS", "Packages import", ", ".join(PACKAGES))


def audio() -> None:
    try:
        import sounddevice as sd

        devs = sd.query_devices()
        outs = [d for d in devs if d["max_output_channels"] > 0]
        default = sd.query_devices(kind="output")
    except Exception as e:
        out("FAIL", "Audio devices", f"{type(e).__name__}: {e}", "check the sound device in Windows sound settings")
        return
    if not outs:
        out("FAIL", "Audio devices", "no output device", "plug in speakers or headphones")
        return
    out("PASS", "Audio devices", f"{len(outs)} outputs; default: {default['name']}")


def config() -> None:
    try:
        from torquehero.config import config_dir

        d = config_dir()
        d.mkdir(parents=True, exist_ok=True)
        probe = d / f".check-{uuid.uuid4().hex}"
        probe.write_text("ok")
        probe.unlink()
    except Exception as e:
        out("FAIL", "Config dir writable", f"{type(e).__name__}: {e}", "check the folder permissions of %APPDATA%")
        return
    out("PASS", "Config dir writable", str(d))


def bindings() -> None:
    try:
        from torquehero.bindings import Bindings, default_path

        path = default_path()
        if not path.exists():
            out("WARN", "Bindings", "none saved yet", "uv run torquehero bind")
            return
        bound = sorted(Bindings.load(path).controls)
    except Exception as e:
        out("FAIL", "Bindings", f"{type(e).__name__}: {e}", "uv run torquehero bind --clear, then bind again")
        return
    if "steer" not in bound:
        out("WARN", "Bindings", f"{len(bound)} bound, steer is not (no force feedback without it)",
            "uv run torquehero bind steer")
    else:
        out("PASS", "Bindings", f"{len(bound)} bound: {' '.join(bound)}")


def cuda() -> None:
    try:
        import torch
    except ImportError:
        out("SKIP", "CUDA (stems)", "stems extra not installed", "only for gen --stems: .\\scripts\\setup.ps1 -Stems")
        return
    except Exception as e:
        out("FAIL", "CUDA (stems)", f"torch does not load: {type(e).__name__}: {e}", ".\\scripts\\setup.ps1 -Stems")
        return
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        out("PASS", "CUDA (stems)", f"torch {torch.__version__}, CUDA {torch.version.cuda}, {name}")
    else:
        out("WARN", "CUDA (stems)", f"torch {torch.__version__} has no CUDA device; gen --stems runs on the CPU",
            "update the NVIDIA driver, then .\\scripts\\setup.ps1 -Stems")


def main() -> int:
    v = sys.version.split()[0]
    if sys.version_info[:2] == (3, 12):
        out("PASS", "Python via uv", v)
    else:
        out("FAIL", "Python via uv", f"{v}, not 3.12", "run .\\scripts\\setup.ps1 again")
    packages()
    audio()
    config()
    bindings()
    cuda()
    return 0


if __name__ == "__main__":
    sys.exit(main())
