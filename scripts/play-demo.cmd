@echo off
rem Torque Hero: the built-in demo song.
rem setup.ps1 puts a shortcut to this file on the desktop.
rem
rem It starts WITHOUT force feedback (--no-ffb).
rem Only after every step of docs\FFB-CHECKLIST.md has passed:
rem change FFB=--no-ffb below to FFB=--ffb-gain 0.2 and save.
set FFB=--no-ffb

rem The song starts paused: click the game window, then press P or Enter.
cd /d "%~dp0.."
uv run torquehero play --demo %FFB%
if errorlevel 1 pause
