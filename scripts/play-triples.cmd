@echo off
rem Torque Hero across the three monitors.
rem Edit the "set" lines below in Notepad (right-click this file, Edit),
rem save, then double-click this file to play.

rem SPAN: 5760x1080 followed by the position of the LEFT monitor.
rem check.ps1 prints it, or read the "monitor N: at ..." lines the game prints.
rem Example: the left monitor is "at -1920+0", so SPAN is 5760x1080-1920+0.
set SPAN=5760x1080-1920+0

rem FFB: force feedback. It starts OFF (--no-ffb).
rem Only after every step of docs\FFB-CHECKLIST.md has passed:
rem change FFB=--no-ffb to FFB=--ffb-gain 0.2 and save.
rem Raise the gain in small steps later (0.05 at a time), one song at a time.
set FFB=--no-ffb

rem CHART: leave empty for the menu, or --demo for the demo song, or the path
rem of a chart, for example "%APPDATA%\torquehero\charts\my-song\chart.json".
set CHART=--demo

rem The song starts paused: click the game window, then press P or Enter.
cd /d "%~dp0.."
uv run torquehero play %CHART% --span %SPAN% %FFB%
if errorlevel 1 pause
