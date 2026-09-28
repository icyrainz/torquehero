<#
.SYNOPSIS
    Sets up Torque Hero on Windows: uv, Python 3.12 and the game's packages.

.DESCRIPTION
    Safe to run again: each step does only what is missing.
    It never changes driver settings, never starts the game and never sends
    force feedback.

.PARAMETER Yes
    Install uv without asking first.

.PARAMETER Stems
    Also install the 'stems' extra (demucs and CUDA torch, about 3 GB) for
    `torquehero gen --stems`, and check that the graphics card is usable.

.PARAMETER NoShortcut
    Do not put the "Torque Hero demo" shortcut on the desktop.

.EXAMPLE
    .\scripts\setup.ps1
.EXAMPLE
    .\scripts\setup.ps1 -Stems -Yes
#>
[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$Stems,
    [switch]$NoShortcut
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$UvInstall = 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'

function Step([string]$Text) { Write-Host "`n== $Text" -ForegroundColor Cyan }
function Ok([string]$Text) { Write-Host "   OK    $Text" -ForegroundColor Green }
function Warn([string]$Text) { Write-Host "   WARN  $Text" -ForegroundColor Yellow }
function Fail([string]$Text) {
    Write-Host "   FAIL  $Text" -ForegroundColor Red
    exit 1
}

function Invoke-Uv {
    & uv @args
    if ($LASTEXITCODE -ne 0) { Fail "uv $($args -join ' ') failed (exit code $LASTEXITCODE)" }
}

function Add-UvToPath {
    # The uv installer puts uv here; this PowerShell window does not see the new PATH yet.
    foreach ($dir in @("$env:USERPROFILE\.local\bin", "$env:USERPROFILE\.cargo\bin")) {
        if ((Test-Path "$dir\uv.exe") -and ($env:Path -notlike "*$dir*")) { $env:Path = "$dir;$env:Path" }
    }
}

Set-Location $Root
Write-Host "Torque Hero setup in $Root"

# --- 1. Windows and graphics card ---
Step 'Windows version'
$build = [Environment]::OSVersion.Version.Build
if ($build -ge 22000) { Ok "Windows 11 (build $build)" }
else { Warn "Windows build $build is not Windows 11. The game is made for Windows 11; it may still work." }

Step 'Graphics card'
$gpus = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
if ($gpus.Count -eq 0) { Warn 'Could not read the graphics cards.' }
foreach ($g in $gpus) { Write-Host "   $g" }
$nvidia = $gpus | Where-Object { $_ -match 'NVIDIA' }
if ($nvidia) { Ok 'NVIDIA card found' }
elseif ($Stems) { Warn 'No NVIDIA card found. -Stems will install, but stem separation will run slowly on the CPU.' }

# --- 2. uv ---
Step 'uv (installs Python and the packages)'
Add-UvToPath
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Ok "$(uv --version)"
} else {
    Write-Host '   uv is not installed. This is the official installer command:'
    Write-Host "`n     $UvInstall`n"
    if (-not $Yes) {
        $answer = Read-Host '   Run it now? [Y/n]'
        if ($answer -and $answer -notmatch '^(y|yes)$') { Fail 'uv is needed. Install it, then run setup again.' }
    }
    Invoke-Expression $UvInstall
    if ($LASTEXITCODE -ne 0) { Fail 'the uv installer failed. See its output above.' }
    Add-UvToPath
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Fail 'uv was installed but is not found. Close PowerShell, open a new one, and run setup again.'
    }
    Ok "$(uv --version)"
}

# --- 3. Python 3.12 ---
Step 'Python 3.12'
Invoke-Uv python install 3.12
Ok 'Python 3.12 is installed (through uv)'

# --- 4. Packages ---
# A plain `uv sync` removes extras that are installed. Keep stems when an earlier run added it.
$hasTorch = Test-Path (Join-Path $Root '.venv\Lib\site-packages\torch')
if ($hasTorch -and -not $Stems) {
    Write-Host '   The stems extra is already installed; keeping it.'
    $Stems = $true
}
if ($Stems) {
    Step 'Game packages with the stems extra (the first time downloads about 3 GB)'
    Invoke-Uv sync --extra stems
} else {
    Step 'Game packages'
    Invoke-Uv sync
}
Ok 'packages installed'

Step 'Game command'
& uv run --no-sync torquehero --version
if ($LASTEXITCODE -ne 0) { Fail 'torquehero does not start. See the error above.' }
Ok 'torquehero runs'

if ($Stems) {
    Step 'CUDA (graphics card support for stem separation)'
    & uv run --no-sync python -c "import torch; ok = torch.cuda.is_available(); print('   torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0) if ok else 'no CUDA device'); raise SystemExit(0 if ok else 3)"
    if ($LASTEXITCODE -eq 0) { Ok 'torch.cuda.is_available() is True' }
    else { Warn 'CUDA is not available: gen --stems will run on the CPU (slow). Update the NVIDIA driver, then run setup -Stems again.' }
}

# --- 5. Desktop shortcut ---
if (-not $NoShortcut) {
    Step 'Desktop shortcut'
    $target = Join-Path $Root 'scripts\play-demo.cmd'
    $link = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Torque Hero demo.lnk'
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut($link)
    $sc.TargetPath = $target
    $sc.WorkingDirectory = $Root
    $sc.Description = 'Torque Hero: the demo song, no force feedback until docs\FFB-CHECKLIST.md passes'
    $sc.Save()
    Ok "$link (starts the demo without force feedback; see scripts\play-demo.cmd)"
}

Write-Host "`nSetup is done. Next:" -ForegroundColor Cyan
Write-Host '   1. .\scripts\check.ps1'
Write-Host '   2. Set the Fanatec driver: rotation 1080 (not Auto), FF strength 30 to 50%'
Write-Host '   3. uv run torquehero probe --live'
Write-Host '   4. uv run torquehero bind'
Write-Host '   5. Every step of docs\FFB-CHECKLIST.md, in order, with a helper, before any song with force feedback'
Write-Host '   See README.md for each step.'
