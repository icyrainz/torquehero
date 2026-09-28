<#
.SYNOPSIS
    Checks what Torque Hero needs on this PC and prints a PASS/FAIL table.

.DESCRIPTION
    Reads only. It installs nothing, changes no settings, opens no game window
    and sends no force feedback. `torquehero probe` reads the controllers once.
    Exit code 1 when any check fails.

.EXAMPLE
    .\scripts\check.ps1
#>
[CmdletBinding()]
param()

# Native tools write to stderr on failure; keep going and report it as a FAIL.
$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Results = New-Object System.Collections.Generic.List[object]

function Add-Result([string]$Result, [string]$Check, [string]$Detail, [string]$Fix = '') {
    $Results.Add([pscustomobject]@{ Result = $Result; Check = $Check; Detail = $Detail; Fix = $Fix })
}

function Format-Offset([int]$V) { if ($V -lt 0) { "$V" } else { "+$V" } }

# --- uv ---
foreach ($dir in @("$env:USERPROFILE\.local\bin", "$env:USERPROFILE\.cargo\bin")) {
    if ((Test-Path "$dir\uv.exe") -and ($env:Path -notlike "*$dir*")) { $env:Path = "$dir;$env:Path" }
}
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Add-Result FAIL 'uv' 'not found' 'run .\scripts\setup.ps1'
} else {
    Add-Result PASS 'uv' "$(uv --version)"
}

# --- Python, packages, audio, config dir, bindings, CUDA (scripts\check_env.py) ---
$venv = Test-Path (Join-Path $Root '.venv')
if ($uv -and $venv) {
    $lines = & uv run --no-sync python (Join-Path $PSScriptRoot 'check_env.py') 2>$null
    $seen = $false
    foreach ($line in $lines) {
        if ($line -match '^(PASS|WARN|FAIL|SKIP)\|([^|]*)\|([^|]*)\|(.*)$') {
            Add-Result $Matches[1] $Matches[2] $Matches[3] $Matches[4]
            $seen = $true
        }
    }
    if (-not $seen) {
        Add-Result FAIL 'Python via uv' 'the game''s Python did not run' 'run .\scripts\setup.ps1 again'
    }
} else {
    Add-Result FAIL 'Python via uv' 'packages not installed (no .venv folder)' 'run .\scripts\setup.ps1'
}

# --- Joysticks through SDL (torquehero probe) ---
if ($uv -and $venv) {
    $probe = (& uv run --no-sync torquehero probe 2>&1 | ForEach-Object { "$_" }) -join "`n"
    $code = $LASTEXITCODE
    $devices = @()
    foreach ($block in ($probe -split "`n\s*`n")) {
        if ($block -match '(?m)^\[(\d+)\] (.+)$') {
            # Save the groups first: the next -match overwrites $Matches.
            $index, $name = [int]$Matches[1], $Matches[2].Trim()
            $haptic = $block -match 'haptic yes'
            $devices += [pscustomobject]@{ Index = $index; Name = $name; Haptic = $haptic }
        }
    }
    $lost = if ($probe -match '(?m)^bound but device not found: (.+)$') { $Matches[1] } else { '' }
    if ($code -ne 0) {
        Add-Result FAIL 'Joysticks (SDL)' ($probe -split "`n" | Select-Object -Last 1) 'run .\scripts\setup.ps1 again'
    } elseif ($devices.Count -eq 0) {
        Add-Result FAIL 'Joysticks (SDL)' 'no devices found' 'plug in the wheel base and USB devices; install the Fanatec driver'
    } else {
        $names = ($devices | ForEach-Object { "[$($_.Index)] $($_.Name)$(if ($_.Haptic) { ' (haptic)' })" }) -join '; '
        Add-Result PASS 'Joysticks (SDL)' "$($devices.Count) found: $names"
        if (-not ($devices | Where-Object { $_.Haptic })) {
            Add-Result WARN 'Force feedback device' 'no device reports haptic support' 'check the wheel base USB cable and the Fanatec driver'
        } else {
            Add-Result PASS 'Force feedback device' (($devices | Where-Object { $_.Haptic } | ForEach-Object { $_.Name }) -join '; ')
        }
    }
    if ($lost) {
        Add-Result WARN 'Bound devices' "bound but not found: $lost" 'plug the device in, or bind those controls again'
    }
} else {
    Add-Result SKIP 'Joysticks (SDL)' 'needs the packages' 'run .\scripts\setup.ps1'
}

# --- Monitors ---
try {
    # Real pixels, not scaled ones, so the positions match what the game prints.
    if (-not ('TorqueHero.Dpi' -as [type])) {
        Add-Type -Namespace TorqueHero -Name Dpi -MemberDefinition '[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();'
    }
    [void][TorqueHero.Dpi]::SetProcessDPIAware()
    Add-Type -AssemblyName System.Windows.Forms
    $screens = @([System.Windows.Forms.Screen]::AllScreens | Sort-Object { $_.Bounds.X })
    $list = ($screens | ForEach-Object {
        $b = $_.Bounds
        "$($b.Width)x$($b.Height) at $(Format-Offset $b.X)$(Format-Offset $b.Y)$(if ($_.Primary) { ' (main)' })"
    }) -join '; '
    Add-Result PASS 'Monitors' "$($screens.Count): $list"

    # Three monitors of one height, edge to edge on one row: suggest the span.
    $span = ''
    foreach ($row in ($screens | Group-Object { "$($_.Bounds.Y) $($_.Bounds.Height)" })) {
        $s = @($row.Group | Sort-Object { $_.Bounds.X })
        for ($i = 0; $i + 2 -lt $s.Count; $i++) {
            $a, $b, $c = $s[$i].Bounds, $s[$i + 1].Bounds, $s[$i + 2].Bounds
            if ($a.Right -eq $b.X -and $b.Right -eq $c.X) {
                $span = "$($a.Width + $b.Width + $c.Width)x$($a.Height)$(Format-Offset $a.X)$(Format-Offset $a.Y)"
                break
            }
        }
        if ($span) { break }
    }
    $wide = $screens | Where-Object { $_.Bounds.Width -ge 5760 }
    if ($span) {
        Add-Result PASS 'Triples span' "--span $span" 'put it in scripts\play-triples.cmd'
    } elseif ($wide) {
        Add-Result PASS 'Triples span' 'one display is 5760 wide (NVIDIA Surround): use --fullscreen'
    } else {
        Add-Result WARN 'Triples span' 'no three side-by-side monitors of one height' 'Settings > System > Display: line up the triples'
    }
} catch {
    Add-Result WARN 'Monitors' "could not read: $($_.Exception.Message)" 'read the "monitor N" lines the game prints'
}

# --- Report ---
$colors = @{ PASS = 'Green'; WARN = 'Yellow'; FAIL = 'Red'; SKIP = 'DarkGray' }
$width = ($Results | ForEach-Object { $_.Check.Length } | Measure-Object -Maximum).Maximum
Write-Host ''
foreach ($r in $Results) {
    Write-Host ('{0,-5} ' -f $r.Result) -ForegroundColor $colors[$r.Result] -NoNewline
    Write-Host ("{0,-$width}  {1}" -f $r.Check, $r.Detail)
    if ($r.Fix -and $r.Result -ne 'PASS') {
        Write-Host ("      {0,-$width}  fix: {1}" -f '', $r.Fix) -ForegroundColor $colors[$r.Result]
    }
}
Write-Host ''
Write-Host 'Not checked here: the Fanatec driver settings (rotation 1080, not Auto) and the force feedback checks in docs\FFB-CHECKLIST.md.'

$failed = @($Results | Where-Object { $_.Result -eq 'FAIL' }).Count
if ($failed) {
    Write-Host "$failed check(s) failed." -ForegroundColor Red
    exit 1
}
Write-Host 'No check failed.' -ForegroundColor Green
exit 0
