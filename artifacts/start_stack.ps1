$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $PSCommandPath
$windowsScript = Join-Path $scriptDir "start_windows.ps1"

if (-not (Test-Path $windowsScript)) {
    throw "Windows launcher not found: $windowsScript"
}

& $windowsScript @args
exit $LASTEXITCODE
