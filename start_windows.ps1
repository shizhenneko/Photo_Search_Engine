$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $PSCommandPath
$launcher = Join-Path $scriptDir "artifacts\start_windows.ps1"

if (-not (Test-Path $launcher)) {
    throw "Windows launcher not found: $launcher"
}

& $launcher @args
exit $LASTEXITCODE
