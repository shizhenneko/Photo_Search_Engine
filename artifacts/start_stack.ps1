$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $PSCommandPath
$projectRoot = Split-Path -Parent $scriptDir
$runtimeDir = Join-Path $projectRoot "artifacts\runtime"
$esHome = "D:\浏览器下载\elasticsearch-9.3.0-windows-x86_64\elasticsearch-9.3.0"
$esBin = Join-Path $esHome "bin\elasticsearch.bat"
$esStdout = Join-Path $runtimeDir "elasticsearch.out.log"
$esStderr = Join-Path $runtimeDir "elasticsearch.err.log"
$frontendLog = Join-Path $runtimeDir "frontend.log"
$frontendErr = Join-Path $runtimeDir "frontend.err.log"
$statusFile = Join-Path $runtimeDir "stack-status.json"
$esLauncher = Join-Path $runtimeDir "launch_elasticsearch.ps1"
$frontendLauncher = Join-Path $runtimeDir "launch_frontend.ps1"
$legacyEsLauncher = Join-Path $runtimeDir "launch_elasticsearch.cmd"
$legacyFrontendLauncher = Join-Path $runtimeDir "launch_frontend.sh"
$esBinDir = Join-Path $esHome "bin"
$esHeapOverrideFile = Join-Path $esHome "config\jvm.options.d\heap-override.options"

function Ensure-Directory([string]$path) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}

function Convert-ToWslPath([string]$windowsPath) {
    $resolved = (Resolve-Path $windowsPath).Path
    $normalized = $resolved -replace "\\", "/"
    if ($normalized -match "^([A-Za-z]):/(.*)$") {
        return "/mnt/$($matches[1].ToLower())/$($matches[2])"
    }
    throw "Unable to convert Windows path to WSL path: $windowsPath"
}

function Test-PortBusy([int]$port) {
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connections | Select-Object -First 1) {
        return $true
    }

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect("127.0.0.1", $port, $null, $null)
        $connected = $async.AsyncWaitHandle.WaitOne(400)
        if ($connected -and $client.Connected) {
            $client.EndConnect($async)
            return $true
        }
    } catch {
    } finally {
        $client.Close()
    }

    return $false
}

function Get-FreePort([int]$startPort, [int]$maxPort) {
    for ($port = $startPort; $port -le $maxPort; $port++) {
        if (-not (Test-PortBusy $port)) {
            return $port
        }
    }
    throw "No free port found in range [$startPort, $maxPort]"
}

function Stop-ProcessByCommandPattern([string]$pattern) {
    $targets = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -like $pattern
    }
    foreach ($proc in $targets) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        } catch {
        }
    }
}

function Wait-ForHttp([string]$url, [int]$timeoutSeconds) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -UseBasicParsing $url -TimeoutSec 3
            if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) {
                return $true
            }
        } catch {
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Write-TextFile([string]$path, [string]$content, [string]$encoding) {
    Set-Content -Path $path -Value $content -Encoding $encoding
}

Ensure-Directory $runtimeDir

$wslProjectRoot = Convert-ToWslPath $projectRoot
$wslRuntimeDir = Convert-ToWslPath $runtimeDir

if (-not (Test-Path $esBin)) {
    throw "Elasticsearch executable not found: $esBin"
}

Write-Host "[STEP 1] Selecting ports..."
$frontendPort = Get-FreePort 10001 10020
$esPort = 9200
$frontendUrl = "http://127.0.0.1:$frontendPort"
$esUrl = "http://127.0.0.1:$esPort"
Write-Host "[INFO] Frontend will use $frontendUrl"
Write-Host "[INFO] Elasticsearch will use $esUrl"

Write-Host "[STEP 2] Stopping old processes..."
Stop-ProcessByCommandPattern "*org.elasticsearch.bootstrap.Elasticsearch*"
Stop-ProcessByCommandPattern "*bin\\elasticsearch.bat*"
Stop-ProcessByCommandPattern "*./.venv/bin/python main.py*"
Stop-ProcessByCommandPattern "*SERVER_PORT=*main.py*"
Stop-ProcessByCommandPattern "*launch_frontend.sh*main.py*"
Start-Sleep -Seconds 2

foreach ($file in @($esStdout, $esStderr, $frontendLog, $frontendErr, $statusFile, $esLauncher, $frontendLauncher, $legacyEsLauncher, $legacyFrontendLauncher)) {
    if (Test-Path $file) {
        try {
            Remove-Item $file -Force -ErrorAction Stop
        } catch {
            Set-Content -Path $file -Value $null -Encoding UTF8
        }
    }
}

Write-Host "[STEP 3] Starting Elasticsearch..."
$esJavaOptsLine = ""
if (-not (Test-Path $esHeapOverrideFile)) {
    $esJavaOptsLine = "`$env:ES_JAVA_OPTS = '-Xms1g -Xmx1g'"
}

$esLauncherContent = @"
`$ErrorActionPreference = 'Stop'
$esJavaOptsLine
Set-Location '$esHome'
Start-Process -FilePath '$esBin' -WorkingDirectory '$esBinDir' -RedirectStandardOutput '$esStdout' -RedirectStandardError '$esStderr' -WindowStyle Minimized
"@
Write-TextFile $esLauncher $esLauncherContent "UTF8"

$null = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $esLauncher `
    -WorkingDirectory $runtimeDir `
    -WindowStyle Hidden `
    -PassThru

if (-not (Wait-ForHttp $esUrl 120)) {
    Write-Host "[ERROR] Elasticsearch failed to become ready."
    if (Test-Path $esStdout) {
        Write-Host "[INFO] Tail of Elasticsearch stdout:"
        Get-Content $esStdout -Tail 120
    }
    if (Test-Path $esStderr) {
        Write-Host "[INFO] Tail of Elasticsearch stderr:"
        Get-Content $esStderr -Tail 120
    }
    throw "Elasticsearch startup failed"
}

Write-Host "[STEP 4] Starting Flask frontend..."
$frontendLauncherContent = @"
`$ErrorActionPreference = 'Stop'
`$frontendCommand = @'
cd '$wslProjectRoot'
if ! [ -x './.venv/bin/python' ] || ! ./.venv/bin/python --version >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv --python 3.12
    uv pip install --python .venv/bin/python -r requirements.txt
  else
    python3 -m venv .venv
    ./.venv/bin/python -m pip install -r requirements.txt
  fi
fi
SERVER_HOST=127.0.0.1 SERVER_PORT=$frontendPort ELASTICSEARCH_HOST=localhost ELASTICSEARCH_PORT=$esPort ./.venv/bin/python main.py
'@
wsl.exe bash -lc `$frontendCommand
"@
Write-TextFile $frontendLauncher $frontendLauncherContent "UTF8"

$frontendProcess = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $frontendLauncher `
    -RedirectStandardOutput $frontendLog `
    -RedirectStandardError $frontendErr `
    -WindowStyle Hidden `
    -PassThru

if (-not (Wait-ForHttp $frontendUrl 120)) {
    Write-Host "[ERROR] Frontend failed to become ready."
    if (Test-Path $frontendLog) {
        Write-Host "[INFO] Tail of frontend stdout:"
        Get-Content $frontendLog -Tail 120
    }
    if (Test-Path $frontendErr) {
        Write-Host "[INFO] Tail of frontend stderr:"
        Get-Content $frontendErr -Tail 120
    }
    throw "Frontend startup failed"
}

$frontendPid = $null
try {
    if ($frontendProcess) {
        $frontendPid = $frontendProcess.Id
    }
} catch {
}

$esPid = $null
try {
    $esProc = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -like "*org.elasticsearch.bootstrap.Elasticsearch*"
    } | Select-Object -First 1
    if ($esProc) {
        $esPid = $esProc.ProcessId
    }
} catch {
}

$status = [PSCustomObject]@{
    frontend_url = $frontendUrl
    frontend_port = $frontendPort
    elasticsearch_url = $esUrl
    elasticsearch_port = $esPort
    elasticsearch_pid = $esPid
    frontend_pid = $frontendPid
    started_at = (Get-Date).ToString("s")
    runtime_dir = $runtimeDir
}

$status | ConvertTo-Json | Set-Content -Path $statusFile -Encoding UTF8

Write-Host ""
Write-Host "[DONE] Elasticsearch: $esUrl"
Write-Host "[DONE] Frontend: $frontendUrl"
Write-Host "[DONE] Status file: $statusFile"
Write-Host "[DONE] Logs:"
Write-Host "  - $esStdout"
Write-Host "  - $esStderr"
Write-Host "  - $frontendLog"
Write-Host "  - $frontendErr"
