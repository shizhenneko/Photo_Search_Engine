$ErrorActionPreference = "Stop"

$statusFile = "C:\Users\86159\Desktop\Photo_Search_Engine\artifacts\runtime\stack-status.json"
$ports = [System.Collections.Generic.List[int]]::new()
foreach ($port in @(9200, 10001, 10002, 10003, 10004, 10005)) {
    if (-not $ports.Contains($port)) {
        $ports.Add($port)
    }
}

if (Test-Path $statusFile) {
    Write-Host "=== STATUS FILE ==="
    $statusRaw = Get-Content $statusFile -Raw
    Write-Host $statusRaw
    try {
        $status = $statusRaw | ConvertFrom-Json
        foreach ($port in @($status.elasticsearch_port, $status.frontend_port)) {
            if ($port -and -not $ports.Contains([int]$port)) {
                $ports.Add([int]$port)
            }
        }
    } catch {
    }
}

Write-Host "NOTE: WSL forwarded ports may show 'No listener' in Get-NetTCPConnection."
Write-Host "NOTE: The HTTP results below are the authoritative health check for the frontend."

foreach ($port in $ports) {
    Write-Host "=== PORT $port ==="
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if (-not $connections) {
        Write-Host "No listener"
        continue
    }

    $listeners = $connections | Where-Object { $_.State -eq "Listen" }
    if ($listeners) {
        foreach ($listener in $listeners) {
            $processId = $listener.OwningProcess
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $processId"
            [PSCustomObject]@{
                LocalAddress = $listener.LocalAddress
                LocalPort = $listener.LocalPort
                State = $listener.State
                OwningProcess = $processId
                ProcessName = $proc.Name
                ExecutablePath = $proc.ExecutablePath
                CommandLine = $proc.CommandLine
            } | Format-List
        }
        continue
    }

    Write-Host "No Windows listener, but local TCP records exist:"
    $connections | Select-Object LocalAddress, LocalPort, RemoteAddress, RemotePort, State, OwningProcess | Format-Table -AutoSize
}

Write-Host "=== HTTP ==="
foreach ($url in @("http://127.0.0.1:9200", "http://127.0.0.1:10001", "http://127.0.0.1:10002")) {
    try {
        $resp = Invoke-WebRequest -UseBasicParsing $url -TimeoutSec 5
        Write-Host "$url -> $($resp.StatusCode)"
    } catch {
        Write-Host "$url -> ERROR: $($_.Exception.Message)"
    }
}
