param(
    [int]$Port = 0,
    [switch]$ElasticsearchOnly
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $PSCommandPath
$projectRoot = Split-Path -Parent $scriptDir
$projectName = Split-Path -Leaf $projectRoot
$envFile = Join-Path $projectRoot ".env"
$envExampleFile = Join-Path $projectRoot ".env.example"
$requirementsFile = Join-Path $projectRoot "requirements.txt"
$venvDir = Join-Path $projectRoot ".venv-windows"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$requirementsHashFile = Join-Path $venvDir "requirements.sha256"
$dataDir = Join-Path $projectRoot "data"
$runtimeDir = Join-Path $projectRoot "artifacts\runtime-windows"
$mainFile = Join-Path $projectRoot "main.py"
$indexPath = Join-Path $dataDir "photo_search.index"
$metadataPath = Join-Path $dataDir "metadata.json"
$stackStatusFile = Join-Path $runtimeDir "stack-status.json"
$managedEsRoot = Join-Path $projectRoot "artifacts\elasticsearch"
$managedEsRuntimeDir = Join-Path $runtimeDir "elasticsearch"
$managedEsConfigDir = Join-Path $managedEsRuntimeDir "config"
$managedEsDataDir = Join-Path $managedEsRuntimeDir "data"
$managedEsLogsDir = Join-Path $managedEsRuntimeDir "logs"
$managedEsTmpDir = Join-Path $managedEsRuntimeDir "tmp"
$managedEsStdout = Join-Path $managedEsRuntimeDir "elasticsearch.stdout.log"
$managedEsStderr = Join-Path $managedEsRuntimeDir "elasticsearch.stderr.log"

function Ensure-Directory([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Get-CommandPathOrNull([string]$Name) {
    try {
        return (Get-Command $Name -ErrorAction Stop).Source
    } catch {
        return $null
    }
}

function Get-FileSha256([string]$Path) {
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash
}

function Get-DotEnvValue([string]$Path, [string]$Key) {
    if (-not (Test-Path $Path)) {
        return $null
    }

    foreach ($line in Get-Content -Path $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        if ($trimmed -match "^$([regex]::Escape($Key))\s*=\s*(.*)$") {
            $value = $matches[1].Trim()
            if (
                ($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))
            ) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            return $value
        }
    }

    return $null
}

function Get-SettingValue([string]$Key, [string]$DefaultValue = $null) {
    $processValue = [System.Environment]::GetEnvironmentVariable($Key, "Process")
    if ($null -ne $processValue) {
        return $processValue
    }

    $fileValue = Get-DotEnvValue -Path $envFile -Key $Key
    if ($null -ne $fileValue) {
        return $fileValue
    }

    return $DefaultValue
}

function Convert-WslPathToWindows([string]$PathValue) {
    if ($PathValue -match "^/mnt/([a-zA-Z])/(.*)$") {
        $drive = $matches[1].ToUpper()
        $rest = $matches[2] -replace "/", "\"
        return "${drive}:\$rest"
    }
    return $PathValue
}

function Resolve-AbsolutePath([string]$PathValue, [string]$BaseDir) {
    if (-not $PathValue) {
        return $null
    }
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $PathValue))
}

function Test-PortBusy([int]$PortNumber) {
    $connections = Get-NetTCPConnection -LocalPort $PortNumber -ErrorAction SilentlyContinue
    return [bool]($connections | Select-Object -First 1)
}

function Ensure-UvPath() {
    $uvPath = Get-CommandPathOrNull "uv"
    if ($uvPath) {
        return $uvPath
    }

    Write-Host "[STEP] Installing uv for native Windows runtime"
    $installScript = Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1"
    Invoke-Expression $installScript

    $candidatePaths = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")
    )

    foreach ($candidate in $candidatePaths) {
        if (Test-Path $candidate) {
            $env:PATH = "$(Split-Path -Parent $candidate);$env:PATH"
            return $candidate
        }
    }

    $uvPath = Get-CommandPathOrNull "uv"
    if ($uvPath) {
        return $uvPath
    }

    throw "uv installation failed on Windows."
}

function Ensure-WindowsPythonEnvironment() {
    $uvPath = Ensure-UvPath
    $needsCreate = -not (Test-Path $venvPython)

    if ($needsCreate) {
        Write-Host "[STEP] Creating Windows virtual environment at $venvDir"
        & $uvPath venv $venvDir --python 3.12
    }

    $requirementsHash = Get-FileSha256 $requirementsFile
    $installedHash = ""
    if (Test-Path $requirementsHashFile) {
        $installedHash = (Get-Content -Path $requirementsHashFile -Raw -Encoding UTF8).Trim()
    }

    if ($needsCreate -or $installedHash -ne $requirementsHash) {
        Write-Host "[STEP] Installing Python dependencies for Windows runtime"
        & $uvPath pip install --python $venvPython -r $requirementsFile
        Set-Content -Path $requirementsHashFile -Value $requirementsHash -Encoding UTF8
    }
}

function Get-PythonElasticsearchVersion() {
    if (-not (Test-Path $venvPython)) {
        return $null
    }

    try {
        $version = & $venvPython -c "import elasticsearch; print(elasticsearch.__versionstr__)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $version) {
            return $version.Trim()
        }
    } catch {
    }

    return $null
}

function Test-IsLocalElasticsearchHost([string]$HostValue) {
    if ($null -eq $HostValue) {
        return $false
    }

    $normalized = $HostValue.Trim().ToLowerInvariant()
    return $normalized -in @("localhost", "127.0.0.1", "::1", "[::1]")
}

function Get-ManagedElasticsearchBindHost([string]$HostValue) {
    $normalized = ""
    if ($null -ne $HostValue) {
        $normalized = $HostValue.Trim().ToLowerInvariant()
    }
    if ($normalized -in @("::1", "[::1]")) {
        return "::1"
    }
    return "127.0.0.1"
}

function Get-HttpStatusCode([string]$Url) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
        return [int]$response.StatusCode
    } catch {
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
            return [int]$_.Exception.Response.StatusCode
        }
        return $null
    }
}

function Test-ElasticsearchReady([string]$HostValue, [int]$PortNumber) {
    $statusCode = Get-HttpStatusCode -Url "http://$HostValue`:$PortNumber"
    return $statusCode -in @(200, 401)
}

function Wait-ForElasticsearch(
    [string]$HostValue,
    [int]$PortNumber,
    [int]$TimeoutSeconds,
    [System.Diagnostics.Process]$Process = $null
) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-ElasticsearchReady -HostValue $HostValue -PortNumber $PortNumber) {
            return $true
        }
        if ($Process -and $Process.HasExited) {
            return $false
        }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Read-StackStatus([string]$Path) {
    if (-not (Test-Path $Path)) {
        return $null
    }

    try {
        return (Get-Content -Path $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Try-Stop-StaleManagedElasticsearch([string]$Path) {
    $status = Read-StackStatus -Path $Path
    if (-not $status) {
        return
    }

    $isManaged = $status.elasticsearch_managed
    $pidValue = $status.elasticsearch_pid
    if (-not $isManaged -or -not $pidValue) {
        return
    }

    try {
        $process = Get-Process -Id ([int]$pidValue) -ErrorAction Stop
        Write-Host "[STEP] Stopping stale managed Elasticsearch process $($process.Id)"
        Stop-Process -Id $process.Id -Force
        Start-Sleep -Seconds 2
    } catch {
    }
}

function Convert-PathToYamlLiteral([string]$PathValue) {
    return ([System.IO.Path]::GetFullPath($PathValue)).Replace("\", "/")
}

function Write-ManagedElasticsearchConfig([string]$EsHome, [string]$BindHost, [int]$PortNumber) {
    Ensure-Directory $managedEsConfigDir
    Ensure-Directory $managedEsDataDir
    Ensure-Directory $managedEsLogsDir
    Ensure-Directory $managedEsTmpDir

    $defaultConfigDir = Join-Path $EsHome "config"
    if (-not (Test-Path $defaultConfigDir -PathType Container)) {
        throw "Elasticsearch config directory not found: $defaultConfigDir"
    }
    Copy-Item -Path (Join-Path $defaultConfigDir "*") -Destination $managedEsConfigDir -Recurse -Force

    $configPath = Join-Path $managedEsConfigDir "elasticsearch.yml"
    $dataPath = Convert-PathToYamlLiteral -PathValue $managedEsDataDir
    $logsPath = Convert-PathToYamlLiteral -PathValue $managedEsLogsDir

    $configContent = @"
cluster.name: photo-search-engine
node.name: photo-search-engine-windows
network.host: $BindHost
http.port: $PortNumber
discovery.type: single-node
xpack.security.enabled: false
xpack.security.http.ssl.enabled: false
xpack.security.transport.ssl.enabled: false
xpack.ml.enabled: false
path.data: '$dataPath'
path.logs: '$logsPath'
"@

    Set-Content -Path $configPath -Value $configContent -Encoding UTF8
}

function Resolve-ConfiguredElasticsearchHome() {
    $configuredBatPath = Get-SettingValue -Key "ELASTICSEARCH_BAT_PATH"
    if (-not [string]::IsNullOrWhiteSpace($configuredBatPath)) {
        $resolvedBatPath = Resolve-AbsolutePath -PathValue $configuredBatPath -BaseDir $projectRoot
        if (-not (Test-Path $resolvedBatPath)) {
            throw "ELASTICSEARCH_BAT_PATH does not exist: $resolvedBatPath"
        }
        return Split-Path -Parent (Split-Path -Parent $resolvedBatPath)
    }

    $configuredHome = Get-SettingValue -Key "ELASTICSEARCH_HOME"
    if (-not [string]::IsNullOrWhiteSpace($configuredHome)) {
        $resolvedHome = Resolve-AbsolutePath -PathValue $configuredHome -BaseDir $projectRoot
        $resolvedBatPath = Join-Path $resolvedHome "bin\elasticsearch.bat"
        if (-not (Test-Path $resolvedBatPath)) {
            throw "ELASTICSEARCH_HOME is missing bin\\elasticsearch.bat: $resolvedHome"
        }
        return $resolvedHome
    }

    return $null
}

function Ensure-ManagedElasticsearchHome([string]$Version) {
    $configuredHome = Resolve-ConfiguredElasticsearchHome
    if ($configuredHome) {
        return $configuredHome
    }

    Ensure-Directory $managedEsRoot

    $zipName = "elasticsearch-$Version-windows-x86_64.zip"
    $zipPath = Join-Path $managedEsRoot $zipName
    $esHome = Join-Path $managedEsRoot "elasticsearch-$Version"
    $esBat = Join-Path $esHome "bin\elasticsearch.bat"

    if (Test-Path $esBat) {
        return $esHome
    }

    $downloadUrl = "https://artifacts.elastic.co/downloads/elasticsearch/$zipName"
    if (-not (Test-Path $zipPath)) {
        Write-Host "[STEP] Downloading Elasticsearch $Version"
        Invoke-WebRequest -UseBasicParsing -Uri $downloadUrl -OutFile $zipPath
    }

    Write-Host "[STEP] Extracting Elasticsearch $Version"
    Expand-Archive -Path $zipPath -DestinationPath $managedEsRoot -Force

    if (-not (Test-Path $esBat)) {
        throw "Elasticsearch archive was extracted, but $esBat was not found."
    }

    return $esHome
}

function Get-ManagedElasticsearchProcessId([int]$WrapperPid) {
    try {
        $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $WrapperPid"
        $javaProcess = $children | Where-Object { $_.Name -ieq "java.exe" } | Select-Object -First 1
        if ($javaProcess) {
            return [int]$javaProcess.ProcessId
        }
    } catch {
    }

    return $WrapperPid
}

function Get-LastLogLines([string]$Path, [int]$Tail = 20) {
    if (-not (Test-Path $Path)) {
        return @()
    }

    try {
        return @(Get-Content -Path $Path -Tail $Tail -Encoding UTF8)
    } catch {
        return @()
    }
}

function Start-ManagedElasticsearch(
    [string]$HostValue,
    [int]$PortNumber,
    [string]$Version
) {
    $esHome = Ensure-ManagedElasticsearchHome -Version $Version
    $bindHost = Get-ManagedElasticsearchBindHost -HostValue $HostValue

    if (Test-ElasticsearchReady -HostValue $HostValue -PortNumber $PortNumber) {
        Write-Host "[INFO] Elasticsearch already reachable at http://$HostValue`:$PortNumber"
        return @{
            Url = "http://$HostValue`:$PortNumber"
            Managed = $false
            Pid = $null
            Version = $Version
        }
    }

    if (Test-PortBusy $PortNumber) {
        Try-Stop-StaleManagedElasticsearch -Path $stackStatusFile
    }

    if (Test-PortBusy $PortNumber) {
        throw "Port $PortNumber is busy, but Elasticsearch is not responding at http://$HostValue`:$PortNumber. Stop the process using this port or change ELASTICSEARCH_PORT."
    }

    Write-ManagedElasticsearchConfig -EsHome $esHome -BindHost $bindHost -PortNumber $PortNumber

    if (Test-Path $managedEsStdout) {
        Remove-Item -Path $managedEsStdout -Force
    }
    if (Test-Path $managedEsStderr) {
        Remove-Item -Path $managedEsStderr -Force
    }

    $previousEsPathConf = [System.Environment]::GetEnvironmentVariable("ES_PATH_CONF", "Process")
    $previousEsJavaOpts = [System.Environment]::GetEnvironmentVariable("ES_JAVA_OPTS", "Process")
    $previousEsTmpDir = [System.Environment]::GetEnvironmentVariable("ES_TMPDIR", "Process")

    [System.Environment]::SetEnvironmentVariable("ES_PATH_CONF", $managedEsConfigDir, "Process")
    [System.Environment]::SetEnvironmentVariable("ES_JAVA_OPTS", "-Xms1g -Xmx1g", "Process")
    [System.Environment]::SetEnvironmentVariable("ES_TMPDIR", $managedEsTmpDir, "Process")

    try {
        Write-Host "[STEP] Starting managed Elasticsearch $Version at http://$HostValue`:$PortNumber"
        $wrapperProcess = Start-Process `
            -FilePath "cmd.exe" `
            -ArgumentList @("/c", "`"$esHome\bin\elasticsearch.bat`"") `
            -WorkingDirectory $esHome `
            -PassThru `
            -RedirectStandardOutput $managedEsStdout `
            -RedirectStandardError $managedEsStderr
    } finally {
        [System.Environment]::SetEnvironmentVariable("ES_PATH_CONF", $previousEsPathConf, "Process")
        [System.Environment]::SetEnvironmentVariable("ES_JAVA_OPTS", $previousEsJavaOpts, "Process")
        [System.Environment]::SetEnvironmentVariable("ES_TMPDIR", $previousEsTmpDir, "Process")
    }

    if (-not (Wait-ForElasticsearch -HostValue $HostValue -PortNumber $PortNumber -TimeoutSeconds 120 -Process $wrapperProcess)) {
        $stdoutLines = Get-LastLogLines -Path $managedEsStdout
        $stderrLines = Get-LastLogLines -Path $managedEsStderr
        $logHint = @()
        if ($stdoutLines.Count -gt 0) {
            $logHint += "STDOUT tail:"
            $logHint += $stdoutLines
        }
        if ($stderrLines.Count -gt 0) {
            $logHint += "STDERR tail:"
            $logHint += $stderrLines
        }
        $details = if ($logHint.Count -gt 0) { "`n" + ($logHint -join "`n") } else { "" }
        throw "Managed Elasticsearch failed to become ready within 120 seconds.$details"
    }

    return @{
        Url = "http://$HostValue`:$PortNumber"
        Managed = $true
        Pid = (Get-ManagedElasticsearchProcessId -WrapperPid $wrapperProcess.Id)
        Version = $Version
    }
}

function Write-StackStatus(
    [int]$FrontendPort,
    [string]$ElasticsearchUrl,
    [int]$ElasticsearchPort,
    [bool]$ElasticsearchManaged,
    [Nullable[int]]$ElasticsearchPid,
    [string]$ElasticsearchVersion
) {
    $status = [PSCustomObject]@{
        generated_at = (Get-Date).ToString("o")
        frontend_port = $FrontendPort
        frontend_url = "http://127.0.0.1:$FrontendPort/"
        elasticsearch_port = $ElasticsearchPort
        elasticsearch_url = $ElasticsearchUrl
        elasticsearch_managed = $ElasticsearchManaged
        elasticsearch_pid = $ElasticsearchPid
        elasticsearch_version = $ElasticsearchVersion
        runtime_dir = $runtimeDir
    }

    $status | ConvertTo-Json -Depth 4 | Set-Content -Path $stackStatusFile -Encoding UTF8
}

Ensure-Directory $dataDir
Ensure-Directory $runtimeDir
Ensure-Directory $managedEsRuntimeDir

if (-not (Test-Path $envFile)) {
    if (Test-Path $envExampleFile) {
        Copy-Item -Path $envExampleFile -Destination $envFile
    }
    throw "Missing .env. A new .env has been created from .env.example. Please set PHOTO_DIR and your API settings, then rerun the script."
}

$rawPhotoDir = $env:PHOTO_DIR
if (-not $rawPhotoDir) {
    $rawPhotoDir = Get-DotEnvValue -Path $envFile -Key "PHOTO_DIR"
}
if (-not $rawPhotoDir) {
    throw "PHOTO_DIR is not set. Please update .env with a Windows photo directory before running this script."
}

$photoDir = Resolve-AbsolutePath -PathValue (Convert-WslPathToWindows $rawPhotoDir) -BaseDir $projectRoot
if (-not (Test-Path $photoDir -PathType Container)) {
    throw "PHOTO_DIR does not exist on Windows: $photoDir"
}

$serverPort = $Port
if ($serverPort -le 0) {
    $rawServerPort = Get-SettingValue -Key "SERVER_PORT" -DefaultValue "10001"
    $serverPort = [int]$rawServerPort
}

if (Test-PortBusy $serverPort) {
    Write-Host "[WARN] Port $serverPort appears to be in use. The app will try a fallback port automatically."
}

Ensure-WindowsPythonEnvironment

$resolvedEsHost = Get-SettingValue -Key "ELASTICSEARCH_HOST" -DefaultValue "localhost"
$resolvedEsPort = [int](Get-SettingValue -Key "ELASTICSEARCH_PORT" -DefaultValue "9200")
$elasticsearchEnabled = -not [string]::IsNullOrWhiteSpace($resolvedEsHost)
$manageLocalElasticsearch = $elasticsearchEnabled -and (Test-IsLocalElasticsearchHost -HostValue $resolvedEsHost)
$elasticsearchInfo = $null

if ($manageLocalElasticsearch) {
    $defaultEsVersion = Get-PythonElasticsearchVersion
    if (-not $defaultEsVersion) {
        $defaultEsVersion = "9.3.1"
    }
    $resolvedEsVersion = Get-SettingValue -Key "ELASTICSEARCH_VERSION" -DefaultValue $defaultEsVersion
    $elasticsearchInfo = Start-ManagedElasticsearch -HostValue $resolvedEsHost -PortNumber $resolvedEsPort -Version $resolvedEsVersion
} elseif ($elasticsearchEnabled) {
    Write-Host "[INFO] Using external Elasticsearch at http://$resolvedEsHost`:$resolvedEsPort"
} else {
    Write-Host "[INFO] Elasticsearch is disabled because ELASTICSEARCH_HOST is empty."
}

$env:PHOTO_DIR = $photoDir
$env:DATA_DIR = $dataDir
$env:RUNTIME_DATA_DIR = $dataDir
$env:INDEX_PATH = $indexPath
$env:METADATA_PATH = $metadataPath
$env:SERVER_HOST = "127.0.0.1"
$env:SERVER_PORT = [string]$serverPort

if ($elasticsearchEnabled) {
    $env:ELASTICSEARCH_HOST = $resolvedEsHost
    $env:ELASTICSEARCH_PORT = [string]$resolvedEsPort
}
if ($manageLocalElasticsearch) {
    $env:ELASTICSEARCH_USERNAME = ""
    $env:ELASTICSEARCH_PASSWORD = ""
}

$elasticsearchUrl = if ($elasticsearchEnabled) { "http://$resolvedEsHost`:$resolvedEsPort" } else { $null }
$elasticsearchManaged = [bool]($elasticsearchInfo -and $elasticsearchInfo.Managed)
$elasticsearchPid = if ($elasticsearchInfo -and $elasticsearchInfo.Pid) { [int]$elasticsearchInfo.Pid } else { $null }
$elasticsearchVersion = if ($elasticsearchInfo -and $elasticsearchInfo.Version) { [string]$elasticsearchInfo.Version } else { $null }

Write-StackStatus `
    -FrontendPort $serverPort `
    -ElasticsearchUrl $elasticsearchUrl `
    -ElasticsearchPort $(if ($elasticsearchEnabled) { $resolvedEsPort } else { 0 }) `
    -ElasticsearchManaged $elasticsearchManaged `
    -ElasticsearchPid $elasticsearchPid `
    -ElasticsearchVersion $elasticsearchVersion

Write-Host ""
Write-Host "[INFO] Project root: $projectRoot"
Write-Host "[INFO] Project name: $projectName"
Write-Host "[INFO] Windows runtime Python: $venvPython"
Write-Host "[INFO] PHOTO_DIR: $photoDir"
Write-Host "[INFO] DATA_DIR: $dataDir"
Write-Host "[INFO] INDEX_PATH: $indexPath"
Write-Host "[INFO] METADATA_PATH: $metadataPath"
if ($elasticsearchEnabled) {
    Write-Host "[INFO] Elasticsearch URL: $elasticsearchUrl"
    if ($elasticsearchManaged) {
        Write-Host "[INFO] Elasticsearch PID: $elasticsearchPid"
        Write-Host "[INFO] Elasticsearch logs: $managedEsLogsDir"
    }
}
Write-Host "[INFO] App URL: http://127.0.0.1:$serverPort/"
Write-Host ""
Write-Host "[INFO] All indexing files will be written inside the project data directory."
Write-Host "[INFO] This launcher runs only on native Windows, installs uv when needed, and can manage a local Elasticsearch instance."
Write-Host ""

if ($ElasticsearchOnly) {
    Write-Host "[INFO] Elasticsearch bootstrap completed. App launch was skipped because -ElasticsearchOnly was specified."
    exit 0
}

Set-Location $projectRoot
& $venvPython $mainFile
exit $LASTEXITCODE
