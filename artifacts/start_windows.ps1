param(
    [int]$Port = 0
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

Ensure-Directory $dataDir
Ensure-Directory $runtimeDir

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
    $rawServerPort = $env:SERVER_PORT
    if (-not $rawServerPort) {
        $rawServerPort = Get-DotEnvValue -Path $envFile -Key "SERVER_PORT"
    }
    if (-not $rawServerPort) {
        $rawServerPort = "10001"
    }
    $serverPort = [int]$rawServerPort
}

if (Test-PortBusy $serverPort) {
    throw "Port $serverPort is already in use on Windows. Stop the existing process or rerun with -Port <new-port>."
}

Ensure-WindowsPythonEnvironment

$env:PHOTO_DIR = $photoDir
$env:DATA_DIR = $dataDir
$env:RUNTIME_DATA_DIR = $dataDir
$env:INDEX_PATH = $indexPath
$env:METADATA_PATH = $metadataPath
$env:SERVER_HOST = "127.0.0.1"
$env:SERVER_PORT = [string]$serverPort

Write-Host ""
Write-Host "[INFO] Project root: $projectRoot"
Write-Host "[INFO] Project name: $projectName"
Write-Host "[INFO] Windows runtime Python: $venvPython"
Write-Host "[INFO] PHOTO_DIR: $photoDir"
Write-Host "[INFO] DATA_DIR: $dataDir"
Write-Host "[INFO] INDEX_PATH: $indexPath"
Write-Host "[INFO] METADATA_PATH: $metadataPath"
Write-Host "[INFO] App URL: http://127.0.0.1:$serverPort/"
Write-Host ""
Write-Host "[INFO] All indexing files will be written inside the project data directory."
Write-Host "[INFO] This launcher runs only on native Windows, installs uv with PowerShell when needed, and does not use WSL."
Write-Host ""

Set-Location $projectRoot
& $venvPython $mainFile
exit $LASTEXITCODE
