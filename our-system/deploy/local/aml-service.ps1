# AML Memory local service (127.0.0.1:8787). Keys never printed to console.
#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Start', 'Stop', 'Status', 'Configure', 'RotateKey', 'ShowKeyOnce')]
    [string]$Action,

    [ValidateSet('Hybrid', 'Fts')]
    [string]$Mode = 'Hybrid'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Security

$script:AmlPython = 'F:\competitions\1AML\our-system\.venv312-tk14\Scripts\python.exe'
$script:AmlWorkDir = 'F:\competitions\1AML\our-system'
$script:AmlProdRoot = 'F:\competitions\.aml-prod'
$script:AmlHost = '127.0.0.1'
$script:AmlPort = 8787
$script:DashScopeEndpoint = 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding'

function Get-AmlProdPaths {
    $root = $script:AmlProdRoot
    @{
        Root   = $root
        DbDir  = Join-Path $root 'db'
        Logs   = Join-Path $root 'logs'
        Run    = Join-Path $root 'run'
        Pid    = Join-Path $root 'run\aml-service.pid'
        State  = Join-Path $root 'run\state.json'
    }
}

function Get-AmlServiceVaultPath {
    Join-Path $env:LOCALAPPDATA 'AML-Memory\credentials\service.dpapi'
}

function Get-AmlDashScopeVaultPath {
    Join-Path $env:LOCALAPPDATA 'AML-Memory\credentials\dashscope.dpapi'
}

function New-AmlUrlSafeKey {
    $bytes = New-Object byte[] 32
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $rng.GetBytes($bytes)
    } finally {
        if ($rng -is [IDisposable]) { $rng.Dispose() }
    }
    $b64 = [Convert]::ToBase64String($bytes)
    [Array]::Clear($bytes, 0, $bytes.Length)
    ($b64.TrimEnd('=') -replace '\+', '-' -replace '/', '_')
}

function Save-AmlServiceKey {
    param(
        [Parameter(Mandatory = $true)][string]$PlainKey,
        [string]$Path = (Get-AmlServiceVaultPath)
    )
    if ([string]::IsNullOrWhiteSpace($PlainKey) -or $PlainKey -match '\s') {
        throw 'invalid_service_key_format'
    }
    $plainBytes = [Text.Encoding]::UTF8.GetBytes($PlainKey)
    $encrypted = [Security.Cryptography.ProtectedData]::Protect(
        $plainBytes, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
    [Array]::Clear($plainBytes, 0, $plainBytes.Length)
    $target = [IO.Path]::GetFullPath($Path)
    $directory = [IO.Path]::GetDirectoryName($target)
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    $temporary = Join-Path $directory ('service-' + [Guid]::NewGuid().ToString('N') + '.tmp')
    try {
        [IO.File]::WriteAllBytes($temporary, $encrypted)
        if ([IO.File]::Exists($target)) {
            [IO.File]::Replace($temporary, $target, [System.Management.Automation.Language.NullString]::Value)
        } else {
            [IO.File]::Move($temporary, $target)
        }
        $temporary = $null
    } finally {
        if ($null -ne $temporary -and [IO.File]::Exists($temporary)) {
            [IO.File]::Delete($temporary)
        }
    }
}

function Ensure-AmlServiceKeyExists {
    $path = Get-AmlServiceVaultPath
    if (-not [IO.File]::Exists($path)) {
        Save-AmlServiceKey -PlainKey (New-AmlUrlSafeKey)
    }
}

function Invoke-AmlWithServiceKey {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [string]$Path = (Get-AmlServiceVaultPath)
    )
    $previous = [Environment]::GetEnvironmentVariable('AML_SERVICE_API_KEY', 'Process')
    $decodedBytes = $null
    try {
        if ([string]::IsNullOrWhiteSpace($previous)) {
            if (-not [IO.File]::Exists($Path)) { throw 'service_key_not_configured' }
            $decodedBytes = [Security.Cryptography.ProtectedData]::Unprotect(
                [IO.File]::ReadAllBytes($Path), $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
            $env:AML_SERVICE_API_KEY = [Text.Encoding]::UTF8.GetString($decodedBytes)
        }
        if ([string]::IsNullOrWhiteSpace($env:AML_SERVICE_API_KEY) -or $env:AML_SERVICE_API_KEY -match '\s') {
            throw 'invalid_service_key_format'
        }
        & $Action
    } finally {
        [Environment]::SetEnvironmentVariable('AML_SERVICE_API_KEY', $previous, 'Process')
        if ($null -ne $decodedBytes) { [Array]::Clear($decodedBytes, 0, $decodedBytes.Length) }
        $previous = $null
    }
}

function Invoke-AmlWithDashScopeKey {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [string]$Path = (Get-AmlDashScopeVaultPath)
    )
    $amlPreviousKey = [Environment]::GetEnvironmentVariable('DASHSCOPE_API_KEY', 'Process')
    $amlDecodedBytes = $null
    try {
        if ([string]::IsNullOrWhiteSpace($amlPreviousKey)) {
            if (-not [IO.File]::Exists($Path)) { throw 'dashscope_key_not_configured' }
            $amlDecodedBytes = [Security.Cryptography.ProtectedData]::Unprotect(
                [IO.File]::ReadAllBytes($Path), $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
            $env:DASHSCOPE_API_KEY = [Text.Encoding]::UTF8.GetString($amlDecodedBytes)
        }
        if ([string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY) -or $env:DASHSCOPE_API_KEY -match '\s') {
            throw 'invalid_credential_format'
        }
        & $Action
    } finally {
        [Environment]::SetEnvironmentVariable('DASHSCOPE_API_KEY', $amlPreviousKey, 'Process')
        if ($null -ne $amlDecodedBytes) { [Array]::Clear($amlDecodedBytes, 0, $amlDecodedBytes.Length) }
        $amlPreviousKey = $null
    }
}

function Get-AmlTiktokenCacheDir {
    foreach ($name in @('TIKTOKEN_CACHE_DIR', 'DATA_GYM_CACHE_DIR')) {
        $value = [Environment]::GetEnvironmentVariable($name, 'User')
        if (-not [string]::IsNullOrWhiteSpace($value)) { return $value }
        $value = [Environment]::GetEnvironmentVariable($name, 'Machine')
        if (-not [string]::IsNullOrWhiteSpace($value)) { return $value }
        $value = [Environment]::GetEnvironmentVariable($name, 'Process')
        if (-not [string]::IsNullOrWhiteSpace($value)) { return $value }
    }
    Join-Path $env:TEMP 'data-gym-cache'
}

function Get-AmlDbPathForMode {
    param([string]$ModeName)
    $paths = Get-AmlProdPaths
    $file = if ($ModeName -eq 'Fts') { 'aml-fts.sqlite' } else { 'aml-hybrid.sqlite' }
    Join-Path $paths.DbDir $file
}

function Set-AmlHybridProcessEnv {
    param([string]$DbPath)
    $cache = Get-AmlTiktokenCacheDir
    $env:AML_PROFILE = 'cycle2_dense'
    $env:AML_RETRIEVAL_MODE = 'raw_hybrid_rrf'
    $env:AML_ENABLE_VECTOR = 'true'
    $env:AML_DB_PATH = $DbPath
    $env:AML_AUTH_MODE = 'bearer'
    $env:AML_SERVICE_KEY_ENV = 'AML_SERVICE_API_KEY'
    $env:AML_EMBEDDING_PROVIDER = 'dashscope'
    $env:AML_EMBEDDING_ENDPOINT = $script:DashScopeEndpoint
    $env:AML_EMBEDDING_REGION = 'cn-beijing'
    $env:AML_EMBEDDING_MODEL = 'text-embedding-v4'
    $env:AML_EMBEDDING_DIMENSIONS = '1024'
    $env:AML_EMBEDDING_REVISION = '1'
    $env:AML_EMBEDDING_KEY_ENV = 'DASHSCOPE_API_KEY'
    $env:AML_EMBEDDING_BATCH_SIZE = '10'
    $env:AML_EMBEDDING_TIMEOUT_SECONDS = '30'
    $env:AML_EMBEDDING_MAX_RETRIES = '0'
    $env:AML_EMBEDDING_MAX_INPUT_BYTES = '8192'
    $env:AML_EMBEDDING_MIN_INTERVAL_SECONDS = '0'
    $env:AML_COORDINATOR_LOCK_TIMEOUT_SECONDS = '30'
    $env:AML_MAX_CONCURRENT_REQUESTS = '32'
    $env:AML_MAX_REQUEST_BYTES = '10485760'
    $env:AML_CANDIDATE_LIMIT = '200'
    $env:AML_LEXICAL_QUOTA = '100'
    $env:AML_DENSE_QUOTA = '100'
    $env:AML_MAX_RETURN_ITEMS = '100'
    $env:AML_RETURN_CHAR_BUDGET = '32000'
    $env:AML_RETURN_TOKEN_BUDGET = '117760'
    $env:AML_PACKING_POLICY = 'prefix'
    $env:AML_ROUTE_POLICY = 'none'
    $env:AML_ENABLE_GOVERNANCE = 'false'
    $env:AML_EXACT_DENSE_ACCELERATION = 'false'
    $env:AML_HOST = $script:AmlHost
    $env:AML_PORT = [string]$script:AmlPort
    $env:TIKTOKEN_CACHE_DIR = $cache
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONUTF8 = '1'
}

function Clear-AmlSecretProcessEnv {
    [Environment]::SetEnvironmentVariable('DASHSCOPE_API_KEY', $null, 'Process')
    [Environment]::SetEnvironmentVariable('AML_SERVICE_API_KEY', $null, 'Process')
}

function Write-AmlState {
    param([string]$ModeName)
    $paths = Get-AmlProdPaths
    $payload = @{
        mode       = $ModeName
        db_path    = (Get-AmlDbPathForMode -ModeName $ModeName)
        updated_at = (Get-Date).ToUniversalTime().ToString('o')
    } | ConvertTo-Json -Compress
    [IO.File]::WriteAllText($paths.State, $payload, [Text.Encoding]::UTF8)
}

function Read-AmlState {
    $paths = Get-AmlProdPaths
    if (-not [IO.File]::Exists($paths.State)) { return $null }
    try {
        return (Get-Content -LiteralPath $paths.State -Raw -Encoding UTF8 | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Test-AmlPidAlive {
    param([int]$ProcessId)
    try {
        $p = Get-Process -Id $ProcessId -ErrorAction Stop
        return $null -ne $p
    } catch {
        return $false
    }
}

function Get-AmlServicePidFromFile {
    $paths = Get-AmlProdPaths
    if (-not [IO.File]::Exists($paths.Pid)) { return $null }
    $text = ([IO.File]::ReadAllText($paths.Pid)).Trim()
    if ($text -match '^\d+$') { return [int]$text }
    return $null
}

function Invoke-AmlRuntimeConfiguration {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $script:AmlPython
    $psi.Arguments = '-m deploy.runtime --configuration'
    $psi.WorkingDirectory = $script:AmlWorkDir
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    foreach ($entry in [Environment]::GetEnvironmentVariables('Process').GetEnumerator()) {
        if ($entry.Key -and $null -ne $entry.Value) {
            $psi.EnvironmentVariables[$entry.Key] = [string]$entry.Value
        }
    }
    $proc = [Diagnostics.Process]::Start($psi)
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    return @{ ExitCode = $proc.ExitCode; StdOut = $stdout; StdErr = $stderr }
}

function Wait-AmlHealth {
    param([int]$TimeoutSeconds = 60)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $lastCode = $null
    $uri = "http://${script:AmlHost}:$($script:AmlPort)/health"
    while ((Get-Date) -lt $deadline) {
        try {
            $sw = [Diagnostics.Stopwatch]::StartNew()
            $response = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            $sw.Stop()
            $lastCode = $response.StatusCode
            if ($response.StatusCode -eq 200) {
                return @{ Ok = $true; StatusCode = 200; ElapsedMs = $sw.ElapsedMilliseconds }
            }
        } catch {
            if ($_.Exception.Response) {
                $lastCode = [int]$_.Exception.Response.StatusCode
            }
        }
        Start-Sleep -Milliseconds 200
    }
    return @{ Ok = $false; StatusCode = $lastCode; ElapsedMs = $null }
}

function Rotate-AmlServiceLogs {
    $paths = Get-AmlProdPaths
    if (-not [IO.Directory]::Exists($paths.Logs)) { return }
    $files = @(Get-ChildItem -LiteralPath $paths.Logs -Filter 'service-*.log' -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike '*.err.log' } | Sort-Object LastWriteTime -Descending)
    if ($files.Count -gt 10) {
        $files | Select-Object -Skip 10 | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue }
    }
    $errFiles = @(Get-ChildItem -LiteralPath $paths.Logs -Filter 'service-*.err.log' -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending)
    if ($errFiles.Count -gt 10) {
        $errFiles | Select-Object -Skip 10 | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue }
    }
}

function Stop-AmlServiceProcessTree {
    param([int]$RootPid)
    if (-not (Test-AmlPidAlive -ProcessId $RootPid)) { return }
    & taskkill.exe /T /PID $RootPid /F 2>&1 | Out-Null
}

function Invoke-AmlServiceConfigure {
    if ($Mode -eq 'Fts') {
        Write-Output 'not_supported'
        exit 2
    }
    $paths = Get-AmlProdPaths
    foreach ($dir in @($paths.Root, $paths.DbDir, $paths.Logs, $paths.Run)) {
        [IO.Directory]::CreateDirectory($dir) | Out-Null
    }
    Ensure-AmlServiceKeyExists
    Write-AmlState -ModeName $Mode
    Write-Output 'configured'
}

switch ($Action) {
    'Configure' {
        Invoke-AmlServiceConfigure
        exit 0
    }
    'RotateKey' {
        Save-AmlServiceKey -PlainKey (New-AmlUrlSafeKey)
        Write-Output 'key_rotated_restart_required'
        exit 0
    }
    'ShowKeyOnce' {
        $path = Get-AmlServiceVaultPath
        if (-not [IO.File]::Exists($path)) {
            Write-Output 'service_key_not_configured'
            exit 2
        }
        $decodedBytes = $null
        try {
            $decodedBytes = [Security.Cryptography.ProtectedData]::Unprotect(
                [IO.File]::ReadAllBytes($path), $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
            $plain = [Text.Encoding]::UTF8.GetString($decodedBytes)
            Set-Clipboard -Value $plain
            $plain = $null
        } finally {
            if ($null -ne $decodedBytes) { [Array]::Clear($decodedBytes, 0, $decodedBytes.Length) }
        }
        Write-Output 'copied_to_clipboard'
        exit 0
    }
    'Stop' {
        $paths = Get-AmlProdPaths
        $pidVal = Get-AmlServicePidFromFile
        if ($null -eq $pidVal) {
            Write-Output 'not_running'
            exit 0
        }
        Stop-AmlServiceProcessTree -RootPid $pidVal
        if ([IO.File]::Exists($paths.Pid)) {
            [IO.File]::Delete($paths.Pid)
        }
        Write-Output 'stopped'
        exit 0
    }
    'Status' {
        $paths = Get-AmlProdPaths
        $state = Read-AmlState
        $modeOut = if ($state -and $state.mode) { [string]$state.mode } else { $Mode }
        $dbOut = if ($state -and $state.db_path) { [string]$state.db_path } else { (Get-AmlDbPathForMode -ModeName $modeOut) }
        $pidVal = Get-AmlServicePidFromFile
        $alive = $false
        if ($null -ne $pidVal) { $alive = Test-AmlPidAlive -ProcessId $pidVal }
        $health = @{ Ok = $false; StatusCode = $null; ElapsedMs = $null }
        if ($alive) {
            $health = Wait-AmlHealth -TimeoutSeconds 5
        }
        $latestLog = $null
        if ([IO.Directory]::Exists($paths.Logs)) {
            $latestLog = Get-ChildItem -LiteralPath $paths.Logs -Filter 'service-*.log' -File -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
        }
        [PSCustomObject]@{
            pid           = $pidVal
            process_alive = $alive
            health_ok     = $health.Ok
            health_status = $health.StatusCode
            health_ms     = $health.ElapsedMs
            mode          = $modeOut
            db_path       = $dbOut
            latest_log    = if ($latestLog) { $latestLog.Name } else { $null }
        } | ConvertTo-Json -Compress
        exit 0
    }
    'Start' {
        if ($Mode -eq 'Fts') {
            Write-Output 'not_supported'
            exit 2
        }
        $paths = Get-AmlProdPaths
        foreach ($dir in @($paths.Root, $paths.DbDir, $paths.Logs, $paths.Run)) {
            [IO.Directory]::CreateDirectory($dir) | Out-Null
        }
        Ensure-AmlServiceKeyExists
        Write-AmlState -ModeName $Mode

        $existing = Get-AmlServicePidFromFile
        if ($null -ne $existing -and (Test-AmlPidAlive -ProcessId $existing)) {
            Write-Output 'already_running'
            exit 2
        }
        if ([IO.File]::Exists($paths.Pid)) {
            [IO.File]::Delete($paths.Pid)
        }

        $dbPath = Get-AmlDbPathForMode -ModeName $Mode
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $logOut = Join-Path $paths.Logs "service-$stamp.log"
        $logErr = Join-Path $paths.Logs "service-$stamp.err.log"
        Rotate-AmlServiceLogs

        $script:StartExitCode = 0
        Invoke-AmlWithDashScopeKey -Action {
            Invoke-AmlWithServiceKey -Action {
                Set-AmlHybridProcessEnv -DbPath $dbPath
                $cfg = Invoke-AmlRuntimeConfiguration
                if ($cfg.ExitCode -ne 0) {
                    $code = 'preflight_failed'
                    try {
                        $parsed = $cfg.StdOut.Trim() | ConvertFrom-Json
                        if ($parsed.code) { $code = [string]$parsed.code }
                    } catch { }
                    Clear-AmlSecretProcessEnv
                    Write-Output $code
                    $script:StartExitCode = 2
                    return
                }

                $p = Start-Process -FilePath $script:AmlPython -ArgumentList '-m', 'deploy.start' `
                    -WorkingDirectory $script:AmlWorkDir `
                    -WindowStyle Hidden `
                    -RedirectStandardOutput $logOut `
                    -RedirectStandardError $logErr `
                    -PassThru

                [IO.File]::WriteAllText($paths.Pid, [string]$p.Id, [Text.Encoding]::ASCII)
                Clear-AmlSecretProcessEnv

                $health = Wait-AmlHealth -TimeoutSeconds 60
                if (-not $health.Ok) {
                    Stop-AmlServiceProcessTree -RootPid $p.Id
                    if ([IO.File]::Exists($paths.Pid)) { [IO.File]::Delete($paths.Pid) }
                    Write-Output 'health_check_failed'
                    $script:StartExitCode = 2
                    return
                }
                Write-Output 'started'
            }
        }
        exit $script:StartExitCode
    }
}
