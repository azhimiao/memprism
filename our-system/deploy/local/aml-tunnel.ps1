# Cloudflare named tunnel for AML Memory (forwards to 127.0.0.1:8787).
#Requires -Version 5.1

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Setup', 'Start', 'Stop', 'Status')]
    [string]$Action,

    [string]$Hostname = 'test.memprism.com',
    [string]$TunnelName = 'aml-test'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:CloudflaredExe = 'C:\Program Files (x86)\cloudflared\cloudflared.exe'
$script:AmlProdRoot = 'F:\competitions\.aml-prod'
$script:AmlServicePort = 8787
$script:CloudflaredUserDir = Join-Path $env:USERPROFILE '.cloudflared'
$script:TemplatePath = 'F:\competitions\1AML\our-system\deploy\local\cloudflared-config.template.yml'

function Get-AmlTunnelPaths {
    $root = $script:AmlProdRoot
    @{
        Root       = $root
        Logs       = Join-Path $root 'logs'
        Run        = Join-Path $root 'run'
        CfDir      = Join-Path $root 'cloudflared'
        Config     = Join-Path $root 'cloudflared\config.yml'
        Pid        = Join-Path $root 'run\aml-tunnel.pid'
        Meta       = Join-Path $root 'run\tunnel-meta.json'
    }
}

function Test-CloudflaredInstalled {
    Test-Path -LiteralPath $script:CloudflaredExe
}

function Get-TunnelPidFromFile {
    $paths = Get-AmlTunnelPaths
    if (-not [IO.File]::Exists($paths.Pid)) { return $null }
    $text = ([IO.File]::ReadAllText($paths.Pid)).Trim()
    if ($text -match '^\d+$') { return [int]$text }
    return $null
}

function Test-TunnelPidAlive {
    param([int]$ProcessId)
    try {
        $p = Get-Process -Id $ProcessId -ErrorAction Stop
        return $null -ne $p
    } catch {
        return $false
    }
}

function Stop-TunnelProcessTree {
    param([int]$RootPid)
    if (-not (Test-TunnelPidAlive -ProcessId $RootPid)) { return }
    & taskkill.exe /T /PID $RootPid /F 2>&1 | Out-Null
}

function Invoke-CloudflaredLogged {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    if (-not (Test-CloudflaredInstalled)) {
        throw 'cloudflared_not_installed'
    }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $script:CloudflaredExe
    $psi.Arguments = ($Arguments -join ' ')
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $proc = [Diagnostics.Process]::Start($psi)
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    $combined = @()
    if ($stdout) { $combined += $stdout }
    if ($stderr) { $combined += $stderr }
    [IO.File]::WriteAllText($LogPath, ($combined -join [Environment]::NewLine), [Text.Encoding]::UTF8)
    return @{ ExitCode = $proc.ExitCode; StdOut = $stdout; StdErr = $stderr }
}

function Get-CloudflaredTunnelListJson {
    if (-not (Test-CloudflaredInstalled)) { return $null }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $script:CloudflaredExe
    $psi.Arguments = 'tunnel list --output json'
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $proc = [Diagnostics.Process]::Start($psi)
    $stdout = $proc.StandardOutput.ReadToEnd()
    $proc.StandardError.ReadToEnd() | Out-Null
    $proc.WaitForExit()
    if ($proc.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($stdout)) { return $null }
    try {
        return ($stdout | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Find-TunnelUuidByName {
    param([string]$Name)
    $list = Get-CloudflaredTunnelListJson
    if ($null -eq $list) { return $null }
    foreach ($item in @($list)) {
        if ($item.name -eq $Name -and $item.id) {
            return [string]$item.id
        }
    }
    return $null
}

function Write-TunnelMeta {
    param(
        [string]$Uuid,
        [string]$Name,
        [string]$Hostname
    )
    $paths = Get-AmlTunnelPaths
    $payload = @{
        tunnel_uuid = $Uuid
        tunnel_name = $Name
        hostname    = $Hostname
        updated_at  = (Get-Date).ToUniversalTime().ToString('o')
    } | ConvertTo-Json -Compress
    [IO.File]::WriteAllText($paths.Meta, $payload, [Text.Encoding]::UTF8)
}

function Read-TunnelMeta {
    $paths = Get-AmlTunnelPaths
    if (-not [IO.File]::Exists($paths.Meta)) { return $null }
    try {
        return (Get-Content -LiteralPath $paths.Meta -Raw -Encoding UTF8 | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function New-TunnelConfigFromTemplate {
    param(
        [string]$Uuid,
        [string]$Hostname,
        [string]$OutPath
    )
    if (-not [IO.File]::Exists($script:TemplatePath)) {
        throw 'template_missing'
    }
    $credFile = Join-Path $script:CloudflaredUserDir ($Uuid + '.json')
    $text = [IO.File]::ReadAllText($script:TemplatePath, [Text.Encoding]::UTF8)
    $text = $text -replace '\{\{TUNNEL_UUID\}\}', $Uuid
    $text = $text -replace '\{\{CREDENTIALS_FILE\}\}', ($credFile -replace '\\', '/')
    $text = $text -replace '\{\{HOSTNAME\}\}', $Hostname
    $dir = [IO.Path]::GetDirectoryName($OutPath)
    [IO.Directory]::CreateDirectory($dir) | Out-Null
    [IO.File]::WriteAllText($OutPath, $text, [Text.Encoding]::UTF8)
}

function Rotate-TunnelLogs {
    param([string]$Prefix)
    $paths = Get-AmlTunnelPaths
    if (-not [IO.Directory]::Exists($paths.Logs)) { return }
    $files = @(Get-ChildItem -LiteralPath $paths.Logs -Filter ($Prefix + '*.log') -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending)
    if ($files.Count -le 10) { return }
    $files | Select-Object -Skip 10 | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue }
}

switch ($Action) {
    'Setup' {
        $paths = Get-AmlTunnelPaths
        foreach ($dir in @($paths.Root, $paths.Logs, $paths.Run, $paths.CfDir)) {
            [IO.Directory]::CreateDirectory($dir) | Out-Null
        }
        $certPath = Join-Path $script:CloudflaredUserDir 'cert.pem'
        if (-not [IO.File]::Exists($certPath)) {
            Write-Output 'cert_missing_run_cloudflared_tunnel_login'
            exit 2
        }
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $setupLog = Join-Path $paths.Logs "tunnel-setup-$stamp.log"

        $uuid = Find-TunnelUuidByName -Name $TunnelName
        $status = 'existing'
        if ([string]::IsNullOrWhiteSpace($uuid)) {
            $create = Invoke-CloudflaredLogged -Arguments @('tunnel', 'create', $TunnelName) -LogPath $setupLog
            if ($create.ExitCode -ne 0) {
                Write-Output 'tunnel_create_failed'
                exit 2
            }
            $uuid = Find-TunnelUuidByName -Name $TunnelName
            if ([string]::IsNullOrWhiteSpace($uuid)) {
                if ($create.StdOut -match '([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})') {
                    $uuid = $Matches[1]
                }
            }
            $status = 'created'
        }

        if ([string]::IsNullOrWhiteSpace($uuid)) {
            Write-Output 'tunnel_uuid_unresolved'
            exit 2
        }

        New-TunnelConfigFromTemplate -Uuid $uuid -Hostname $Hostname -OutPath $paths.Config
        $routeLog = Join-Path $paths.Logs "tunnel-setup-route-$stamp.log"
        $route = Invoke-CloudflaredLogged -Arguments @('tunnel', 'route', 'dns', $TunnelName, $Hostname) -LogPath $routeLog
        if ($route.ExitCode -ne 0) {
            Write-Output 'tunnel_route_failed'
            exit 2
        }

        Write-TunnelMeta -Uuid $uuid -Name $TunnelName -Hostname $Hostname
        [PSCustomObject]@{
            tunnel_uuid = $uuid
            status      = $status
        } | ConvertTo-Json -Compress
        exit 0
    }
    'Start' {
        $paths = Get-AmlTunnelPaths
        if (-not [IO.File]::Exists($paths.Config)) {
            Write-Output 'config_missing_run_setup'
            exit 2
        }
        $existing = Get-TunnelPidFromFile
        if ($null -ne $existing -and (Test-TunnelPidAlive -ProcessId $existing)) {
            Write-Output 'already_running'
            exit 2
        }
        if ([IO.File]::Exists($paths.Pid)) {
            [IO.File]::Delete($paths.Pid)
        }
        if (-not (Test-CloudflaredInstalled)) {
            Write-Output 'cloudflared_not_installed'
            exit 2
        }
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $logOut = Join-Path $paths.Logs "tunnel-$stamp.log"
        $logErr = Join-Path $paths.Logs "tunnel-$stamp.err.log"
        Rotate-TunnelLogs -Prefix 'tunnel-'

        $procArgs = @('tunnel', '--config', $paths.Config, 'run', $TunnelName)
        $p = Start-Process -FilePath $script:CloudflaredExe -ArgumentList $procArgs `
            -WindowStyle Hidden `
            -RedirectStandardOutput $logOut `
            -RedirectStandardError $logErr `
            -PassThru
        [IO.File]::WriteAllText($paths.Pid, [string]$p.Id, [Text.Encoding]::ASCII)
        Write-Output 'started'
        exit 0
    }
    'Stop' {
        $paths = Get-AmlTunnelPaths
        $pidVal = Get-TunnelPidFromFile
        if ($null -eq $pidVal) {
            Write-Output 'not_running'
            exit 0
        }
        Stop-TunnelProcessTree -RootPid $pidVal
        if ([IO.File]::Exists($paths.Pid)) {
            [IO.File]::Delete($paths.Pid)
        }
        Write-Output 'stopped'
        exit 0
    }
    'Status' {
        $paths = Get-AmlTunnelPaths
        $meta = Read-TunnelMeta
        $hostName = if ($meta -and $meta.hostname) { [string]$meta.hostname } else { $Hostname }
        $pidVal = Get-TunnelPidFromFile
        $alive = $false
        if ($null -ne $pidVal) { $alive = Test-TunnelPidAlive -ProcessId $pidVal }
        $publicStatus = $null
        $publicMs = $null
        if ($alive) {
            try {
                $sw = [Diagnostics.Stopwatch]::StartNew()
                $response = Invoke-WebRequest -Uri "https://$hostName/health" -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
                $sw.Stop()
                $publicStatus = $response.StatusCode
                $publicMs = $sw.ElapsedMilliseconds
            } catch {
                if ($_.Exception.Response) {
                    $publicStatus = [int]$_.Exception.Response.StatusCode
                }
            }
        }
        [PSCustomObject]@{
            pid                = $pidVal
            process_alive      = $alive
            tunnel_name        = $TunnelName
            hostname           = $hostName
            config_path        = $paths.Config
            public_health_code = $publicStatus
            public_health_ms   = $publicMs
        } | ConvertTo-Json -Compress
        exit 0
    }
}
