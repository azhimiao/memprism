param([string]$Target = 'root@47.83.232.55')
# Streams configure-and-deploy.sh to the host over ssh, with the DashScope key from the local DPAPI vault.
# The key travels only inside the ssh session; nothing is written locally or printed.
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
. (Join-Path $workspace 'experiments\dashscope-credentials.ps1')
$body = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'configure-and-deploy.sh'))
if ($body.Contains("`r") -or $body -match '[^\x00-\x7F]') {
    throw 'configure-and-deploy.sh must be ASCII with LF line endings'
}

$script:remoteExit = 1
Invoke-AmlWithDashScopeKey -Action {
    if ($env:DASHSCOPE_API_KEY -notmatch '^[A-Za-z0-9._-]{8,256}$') {
        throw 'unexpected DashScope key format in the local vault'
    }
    $payload = "AML_DASHSCOPE_KEY_INPUT='" + $env:DASHSCOPE_API_KEY + "'`n" + $body
    $payload | ssh -T $Target 'bash -s'
    $script:remoteExit = $LASTEXITCODE
    $payload = $null
}
exit $script:remoteExit
