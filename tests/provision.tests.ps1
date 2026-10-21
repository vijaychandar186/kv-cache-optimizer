# Offline contract test: every Azure invocation is replaced by this in-process mock.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$global:KvLabAzCalls = [System.Collections.Generic.List[object]]::new()
function global:az {
    $global:KvLabAzCalls.Add([string[]]$args)
    $global:LASTEXITCODE = 0
    if ($args.Count -gt 2 -and $args[0] -eq 'vm' -and $args[1] -eq 'image') {
        return '24.04.2026100101'
    }
}
function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
$scratch = Join-Path $PSScriptRoot '..\.pytest-tmp\provision'
New-Item -ItemType Directory -Path $scratch -Force | Out-Null
$keyPath = Join-Path $scratch 'test.pub'
Set-Content -LiteralPath $keyPath -Value 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest offline-test'
$provision = Join-Path $PSScriptRoot '..\scripts\provision.ps1'
try {
    & $provision -Subscription 'test-subscription' -Location 'test-region' `
        -AdminSourceCidr '203.0.113.10/32' -SshPublicKeyPath $keyPath | Out-Null
    $deployment = $null
    foreach ($call in $global:KvLabAzCalls) {
        Assert-True ($call.Count -ge 3) 'The wrapper dropped Azure CLI arguments.'
        if ($call[0] -eq 'deployment') { $deployment = $call }
        Assert-True (-not ($call -contains 'auto-shutdown')) 'Preview configured a VM.'
    }
    Assert-True ($null -ne $deployment) 'Preview did not invoke deployment.'
    Assert-True ($deployment[2] -eq 'what-if') 'Preview must use what-if.'
    Assert-True ($deployment -contains 'location=test-region') 'Location parameter missing.'
    Assert-True ($deployment -contains 'imageVersion=24.04.2026100101') 'Image version not pinned.'
    Assert-True ($deployment -contains 'sshPublicKey=ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest offline-test') 'Public key was split.'
    $global:KvLabAzCalls.Clear()
    & $provision -Subscription 'test-subscription' -Location 'test-region' `
        -AdminSourceCidr '203.0.113.10/32' -SshPublicKeyPath $keyPath `
        -ImageVersion '24.04.2026100101' -Deploy | Out-Null
    $created = $shutdown = $false
    foreach ($call in $global:KvLabAzCalls) {
        if ($call[0] -eq 'deployment' -and $call[2] -eq 'create') { $created = $true }
        if ($call -contains 'auto-shutdown') { $shutdown = $true }
    }
    Assert-True $created 'Deploy did not create the deployment.'
    Assert-True $shutdown 'Deploy did not configure auto-shutdown.'
    Write-Output 'Provisioning script contract checks passed (Azure CLI mocked; no cloud calls).'
} finally {
    Remove-Item Function:\az
    Remove-Variable -Name KvLabAzCalls -Scope Global
}
