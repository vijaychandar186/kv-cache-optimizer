[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Subscription,
    [Parameter(Mandatory)][string]$Location,
    [Parameter(Mandatory)][string]$AdminSourceCidr,
    [Parameter(Mandatory)][string]$SshPublicKeyPath,
    [string]$ResourceGroup = 'rg-olmo-kv-lab',
    [string]$VmName = 'olmo-kv-lab',
    [string]$VmSize = 'Standard_ND96isr_H100_v5',
    [string]$ImageVersion,
    [switch]$Deploy
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
function Invoke-Az {
    param([string[]]$Arguments)
    $result = & az @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Azure CLI failed: az $($Arguments[0])" }
    return $result
}
if (-not (Get-Command az -ErrorAction SilentlyContinue)) { throw 'Install Azure CLI and run az login.' }
if ($AdminSourceCidr -notmatch '^\d{1,3}(\.\d{1,3}){3}/32$') {
    throw 'AdminSourceCidr must restrict SSH to a single public IPv4 /32 address.'
}
$adminIp = $null
if (-not [System.Net.IPAddress]::TryParse($AdminSourceCidr.Split('/')[0], [ref]$adminIp) -or
    $adminIp.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) {
    throw 'AdminSourceCidr must contain a valid IPv4 address.'
}
$key = (Get-Content -LiteralPath (Resolve-Path -LiteralPath $SshPublicKeyPath) -Raw).Trim()
if ($key -notmatch '^ssh-(ed25519|rsa) ') { throw 'Supply an OpenSSH public key (.pub).' }
Invoke-Az -Arguments @('account', 'set', '--subscription', $Subscription) | Out-Null
Invoke-Az -Arguments @('vm', 'list-usage', '--location', $Location, '--output', 'table')
Invoke-Az -Arguments @('vm', 'list-skus', '--location', $Location, '--size', $VmSize,
    '--all', '--query', '[].{name:name,restrictions:restrictions}', '--output', 'json')
if (-not $ImageVersion) {
    $ImageVersion = (Invoke-Az -Arguments @('vm', 'image', 'list', '--location', $Location,
        '--publisher', 'microsoft-dsvm', '--offer', 'ubuntu-hpc', '--sku', '2404', '--all',
        '--query', 'sort_by(@, &version)[-1].version', '--output', 'tsv') | Out-String).Trim()
}
if (-not $ImageVersion -or $ImageVersion -eq 'None') { throw 'No Ubuntu HPC 2404 image found.' }
Write-Output "Pinned Azure image version: $ImageVersion"
$template = Join-Path $PSScriptRoot '..\infra\main.bicep'
$parameters = @("location=$Location", "vmName=$VmName", "vmSize=$VmSize",
    "adminSourceCidr=$AdminSourceCidr", "sshPublicKey=$key", "imageVersion=$ImageVersion")
# Group creation is the only mutation in preview mode; it allocates no GPU resources.
Invoke-Az -Arguments @('group', 'create', '--name', $ResourceGroup, '--location', $Location,
    '--tags', 'project=olmo-kv-cache') | Out-Null
$operation = 'what-if'
if ($Deploy) { $operation = 'create' }
Invoke-Az -Arguments (@('deployment', 'group', $operation, '--resource-group', $ResourceGroup,
    '--template-file', $template, '--parameters') + $parameters)
if ($Deploy) {
    Invoke-Az -Arguments @('vm', 'auto-shutdown', '--resource-group', $ResourceGroup,
        '--name', $VmName, '--time', '0300') | Out-Null
    Write-Output 'Auto-shutdown configured for 03:00 UTC. Deallocate when finished to stop VM compute charges.'
}
