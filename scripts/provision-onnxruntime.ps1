[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $Destination,
    [string] $Manifest,
    [switch] $ConfirmInstall
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Manifest)) {
    $Manifest = Join-Path $PSScriptRoot '..\reference\runtime\onnxruntime-1.23.0-windows-x64.json'
}
$contract = Get-Content -LiteralPath ([IO.Path]::GetFullPath($Manifest)) -Raw | ConvertFrom-Json
$destinationPath = [IO.Path]::GetFullPath($Destination)
if ($contract.schema_version -ne 1 -or $contract.id -notmatch '^onnxruntime-(gpu-)?1\.[0-9]+\.[0-9]+-windows-x64$') {
    throw 'unsupported ONNX Runtime provisioning manifest'
}
if (-not $ConfirmInstall) {
    [pscustomobject]@{ Action = 'Plan'; Destination = $destinationPath; Package = $contract.package.source; Sha256 = $contract.package.sha256 }
    return
}
# Versioned directories are installed whole. Never replace a loaded DLL, nor
# publish one DLL before its companions and license notices have verified.
if (Test-Path -LiteralPath $destinationPath) { throw 'Destination already exists; verify it or choose a new versioned directory' }
$parent = Split-Path $destinationPath -Parent
if (-not (Test-Path -LiteralPath $parent -PathType Container)) { throw 'Destination parent must exist' }
$current = Get-Item -LiteralPath $parent
while ($null -ne $current) {
    if ($current.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Destination traverses a reparse point' }
    $current = $current.Parent
}
$temporaryRoot = Join-Path $parent ('.xrt-onnxruntime-stage-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
try {
    $packagePath = Join-Path $temporaryRoot 'onnxruntime.zip'
    Invoke-WebRequest -Uri $contract.package.source -OutFile $packagePath -UseBasicParsing
    if ((Get-Item -LiteralPath $packagePath).Length -ne $contract.package.size_bytes -or
        (Get-FileHash -LiteralPath $packagePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $contract.package.sha256) {
        throw 'ONNX Runtime package size or SHA-256 mismatch'
    }
    $expandedPath = Join-Path $temporaryRoot 'expanded'
    Expand-Archive -LiteralPath $packagePath -DestinationPath $expandedPath
    $payload = Join-Path $temporaryRoot 'payload'
    New-Item -ItemType Directory -Path $payload | Out-Null
    foreach ($entry in @($contract.dll) + @($contract.companion_dlls)) {
        if ($null -eq $entry) { continue }
        if ($entry.file_name -notmatch '^[A-Za-z0-9_.-]+\.dll$' -or $entry.archive_path -match '(^|[/\\])\.\.([/\\]|$)') {
            throw 'Invalid native runtime manifest path'
        }
        $source = Join-Path $expandedPath ($entry.archive_path -replace '/', '\')
        if ((Get-Item -LiteralPath $source).Length -ne $entry.size_bytes -or
            (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
            throw "Native DLL identity mismatch: $($entry.file_name)"
        }
        $target = Join-Path $payload $entry.file_name
        Copy-Item -LiteralPath $source -Destination $target
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
            throw 'Staged DLL failed verification'
        }
    }
    Copy-Item -LiteralPath (Join-Path $expandedPath 'LICENSE') -Destination (Join-Path $payload 'onnxruntime.LICENSE')
    Copy-Item -LiteralPath (Join-Path $expandedPath 'ThirdPartyNotices.txt') -Destination (Join-Path $payload 'onnxruntime.ThirdPartyNotices.txt')
    # Directory.Move refuses an existing destination, including a competing
    # installer. Staging is on the same volume, so visibility is atomic.
    [IO.Directory]::Move($payload, $destinationPath)
    [pscustomobject]@{ Action = 'Installed'; Path = $destinationPath; Runtime = $contract.id }
}
finally {
    $links = @(Get-ChildItem -LiteralPath $temporaryRoot -Recurse -Force -Attributes ReparsePoint)
    if ($links.Count -eq 0) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -Confirm:$false
    } else {
        Write-Warning "Leaving staging intact because it contains reparse points: $temporaryRoot"
    }
}
