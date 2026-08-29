[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Destination,

    [string] $Manifest = (Join-Path $PSScriptRoot '..\reference\runtime\onnxruntime-1.20.0-windows-x64.json')
)

$ErrorActionPreference = 'Stop'
$manifestPath = [System.IO.Path]::GetFullPath($Manifest)
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$contract = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($contract.schema_version -ne 1 -or $contract.id -ne 'onnxruntime-1.20.0-windows-x64') {
    throw 'unsupported ONNX Runtime provisioning manifest'
}
$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('xrt-onnxruntime-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
try {
    $packagePath = Join-Path $temporaryRoot 'onnxruntime.nupkg'
    Invoke-WebRequest -Uri $contract.package.source -OutFile $packagePath
    $packageHash = (Get-FileHash -LiteralPath $packagePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($packageHash -ne $contract.package.sha256) {
        throw "ONNX Runtime package SHA-256 mismatch: $packageHash"
    }

    $zipPath = Join-Path $temporaryRoot 'onnxruntime.zip'
    Copy-Item -LiteralPath $packagePath -Destination $zipPath
    $expandedPath = Join-Path $temporaryRoot 'expanded'
    Expand-Archive -LiteralPath $zipPath -DestinationPath $expandedPath
    $sourceDll = Join-Path $expandedPath ($contract.dll.archive_path -replace '/', '\')
    $dll = Get-Item -LiteralPath $sourceDll
    $dllHash = (Get-FileHash -LiteralPath $sourceDll -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($dll.Length -ne $contract.dll.size_bytes -or $dllHash -ne $contract.dll.sha256) {
        throw "ONNX Runtime DLL identity mismatch: bytes=$($dll.Length) sha256=$dllHash"
    }

    New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null
    Copy-Item -LiteralPath $sourceDll -Destination (Join-Path $destinationPath $contract.dll.file_name) -Force
    Copy-Item -LiteralPath (Join-Path $expandedPath 'LICENSE') -Destination (Join-Path $destinationPath 'onnxruntime.LICENSE') -Force
    Copy-Item -LiteralPath (Join-Path $expandedPath 'ThirdPartyNotices.txt') -Destination (Join-Path $destinationPath 'onnxruntime.ThirdPartyNotices.txt') -Force

    $installed = Get-Item -LiteralPath (Join-Path $destinationPath $contract.dll.file_name)
    $installedHash = (Get-FileHash -LiteralPath $installed.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($installed.Length -ne $contract.dll.size_bytes -or $installedHash -ne $contract.dll.sha256) {
        throw 'provisioned ONNX Runtime DLL failed the post-copy integrity check'
    }
    [pscustomobject]@{
        Path = $installed.FullName
        SizeBytes = $installed.Length
        Sha256 = $installedHash
        FileVersion = $installed.VersionInfo.FileVersion
    }
}
finally {
    $resolvedTemporaryRoot = [System.IO.Path]::GetFullPath($temporaryRoot)
    $systemTemporaryRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    if ($resolvedTemporaryRoot.StartsWith($systemTemporaryRoot, [System.StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path $resolvedTemporaryRoot -Leaf).StartsWith('xrt-onnxruntime-')) {
        Remove-Item -LiteralPath $resolvedTemporaryRoot -Recurse -Force
    }
}
