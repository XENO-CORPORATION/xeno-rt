[CmdletBinding()]
param(
    [string]$CacheRoot = 'X:\ai\models\llm',
    [int]$TimeoutSeconds = 1800
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Net.Http

if ($TimeoutSeconds -lt 60) {
    throw "-TimeoutSeconds must be at least 60"
}

$awqRevision = "cb07c13df107486a6d99bd487a819dd8905510e9"
$ggufRevision = "df5bf01389a39c743ab467d734bf501681e041c5"
$awqDirectoryName = "Qwen2.5-0.5B-Instruct-AWQ-$($awqRevision.Substring(0, 8))"
$ggufDirectoryName = "Qwen2.5-0.5B-Instruct-GGUF-$($ggufRevision.Substring(0, 8))"

$cacheRootPath = [IO.Path]::GetFullPath($CacheRoot)
New-Item -ItemType Directory -Force -Path $cacheRootPath | Out-Null
$cacheRootPath = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $cacheRootPath).Path)

function Resolve-CachePath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $path = [IO.Path]::GetFullPath((Join-Path $cacheRootPath $RelativePath))
    $prefix = $cacheRootPath.TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    ) + [IO.Path]::DirectorySeparatorChar
    if (-not $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "fixture path escapes cache root: $path"
    }
    $path
}

function Get-FileSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-PinnedFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][long]$ExpectedBytes,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )

    $actualBytes = (Get-Item -LiteralPath $Path).Length
    if ($actualBytes -ne $ExpectedBytes) {
        throw "fixture file '$Path' has $actualBytes bytes, expected $ExpectedBytes"
    }
    $actualSha256 = Get-FileSha256 $Path
    if ($actualSha256 -ne $ExpectedSha256) {
        throw "fixture file '$Path' has SHA256 $actualSha256, expected $ExpectedSha256"
    }
}

function Receive-PinnedFile {
    param(
        [Parameter(Mandatory = $true)][System.Net.Http.HttpClient]$Client,
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][long]$ExpectedBytes,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )

    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        Assert-PinnedFile $Destination $ExpectedBytes $ExpectedSha256
        Write-Host "verified cached fixture: $Destination"
        return
    }
    if (Test-Path -LiteralPath $Destination) {
        throw "fixture destination exists but is not a file: $Destination"
    }

    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $partial = "$Destination.partial"
    if (Test-Path -LiteralPath $partial) {
        Remove-Item -LiteralPath $partial -Force
    }

    Write-Host "downloading pinned fixture: $Uri"
    $request = [System.Net.Http.HttpRequestMessage]::new(
        [System.Net.Http.HttpMethod]::Get,
        $Uri
    )
    $response = $null
    $source = $null
    $target = $null
    try {
        $response = $Client.SendAsync(
            $request,
            [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead
        ).GetAwaiter().GetResult()
        $response.EnsureSuccessStatusCode() | Out-Null
        $contentLength = $response.Content.Headers.ContentLength
        if ($null -ne $contentLength -and $contentLength -ne $ExpectedBytes) {
            throw "download declared $contentLength bytes, expected $ExpectedBytes for $Uri"
        }

        $source = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $target = [IO.File]::Open(
            $partial,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,
            [IO.FileShare]::None
        )
        $buffer = New-Object byte[] (1024 * 1024)
        [long]$written = 0
        while (($read = $source.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $written += $read
            if ($written -gt $ExpectedBytes) {
                throw "download exceeded expected size $ExpectedBytes for $Uri"
            }
            $target.Write($buffer, 0, $read)
        }
        $target.Flush()
        if ($written -ne $ExpectedBytes) {
            throw "download wrote $written bytes, expected $ExpectedBytes for $Uri"
        }
    } catch {
        if ($target) {
            $target.Dispose()
            $target = $null
        }
        if (Test-Path -LiteralPath $partial) {
            Remove-Item -LiteralPath $partial -Force
        }
        throw
    } finally {
        if ($target) {
            $target.Dispose()
        }
        if ($source) {
            $source.Dispose()
        }
        if ($response) {
            $response.Dispose()
        }
        $request.Dispose()
    }

    Assert-PinnedFile $partial $ExpectedBytes $ExpectedSha256
    Move-Item -LiteralPath $partial -Destination $Destination
    Write-Host "installed verified fixture: $Destination"
}

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
$client.DefaultRequestHeaders.UserAgent.ParseAdd("xeno-rt-cuda-validation/1.0")

$awqBase = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ/resolve/$awqRevision"
$ggufBase = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/$ggufRevision"
$files = @(
    [pscustomobject]@{
        RelativePath = "$awqDirectoryName\config.json"
        Uri = "$awqBase/config.json"
        Bytes = 837L
        Sha256 = "bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090"
    },
    [pscustomobject]@{
        RelativePath = "$awqDirectoryName\merges.txt"
        Uri = "$awqBase/merges.txt"
        Bytes = 1671839L
        Sha256 = "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"
    },
    [pscustomobject]@{
        RelativePath = "$awqDirectoryName\tokenizer_config.json"
        Uri = "$awqBase/tokenizer_config.json"
        Bytes = 7305L
        Sha256 = "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
    },
    [pscustomobject]@{
        RelativePath = "$awqDirectoryName\vocab.json"
        Uri = "$awqBase/vocab.json"
        Bytes = 2776833L
        Sha256 = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
    },
    [pscustomobject]@{
        RelativePath = "$awqDirectoryName\model.safetensors"
        Uri = "$awqBase/model.safetensors"
        Bytes = 730652248L
        Sha256 = "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
    },
    [pscustomobject]@{
        RelativePath = "$ggufDirectoryName\qwen2.5-0.5b-instruct-q8_0.gguf"
        Uri = "$ggufBase/qwen2.5-0.5b-instruct-q8_0.gguf"
        Bytes = 675710816L
        Sha256 = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
    }
)

try {
    foreach ($file in $files) {
        Receive-PinnedFile `
            -Client $client `
            -Uri $file.Uri `
            -Destination (Resolve-CachePath $file.RelativePath) `
            -ExpectedBytes $file.Bytes `
            -ExpectedSha256 $file.Sha256
    }
} finally {
    $client.Dispose()
}

[pscustomobject]@{
    AwqModelDirectory = Resolve-CachePath $awqDirectoryName
    GgufModelPath = Resolve-CachePath "$ggufDirectoryName\qwen2.5-0.5b-instruct-q8_0.gguf"
    AwqRevision = $awqRevision
    GgufRevision = $ggufRevision
}
