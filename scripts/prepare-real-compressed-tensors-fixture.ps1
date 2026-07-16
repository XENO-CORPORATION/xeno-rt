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

$compressedRevision = "81f31585caa4e516d62f8e6c132a1ad4076b402d"
$denseRevision = "060db6499f32faf8b98477b0a26969ef7d8b9987"
$compressedDirectoryName = "Qwen2.5-0.5B-compressed-tensors-W4A16-$($compressedRevision.Substring(0, 8))"
$denseDirectoryName = "Qwen2.5-0.5B-BF16-$($denseRevision.Substring(0, 8))"

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

$compressedBase = "https://huggingface.co/RedHatAI/Qwen2.5-0.5B-quantized.w4a16/resolve/$compressedRevision"
$denseBase = "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/$denseRevision"
$files = @(
    [pscustomobject]@{
        RelativePath = "$compressedDirectoryName\config.json"
        Uri = "$compressedBase/config.json"
        Bytes = 1830L
        Sha256 = "6d8da94d5f142d1d422256b828dd05d252013123fab8a68da8a7c865958ecc9c"
    },
    [pscustomobject]@{
        RelativePath = "$compressedDirectoryName\merges.txt"
        Uri = "$compressedBase/merges.txt"
        Bytes = 1671853L
        Sha256 = "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"
    },
    [pscustomobject]@{
        RelativePath = "$compressedDirectoryName\tokenizer_config.json"
        Uri = "$compressedBase/tokenizer_config.json"
        Bytes = 7229L
        Sha256 = "cefaa66de8fae4a09ca18a9c3a7fd8b61311ed568e5f4e634f6a3d95a2a9e889"
    },
    [pscustomobject]@{
        RelativePath = "$compressedDirectoryName\vocab.json"
        Uri = "$compressedBase/vocab.json"
        Bytes = 2776833L
        Sha256 = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
    },
    [pscustomobject]@{
        RelativePath = "$compressedDirectoryName\model.safetensors"
        Uri = "$compressedBase/model.safetensors"
        Bytes = 735851328L
        Sha256 = "75ccc597de25f987dd7b841c0f37d2424472b83e57dd67478b3acb6080bd753f"
    },
    [pscustomobject]@{
        RelativePath = "$denseDirectoryName\config.json"
        Uri = "$denseBase/config.json"
        Bytes = 681L
        Sha256 = "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b"
    },
    [pscustomobject]@{
        RelativePath = "$denseDirectoryName\merges.txt"
        Uri = "$denseBase/merges.txt"
        Bytes = 1671839L
        Sha256 = "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"
    },
    [pscustomobject]@{
        RelativePath = "$denseDirectoryName\tokenizer_config.json"
        Uri = "$denseBase/tokenizer_config.json"
        Bytes = 7228L
        Sha256 = "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9"
    },
    [pscustomobject]@{
        RelativePath = "$denseDirectoryName\vocab.json"
        Uri = "$denseBase/vocab.json"
        Bytes = 2776833L
        Sha256 = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
    },
    [pscustomobject]@{
        RelativePath = "$denseDirectoryName\model.safetensors"
        Uri = "$denseBase/model.safetensors"
        Bytes = 988097824L
        Sha256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
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
    CompressedTensorsModelDirectory = Resolve-CachePath $compressedDirectoryName
    DenseModelDirectory = Resolve-CachePath $denseDirectoryName
    CompressedTensorsRevision = $compressedRevision
    DenseRevision = $denseRevision
}
