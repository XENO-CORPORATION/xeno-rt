[CmdletBinding()]
param(
    [string]$CacheRoot = 'X:\ai\models\llm',
    [int]$TimeoutSeconds = 1800,
    [ValidateSet("gemm", "gemv")]
    [string]$Format = "gemm"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Net.Http

if ($TimeoutSeconds -lt 60) {
    throw "-TimeoutSeconds must be at least 60"
}

if ($Format -eq "gemm") {
    $awqRepository = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
    $awqRevision = "cb07c13df107486a6d99bd487a819dd8905510e9"
    $ggufRepository = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    $ggufRevision = "df5bf01389a39c743ab467d734bf501681e041c5"
    $awqDirectoryName = "Qwen2.5-0.5B-Instruct-AWQ-$($awqRevision.Substring(0, 8))"
    $ggufDirectoryName = "Qwen2.5-0.5B-Instruct-GGUF-$($ggufRevision.Substring(0, 8))"
    $ggufFileName = "qwen2.5-0.5b-instruct-q8_0.gguf"
    $pins = [ordered]@{
        Config = [pscustomobject]@{ Bytes = 837L; Sha256 = "bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090" }
        Merges = [pscustomobject]@{ Bytes = 1671839L; Sha256 = "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3" }
        TokenizerConfig = [pscustomobject]@{ Bytes = 7305L; Sha256 = "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583" }
        Vocab = [pscustomobject]@{ Bytes = 2776833L; Sha256 = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910" }
        Model = [pscustomobject]@{ Bytes = 730652248L; Sha256 = "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b" }
        Gguf = [pscustomobject]@{ Bytes = 675710816L; Sha256 = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e" }
    }
} else {
    $awqRepository = "casimiir/Qwen3-0.6B-Base-awq-gemv-w4"
    $awqRevision = "ad0963720d88c62b49f93b1bcec0db146576d1f1"
    $ggufRepository = "Qwen/Qwen3-0.6B-GGUF"
    $ggufRevision = "23749fefcc72300e3a2ad315e1317431b06b590a"
    $awqDirectoryName = "Qwen3-0.6B-Base-AWQ-GEMV-$($awqRevision.Substring(0, 8))"
    $ggufDirectoryName = "Qwen3-0.6B-GGUF-$($ggufRevision.Substring(0, 8))"
    $ggufFileName = "Qwen3-0.6B-Q8_0.gguf"
    $pins = [ordered]@{
        Config = [pscustomobject]@{ Bytes = 1041L; Sha256 = "a802d41ed37f50ab135c30ab6704b53d4b9e1625d695b575ae139f1b1b9d463b" }
        Merges = [pscustomobject]@{ Bytes = 1671853L; Sha256 = "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5" }
        TokenizerConfig = [pscustomobject]@{ Bytes = 5407L; Sha256 = "67e5a0a11cd35f9c00ee52e0af4cdc0baa75fea0cb5fce7d1beb251b4621d15c" }
        Vocab = [pscustomobject]@{ Bytes = 2776833L; Sha256 = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910" }
        Model = [pscustomobject]@{ Bytes = 540176192L; Sha256 = "013213ce008475fa62e752092d3e1352375aa1a5b1d855cb1aa914e5bfa1595f" }
        Gguf = [pscustomobject]@{ Bytes = 639446688L; Sha256 = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031" }
    }
}

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

$awqBase = "https://huggingface.co/$awqRepository/resolve/$awqRevision"
$ggufBase = "https://huggingface.co/$ggufRepository/resolve/$ggufRevision"
$files = @(
    [pscustomobject]@{
        RelativePath = "$awqDirectoryName\config.json"
        Uri = "$awqBase/config.json"
        Bytes = $pins.Config.Bytes
        Sha256 = $pins.Config.Sha256
    },
    [pscustomobject]@{
        RelativePath = "$awqDirectoryName\merges.txt"
        Uri = "$awqBase/merges.txt"
        Bytes = $pins.Merges.Bytes
        Sha256 = $pins.Merges.Sha256
    },
    [pscustomobject]@{
        RelativePath = "$awqDirectoryName\tokenizer_config.json"
        Uri = "$awqBase/tokenizer_config.json"
        Bytes = $pins.TokenizerConfig.Bytes
        Sha256 = $pins.TokenizerConfig.Sha256
    },
    [pscustomobject]@{
        RelativePath = "$awqDirectoryName\vocab.json"
        Uri = "$awqBase/vocab.json"
        Bytes = $pins.Vocab.Bytes
        Sha256 = $pins.Vocab.Sha256
    },
    [pscustomobject]@{
        RelativePath = "$awqDirectoryName\model.safetensors"
        Uri = "$awqBase/model.safetensors"
        Bytes = $pins.Model.Bytes
        Sha256 = $pins.Model.Sha256
    },
    [pscustomobject]@{
        RelativePath = "$ggufDirectoryName\$ggufFileName"
        Uri = "$ggufBase/$ggufFileName"
        Bytes = $pins.Gguf.Bytes
        Sha256 = $pins.Gguf.Sha256
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
    GgufModelPath = Resolve-CachePath "$ggufDirectoryName\$ggufFileName"
    AwqRevision = $awqRevision
    GgufRevision = $ggufRevision
    Format = $Format
}
