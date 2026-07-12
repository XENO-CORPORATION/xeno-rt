[CmdletBinding()]
param(
    [string]$CacheRoot = 'X:\ai\models\llm',
    [int]$TimeoutSeconds = 1800
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Net.Http
Add-Type -TypeDefinition @"
using System;

public static class GptqPackedZeroConverter
{
    public static void V1ToV2(byte[] bytes)
    {
        if (bytes == null || bytes.Length % 4 != 0)
        {
            throw new ArgumentException("GPTQ qzeros payload must contain complete I32 words");
        }
        for (int offset = 0; offset < bytes.Length; offset += 4)
        {
            uint word = BitConverter.ToUInt32(bytes, offset);
            uint converted = 0;
            for (int nibble = 0; nibble < 8; nibble++)
            {
                int shift = nibble * 4;
                uint encodedZero = (word >> shift) & 0xFu;
                uint directZero = (encodedZero + 1u) & 0xFu;
                converted |= directZero << shift;
            }
            byte[] encoded = BitConverter.GetBytes(converted);
            Buffer.BlockCopy(encoded, 0, bytes, offset, 4);
        }
    }
}
"@

if ($TimeoutSeconds -lt 60) {
    throw "-TimeoutSeconds must be at least 60"
}

$actOrderRevision = "46e6f58dadc81c981175388a91d010f4c37fbfba"
$denseRevision = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
$actOrderDirectoryName = "Qwen2.5-1.5B-Instruct-GPTQ-act-order-$($actOrderRevision.Substring(0, 8))"
$denseDirectoryName = "Qwen2.5-1.5B-Instruct-dense-$($denseRevision.Substring(0, 8))"
$derivedV2DirectoryName = "Qwen2.5-0.5B-Instruct-GPTQ-v2-derived-c34a4a91"

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

function Set-JsonProperty {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)]$Value
    )

    if ($Object.PSObject.Properties.Name -contains $Name) {
        $Object.$Name = $Value
    } else {
        $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    }
}

function Read-Exactly {
    param(
        [Parameter(Mandatory = $true)][IO.Stream]$Stream,
        [Parameter(Mandatory = $true)][byte[]]$Buffer
    )

    $offset = 0
    while ($offset -lt $Buffer.Length) {
        $read = $Stream.Read($Buffer, $offset, $Buffer.Length - $offset)
        if ($read -le 0) {
            throw "unexpected end of file while reading GPTQ fixture"
        }
        $offset += $read
    }
}

function Convert-GptqV1ModelToV2 {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string]$ExpectedSha256 = ""
    )

    $expectedBytes = (Get-Item -LiteralPath $Source).Length
    if ($ExpectedSha256 -and (Test-Path -LiteralPath $Destination -PathType Leaf)) {
        Assert-PinnedFile $Destination $expectedBytes $ExpectedSha256
        Write-Host "verified cached derived GPTQ v2 fixture: $Destination"
        return
    }

    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $partial = "$Destination.partial"
    if (Test-Path -LiteralPath $partial) {
        Remove-Item -LiteralPath $partial -Force
    }
    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Force
    }
    Copy-Item -LiteralPath $Source -Destination $partial

    $stream = [IO.File]::Open(
        $partial,
        [IO.FileMode]::Open,
        [IO.FileAccess]::ReadWrite,
        [IO.FileShare]::None
    )
    try {
        $lengthBytes = New-Object byte[] 8
        Read-Exactly $stream $lengthBytes
        [long]$headerLength = [BitConverter]::ToUInt64($lengthBytes, 0)
        if ($headerLength -le 0 -or $headerLength -gt 16MB) {
            throw "invalid SafeTensors header length $headerLength"
        }
        $headerBytes = New-Object byte[] ([int]$headerLength)
        Read-Exactly $stream $headerBytes
        $headerJson = [Text.Encoding]::UTF8.GetString($headerBytes).TrimEnd(' ')
        $header = $headerJson | ConvertFrom-Json
        [long]$dataBase = 8 + $headerLength
        $qzeroProperties = @(
            $header.PSObject.Properties |
                Where-Object { $_.Name.EndsWith(".qzeros") }
        )
        if (-not $qzeroProperties) {
            throw "source GPTQ fixture has no qzeros tensors"
        }

        foreach ($property in $qzeroProperties) {
            $offsets = $property.Value.data_offsets
            [long]$start = $offsets[0]
            [long]$end = $offsets[1]
            [long]$length = $end - $start
            if ($start -lt 0 -or $length -le 0 -or $length % 4 -ne 0 -or $length -gt [int]::MaxValue) {
                throw "invalid qzeros data range for $($property.Name): [$start, $end)"
            }
            $buffer = New-Object byte[] ([int]$length)
            [void]$stream.Seek($dataBase + $start, [IO.SeekOrigin]::Begin)
            Read-Exactly $stream $buffer
            [GptqPackedZeroConverter]::V1ToV2($buffer)
            [void]$stream.Seek($dataBase + $start, [IO.SeekOrigin]::Begin)
            $stream.Write($buffer, 0, $buffer.Length)
        }
        $stream.Flush()
    } finally {
        $stream.Dispose()
    }

    $actualBytes = (Get-Item -LiteralPath $partial).Length
    if ($actualBytes -ne $expectedBytes) {
        throw "derived GPTQ v2 file has $actualBytes bytes, expected $expectedBytes"
    }
    $actualSha256 = Get-FileSha256 $partial
    if ($ExpectedSha256 -and $actualSha256 -ne $ExpectedSha256) {
        throw "derived GPTQ v2 file has SHA256 $actualSha256, expected $ExpectedSha256"
    }
    Move-Item -LiteralPath $partial -Destination $Destination
    Write-Host "installed derived GPTQ v2 fixture: $Destination"
    Write-Host "derived GPTQ v2 model SHA256: $actualSha256"
}

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
$client.DefaultRequestHeaders.UserAgent.ParseAdd("xeno-rt-cuda-validation/1.0")

$actOrderBase = "https://huggingface.co/Mohaaxa/qwen2.5-1.5b-gptq-4bit-v2/resolve/$actOrderRevision"
$denseBase = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/$denseRevision"
$files = @(
    [pscustomobject]@{ RelativePath = "$actOrderDirectoryName\config.json"; Uri = "$actOrderBase/config.json"; Bytes = 205528L; Sha256 = "e4922014526ef5d2fcee33db5fec3c1123743f8d7aaa3aa8a22d5e697d5de79d" },
    [pscustomobject]@{ RelativePath = "$actOrderDirectoryName\merges.txt"; Uri = "$actOrderBase/merges.txt"; Bytes = 1671853L; Sha256 = "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5" },
    [pscustomobject]@{ RelativePath = "$actOrderDirectoryName\tokenizer_config.json"; Uri = "$actOrderBase/tokenizer_config.json"; Bytes = 7306L; Sha256 = "7e88129d9769a0b14b1587a7d5e829fe93ac0e1511636471fdfc0811951418e6" },
    [pscustomobject]@{ RelativePath = "$actOrderDirectoryName\vocab.json"; Uri = "$actOrderBase/vocab.json"; Bytes = 2776833L; Sha256 = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910" },
    [pscustomobject]@{ RelativePath = "$actOrderDirectoryName\model.safetensors"; Uri = "$actOrderBase/model.safetensors"; Bytes = 1176640304L; Sha256 = "db2f1663e71306db8f4607ebd6166fbf663013aaa61e42cc0c1c6ce3f16c11c1" },
    [pscustomobject]@{ RelativePath = "$denseDirectoryName\config.json"; Uri = "$denseBase/config.json"; Bytes = 660L; Sha256 = "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670" },
    [pscustomobject]@{ RelativePath = "$denseDirectoryName\merges.txt"; Uri = "$denseBase/merges.txt"; Bytes = 1671839L; Sha256 = "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3" },
    [pscustomobject]@{ RelativePath = "$denseDirectoryName\tokenizer_config.json"; Uri = "$denseBase/tokenizer_config.json"; Bytes = 7305L; Sha256 = "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583" },
    [pscustomobject]@{ RelativePath = "$denseDirectoryName\vocab.json"; Uri = "$denseBase/vocab.json"; Bytes = 2776833L; Sha256 = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910" },
    [pscustomobject]@{ RelativePath = "$denseDirectoryName\model.safetensors"; Uri = "$denseBase/model.safetensors"; Bytes = 3087467144L; Sha256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee" }
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

$baseFixture = & (Join-Path $PSScriptRoot "prepare-real-gptq-fixture.ps1") `
    -CacheRoot $cacheRootPath `
    -TimeoutSeconds $TimeoutSeconds
$sourceV1Directory = [IO.Path]::GetFullPath($baseFixture.GptqModelDirectory)
$derivedV2Directory = Resolve-CachePath $derivedV2DirectoryName
New-Item -ItemType Directory -Force -Path $derivedV2Directory | Out-Null
foreach ($asset in @("merges.txt", "tokenizer_config.json", "vocab.json")) {
    Copy-Item `
        -LiteralPath (Join-Path $sourceV1Directory $asset) `
        -Destination (Join-Path $derivedV2Directory $asset) `
        -Force
}

$sourceConfigPath = Join-Path $sourceV1Directory "config.json"
$derivedConfigPath = Join-Path $derivedV2Directory "config.json"
$config = Get-Content -LiteralPath $sourceConfigPath -Raw | ConvertFrom-Json
$quantization = $config.quantization_config
Set-JsonProperty $quantization "checkpoint_format" "gptq_v2"
Set-JsonProperty $quantization "format" "gptq_v2"
if (
    -not ($quantization.PSObject.Properties.Name -contains "meta") -or
    $null -eq $quantization.meta
) {
    Set-JsonProperty $quantization "meta" ([pscustomobject]@{})
}
Set-JsonProperty $quantization.meta "v2" $true
$configJson = $config | ConvertTo-Json -Depth 40
[IO.File]::WriteAllText(
    $derivedConfigPath,
    $configJson,
    [Text.UTF8Encoding]::new($false)
)
[IO.File]::WriteAllText(
    (Join-Path $derivedV2Directory "quantize_config.json"),
    ($quantization | ConvertTo-Json -Depth 40),
    [Text.UTF8Encoding]::new($false)
)

$derivedModelPath = Join-Path $derivedV2Directory "model.safetensors"
Convert-GptqV1ModelToV2 `
    -Source (Join-Path $sourceV1Directory "model.safetensors") `
    -Destination $derivedModelPath

[pscustomobject]@{
    ActOrderModelDirectory = Resolve-CachePath $actOrderDirectoryName
    DenseModelDirectory = Resolve-CachePath $denseDirectoryName
    GptqV1ModelDirectory = $sourceV1Directory
    GptqV2ModelDirectory = $derivedV2Directory
    GptqV2ModelSha256 = Get-FileSha256 $derivedModelPath
    ActOrderRevision = $actOrderRevision
    DenseRevision = $denseRevision
}
