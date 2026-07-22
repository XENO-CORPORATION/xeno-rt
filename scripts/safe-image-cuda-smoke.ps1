param(
    [Parameter(Mandatory = $true)]
    [string]$BundlePath,
    [Parameter(Mandatory = $true)]
    [switch]$ConfirmLargeModelRun,
    [ValidateSet("q4_k_m", "q6_k", "q5_k_m")]
    [string]$Tier = "q4_k_m",
    [ValidateSet("correctness-16x16-s2", "comparator-512x512-s4")]
    [string]$Workload = "correctness-16x16-s2",
    [switch]$AcceptUnpinnedCandidate,
    [ValidateRange(0, 15)]
    [int]$Device = 0,
    [ValidateRange(16384, 24576)]
    [int]$MinimumFreeVramMiB = 18432,
    [ValidateRange(16, 128)]
    [int]$MinimumAvailableHostMemoryGiB = 32,
    [ValidateRange(60, 3600)]
    [int]$BuildTimeoutSeconds = 1200,
    [ValidateRange(300, 7200)]
    [int]$RunTimeoutSeconds = 1800,
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"
$profiles = @{
    q4_k_m = [ordered]@{
        Id = "qwen-image-2512-q4_k_m"
        Quantization = "Q4_K_M"
        ManifestSha256 = "c89870d6d8b6196b7bb0e3fad17a5ca047abebd55d9cd2fb74162d6c1baebbfc"
        TransformerArtifactBytes = 13244758899L
        ExpectedResidentWeightBytes = 13649426688L
    }
    q6_k = [ordered]@{
        Id = "qwen-image-2512-q6_k"
        Quantization = "Q6_K"
        ManifestSha256 = "687da89af54bcf577f41cb98a0e7e2b3dba50eff357e944e57cf1c94a7989140"
        TransformerArtifactBytes = 16824990240L
        ExpectedResidentWeightBytes = 17224349952L
    }
    q5_k_m = [ordered]@{
        Id = "qwen-image-2512-q5_k_m"
        Quantization = "Q5_K_M"
        ManifestSha256 = "7a30d9fa039144817673f525ad243a5adcf45daf2e3e7db24a7dacfeb2436801"
        TransformerArtifactBytes = 15000074784L
        ExpectedResidentWeightBytes = 15399434496L
    }
}
$workloads = @{
    "correctness-16x16-s2" = [ordered]@{
        FileStem = "cuda-smoke-16x16-s2"
        Prompt = "a"
        NegativePrompt = $null
        Width = 16
        Height = 16
        Steps = 2
        TrueCfgScale = 1.0
        Seed = 0L
        RetainFirstOutput = $false
        AllowedTiers = @("q4_k_m", "q6_k", "q5_k_m")
        ExpectedPngSha256 = @{
            q4_k_m = "82a69a3d50c4502f1166657b8c9df9e6e25848b13f9e00085c29ebc326b1ca71"
            q6_k = "c36d8715b331f2f52e20b218b519e24abf923df92674e28a4d7f8609a2b9c433"
            q5_k_m = "94177744e138df57d8aabf33edbd5c1d46f996b330c8f961f6f867a187e801a9"
        }
    }
    "comparator-512x512-s4" = [ordered]@{
        FileStem = "cuda-comparator-512x512-s4-seed424242"
        Prompt = "A cobalt mechanical keyboard on a walnut desk, precise product photograph."
        NegativePrompt = " "
        Width = 512
        Height = 512
        Steps = 4
        TrueCfgScale = 4.0
        Seed = 424242L
        RetainFirstOutput = $true
        AllowedTiers = @("q4_k_m")
        ExpectedPngSha256 = @{
            q4_k_m = "16d53f008029550757b257e2b40db234a7b913d26615573b698c4a77d015ade9"
        }
    }
}
$profile = $profiles[$Tier]
$workloadProfile = $workloads[$Workload]
if ($workloadProfile.AllowedTiers -notcontains $Tier) {
    throw "workload $Workload is not pinned for tier $Tier"
}
$expectedPngSha256 = $workloadProfile.ExpectedPngSha256[$Tier]
$candidateMode = [string]::IsNullOrWhiteSpace($expectedPngSha256)

if (-not $ConfirmLargeModelRun) {
    throw "this command loads the complete $($profile.Quantization) bundle and executes CUDA; rerun with -ConfirmLargeModelRun"
}
if ($candidateMode -and -not $AcceptUnpinnedCandidate) {
    throw "$($profile.Quantization) has no locked CUDA output yet; use -AcceptUnpinnedCandidate for one non-admission capture"
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$bundleMetadata = Get-Item -LiteralPath $BundlePath -Force
if (-not $bundleMetadata.PSIsContainer -or $bundleMetadata.LinkType) {
    throw "-BundlePath must be a real bundle directory, not a file or symlink"
}
$bundleRoot = [IO.Path]::GetFullPath($bundleMetadata.FullName)
$manifestPath = Join-Path $bundleRoot "xrt.bundle.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "bundle is missing xrt.bundle.json"
}
$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifestHash -ne $profile.ManifestSha256) {
    throw "safe CUDA smoke accepts only the pinned $($profile.Id) manifest $($profile.ManifestSha256)"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.id -ne $profile.Id -or $manifest.quantization -ne $profile.Quantization) {
    throw "pinned image CUDA smoke requires $($profile.Id) with $($profile.Quantization) quantization"
}

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
    $OutputPath = Join-Path $repoRoot "benchmark-results\image\native\$($profile.Id)-$($workloadProfile.FileStem)-$stamp.json"
} elseif (-not [IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $repoRoot $OutputPath
}
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
if ([IO.Path]::GetExtension($OutputPath) -ne ".json") {
    throw "-OutputPath must end in .json"
}
if (Test-Path -LiteralPath $OutputPath) {
    throw "refusing to overwrite existing evidence file $OutputPath"
}
$outputDirectory = Split-Path -Parent $OutputPath
if (-not (Test-Path -LiteralPath $outputDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}
$candidatePath = [IO.Path]::ChangeExtension($OutputPath, ".candidate.json")
if (Test-Path -LiteralPath $candidatePath) {
    throw "refusing to overwrite retained candidate report $candidatePath"
}
$retainedPngPath = if ($workloadProfile.RetainFirstOutput) {
    [IO.Path]::ChangeExtension($OutputPath, ".png")
} else {
    $null
}
if ($retainedPngPath -and (Test-Path -LiteralPath $retainedPngPath)) {
    throw "refusing to overwrite retained benchmark image $retainedPngPath"
}

$nvidiaSmiCommand = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidiaSmiCommand) {
    throw "nvidia-smi.exe is required for bounded CUDA admission checks"
}
$nvidiaSmi = $nvidiaSmiCommand.Source

$env:CARGO_BUILD_JOBS = "1"
$env:RUST_TEST_THREADS = "1"
$env:RAYON_NUM_THREADS = "32"
$env:XRT_CUDA_DEVICE = "$Device"
$env:XRT_GPU_MEMORY_FRACTION = "0.9"
$env:XRT_GPU_RESERVED_MB = "1024"
$env:XRT_CUDA_GRAPH = "0"
Remove-Item Env:XRT_CUDA_PROFILE -ErrorAction SilentlyContinue
Remove-Item Env:XRT_CUDA_KQUANT_MMQ -ErrorAction SilentlyContinue
Remove-Item Env:XRT_CPU_FLOAT_ACTIVATION_REFERENCE -ErrorAction SilentlyContinue

$rustupCargo = Join-Path $env:USERPROFILE ".rustup\toolchains\stable-x86_64-pc-windows-msvc\bin\cargo.exe"
$cargo = "cargo"
if (Test-Path -LiteralPath $rustupCargo -PathType Leaf) {
    $cargo = $rustupCargo
} else {
    $cargoCommand = Get-Command cargo -ErrorAction SilentlyContinue
    if ($cargoCommand) {
        $cargo = $cargoCommand.Source
    }
}
$targetRoot = if ($env:CARGO_TARGET_DIR) {
    [IO.Path]::GetFullPath($env:CARGO_TARGET_DIR)
} else {
    Join-Path $repoRoot "target"
}

function Get-ImageWorkloadProcess {
    $rustProcesses = @(Get-Process -Name cargo, rustc, xrt-cli, xrt-server, xrt-runtime, sd-cli -ErrorAction SilentlyContinue)
    $referenceProcesses = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -eq "python.exe" -and
        ($_.CommandLine -like "*reference\image\qwen*" -or $_.CommandLine -like "*run_native_comparator.py*")
    })
    @($rustProcesses | ForEach-Object {
        [pscustomobject]@{ ProcessId = $_.Id; Name = $_.ProcessName }
    }) + @($referenceProcesses | ForEach-Object {
        [pscustomobject]@{ ProcessId = $_.ProcessId; Name = $_.Name }
    })
}

function Stop-ProcessTree {
    param([int[]]$Ids)

    foreach ($id in $Ids) {
        $oldPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & taskkill.exe /T /F /PID $id *> $null | Out-Null
        } finally {
            $ErrorActionPreference = $oldPreference
            $global:LASTEXITCODE = 0
        }
    }
}

function Assert-ImageWorkloadQuiet {
    param([string]$Message)

    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        $processes = Get-ImageWorkloadProcess
        if (-not $processes) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    $processes = Get-ImageWorkloadProcess
    if ($processes) {
        $processes | Format-Table -AutoSize
        throw $Message
    }
}

function Wait-ImageWorkloadQuietOrKillNew {
    param(
        [string]$Message,
        [int[]]$KnownIds
    )

    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        $processes = Get-ImageWorkloadProcess
        if (-not $processes) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    $processes = Get-ImageWorkloadProcess
    $newProcesses = @($processes | Where-Object { $KnownIds -notcontains $_.ProcessId })
    if ($newProcesses) {
        $newProcesses | Format-Table -AutoSize
        Stop-ProcessTree @($newProcesses | ForEach-Object { $_.ProcessId })
        Start-Sleep -Seconds 2
    }
    $remaining = Get-ImageWorkloadProcess
    if ($remaining) {
        $remaining | Format-Table -AutoSize
        throw $Message
    }
}

function Join-ProcessArguments {
    param([string[]]$Arguments)

    ($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }) -join " "
}

function Invoke-BoundedProcess {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [int]$TimeoutSeconds,
        [switch]$Capture
    )

    Write-Host "$FilePath $($Arguments -join ' ')"
    $knownIds = @(Get-ImageWorkloadProcess | ForEach-Object { $_.ProcessId })
    $process = [Diagnostics.Process]::new()
    $process.StartInfo.FileName = $FilePath
    $process.StartInfo.Arguments = Join-ProcessArguments $Arguments
    $process.StartInfo.WorkingDirectory = $repoRoot
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    $failure = $null
    $stdoutTask = $null
    $stderrTask = $null
    $exitCode = $null
    try {
        [void]$process.Start()
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            Stop-ProcessTree @($process.Id)
            [void]$process.WaitForExit(5000)
            $failure = "process timed out after ${TimeoutSeconds}s"
        } else {
            $exitCode = $process.ExitCode
            if ($exitCode -ne 0) {
                $failure = "process failed with exit code $exitCode"
            }
        }
    } finally {
        $stdout = if ($stdoutTask) { $stdoutTask.GetAwaiter().GetResult() } else { "" }
        $stderr = if ($stderrTask) { $stderrTask.GetAwaiter().GetResult() } else { "" }
        if (-not [string]::IsNullOrWhiteSpace($stderr)) {
            Write-Host $stderr.TrimEnd()
        }
        if (-not $Capture -and -not [string]::IsNullOrWhiteSpace($stdout)) {
            Write-Host $stdout.TrimEnd()
        }
        $process.Dispose()
        Wait-ImageWorkloadQuietOrKillNew "leftover image workload process after bounded command" $knownIds
    }
    if ($failure) {
        if (-not [string]::IsNullOrWhiteSpace($stdout)) {
            Write-Host $stdout.TrimEnd()
        }
        throw $failure
    }
    if ($Capture) {
        return [pscustomobject]@{
            Stdout = $stdout.Trim()
            StderrBytes = [Text.Encoding]::UTF8.GetByteCount($stderr)
            StderrSha256 = if ($stderr) {
                $bytes = [Text.Encoding]::UTF8.GetBytes($stderr)
                ([BitConverter]::ToString([Security.Cryptography.SHA256]::HashData($bytes))).Replace("-", "").ToLowerInvariant()
            } else {
                $null
            }
        }
    }
}

function Get-GpuSnapshot {
    $row = @(& $nvidiaSmi -i $Device --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null)
    if ($LASTEXITCODE -ne 0 -or $row.Count -ne 1) {
        throw "unable to capture one nvidia-smi row for device $Device"
    }
    $fields = @($row[0].Split(',') | ForEach-Object { $_.Trim() })
    if ($fields.Count -ne 8) {
        throw "unexpected nvidia-smi GPU snapshot shape"
    }
    $processRows = @(& $nvidiaSmi -i $Device --query-compute-apps=pid --format=csv,noheader,nounits 2>$null |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($LASTEXITCODE -ne 0) {
        throw "unable to enumerate registered GPU processes for device $Device"
    }
    [pscustomobject]@{
        name = $fields[0]
        driver_version = $fields[1]
        total_mib = [int]$fields[2]
        used_mib = [int]$fields[3]
        free_mib = [int]$fields[4]
        utilization_percent = [int]$fields[5]
        temperature_c = [int]$fields[6]
        power_draw_w = $fields[7]
        registered_process_rows = $processRows.Count
    }
}

function Get-ToolVersion {
    param([string]$Tool)
    try {
        (& $Tool --version 2>$null | Select-Object -First 1).Trim()
    } catch {
        $null
    }
}

Assert-ImageWorkloadQuiet "pre-existing image/Cargo workload detected; the CUDA smoke must run serially"
$hostMemory = Get-CimInstance Win32_OperatingSystem
$availableHostGiB = [math]::Round(($hostMemory.FreePhysicalMemory * 1KB) / 1GB, 2)
if ($availableHostGiB -lt $MinimumAvailableHostMemoryGiB) {
    throw "only $availableHostGiB GiB host memory is available; at least $MinimumAvailableHostMemoryGiB GiB is required"
}
$gpuBefore = Get-GpuSnapshot
if ($gpuBefore.free_mib -lt $MinimumFreeVramMiB) {
    throw "device $Device has $($gpuBefore.free_mib) MiB free; at least $MinimumFreeVramMiB MiB is required"
}
$deviceTotalBytes = [int64]$gpuBefore.total_mib * 1MB
$observedNonXenoBytes = [int64]$gpuBefore.used_mib * 1MB
$reserveAwareCapBytes = $deviceTotalBytes - $observedNonXenoBytes - 2GB
$reserveAwareCapBytes = [math]::Min($reserveAwareCapBytes, 22GB)
if ($reserveAwareCapBytes -le 0 -or $profile.TransformerArtifactBytes -gt $reserveAwareCapBytes) {
    throw "$($profile.Quantization) transformer artifact $($profile.TransformerArtifactBytes) exceeds the current reserve-aware CUDA cap $reserveAwareCapBytes"
}

Invoke-BoundedProcess $cargo @(
    "build", "-p", "xrt-cli", "--release", "--features", "image-generation,cuda", "--quiet"
) $BuildTimeoutSeconds

$cli = Join-Path $targetRoot "release\xrt-cli.exe"
if (-not (Test-Path -LiteralPath $cli -PathType Leaf)) {
    throw "missing release image CLI at $cli"
}
$runArguments = @(
    "image", "bench",
    "--model-path", $bundleRoot,
    "--prompt", $workloadProfile.Prompt,
    "--size", "$($workloadProfile.Width)x$($workloadProfile.Height)",
    "--steps", "$($workloadProfile.Steps)",
    "--true-cfg-scale", "$($workloadProfile.TrueCfgScale)",
    "--seed", "$($workloadProfile.Seed)",
    "--n", "1",
    "--backend", "cuda",
    "--offload", "sequential",
    "--repetitions", "1",
    "--json"
)
if ($null -ne $workloadProfile.NegativePrompt) {
    $runArguments += @("--negative-prompt", $workloadProfile.NegativePrompt)
}
if ($retainedPngPath) {
    $runArguments += @("--retain-first-output", $retainedPngPath)
}
$run = Invoke-BoundedProcess $cli $runArguments $RunTimeoutSeconds -Capture

if ([string]::IsNullOrWhiteSpace($run.Stdout)) {
    throw "image CUDA benchmark returned no JSON"
}
$objectMarker = $run.Stdout.IndexOf('"object": "xrt.image.benchmark"')
if ($objectMarker -lt 0) {
    Write-Host $run.Stdout
    throw "image CUDA benchmark output does not contain an xrt.image.benchmark report"
}
$jsonStart = $run.Stdout.LastIndexOf([char]'{', $objectMarker)
$jsonEnd = $run.Stdout.LastIndexOf([char]'}')
if ($jsonStart -lt 0 -or $jsonEnd -le $jsonStart) {
    Write-Host $run.Stdout
    throw "image CUDA benchmark output contains an incomplete JSON report"
}
$jsonPayload = $run.Stdout.Substring($jsonStart, $jsonEnd - $jsonStart + 1)
$report = $jsonPayload | ConvertFrom-Json
[IO.File]::WriteAllText(
    $candidatePath,
    $jsonPayload + [Environment]::NewLine,
    [Text.UTF8Encoding]::new($false)
)
if ($report.object -ne "xrt.image.benchmark" -or $report.schema_version -ne 1) {
    throw "image CUDA benchmark returned an unsupported report schema"
}
if ($report.model -ne $profile.Id -or $report.requested_backend -ne "cuda") {
    throw "image CUDA benchmark did not execute the pinned model on the explicit CUDA backend"
}
if ($report.plan.backend -ne "cuda" -or $report.plan.offload -ne "sequential") {
    throw "image CUDA benchmark returned an unexpected execution plan"
}
if ([int64]$report.plan.estimated_device_bytes -gt $reserveAwareCapBytes) {
    throw "image CUDA plan estimates $($report.plan.estimated_device_bytes) device bytes above the current reserve-aware cap $reserveAwareCapBytes"
}
$samples = @($report.repetitions)
if ($samples.Count -ne 1 -or $samples[0].output_count -ne 1) {
    throw "image CUDA smoke must produce exactly one measured image"
}
if (-not $candidateMode -and $samples[0].first_output_sha256 -ne $expectedPngSha256) {
    throw "image CUDA smoke hash drifted: expected $expectedPngSha256, got $($samples[0].first_output_sha256)"
}
if ($retainedPngPath) {
    if (-not (Test-Path -LiteralPath $retainedPngPath -PathType Leaf)) {
        throw "image CUDA benchmark did not retain its first output"
    }
    $retainedPngSha256 = (Get-FileHash -LiteralPath $retainedPngPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($retainedPngSha256 -ne $samples[0].first_output_sha256) {
        throw "retained image hash does not match the benchmark report"
    }
    $rootPrefix = $repoRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $retainedPngEvidencePath = if ($retainedPngPath.StartsWith(
            $rootPrefix,
            [StringComparison]::OrdinalIgnoreCase
        )) {
        $retainedPngPath.Substring($rootPrefix.Length).Replace('\', '/')
    } else {
        [IO.Path]::GetFileName($retainedPngPath)
    }
}
if (-not $report.gpu_resource.cuda_feature_enabled -or -not $report.gpu_resource.cuda_available) {
    throw "image CUDA smoke did not report an available compiled CUDA backend"
}
$residentWeightBytes = [int64]$report.gpu_resource.arena_allocations.image_component_weight_bytes
if ($null -ne $profile.ExpectedResidentWeightBytes -and
    $residentWeightBytes -ne $profile.ExpectedResidentWeightBytes) {
    throw "image CUDA smoke reported unexpected resident transformer bytes: expected $($profile.ExpectedResidentWeightBytes), got $residentWeightBytes"
}
if ($residentWeightBytes -lt $profile.TransformerArtifactBytes) {
    throw "image CUDA smoke resident weights are below the pinned transformer artifact size"
}
if ($report.gpu_resource.arena_peak_allocated_bytes -lt $residentWeightBytes) {
    throw "image CUDA smoke peak allocation is below its resident weight allocation"
}

$gpuAfter = Get-GpuSnapshot
for ($attempt = 0; $attempt -lt 6; $attempt++) {
    Start-Sleep -Seconds 1
    Wait-ImageWorkloadQuietOrKillNew "delayed image workload process after CUDA smoke" @()
}

$gitHead = (& git -C $repoRoot rev-parse HEAD 2>$null | Select-Object -First 1).Trim()
$dirtyEntries = @(& git -C $repoRoot status --porcelain 2>$null).Count
$nonQuiet = $gpuBefore.registered_process_rows -gt 0
$evidence = [ordered]@{
    schema_version = 1
    status = if ($candidateMode) {
        if ($nonQuiet) { "captured_unpinned_cuda_candidate_non_quiet" } else { "captured_unpinned_cuda_candidate" }
    } elseif ($nonQuiet) {
        "passed_experimental_correctness_smoke_non_quiet"
    } else {
        "passed_experimental_correctness_smoke"
    }
    captured_at = [DateTime]::UtcNow.ToString("o")
    engine = "xeno-rt"
    runtime = "native_rust_cuda"
    model = [ordered]@{
        id = $profile.Id
        quantization = $profile.Quantization
        bundle_manifest_sha256 = $manifestHash
        official_revision = "25468b98e3276ca6700de15c6628e51b7de54a26"
    }
    request = [ordered]@{
        workload = $Workload
        prompt = $workloadProfile.Prompt
        negative_prompt = $workloadProfile.NegativePrompt
        seed = $workloadProfile.Seed
        width = $workloadProfile.Width
        height = $workloadProfile.Height
        steps = $workloadProfile.Steps
        true_cfg_scale = $workloadProfile.TrueCfgScale
        outputs = 1
        backend = "cuda"
        offload = "sequential"
    }
    build = [ordered]@{
        git_commit = $gitHead
        dirty = $dirtyEntries -gt 0
        dirty_entries = $dirtyEntries
        profile = "release"
        rustc = Get-ToolVersion "rustc"
        cargo = Get-ToolVersion $cargo
    }
    host = [ordered]@{
        available_memory_gib_before = $availableHostGiB
        logical_processors = [Environment]::ProcessorCount
    }
    gpu = [ordered]@{
        device = $Device
        before = $gpuBefore
        after = $gpuAfter
        reserve_aware_cap_bytes = $reserveAwareCapBytes
        quiet_performance_gate = -not $nonQuiet
        privacy_note = "Only the count of registered GPU process rows is retained; unrelated process IDs and paths are omitted."
    }
    command = "scripts/safe-image-cuda-smoke.ps1 -Tier $Tier -Workload $Workload -BundlePath '<pinned bundle>' -ConfirmLargeModelRun$($(if ($candidateMode) { ' -AcceptUnpinnedCandidate' } else { '' }))"
    benchmark = $report
    retained_output = if ($retainedPngPath) {
        [ordered]@{
            path = $retainedPngEvidencePath
            sha256 = $retainedPngSha256
            bytes = (Get-Item -LiteralPath $retainedPngPath).Length
        }
    } else {
        $null
    }
    process_stderr = [ordered]@{
        bytes = $run.StderrBytes
        sha256 = $run.StderrSha256
    }
    validation = [ordered]@{
        expected_png_sha256 = $expectedPngSha256
        observed_png_sha256 = $samples[0].first_output_sha256
        exact_output_hash = -not $candidateMode
        expected_resident_weight_bytes = $profile.ExpectedResidentWeightBytes
        measured_resident_weight_bytes = $residentWeightBytes
        measured_peak_allocation_bytes = $report.gpu_resource.arena_peak_allocated_bytes
    }
    admission = [ordered]@{
        bounded_cuda_correctness_smoke = if ($candidateMode) { "not_claimed_until_output_is_locked_and_repeated" } else { "passed" }
        production_support = $false
        performance_admission = $false
        reason = if ($candidateMode) {
            "The bounded CUDA run completed, but this first observed output is a candidate only; lock the hash and repeat before claiming correctness."
        } elseif ($nonQuiet) {
            "The exact bounded CUDA output passed, but registered non-XENO GPU workloads make this unsuitable as a quiet performance baseline; full-resolution quality and performance gates remain open."
        } else {
            "The exact bounded CUDA output passed; full-resolution quality and performance gates remain open."
        }
    }
}
$json = $evidence | ConvertTo-Json -Depth 20
[IO.File]::WriteAllText($OutputPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
Remove-Item -LiteralPath $candidatePath -Force
if ($candidateMode) {
    Write-Host "safe image CUDA candidate captured without admission: $OutputPath"
} else {
    Write-Host "safe image CUDA smoke passed: $OutputPath"
}
