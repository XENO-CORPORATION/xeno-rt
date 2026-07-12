param(
    [string]$Model = "vibethinker-3b-q4",
    [string]$Prompt = "Hello",
    [string]$CacheMode = "f32",
    [ValidateSet("0", "1", "auto")]
    [string]$CudaGraphMode = "auto",
    [ValidateSet("0", "1")]
    [string]$PrefixCacheMode = "1",
    [int]$MaxTokens = 1,
    [int]$Repetitions = 1,
    [ValidateRange(1, 8)]
    [int]$Concurrency = 1,
    [ValidateRange(1, 65536)]
    [int]$PrefillChunkTokens = 128,
    [ValidateRange(1, 1024)]
    [int]$MaxDecodeTurnsBeforePrefill = 8,
    [ValidateRange(1, 8)]
    [int]$MaxDecodeBatchSize = 4,
    [ValidateRange(0, 1000000)]
    [int]$DecodeBatchWaitMicros = 20000,
    [int]$BuildTimeoutSeconds = 240,
    [int]$RunTimeoutSeconds = 180,
    [ValidateRange(0, 1048576)]
    [int]$MaxInitialGpuMemoryUsedMB = 4096,
    [switch]$ConfirmGpuRun,
    [switch]$CompareCpu,
    [switch]$Profile
)

$ErrorActionPreference = "Stop"
$allowedCacheModes = @("f32", "float", "float32", "q8", "int8", "kq4_vq8", "kq4", "key_q4_value_q8", "key-q4-value-q8", "q4_keys_q8_values", "agent_adaptive", "agent-adaptive", "adaptive", "agent")
if ($allowedCacheModes -notcontains $CacheMode.Trim().ToLowerInvariant()) {
    throw "unsupported -CacheMode '$CacheMode'; use f32, q8, kq4_vq8, or agent_adaptive"
}
if ($Repetitions -lt 1) {
    throw "-Repetitions must be at least 1"
}
if (-not $ConfirmGpuRun) {
    throw "safe CUDA smoke runs a real GPU/model workload; rerun with -ConfirmGpuRun"
}
$env:CARGO_BUILD_JOBS = "1"
$env:RUST_TEST_THREADS = "1"
$env:XRT_CUDA_GRAPH = $CudaGraphMode
$env:XRT_PREFIX_CACHE = $PrefixCacheMode
if ($Profile) {
    $env:XRT_CUDA_PROFILE = "1"
} else {
    Remove-Item Env:XRT_CUDA_PROFILE -ErrorAction SilentlyContinue
}

$rustupCargo = Join-Path $env:USERPROFILE ".rustup\toolchains\stable-x86_64-pc-windows-msvc\bin\cargo.exe"
$cargo = "cargo"
if (Test-Path $rustupCargo) {
    $cargo = $rustupCargo
} else {
    $cargoCommand = Get-Command $cargo -ErrorAction SilentlyContinue
    if ($cargoCommand) {
        $cargo = $cargoCommand.Source
    }
}
$targetRoot = if ($env:CARGO_TARGET_DIR) {
    [IO.Path]::GetFullPath($env:CARGO_TARGET_DIR)
} else {
    Join-Path (Get-Location) "target"
}

function Get-RustXrtProcess {
    Get-Process -Name cargo, rustc, xrt-cli, xrt-server, xrt-runtime -ErrorAction SilentlyContinue |
        ForEach-Object {
            [pscustomobject]@{
                ProcessId    = $_.Id
                Name         = $_.ProcessName
                WorkingSetMB = [math]::Round($_.WorkingSet64 / 1MB, 1)
            }
        }
}

function Get-RustXrtProcessIds {
    @(Get-RustXrtProcess | ForEach-Object { $_.ProcessId })
}

function Stop-RustXrtProcessTree {
    param([int[]]$ProcessIds)

    foreach ($processId in $ProcessIds) {
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & taskkill.exe /T /F /PID $processId *> $null | Out-Null
        } catch {
        } finally {
            $ErrorActionPreference = $oldErrorActionPreference
            $global:LASTEXITCODE = 0
        }
    }
}

function Assert-RustXrtQuiet {
    param([string]$Message)

    for ($i = 0; $i -lt 40; $i++) {
        $processes = Get-RustXrtProcess
        if (-not $processes) {
            return
        }
        Start-Sleep -Milliseconds 500
    }

    $processes = Get-RustXrtProcess
    if ($processes) {
        $processes | Format-Table -AutoSize
        throw $Message
    }
}

Assert-RustXrtQuiet "pre-existing Rust/xrt process detected"

function Assert-GpuHeadroom {
    $memoryLine = & nvidia-smi `
        --query-gpu=memory.used,memory.total `
        --format=csv,noheader,nounits 2>&1
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($memoryLine)) {
        throw "failed to query GPU memory before CUDA smoke: $memoryLine"
    }
    $firstGpu = @($memoryLine)[0].ToString().Split(',')
    if ($firstGpu.Count -ne 2) {
        throw "unexpected nvidia-smi memory output: $memoryLine"
    }
    $usedMB = 0
    $totalMB = 0
    if (-not [int]::TryParse($firstGpu[0].Trim(), [ref]$usedMB) -or
        -not [int]::TryParse($firstGpu[1].Trim(), [ref]$totalMB)) {
        throw "invalid nvidia-smi memory output: $memoryLine"
    }
    if ($usedMB -gt $MaxInitialGpuMemoryUsedMB) {
        throw "GPU is busy before CUDA smoke: ${usedMB} MiB / ${totalMB} MiB is already used; limit is ${MaxInitialGpuMemoryUsedMB} MiB. Close GPU-heavy apps before retrying."
    }
    Write-Host "GPU headroom preflight passed: ${usedMB} MiB / ${totalMB} MiB used"
}

Assert-GpuHeadroom

function Wait-RustXrtQuietOrKillNew {
    param(
        [string]$Message,
        [int[]]$KnownIds
    )

    for ($i = 0; $i -lt 40; $i++) {
        $processes = Get-RustXrtProcess
        if (-not $processes) {
            return
        }
        Start-Sleep -Milliseconds 500
    }

    $processes = Get-RustXrtProcess
    $newProcesses = @($processes | Where-Object { $KnownIds -notcontains $_.ProcessId })
    if ($newProcesses) {
        $newProcesses | Format-Table -AutoSize
        Stop-RustXrtProcessTree @($newProcesses | ForEach-Object { $_.ProcessId })
        Start-Sleep -Seconds 2
    }

    $remaining = Get-RustXrtProcess
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
        [switch]$CaptureStdout
    )

    Write-Host "$FilePath $($Arguments -join ' ')"
    $knownIds = Get-RustXrtProcessIds
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo.FileName = $FilePath
    $process.StartInfo.Arguments = Join-ProcessArguments $Arguments
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    $failureMessage = $null
    $capturedStdout = $null
    $stdoutTask = $null
    $stderrTask = $null
    try {
        [void]$process.Start()
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            Stop-RustXrtProcessTree @($process.Id)
            Stop-RustXrtProcessTree @((Get-Process -Name xrt-cli -ErrorAction SilentlyContinue | ForEach-Object { $_.Id }))
            [void]$process.WaitForExit(5000)
            $failureMessage = "process timed out after ${TimeoutSeconds}s"
        } elseif ($process.ExitCode -ne 0) {
            $failureMessage = "process failed with exit code $($process.ExitCode)"
        }
    } finally {
        if ($stdoutTask) {
            $stdout = $stdoutTask.GetAwaiter().GetResult()
            if (-not [string]::IsNullOrWhiteSpace($stdout)) {
                $stdout = $stdout.TrimEnd()
                if ($CaptureStdout) {
                    $capturedStdout = $stdout
                    Write-Host $stdout
                } else {
                    Write-Output $stdout
                }
            }
        }
        if ($stderrTask) {
            $stderr = $stderrTask.GetAwaiter().GetResult()
            if (-not [string]::IsNullOrWhiteSpace($stderr)) {
                $stderr = $stderr.TrimEnd()
                if ($CaptureStdout) {
                    Write-Host $stderr
                } else {
                    Write-Output $stderr
                }
            }
        }
        $process.Dispose()
        Wait-RustXrtQuietOrKillNew "leftover Rust/xrt process detected after: $FilePath $($Arguments -join ' ')" $knownIds
    }
    if ($failureMessage) {
        throw $failureMessage
    }
    if ($CaptureStdout) {
        return $capturedStdout
    }
}

function Assert-BenchmarkTransferTelemetry {
    param([string]$Json)

    if ([string]::IsNullOrWhiteSpace($Json)) {
        throw "benchmark produced no JSON output"
    }
    $objectMarker = $Json.IndexOf('"object": "xrt.bench"')
    if ($objectMarker -lt 0) {
        throw "benchmark output does not contain an xrt.bench JSON object"
    }
    $jsonStart = $Json.LastIndexOf([char]'{', $objectMarker)
    $jsonEnd = $Json.LastIndexOf([char]'}')
    if ($jsonStart -lt 0 -or $jsonEnd -le $jsonStart) {
        throw "benchmark output contains an incomplete xrt.bench JSON object"
    }
    $jsonPayload = $Json.Substring($jsonStart, $jsonEnd - $jsonStart + 1)
    $report = $jsonPayload | ConvertFrom-Json
    if ($report.object -ne "xrt.bench") {
        throw "benchmark output object must be xrt.bench"
    }

    $cudaResults = @($report.results | Where-Object { $_.active_backend -eq "cuda-resident" })
    if ($cudaResults.Count -eq 0) {
        throw "benchmark output contains no active cuda-resident result"
    }
    foreach ($result in $cudaResults) {
        if ($null -ne $result.error) {
            throw "CUDA benchmark returned an error: $($result.error)"
        }
        if ($null -eq $result.transfer_delta) {
            throw "CUDA benchmark did not report a transfer delta"
        }
        if ($null -eq $result.gpu_resource.transfer_totals) {
            throw "CUDA benchmark did not report cumulative transfer totals"
        }
        if ($null -eq $result.gpu_resource.allocation_totals) {
            throw "CUDA benchmark did not report cumulative allocation totals"
        }
        if ($null -eq $result.allocation_delta) {
            throw "CUDA benchmark did not report an allocation interval"
        }
        if ($result.gpu_resource.model_weight_bytes -le 0) {
            throw "CUDA benchmark did not report resident model bytes"
        }
        if ($result.transfer_delta.host_to_device_bytes -ge $result.gpu_resource.model_weight_bytes) {
            throw "CUDA generation transferred model-sized host data"
        }
        if ($result.output_tokens -le 0) {
            throw "CUDA benchmark generated no tokens"
        }
        if ($result.transfer_delta.device_to_host_calls -ne $result.output_tokens) {
            throw "CUDA generation must download exactly one logits vector per output token"
        }
        if ($result.transfer_delta.device_to_host_bytes -le 0 -or
            $result.transfer_delta.device_to_host_bytes % $result.output_tokens -ne 0) {
            throw "CUDA logits download bytes must be positive and divisible by output tokens"
        }
        if ($result.allocation_delta.baseline_bytes -le 0 -or
            $result.allocation_delta.final_bytes -lt $result.allocation_delta.baseline_bytes -or
            $result.allocation_delta.peak_bytes -lt $result.allocation_delta.final_bytes) {
            throw "CUDA benchmark allocation baseline/final/peak ordering is invalid"
        }
        if ($PrefixCacheMode -eq "0" -and
            $result.allocation_delta.final_bytes -ne $result.allocation_delta.baseline_bytes) {
            throw "CUDA benchmark without prefix retention must return to its allocation baseline"
        }
        if ($result.allocation_delta.peak_bytes -lt $result.allocation_delta.baseline_bytes -or
            $result.allocation_delta.allocation_calls -le 0 -or
            $result.allocation_delta.allocated_bytes -le 0) {
            throw "CUDA benchmark allocation interval is incomplete"
        }
    }

    $cpuResults = @($report.results | Where-Object { $_.active_backend -eq "cpu" })
    foreach ($result in $cpuResults) {
        if ($null -ne $result.transfer_delta -or
            $null -ne $result.allocation_delta -or
            $null -ne $result.gpu_resource.transfer_totals -or
            $null -ne $result.gpu_resource.allocation_totals) {
            throw "CPU benchmark must not report CUDA transfer or allocation telemetry"
        }
    }
}

function Assert-CleanExitSoak {
    for ($i = 0; $i -lt 18; $i++) {
        Start-Sleep -Seconds 5
        Wait-RustXrtQuietOrKillNew "leftover Rust/xrt process detected during clean-exit soak" @()
    }
}

Invoke-BoundedProcess $cargo @("build", "-p", "xrt-cli", "--features", "cuda") $BuildTimeoutSeconds

$cli = Join-Path $targetRoot "debug\xrt-cli.exe"
if (-not (Test-Path $cli)) {
    throw "missing built CLI at $cli"
}
$backends = if ($CompareCpu) { "cpu,cuda" } else { "cuda" }

$benchJson = Invoke-BoundedProcess $cli @(
    "bench",
    "--model", $Model,
    "--prompt", $Prompt,
    "--max-tokens", "$MaxTokens",
    "--repetitions", "$Repetitions",
    "--concurrency", "$Concurrency",
    "--prefill-chunk-tokens", "$PrefillChunkTokens",
    "--max-decode-turns-before-prefill", "$MaxDecodeTurnsBeforePrefill",
    "--max-decode-batch-size", "$MaxDecodeBatchSize",
    "--decode-batch-wait-micros", "$DecodeBatchWaitMicros",
    "--cache-modes", $CacheMode,
    "--backends", $backends,
    "--seed", "1",
    "--json"
) $RunTimeoutSeconds -CaptureStdout
Assert-BenchmarkTransferTelemetry $benchJson

Assert-CleanExitSoak

Write-Host "safe CUDA smoke passed"
