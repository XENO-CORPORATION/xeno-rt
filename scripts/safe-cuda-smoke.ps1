param(
    [string]$Model = "vibethinker-3b-q4",
    [string]$Prompt = "Hello",
    [string]$CacheMode = "f32",
    [int]$MaxTokens = 1,
    [int]$BuildTimeoutSeconds = 240,
    [int]$RunTimeoutSeconds = 180,
    [switch]$ConfirmGpuRun,
    [switch]$CompareCpu,
    [switch]$Profile
)

$ErrorActionPreference = "Stop"
$allowedCacheModes = @("f32", "float", "float32", "q8", "int8", "kq4_vq8", "kq4", "key_q4_value_q8", "key-q4-value-q8", "q4_keys_q8_values", "agent_adaptive", "agent-adaptive", "adaptive", "agent")
if ($allowedCacheModes -notcontains $CacheMode.Trim().ToLowerInvariant()) {
    throw "unsupported -CacheMode '$CacheMode'; use f32, q8, kq4_vq8, or agent_adaptive"
}
if (-not $ConfirmGpuRun) {
    throw "safe CUDA smoke runs a real GPU/model workload; rerun with -ConfirmGpuRun"
}
$env:CARGO_BUILD_JOBS = "1"
$env:RUST_TEST_THREADS = "1"
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
        [int]$TimeoutSeconds
    )

    Write-Host "$FilePath $($Arguments -join ' ')"
    $knownIds = Get-RustXrtProcessIds
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo.FileName = $FilePath
    $process.StartInfo.Arguments = Join-ProcessArguments $Arguments
    $process.StartInfo.UseShellExecute = $false
    $failureMessage = $null
    try {
        [void]$process.Start()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            Stop-RustXrtProcessTree @($process.Id)
            Stop-RustXrtProcessTree @((Get-Process -Name xrt-cli -ErrorAction SilentlyContinue | ForEach-Object { $_.Id }))
            $failureMessage = "process timed out after ${TimeoutSeconds}s"
        } elseif ($process.ExitCode -ne 0) {
            $failureMessage = "process failed with exit code $($process.ExitCode)"
        }
    } finally {
        $process.Dispose()
        Wait-RustXrtQuietOrKillNew "leftover Rust/xrt process detected after: $FilePath $($Arguments -join ' ')" $knownIds
    }
    if ($failureMessage) {
        throw $failureMessage
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

Invoke-BoundedProcess $cli @(
    "bench",
    "--model", $Model,
    "--prompt", $Prompt,
    "--max-tokens", "$MaxTokens",
    "--cache-modes", $CacheMode,
    "--backends", $backends,
    "--seed", "1",
    "--json"
) $RunTimeoutSeconds

Assert-CleanExitSoak

Write-Host "safe CUDA smoke passed"
