param(
    [ValidateRange(60, 3600)]
    [int]$BuildTimeoutSeconds = 900,
    [ValidateRange(30, 900)]
    [int]$RunTimeoutSeconds = 180,
    [switch]$CompileOnly,
    [switch]$ConfirmGpuRun
)

$ErrorActionPreference = "Stop"

if (-not $CompileOnly -and -not $ConfirmGpuRun) {
    throw "image CUDA parity executes GPU kernels; rerun with -ConfirmGpuRun or use -CompileOnly"
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$env:CARGO_BUILD_JOBS = "1"
$env:RUST_TEST_THREADS = "1"
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

function Get-ImageRustProcess {
    @(Get-Process -Name cargo, rustc, xrt-cli, xrt-server, xrt-runtime -ErrorAction SilentlyContinue |
        ForEach-Object {
            [pscustomobject]@{
                ProcessId = $_.Id
                Name = $_.ProcessName
                WorkingSetMB = [math]::Round($_.WorkingSet64 / 1MB, 1)
            }
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

function Assert-ImageRustQuiet {
    param([string]$Message)

    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        $processes = Get-ImageRustProcess
        if (-not $processes) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    $processes = Get-ImageRustProcess
    if ($processes) {
        $processes | Format-Table -AutoSize
        throw $Message
    }
}

function Wait-ImageRustQuietOrKillNew {
    param(
        [string]$Message,
        [int[]]$KnownIds
    )

    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        $processes = Get-ImageRustProcess
        if (-not $processes) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    $processes = Get-ImageRustProcess
    $newProcesses = @($processes | Where-Object { $KnownIds -notcontains $_.ProcessId })
    if ($newProcesses) {
        $newProcesses | Format-Table -AutoSize
        Stop-ProcessTree @($newProcesses | ForEach-Object { $_.ProcessId })
        Start-Sleep -Seconds 2
    }
    $remaining = Get-ImageRustProcess
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
    $knownIds = @(Get-ImageRustProcess | ForEach-Object { $_.ProcessId })
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
    $stdout = ""
    try {
        [void]$process.Start()
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            Stop-ProcessTree @($process.Id)
            [void]$process.WaitForExit(5000)
            $failure = "process timed out after ${TimeoutSeconds}s"
        } elseif ($process.ExitCode -ne 0) {
            $failure = "process failed with exit code $($process.ExitCode)"
        }
    } finally {
        if ($stdoutTask) {
            $stdout = $stdoutTask.GetAwaiter().GetResult()
            if (-not [string]::IsNullOrWhiteSpace($stdout)) {
                if (-not $Capture) {
                    Write-Host $stdout.TrimEnd()
                }
            }
        }
        if ($stderrTask) {
            $stderr = $stderrTask.GetAwaiter().GetResult()
            if (-not [string]::IsNullOrWhiteSpace($stderr)) {
                Write-Host $stderr.TrimEnd()
            }
        }
        $process.Dispose()
        Wait-ImageRustQuietOrKillNew "leftover Rust/XRT process after bounded image command" $knownIds
    }
    if ($failure) {
        if ($Capture -and -not [string]::IsNullOrWhiteSpace($stdout)) {
            Write-Host $stdout.TrimEnd()
        }
        throw $failure
    }
    if ($Capture) {
        return $stdout.Trim()
    }
}

Assert-ImageRustQuiet "pre-existing Rust/XRT process detected; image CUDA checks must run serially"

Invoke-BoundedProcess $cargo @(
    "test", "-p", "xrt-image", "--features", "cuda", "--tests", "--no-run", "--quiet"
) $BuildTimeoutSeconds
Invoke-BoundedProcess $cargo @(
    "build", "-p", "xrt-cli", "--features", "image-generation,cuda", "--quiet"
) $BuildTimeoutSeconds

if (-not $CompileOnly) {
    $testRoot = Join-Path $targetRoot "debug\deps"
    $testExecutable = $null
    foreach ($candidate in @(Get-ChildItem -LiteralPath $testRoot -Filter "cuda_qwen_edit-*.exe" -File |
        Sort-Object LastWriteTimeUtc -Descending)) {
        $listing = Invoke-BoundedProcess $candidate.FullName @("--list") 30 -Capture
        if ($listing -match "tiny_generation_cuda_stays_in_cpu_parity" -and
            $listing -match "tiny_zero_conditioned_edit_cuda_matches_cpu" -and
            $listing -match "tiled_attention_above_portable_shared_memory_matches_cpu") {
            $testExecutable = $candidate
            break
        }
    }
    if (-not $testExecutable) {
        throw "no compiled cuda_qwen_edit executable contains all expected CUDA parity tests under $testRoot"
    }
    if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
        throw "nvidia-smi.exe is required before executing image CUDA parity tests"
    }
    $testOutput = Invoke-BoundedProcess $testExecutable.FullName @(
        "--ignored", "--nocapture", "--test-threads=1"
    ) $RunTimeoutSeconds -Capture
    if ($testOutput -notmatch "test result: ok\. 3 passed; 0 failed") {
        if ($testOutput) {
            Write-Host $testOutput
        }
        throw "image CUDA parity wrapper did not execute exactly three passing tests"
    }
    Write-Host $testOutput
}

for ($attempt = 0; $attempt -lt 6; $attempt++) {
    Start-Sleep -Seconds 1
    Wait-ImageRustQuietOrKillNew "delayed Rust/XRT process after image CUDA checks" @()
}

$mode = if ($CompileOnly) { "compile-only" } else { "compile plus tiny CUDA parity" }
Write-Host "safe image CUDA check passed ($mode)"
