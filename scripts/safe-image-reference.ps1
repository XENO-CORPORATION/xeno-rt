param(
    [Parameter(Mandatory = $true)]
    [switch]$ConfirmLargeModelRun,
    [Parameter(Mandatory = $true)]
    [ValidateSet("diffusers", "native")]
    [string]$Engine,
    [ValidateSet("smoke", "release")]
    [string]$Profile = "smoke",
    [ValidateRange(300, 21600)]
    [int]$TimeoutSeconds = 7200,
    [string]$RunId = "",
    [switch]$NoRehash
)

$ErrorActionPreference = "Stop"

if (-not $ConfirmLargeModelRun) {
    throw "-ConfirmLargeModelRun is required because this command loads tens of GiB and uses the GPU"
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$referencePython = Join-Path $repoRoot "reference\image\qwen\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $referencePython -PathType Leaf)) {
    throw "missing pinned reference environment; run uv sync --project reference/image/qwen --python 3.11 --frozen"
}

function Get-ImageReferenceProcess {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @("sd-cli.exe", "xrt-server.exe", "xrt-cli.exe", "cargo.exe", "rustc.exe") -or
        $_.CommandLine -like "*run_diffusers_reference.py*" -or
        $_.CommandLine -like "*run_native_comparator.py*"
    })
}

function Stop-ProcessTree {
    param([int]$ProcessId)

    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & taskkill.exe /T /F /PID $ProcessId *> $null
    } finally {
        $ErrorActionPreference = $oldPreference
        $global:LASTEXITCODE = 0
    }
}

function Assert-Quiet {
    param(
        [string]$Message,
        [int[]]$AllowedIds = @()
    )

    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        $unexpected = @(Get-ImageReferenceProcess | Where-Object { $AllowedIds -notcontains $_.ProcessId })
        if (-not $unexpected) {
            return
        }
        Start-Sleep -Milliseconds 500
    }
    $unexpected = @(Get-ImageReferenceProcess | Where-Object { $AllowedIds -notcontains $_.ProcessId })
    if ($unexpected) {
        $unexpected | Select-Object ProcessId, Name, CommandLine | Format-Table -AutoSize
        throw $Message
    }
}

function Join-Arguments {
    param([string[]]$Values)

    ($Values | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }) -join " "
}

$knownIds = @(Get-ImageReferenceProcess | ForEach-Object { $_.ProcessId })
if ($knownIds) {
    Assert-Quiet "pre-existing image/Cargo/GPU process detected"
}

$script = if ($Engine -eq "diffusers") {
    Join-Path $repoRoot "reference\image\qwen\run_diffusers_reference.py"
} else {
    Join-Path $repoRoot "reference\image\qwen\run_native_comparator.py"
}
$arguments = @($script, "--profile", $Profile)
if ($RunId) {
    $arguments += @("--run-id", $RunId)
}
if ($NoRehash) {
    $arguments += if ($Engine -eq "diffusers") { "--no-rehash-bundle" } else { "--no-rehash-artifacts" }
}
if ($Engine -eq "native") {
    $arguments += @("--timeout-seconds", "$TimeoutSeconds")
}

Write-Host "Running pinned $Engine $Profile image reference with a ${TimeoutSeconds}s outer bound"
$process = [Diagnostics.Process]::new()
$process.StartInfo.FileName = $referencePython
$process.StartInfo.Arguments = Join-Arguments $arguments
$process.StartInfo.WorkingDirectory = $repoRoot
$process.StartInfo.UseShellExecute = $false
$failure = $null
try {
    [void]$process.Start()
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        Stop-ProcessTree $process.Id
        $failure = "image reference timed out after ${TimeoutSeconds}s"
    } elseif ($process.ExitCode -ne 0) {
        $failure = "image reference failed with exit code $($process.ExitCode)"
    }
} finally {
    $processId = $process.Id
    $process.Dispose()
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        $leftovers = @(Get-ImageReferenceProcess | Where-Object { $knownIds -notcontains $_.ProcessId })
        if (-not $leftovers) {
            break
        }
        Start-Sleep -Milliseconds 500
    }
    $leftovers = @(Get-ImageReferenceProcess | Where-Object { $knownIds -notcontains $_.ProcessId })
    foreach ($leftover in $leftovers) {
        Stop-ProcessTree $leftover.ProcessId
    }
    if ($leftovers -and -not $failure) {
        $failure = "image reference left child processes after parent $processId exited"
    }
}

if ($failure) {
    throw $failure
}

Write-Host "safe image reference passed"
