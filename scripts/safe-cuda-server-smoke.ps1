param(
    [string]$Model = "vibethinker-3b-q4",
    [ValidateRange(2, 8)]
    [int]$Concurrency = 2,
    [ValidateRange(1, 128)]
    [int]$MaxTokens = 8,
    [ValidateRange(1, 65536)]
    [int]$PrefillChunkTokens = 128,
    [ValidateRange(1, 1024)]
    [int]$MaxDecodeTurnsBeforePrefill = 8,
    [ValidateRange(1, 8)]
    [int]$MaxDecodeBatchSize = 4,
    [ValidateRange(0, 1000000)]
    [int]$DecodeBatchWaitMicros = 2000,
    [int]$BuildTimeoutSeconds = 600,
    [int]$RunTimeoutSeconds = 300,
    [switch]$ConfirmGpuRun
)

$ErrorActionPreference = "Stop"
if (-not $ConfirmGpuRun) {
    throw "safe CUDA server smoke runs a real GPU/model workload; rerun with -ConfirmGpuRun"
}
$env:CARGO_BUILD_JOBS = "1"
$env:RUST_TEST_THREADS = "1"
$env:XRT_CUDA_GRAPH = "auto"
Remove-Item Env:XRT_CUDA_PROFILE -ErrorAction SilentlyContinue

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
    Join-Path (Get-Location) "target"
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

function Stop-ProcessTree {
    param([int]$ProcessId)

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & taskkill.exe /T /F /PID $ProcessId *> $null | Out-Null
    } finally {
        $ErrorActionPreference = $previousPreference
        $global:LASTEXITCODE = 0
    }
}

function Invoke-BoundedProcess {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [int]$TimeoutSeconds
    )

    Write-Host "$FilePath $($Arguments -join ' ')"
    $process = [Diagnostics.Process]::new()
    $process.StartInfo.FileName = $FilePath
    $process.StartInfo.Arguments = Join-ProcessArguments $Arguments
    $process.StartInfo.UseShellExecute = $false
    try {
        [void]$process.Start()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            Stop-ProcessTree $process.Id
            throw "process timed out after ${TimeoutSeconds}s"
        }
        if ($process.ExitCode -ne 0) {
            throw "process failed with exit code $($process.ExitCode)"
        }
    } finally {
        $process.Dispose()
    }
}

if (Get-Process -Name xrt-server -ErrorAction SilentlyContinue) {
    throw "pre-existing xrt-server process detected"
}

Invoke-BoundedProcess $cargo @("build", "-p", "xrt-server", "--features", "cuda") $BuildTimeoutSeconds

$serverExe = Join-Path $targetRoot "debug\xrt-server.exe"
if (-not (Test-Path -LiteralPath $serverExe -PathType Leaf)) {
    throw "missing built server at $serverExe"
}

$portProbe = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
$portProbe.Start()
$port = ([Net.IPEndPoint]$portProbe.LocalEndpoint).Port
$portProbe.Stop()

$artifactRoot = Join-Path (Get-Location) "artifacts"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$stdoutPath = Join-Path $artifactRoot "cuda-server-smoke.stdout.log"
$stderrPath = Join-Path $artifactRoot "cuda-server-smoke.stderr.log"
Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue

$serverArguments = @(
    "--model", $Model,
    "--backend", "cuda",
    "--host", "127.0.0.1",
    "--port", "$port",
    "--max-active-sequences", "$Concurrency",
    "--max-queued-sequences", "$Concurrency",
    "--stream-buffer-capacity", "4",
    "--prefill-chunk-tokens", "$PrefillChunkTokens",
    "--max-decode-turns-before-prefill", "$MaxDecodeTurnsBeforePrefill"
    "--max-decode-batch-size", "$MaxDecodeBatchSize"
    "--decode-batch-wait-micros", "$DecodeBatchWaitMicros"
)
$server = Start-Process `
    -FilePath $serverExe `
    -ArgumentList $serverArguments `
    -WorkingDirectory (Get-Location).Path `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

Add-Type -AssemblyName System.Net.Http
[Net.ServicePointManager]::DefaultConnectionLimit = [Math]::Max(8, $Concurrency + 2)
$client = [Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds($RunTimeoutSeconds)
$baseUrl = "http://127.0.0.1:$port"
$responses = @()
$contents = @()

function Start-ChatRequest {
    param(
        [int]$Index,
        [string]$Prompt
    )

    $payload = @{
        model = $Model
        messages = @(
            @{
                role = "user"
                content = $Prompt
            }
        )
        max_tokens = $MaxTokens
        temperature = 0.0
        top_k = 1
        stream = $true
        seed = 1
    } | ConvertTo-Json -Depth 8 -Compress
    $content = [Net.Http.StringContent]::new(
        $payload,
        [Text.Encoding]::UTF8,
        "application/json"
    )
    [pscustomobject]@{
        Index = $Index
        Content = $content
        Task = $client.PostAsync("$baseUrl/v1/chat/completions", $content)
    }
}

try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        if ($server.HasExited) {
            throw "xrt-server exited during startup with code $($server.ExitCode)"
        }
        try {
            $statusResponse = $client.GetAsync("$baseUrl/v1/runtime/status").GetAwaiter().GetResult()
            if ($statusResponse.IsSuccessStatusCode) {
                $status = $statusResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult() |
                    ConvertFrom-Json
                $statusResponse.Dispose()
                if ($status.ready -and $status.active_backend -eq "cuda-resident") {
                    $ready = $true
                    break
                }
            } else {
                $statusResponse.Dispose()
            }
        } catch {
        }
        Start-Sleep -Seconds 1
    }
    if (-not $ready) {
        throw "xrt-server did not become CUDA-ready"
    }

    $tasks = @()
    $longPrompt = (("This is bounded scheduler context segment 1. " * 32) -join "") +
        "Reply with a short greeting after reading the context."
    $longRequest = Start-ChatRequest 1 $longPrompt
    $contents += $longRequest.Content
    $tasks += $longRequest.Task

    $longPrefillStarted = $false
    for ($attempt = 0; $attempt -lt 300; $attempt++) {
        $overlapStatusResponse = $client.GetAsync("$baseUrl/v1/runtime/status").GetAwaiter().GetResult()
        if ($overlapStatusResponse.IsSuccessStatusCode) {
            $overlapStatus = $overlapStatusResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult() |
                ConvertFrom-Json
            if ($overlapStatus.scheduler.active_execution_phase -eq "prefill") {
                $longPrefillStarted = $true
                $overlapStatusResponse.Dispose()
                break
            }
        }
        $overlapStatusResponse.Dispose()
        Start-Sleep -Milliseconds 100
    }
    if (-not $longPrefillStarted) {
        throw "long request did not enter a prefill turn"
    }

    $shortRequest = Start-ChatRequest 0 "Reply with a short greeting for concurrent request 0."
    $contents += $shortRequest.Content
    $tasks += $shortRequest.Task
    for ($index = 2; $index -lt $Concurrency; $index++) {
        $request = Start-ChatRequest $index "Reply with a short greeting for request $index."
        $contents += $request.Content
        $tasks += $request.Task
    }

    foreach ($task in $tasks) {
        if (-not $task.Wait($RunTimeoutSeconds * 1000)) {
            throw "concurrent OpenAI streaming request timed out"
        }
        $response = $task.Result
        $responses += $response
        if (-not $response.IsSuccessStatusCode) {
            $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            throw "streaming request failed with HTTP $([int]$response.StatusCode): $body"
        }
        $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        if ($body -notmatch 'chat\.completion\.chunk') {
            throw "streaming response did not contain chat completion chunks"
        }
        if ($body -notmatch 'data:\s*\[DONE\]') {
            throw "streaming response did not terminate with [DONE]"
        }
    }

    $finalStatusResponse = $client.GetAsync("$baseUrl/v1/runtime/status").GetAwaiter().GetResult()
    if (-not $finalStatusResponse.IsSuccessStatusCode) {
        throw "failed to read final runtime status"
    }
    $finalStatus = $finalStatusResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult() |
        ConvertFrom-Json
    $finalStatusResponse.Dispose()
    $scheduler = $finalStatus.scheduler
    if ($scheduler.active_sequences -ne 0 -or $scheduler.queued_sequences -ne 0) {
        throw "scheduler did not drain active/queued sequences"
    }
    if ($scheduler.kv_reserved_bytes -ne 0) {
        throw "scheduler leaked $($scheduler.kv_reserved_bytes) reserved KV bytes"
    }
    if ($scheduler.active_prefill_sequences -ne 0) {
        throw "scheduler leaked $($scheduler.active_prefill_sequences) prefill registrations"
    }
    if ($scheduler.admitted_total -lt $Concurrency) {
        throw "scheduler admitted only $($scheduler.admitted_total) of $Concurrency requests"
    }
    if ($scheduler.completed_prefill_turns -le $Concurrency) {
        throw "long prompt did not produce chunked prefill turns: $($scheduler.completed_prefill_turns)"
    }
    if ($scheduler.decode_turns_with_active_prefill -lt 1) {
        throw "no decode turn ran while the long request remained in prefill"
    }
    if ($Concurrency -gt 1 -and $MaxDecodeBatchSize -gt 1) {
        if ($scheduler.max_observed_decode_batch_size -lt 2) {
            throw "decode rendezvous never formed a multi-sequence batch"
        }
        if ($scheduler.completed_fused_decode_batches -lt 1) {
            throw "no multi-sequence CUDA decode graph replay completed"
        }
    }
    $captureCount = @(
        Select-String `
            -LiteralPath $stderrPath `
            -Pattern "captured CUDA batch-1 decode graph" `
            -ErrorAction SilentlyContinue
    ).Count
    if ($captureCount -gt $Concurrency) {
        throw "prefill captured decode graphs: captures=$captureCount concurrency=$Concurrency"
    }
    $batchCaptureCount = @(
        Select-String `
            -LiteralPath $stderrPath `
            -Pattern "captured CUDA multi-sequence decode graph" `
            -ErrorAction SilentlyContinue
    ).Count

    Write-Host (
        "concurrent CUDA server smoke passed: requests={0} prefill_turns={1} decode_batches={2} fused_batches={3} max_batch={4} decode_during_active_prefill={5} batch_graph_captures={6} batch1_graph_captures={7}" -f
            $Concurrency,
            $scheduler.completed_prefill_turns,
            $scheduler.completed_decode_turns,
            $scheduler.completed_fused_decode_batches,
            $scheduler.max_observed_decode_batch_size,
            $scheduler.decode_turns_with_active_prefill,
            $batchCaptureCount,
            $captureCount
    )
} catch {
    if (Test-Path -LiteralPath $stdoutPath) {
        Get-Content -LiteralPath $stdoutPath -Tail 80
    }
    if (Test-Path -LiteralPath $stderrPath) {
        Get-Content -LiteralPath $stderrPath -Tail 80
    }
    throw
} finally {
    foreach ($response in $responses) {
        $response.Dispose()
    }
    foreach ($content in $contents) {
        $content.Dispose()
    }
    $client.Dispose()
    if (-not $server.HasExited) {
        Stop-ProcessTree $server.Id
        [void]$server.WaitForExit(30000)
    }
    $server.Dispose()
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        if (-not (Get-Process -Name xrt-server -ErrorAction SilentlyContinue)) {
            break
        }
        Start-Sleep -Milliseconds 500
    }
    if (Get-Process -Name xrt-server -ErrorAction SilentlyContinue) {
        throw "xrt-server process remained after concurrent smoke cleanup"
    }
}
