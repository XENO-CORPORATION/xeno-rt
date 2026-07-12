param(
    [int]$TimeoutSeconds = 240,
    [switch]$RunGpuParity,
    [switch]$RunLayerDiagnostics,
    [switch]$RunRealSafeTensorsCuda,
    [switch]$RunRealAwqCuda,
    [switch]$RunRealGptqCuda,
    [switch]$RunRealCompressedTensorsCuda,
    [string]$RealModelPath = "",
    [string]$RealSafeTensorsPath = "",
    [string]$RealAwqPath = "",
    [string]$RealAwqGgufPath = "",
    [string]$RealGptqPath = "",
    [string]$RealGptqGgufPath = "",
    [string]$RealCompressedTensorsPath = "",
    [string]$RealDenseHfPath = ""
)

$ErrorActionPreference = "Stop"
$env:CARGO_BUILD_JOBS = "1"
$env:RUST_TEST_THREADS = "1"
Remove-Item Env:XRT_CUDA_PROFILE -ErrorAction SilentlyContinue

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
$testDepsRoot = Join-Path $targetRoot "debug\deps"

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

    for ($round = 0; $round -lt 12; $round++) {
        $processes = Get-RustXrtProcess
        if (-not $processes) {
            return
        }

        $newProcesses = @($processes | Where-Object { $KnownIds -notcontains $_.ProcessId })
        if ($newProcesses) {
            $newProcesses | Format-Table -AutoSize
            Stop-RustXrtProcessTree @($newProcesses | ForEach-Object { $_.ProcessId })
        }
        Start-Sleep -Seconds 5
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

function Invoke-SafeProcess {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [int]$ProcessTimeoutSeconds = $TimeoutSeconds
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
        if (-not $process.WaitForExit($ProcessTimeoutSeconds * 1000)) {
            Stop-RustXrtProcessTree @($process.Id)
            $failureMessage = "process timed out after ${ProcessTimeoutSeconds}s"
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
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 5
        Wait-RustXrtQuietOrKillNew "leftover Rust/xrt process detected during clean-exit soak" @()
    }
    Start-Sleep -Seconds 15
    Wait-RustXrtQuietOrKillNew "leftover Rust/xrt process detected during final quiet check" @()
}

function Invoke-SafeCargo {
    param(
        [string[]]$Arguments,
        [int]$ProcessTimeoutSeconds = $TimeoutSeconds
    )

    Invoke-SafeProcess $cargo $Arguments $ProcessTimeoutSeconds
}

function Get-LatestTestExe {
    param([string]$Prefix)

    $exe = Get-ChildItem -Path $testDepsRoot -Filter "$Prefix-*.exe" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $exe) {
        throw "missing test executable for prefix $Prefix"
    }
    $exe.FullName
}

function Get-TestExeWithFilter {
    param(
        [string]$Prefix,
        [string]$Filter
    )

    $exes = Get-ChildItem -Path $testDepsRoot -Filter "$Prefix-*.exe" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    foreach ($exe in $exes) {
        $list = & $exe.FullName "--list"
        if ($list -match [regex]::Escape($Filter)) {
            return $exe.FullName
        }
    }
    throw "missing test executable for prefix $Prefix containing filter $Filter"
}

function Invoke-TestExe {
    param(
        [string]$Exe,
        [string]$Filter
    )

    Invoke-SafeProcess $Exe @($Filter)
}

function Invoke-IgnoredGpuTestExe {
    param(
        [string]$Exe,
        [string]$Filter
    )

    Invoke-SafeProcess $Exe @(
        $Filter,
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1"
    )
}

$gpuParityFailures = [Collections.Generic.List[string]]::new()

function Invoke-GpuParityCase {
    param(
        [string]$Exe,
        [string]$Filter
    )

    try {
        Invoke-IgnoredGpuTestExe $Exe $Filter
    } catch {
        if ($_.Exception.Message -notlike "process failed with exit code*") {
            throw
        }
        Write-Host "CUDA parity case failed: $Filter"
        $script:gpuParityFailures.Add($Filter)
    }
}

function Invoke-TestFilter {
    param(
        [string]$Prefix,
        [string]$Filter
    )

    $exe = Get-TestExeWithFilter $Prefix $Filter
    Invoke-TestExe $exe $Filter
}

function Assert-SmokeRejectsInvalidCacheMode {
    try {
        & ".\scripts\safe-cuda-smoke.ps1" -CacheMode "__invalid__"
        throw "safe CUDA smoke accepted invalid cache mode"
    } catch {
        if ($_.Exception.Message -notlike "*unsupported -CacheMode*") {
            throw
        }
    }
}

Assert-SmokeRejectsInvalidCacheMode
Invoke-SafeCargo @("test", "-p", "xrt-safetensors")
Invoke-SafeCargo @("test", "-p", "xrt-tokenizer", "hf_bpe_loader_")
if ($RealSafeTensorsPath) {
    if (-not (Test-Path -LiteralPath $RealSafeTensorsPath -PathType Container)) {
        throw "missing real SafeTensors model directory: $RealSafeTensorsPath"
    }
    if (-not $RealModelPath -or -not (Test-Path -LiteralPath $RealModelPath -PathType Leaf)) {
        throw "real SafeTensors tokenizer parity requires the equivalent -RealModelPath GGUF"
    }
    $env:XRT_REAL_HF_MODEL_DIR = [IO.Path]::GetFullPath($RealSafeTensorsPath)
    $env:XRT_REAL_GGUF = [IO.Path]::GetFullPath($RealModelPath)
    try {
        Invoke-SafeCargo @(
            "test",
            "-p",
            "xrt-safetensors",
            "tests::real_hf_bundle_validates_shards_and_qwen2_tensor_metadata",
            "--",
            "--ignored",
            "--exact",
            "--nocapture"
        )
        Invoke-SafeCargo @(
            "test",
            "-p",
            "xrt-runtime",
            "--features",
            "cuda",
            "resident_tensor::tests::real_hf_qwen2_source_maps_every_dense_tensor",
            "--",
            "--ignored",
            "--exact",
            "--nocapture"
        )
        if ($RunRealSafeTensorsCuda) {
            Invoke-SafeCargo -ProcessTimeoutSeconds 1200 -Arguments @(
                "test",
                "--release",
                "-p",
                "xrt-workspace-tests",
                "--features",
                "cuda",
                "--test",
                "smoke_e2e",
                "cuda_real_safetensors_qwen2_matches_equivalent_gguf_top_tokens",
                "--",
                "--ignored",
                "--exact",
                "--nocapture"
            )
        }
        Invoke-SafeCargo @(
            "test",
            "-p",
            "xrt-tokenizer",
            "tests::real_hf_tokenizer_matches_the_equivalent_gguf_tokenizer",
            "--",
            "--ignored",
            "--exact",
            "--nocapture"
        )
    } finally {
        Remove-Item Env:XRT_REAL_HF_MODEL_DIR -ErrorAction SilentlyContinue
        Remove-Item Env:XRT_REAL_GGUF -ErrorAction SilentlyContinue
    }
}
if ($RunRealAwqCuda) {
    if (-not $RealAwqPath -or -not (Test-Path -LiteralPath $RealAwqPath -PathType Container)) {
        throw "real AutoAWQ CUDA parity requires -RealAwqPath"
    }
    if (-not $RealAwqGgufPath -or -not (Test-Path -LiteralPath $RealAwqGgufPath -PathType Leaf)) {
        throw "real AutoAWQ CUDA parity requires -RealAwqGgufPath"
    }
    $env:XRT_REAL_AWQ_MODEL_DIR = [IO.Path]::GetFullPath($RealAwqPath)
    $env:XRT_REAL_AWQ_GGUF = [IO.Path]::GetFullPath($RealAwqGgufPath)
    try {
        Invoke-SafeCargo @(
            "test",
            "-p",
            "xrt-runtime",
            "--features",
            "cuda",
            "resident_tensor::tests::real_autoawq_qwen2_source_maps_every_packed_tensor",
            "--",
            "--ignored",
            "--exact",
            "--nocapture"
        )
        Invoke-SafeCargo -ProcessTimeoutSeconds 1200 -Arguments @(
            "test",
            "--release",
            "-p",
            "xrt-workspace-tests",
            "--features",
            "cuda",
            "--test",
            "smoke_e2e",
            "cuda_real_autoawq_qwen2_matches_equivalent_gguf_top_tokens",
            "--",
            "--ignored",
            "--exact",
            "--nocapture"
        )
    } finally {
        Remove-Item Env:XRT_REAL_AWQ_MODEL_DIR -ErrorAction SilentlyContinue
        Remove-Item Env:XRT_REAL_AWQ_GGUF -ErrorAction SilentlyContinue
    }
}
if ($RunRealGptqCuda) {
    if (-not $RealGptqPath -or -not (Test-Path -LiteralPath $RealGptqPath -PathType Container)) {
        throw "real GPTQ CUDA parity requires -RealGptqPath"
    }
    if (-not $RealGptqGgufPath -or -not (Test-Path -LiteralPath $RealGptqGgufPath -PathType Leaf)) {
        throw "real GPTQ CUDA parity requires -RealGptqGgufPath"
    }
    $env:XRT_REAL_GPTQ_MODEL_DIR = [IO.Path]::GetFullPath($RealGptqPath)
    $env:XRT_REAL_GPTQ_GGUF = [IO.Path]::GetFullPath($RealGptqGgufPath)
    try {
        Invoke-SafeCargo @(
            "test",
            "-p",
            "xrt-runtime",
            "--features",
            "cuda",
            "resident_tensor::tests::real_gptq_v1_qwen2_source_maps_every_packed_tensor",
            "--",
            "--ignored",
            "--exact",
            "--nocapture"
        )
        Invoke-SafeCargo @(
            "test",
            "-p",
            "xrt-runtime",
            "--features",
            "cuda",
            "resident_tensor::tests::real_gptq_v1_qwen2_kernels_match_host_dequantization",
            "--",
            "--ignored",
            "--exact",
            "--nocapture"
        )
        Invoke-SafeCargo -ProcessTimeoutSeconds 1200 -Arguments @(
            "test",
            "--release",
            "-p",
            "xrt-workspace-tests",
            "--features",
            "cuda",
            "--test",
            "smoke_e2e",
            "cuda_real_gptq_v1_qwen2_matches_equivalent_gguf_semantics",
            "--",
            "--ignored",
            "--exact",
            "--nocapture"
        )
    } finally {
        Remove-Item Env:XRT_REAL_GPTQ_MODEL_DIR -ErrorAction SilentlyContinue
        Remove-Item Env:XRT_REAL_GPTQ_GGUF -ErrorAction SilentlyContinue
    }
}
if ($RunRealCompressedTensorsCuda) {
    if (-not $RealCompressedTensorsPath -or -not (Test-Path -LiteralPath $RealCompressedTensorsPath -PathType Container)) {
        throw "real compressed-tensors CUDA parity requires -RealCompressedTensorsPath"
    }
    if (-not $RealDenseHfPath -or -not (Test-Path -LiteralPath $RealDenseHfPath -PathType Container)) {
        throw "real compressed-tensors CUDA parity requires -RealDenseHfPath"
    }
    $env:XRT_REAL_COMPRESSED_TENSORS_MODEL_DIR = [IO.Path]::GetFullPath($RealCompressedTensorsPath)
    $env:XRT_REAL_DENSE_HF_MODEL_DIR = [IO.Path]::GetFullPath($RealDenseHfPath)
    try {
        Invoke-SafeCargo @(
            "test",
            "-p",
            "xrt-runtime",
            "--features",
            "cuda",
            "resident_tensor::tests::real_compressed_tensors_qwen2_source_maps_every_packed_tensor",
            "--",
            "--ignored",
            "--exact",
            "--nocapture"
        )
        Invoke-SafeCargo @(
            "test",
            "-p",
            "xrt-runtime",
            "--features",
            "cuda",
            "resident_tensor::tests::real_compressed_tensors_qwen2_kernels_match_host_dequantization",
            "--",
            "--ignored",
            "--exact",
            "--nocapture"
        )
        Invoke-SafeCargo -ProcessTimeoutSeconds 1200 -Arguments @(
            "test",
            "--release",
            "-p",
            "xrt-workspace-tests",
            "--features",
            "cuda",
            "--test",
            "smoke_e2e",
            "cuda_real_compressed_tensors_qwen2_matches_dense_bf16_semantics",
            "--",
            "--ignored",
            "--exact",
            "--nocapture"
        )
    } finally {
        Remove-Item Env:XRT_REAL_COMPRESSED_TENSORS_MODEL_DIR -ErrorAction SilentlyContinue
        Remove-Item Env:XRT_REAL_DENSE_HF_MODEL_DIR -ErrorAction SilentlyContinue
    }
}
Invoke-SafeCargo @(
    "test",
    "-p",
    "xrt-tokenizer",
    "mapping_get_preprocessing_supports_present_and_default_values"
)
Invoke-SafeCargo @("check", "-p", "xrt-runtime", "--features", "cuda")
Invoke-SafeCargo @("test", "-p", "xrt-runtime", "--features", "cuda", "--no-run")
$cudaRuntimeFeatureTest = Get-TestExeWithFilter "xrt_runtime" "cuda_feature_session_can_select_quantized_gpu_kv"
Invoke-TestExe $cudaRuntimeFeatureTest "cuda_feature_session_can_select_quantized_gpu_kv"
Invoke-TestExe $cudaRuntimeFeatureTest "cuda_adaptive_position_routing_matches_policy"
Invoke-TestExe $cudaRuntimeFeatureTest "cuda_adaptive_route_migration_needed_detects_mask_drift"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_autoawq_gemm_source_maps_versioned_tensor_groups"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_autoawq_source_rejects_wrong_packed_geometry"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_gptq_v1_source_maps_versioned_tensor_groups"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_gptq_source_rejects_nonstandard_groups_without_desc_act"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_gptq_v1_act_order_source_maps_explicit_groups"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_gptq_source_rejects_nonzero_auxiliary_bias"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_gptq_v2_source_maps_direct_zero_encoding"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_gptq_source_rejects_missing_desc_act"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_compressed_tensors_w4a16_source_maps_permuted_groups"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_compressed_tensors_source_rejects_wrong_format"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_compressed_tensors_source_rejects_activation_quantization"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_compressed_tensors_source_rejects_asymmetric_weights"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_compressed_tensors_source_rejects_malformed_group_indices"
Invoke-TestExe $cudaRuntimeFeatureTest "resident_tensor::tests::synthetic_compressed_tensors_source_rejects_shape_payload_mismatch"
Invoke-SafeCargo @("test", "-p", "xrt-runtime", "cuda_session_adaptive_router_uses_retained_policy_metadata")
Invoke-SafeCargo @(
    "test",
    "-p",
    "xrt-models",
    "hf_qwen2_autoawq_reuses_standard_model_geometry"
)
Invoke-SafeCargo @(
    "test",
    "-p",
    "xrt-models",
    "hf_qwen2_gptq_v1_reuses_standard_model_geometry"
)
Invoke-SafeCargo @(
    "test",
    "-p",
    "xrt-models",
    "hf_qwen2_compressed_tensors_reuses_standard_model_geometry"
)
Invoke-SafeCargo @("check", "-p", "xrt-cli", "--features", "cuda")
Invoke-SafeCargo @("test", "-p", "xrt-cli", "--no-run")
Invoke-TestFilter "xrt_cli" "concurrent_bench_helpers_report_aggregate_metrics"
Invoke-SafeCargo @("check", "-p", "xrt-server", "--features", "cuda")
Invoke-SafeCargo @("test", "-p", "xrt-server", "--no-run")
Invoke-TestFilter "xrt_server" "multipart_request_parts_parse_expected_fields"
Invoke-SafeCargo @("test", "-p", "xrt-workspace-tests", "--features", "cuda", "--no-run")
Invoke-SafeCargo @("test", "-p", "xrt-workspace-tests", "--no-run")
Invoke-TestFilter "smoke_e2e" "gpu_resource_status_tracks_active_sessions"
Invoke-TestFilter "smoke_e2e" "synthetic_float_fixtures_decode_on_cpu"
Invoke-TestFilter "smoke_e2e" "scheduled_chunked_prefill_matches_unscheduled_generation"
Invoke-TestFilter "smoke_e2e" "repeated_cpu_prompt_reuses_an_immutable_prefix_snapshot"

Invoke-SafeCargo @("test", "-p", "xrt-runtime", "--no-run")
Invoke-TestFilter "xrt_runtime" "f32_prefix_pages_are_shared_until_the_suffix_is_written"
Invoke-TestFilter "xrt_runtime" "quantized_prefix_pages_copy_on_write_without_mutating_the_snapshot"
Invoke-TestFilter "xrt_runtime" "structural_prefix_stops_before_the_first_user_span"
Invoke-TestFilter "xrt_runtime" "exact_key_dimensions_control_hits"
Invoke-TestFilter "xrt_runtime" "lru_eviction_keeps_the_recent_entry"
Invoke-TestFilter "xrt_runtime" "invalid_config_values_use_bounded_defaults"
Invoke-TestFilter "xrt_runtime" "scheduler_config_rejects_zero_active_and_stream_capacity"
Invoke-TestFilter "xrt_runtime" "execution_turns_prioritize_decode_and_bound_prefill_wait"
Invoke-TestFilter "xrt_runtime" "same_phase_execution_turns_are_fifo"
Invoke-TestFilter "xrt_runtime" "kv_reservations_enforce_aggregate_budget_and_release_on_drop"
Invoke-TestFilter "xrt_runtime" "external_prefix_kv_bytes_reduce_the_scheduler_budget"
Invoke-TestFilter "xrt_runtime" "scheduler_bounds_active_and_queued_requests"
Invoke-TestFilter "xrt_runtime" "cancelled_waiter_releases_queue_capacity"
Invoke-TestFilter "xrt_runtime" "invalid_values_fall_back_to_safe_defaults"
Invoke-TestFilter "xrt_runtime" "total_len_after_batch_checks_overflow"
Invoke-TestFilter "xrt_runtime" "logits_for_position_checks_bounds"
Invoke-TestFilter "xrt_runtime" "checked_add_reports_overflow"
Invoke-TestFilter "xrt_runtime" "chunk_embedding_overrides_move_and_remap_local_positions"
Invoke-TestFilter "xrt_runtime" "cuda_profile_value_requires_truthy_value"
Invoke-TestFilter "xrt_runtime" "embedding_overrides_validate_before_cuda_prefill"
Invoke-TestFilter "xrt_runtime" "resident_linear_dtype_support_includes_f32_and_current_quant_formats"
Invoke-TestFilter "xrt_runtime" "q8_0_probe_status_requires_q8_embedding_and_output"
Invoke-TestFilter "xrt_runtime" "dense_decode_status_requires_gpu_resident_embedding_and_all_layers"
Invoke-TestFilter "xrt_runtime" "cuda_decode_unsupported_message_lists_current_dense_formats"
Invoke-TestFilter "xrt_runtime" "cuda_all_logits_output_len_checks_overflow"
Invoke-TestFilter "xrt_runtime" "cuda_model_upload_budget_applies_fraction_free_and_reserve"
Invoke-TestFilter "xrt_runtime" "cuda_kv_budget_uses_remaining_safe_vram_fraction"
Invoke-TestFilter "xrt_runtime" "cuda_session_kv_byte_estimate_matches_cache_modes"
Invoke-TestFilter "xrt_runtime" "cuda_extra_resident_tensor_bytes_accounts_for_expanded_and_tied_formats"
Invoke-TestFilter "xrt_runtime" "cuda_k_quant_embedding_layout_caps_expanded_residency"
Invoke-TestFilter "xrt_runtime" "cuda_position_helpers_check_overflow"
Invoke-TestFilter "xrt_runtime" "layer0_projection_probe_rejects_nonzero_position"
Invoke-TestFilter "xrt_runtime" "cuda_session_rejects_lengths_beyond_context_before_allocating"
Invoke-TestFilter "xrt_runtime" "cuda_session_zero_length_prepare_stays_unallocated"
Invoke-TestFilter "xrt_runtime" "session_kv_reservation_estimate_covers_growth_peak"
Invoke-TestFilter "xrt_runtime" "cuda_session_rejects_kv_allocation_over_budget_before_allocating"
Invoke-TestFilter "xrt_runtime" "non_cuda_session_maps_quantized_modes_to_f32"
Invoke-TestFilter "xrt_runtime" "cuda_session_retains_policy_metadata_for_future_adaptive_router"
Invoke-TestFilter "xrt_runtime" "cuda_adaptive_route_migration_needed_detects_mask_drift"
Invoke-TestFilter "xrt_runtime" "cuda_cache_layout_changes_when_mode_or_shape_changes"
Invoke-TestFilter "xrt_runtime" "parses_all_cuda_graph_modes"
Invoke-TestFilter "xrt_runtime" "runtime_level_status_has_no_session_cache_mode"
Invoke-TestFilter "xrt_runtime" "cuda_replace_cache_updates_shape_without_replacing_context_len"

Invoke-SafeCargo @("test", "-p", "xrt-cuda", "--no-run")
$cudaDefaultTest = Get-TestExeWithFilter "xrt_cuda" "resident_api_stubs_fail_clearly_without_cuda_feature"
Invoke-TestExe $cudaDefaultTest "resident_api_stubs_fail_clearly_without_cuda_feature"

Invoke-SafeCargo @("test", "-p", "xrt-cuda", "--features", "cuda", "--no-run")
$cudaFeatureTest = Get-TestExeWithFilter "xrt_cuda" "float_tensor_bytes_decode_supported_dtypes_without_cuda_device"
Invoke-TestExe $cudaFeatureTest "float_tensor_bytes_decode_supported_dtypes_without_cuda_device"
Invoke-TestExe $cudaFeatureTest "q8_kv_allocated_bytes_formula_is_smaller_than_f32"
Invoke-TestExe $cudaFeatureTest "kq4_vq8_kv_allocated_bytes_formula_is_smaller_than_q8"
Invoke-TestExe $cudaFeatureTest "q4_k_matrix_dequantizes_to_transposed_cpu_layout_without_cuda_device"
Invoke-TestExe $cudaFeatureTest "q5_k_matrix_dequantizes_to_transposed_cpu_layout_without_cuda_device"
Invoke-TestExe $cudaFeatureTest "q6_k_matrix_dequantizes_to_transposed_cpu_layout_without_cuda_device"

if ($RunGpuParity) {
    Write-Host "running serial CUDA kernel parity tests"
    foreach ($filter in @(
        "tests::cuda_graph_replays_stable_buffers_with_updated_inputs",
        "tests::cuda_parallel_child_graphs_replay_independent_buffers",
        "tests::cuda_graph_decode_params_advance_rope_paged_kv_and_attention",
        "tests::paged_kv_clones_preserve_remapped_prefixes_and_are_independent",
        "tests::resident_f32_kernels_match_host_upload_path",
        "tests::silu_mul_device_path_matches_scalar_reference",
        "tests::gemma4_activation_primitives_match_cpu_reference",
        "tests::rope_device_path_matches_scalar_reference",
        "tests::repeat_kv_for_gqa_device_matches_scalar_reference",
        "tests::single_query_attention_device_matches_scalar_reference",
        "tests::q8_layer_kv_append_dequantize_matches_scalar_reference",
        "tests::kq4_vq8_layer_kv_append_dequantize_matches_scalar_reference",
        "tests::q8_0_matvec_kernel_matches_scalar_reference",
        "tests::awq_gemm4_matvec_kernel_matches_scalar_reference",
        "tests::gptq_gemm4_matvec_kernel_matches_scalar_reference",
        "tests::gptq_explicit_gemm4_matvec_kernel_matches_v1_and_v2_references",
        "tests::compressed_tensors_w4a16_matvec_kernel_matches_scalar_reference"
    )) {
        Invoke-GpuParityCase $cudaFeatureTest $filter
    }
    Invoke-GpuParityCase `
        $cudaRuntimeFeatureTest `
        "resident_tensor::tests::synthetic_autoawq_runtime_executes_full_cuda_decode"
    Invoke-GpuParityCase `
        $cudaRuntimeFeatureTest `
        "resident_tensor::tests::synthetic_gptq_runtime_executes_full_cuda_decode"
    Invoke-GpuParityCase `
        $cudaRuntimeFeatureTest `
        "resident_tensor::tests::synthetic_gptq_explicit_runtime_executes_v1_act_order_and_v2_decode"
    Invoke-GpuParityCase `
        $cudaRuntimeFeatureTest `
        "resident_tensor::tests::synthetic_compressed_tensors_w4a16_runtime_executes_full_cuda_decode"

    Write-Host "running serial CUDA runtime parity tests"
    $workspaceCudaTest = Get-TestExeWithFilter "smoke_e2e" "cuda_q8_0_runtime_matches_cpu_logits"
    foreach ($filter in @(
        "cuda_q8_0_runtime_matches_cpu_logits",
        "cuda_repeated_prompt_reuses_immutable_prefix_kv",
        "cuda_multi_sequence_decode_graph_matches_cpu_logits",
        "cuda_q8_0_tied_output_runtime_matches_cpu_logits",
        "cuda_q8_0_quantized_kv_modes_decode",
        "cuda_gemma4_f32_runtime_matches_cpu_logits",
        "cuda_gemma4_quantized_kv_runtime_matches_cpu_logits",
        "cuda_f16_runtime_matches_cpu_logits",
        "cuda_bf16_runtime_matches_cpu_logits",
        "cuda_q4_0_runtime_matches_cpu_logits",
        "cuda_q4_k_runtime_matches_cpu_logits",
        "cuda_q5_k_runtime_matches_cpu_logits",
        "cuda_q6_k_runtime_matches_cpu_logits"
    )) {
        Invoke-GpuParityCase $workspaceCudaTest $filter
    }

    if ($RealModelPath) {
        if (-not (Test-Path -LiteralPath $RealModelPath -PathType Leaf)) {
            throw "real-model parity GGUF not found: $RealModelPath"
        }
        $previousRealGguf = $env:XRT_REAL_GGUF
        $previousLayerDiagnostics = $env:XRT_REAL_GGUF_LAYER_DIAGNOSTICS
        try {
            $env:XRT_REAL_GGUF = (Resolve-Path -LiteralPath $RealModelPath).Path
            if ($RunLayerDiagnostics) {
                $env:XRT_REAL_GGUF_LAYER_DIAGNOSTICS = "1"
            } else {
                Remove-Item Env:XRT_REAL_GGUF_LAYER_DIAGNOSTICS -ErrorAction SilentlyContinue
            }
            Invoke-GpuParityCase $workspaceCudaTest "cuda_real_model_first_token_logits_choose_same_top_token_as_cpu"
        } finally {
            if ($null -eq $previousRealGguf) {
                Remove-Item Env:XRT_REAL_GGUF -ErrorAction SilentlyContinue
            } else {
                $env:XRT_REAL_GGUF = $previousRealGguf
            }
            if ($null -eq $previousLayerDiagnostics) {
                Remove-Item Env:XRT_REAL_GGUF_LAYER_DIAGNOSTICS -ErrorAction SilentlyContinue
            } else {
                $env:XRT_REAL_GGUF_LAYER_DIAGNOSTICS = $previousLayerDiagnostics
            }
        }
    }

    if ($gpuParityFailures.Count -gt 0) {
        throw "CUDA parity failures: $($gpuParityFailures -join ', ')"
    }
}

Assert-CleanExitSoak

Write-Host "safe CUDA check passed"
