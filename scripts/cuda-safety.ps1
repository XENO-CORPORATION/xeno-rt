function Assert-XrtGpuHeadroom {
    [CmdletBinding()]
    param(
        [ValidateRange(0, 1048576)]
        [int]$MaxInitialGpuMemoryUsedMB = 4096,
        [ValidateNotNullOrEmpty()]
        [string]$WorkloadName = "CUDA workload"
    )

    $deviceOrdinal = 0
    if (-not [string]::IsNullOrWhiteSpace($env:XRT_CUDA_DEVICE)) {
        if (-not [int]::TryParse($env:XRT_CUDA_DEVICE.Trim(), [ref]$deviceOrdinal) -or
            $deviceOrdinal -lt 0) {
            throw "invalid XRT_CUDA_DEVICE '$env:XRT_CUDA_DEVICE'"
        }
    }

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $nvidiaSmi) {
        throw "nvidia-smi is required for the $WorkloadName GPU headroom preflight"
    }

    $memoryLines = & $nvidiaSmi.Source `
        --query-gpu=index,memory.used,memory.total `
        --format=csv,noheader,nounits 2>&1
    $queryExitCode = $LASTEXITCODE
    if ($queryExitCode -ne 0 -or -not $memoryLines) {
        throw "failed to query GPU memory before ${WorkloadName}: $memoryLines"
    }

    $selectedGpu = $null
    foreach ($memoryLine in @($memoryLines)) {
        $fields = $memoryLine.ToString().Split(',')
        if ($fields.Count -ne 3) {
            continue
        }

        $index = 0
        $usedMB = 0
        $totalMB = 0
        if (-not [int]::TryParse($fields[0].Trim(), [ref]$index) -or
            -not [int]::TryParse($fields[1].Trim(), [ref]$usedMB) -or
            -not [int]::TryParse($fields[2].Trim(), [ref]$totalMB)) {
            continue
        }
        if ($index -eq $deviceOrdinal) {
            $selectedGpu = [pscustomobject]@{
                Index = $index
                UsedMB = $usedMB
                TotalMB = $totalMB
            }
            break
        }
    }

    if ($null -eq $selectedGpu) {
        throw "nvidia-smi did not report CUDA device $deviceOrdinal before ${WorkloadName}: $memoryLines"
    }
    if ($selectedGpu.UsedMB -gt $MaxInitialGpuMemoryUsedMB) {
        throw "GPU $deviceOrdinal is busy before ${WorkloadName}: $($selectedGpu.UsedMB) MiB / $($selectedGpu.TotalMB) MiB is already used; limit is ${MaxInitialGpuMemoryUsedMB} MiB. Close GPU-heavy apps before retrying."
    }

    Write-Host (
        "GPU {0} headroom preflight passed for {1}: {2} MiB / {3} MiB used" -f
            $deviceOrdinal,
            $WorkloadName,
            $selectedGpu.UsedMB,
            $selectedGpu.TotalMB
    )
}
