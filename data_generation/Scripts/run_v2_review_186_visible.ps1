$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Collection = Join-Path $ProjectRoot "Artifacts\V2Review186SingleWorker-$RunStamp"
$Python = (Get-Command python).Source
$Unreal = if ($env:HE_GRENADE_UNREAL_CMD) {
    $env:HE_GRENADE_UNREAL_CMD
} else {
    "C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe"
}
$Ffmpeg = (& $Python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())").Trim()
if ($LASTEXITCODE -ne 0 -or -not $Ffmpeg) {
    throw "Could not resolve ffmpeg through imageio_ffmpeg."
}

Set-Location $ProjectRoot
$Host.UI.RawUI.WindowTitle = "V2 186-Video Review Generation"

try {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "V2 186-VIDEO REVIEW GENERATION" -ForegroundColor Cyan
    Write-Host "Output: $Collection"
    Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "============================================================" -ForegroundColor Cyan

    if (Test-Path -LiteralPath $Collection) {
        throw "Output directory already exists: $Collection"
    }

    Write-Host "[1/5] Creating immutable 186-recipe review plan..." -ForegroundColor Yellow
    & $Python "Scripts/v2_dataset_controller.py" review-plan $Collection `
        --observation-rate 20 --workers 1
    if ($LASTEXITCODE -ne 0) { throw "Review-plan creation failed." }

    Write-Host "[2/5] Batch-certifying all 186 recipes without recording..." -ForegroundColor Yellow
    & $Python "Scripts/certify_v2_plan.py" $Collection `
        --executable $Unreal --output (Join-Path $Collection "certification")
    if ($LASTEXITCODE -ne 0) { throw "Full-plan construction certification failed. No recording was started." }

    Write-Host "[3/5] Initializing execution binding and first recipe..." -ForegroundColor Yellow
    & $Python "Scripts/dataset_worker.py" $Collection `
        --executable $Unreal --worker-id 0 --executor-id "visible-worker-0-init" --one
    if ($LASTEXITCODE -ne 0) { throw "Initial recipe failed." }

    Write-Host "[4/5] Running one Unreal worker..." -ForegroundColor Yellow
    $Jobs = @()
    foreach ($WorkerId in 0..0) {
        $ExecutorId = "visible-worker-$WorkerId"
        $Jobs += Start-Job -Name "v2-review-worker-$WorkerId" `
            -ArgumentList $ProjectRoot, $Python, $Collection, $Unreal, $WorkerId, $ExecutorId `
            -ScriptBlock {
            param($ProjectRoot, $Python, $Collection, $Unreal, $WorkerId, $ExecutorId)
            Set-Location $ProjectRoot
            & $Python "Scripts/dataset_worker.py" $Collection `
                --executable $Unreal --worker-id $WorkerId `
                --executor-id $ExecutorId
            if ($LASTEXITCODE -ne 0) {
                throw "Worker $WorkerId exited with code $LASTEXITCODE"
            }
        }
    }

    while (($Jobs | Where-Object State -eq "Running").Count -gt 0) {
        $Validated = 0
        $Failed = 0
        Get-ChildItem -LiteralPath (Join-Path $Collection "results") -Filter "*.json" `
            -ErrorAction SilentlyContinue | ForEach-Object {
                try {
                    $Result = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
                    if ($Result.technical_result -eq "validated") { $Validated++ }
                    elseif ($Result.technical_result) { $Failed++ }
                } catch {}
            }
        $Running = ($Jobs | Where-Object State -eq "Running").Count
        Write-Host ("[{0}] Validated {1}/186 | Failed attempts {2} | Workers running {3}" -f `
            (Get-Date -Format "HH:mm:ss"), $Validated, $Failed, $Running)
        Start-Sleep -Seconds 10
    }

    $Jobs | Receive-Job
    $FailedJobCount = @($Jobs | Where-Object State -eq "Failed").Count
    $Jobs | Remove-Job -Force
    if ($FailedJobCount -gt 0) {
        throw "$FailedJobCount generation worker(s) failed."
    }

    Write-Host "[5/5] Validating all assignments and rendering 186 videos..." -ForegroundColor Yellow
    & $Python "Scripts/build_v2_review_set.py" $Collection `
        --output (Join-Path $Collection "videos") --ffmpeg $Ffmpeg
    if ($LASTEXITCODE -ne 0) { throw "Final validation or video rendering failed." }

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "FINISHED: ALL 186 REVIEW VIDEOS GENERATED" -ForegroundColor Green
    Write-Host "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
    Write-Host "Videos: $(Join-Path $Collection 'videos')" -ForegroundColor Green
    Write-Host "The review set is ready for inspection." -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Inspect the terminal output and collection results." -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to close this terminal"
