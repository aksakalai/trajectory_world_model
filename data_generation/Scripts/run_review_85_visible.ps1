param([string]$ResumeRoot = "")
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Root = if ($ResumeRoot) { (Resolve-Path $ResumeRoot).Path } else { Join-Path $ProjectRoot "Artifacts\RepresentativeReview85-$Stamp" }
$Python = (Get-Command python).Source
$Unreal = "C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe"
$Project = Join-Path $ProjectRoot "he_grenade_game.uproject"
$Host.UI.RawUI.WindowTitle = "85-Video V1 + V2 Representative Review"
Set-Location $ProjectRoot

function Run-Step([string]$Label, [scriptblock]$Action) {
    Write-Host $Label -ForegroundColor Yellow
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

function Test-CompletedDataset([string]$Path) {
    $Manifest = Join-Path $Path "dataset.json"
    if (-not (Test-Path $Manifest)) { return $false }
    try { return [bool]((Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json).complete) }
    catch { return $false }
}

try {
    if (-not (Test-Path $Root)) { New-Item -ItemType Directory -Path $Root | Out-Null }
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "85-VIDEO REPRESENTATIVE REVIEW (19 V1 + 66 V2)" -ForegroundColor Cyan
    Write-Host "Output: $Root"
    Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "============================================================" -ForegroundColor Cyan

    $V1GuidedOutput = Join-Path $Root "v1-guided"
    if (-not (Test-Path (Join-Path $V1GuidedOutput "dataset.json"))) {
        Run-Step "[1/9] Recording deterministic 60-case V1 guided inspection plan..." {
            & $Unreal $Project -game -GenerateDataset -Stage=movement -MissionReviewSuite `
                -Episodes=60 -ObservationRate=20 -Width=384 -Height=384 -StorageFormat=webp_parquet `
                -SeedStart=310000 "-Output=$V1GuidedOutput" -RenderOffscreen -unattended -nosound -NoSplash -NoVSync
        }
    } else { Write-Host "[1/9] Reusing completed V1 guided capture." -ForegroundColor Yellow }
    Run-Step "[2/9] Finalizing and verifying V1 guided capture..." {
        if (-not (Test-CompletedDataset $V1GuidedOutput)) {
            & $Python Scripts/finalize_production_dataset.py $V1GuidedOutput
            if ($LASTEXITCODE -ne 0) { return }
        }
        & $Python Scripts/review_dataset.py $V1GuidedOutput --validate-only
    }
    Run-Step "[3/9] Recording four seeded V1 semi-Markov examples..." {
        $V1SemiOutput = Join-Path $Root "v1-semi-markov"
        if (-not (Test-Path (Join-Path $V1SemiOutput "dataset.json"))) {
            & $Unreal $Project -game -GenerateDataset -Stage=movement -Mission=semi_markov `
                -Episodes=4 -EpisodeSeconds=150 -ObservationRate=20 -Width=384 -Height=384 -StorageFormat=webp_parquet `
                -SeedStart=320000 "-Output=$V1SemiOutput" -RenderOffscreen -unattended -nosound -NoSplash -NoVSync
        } else { Write-Host "Reusing existing V1 semi-Markov capture." -ForegroundColor Yellow }
    }
    Run-Step "[4/9] Finalizing and verifying V1 semi-Markov capture..." {
        $V1SemiOutput = Join-Path $Root "v1-semi-markov"
        if (-not (Test-CompletedDataset $V1SemiOutput)) {
            & $Python Scripts/finalize_production_dataset.py $V1SemiOutput
            if ($LASTEXITCODE -ne 0) { return }
        }
        & $Python Scripts/review_dataset.py $V1SemiOutput --validate-only
    }
    $V2 = Join-Path $Root "v2"
    Run-Step "[5/9] Creating and structurally verifying immutable V2 review source plan..." {
        & $Python Scripts/v2_dataset_controller.py plan $V2 --frame-budget 33280 --workers 1 `
            --episode-seconds 150 --observation-rate 20 --width 384 --height 384 --seed-start 330000
        if ($LASTEXITCODE -eq 0) { & $Python Scripts/v2_dataset_controller.py verify-plan $V2 }
    }
    Run-Step "[6/9] Batch-certifying the exact V2 plan without recording..." {
        & $Python Scripts/certify_v2_plan.py $V2 --executable $Unreal --output (Join-Path $V2 "certification")
    }
    Run-Step "[7/9] Recording immutable V2 assignments with one worker..." {
        & $Python Scripts/dataset_worker.py $V2 --executable $Unreal --worker-id 0 --executor-id review-85-visible
    }
    Run-Step "[8/9] Checking final V2 inventory..." {
        & $Python Scripts/v2_dataset_controller.py inventory $V2 --write-snapshot
    }
    Run-Step "[9/9] Rendering the exact 85 selected videos..." {
        & $Python Scripts/build_review_85.py $Root
    }
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "FINISHED: ALL 85 VIDEOS GENERATED" -ForegroundColor Green
    Write-Host "Videos: $(Join-Path $Root 'videos')" -ForegroundColor Green
    Write-Host "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "The immutable plans and completed evidence remain under: $Root" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
}
Read-Host "Press Enter to close this terminal"
