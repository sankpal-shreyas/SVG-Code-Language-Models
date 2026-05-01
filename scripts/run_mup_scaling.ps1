$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

if (-not (Test-Path "results/mup_best_lr.json")) {
    Write-Error "results/mup_best_lr.json missing. Run scripts/run_mup_sweep.ps1 first."
    exit 1
}
$best = (Get-Content results/mup_best_lr.json | ConvertFrom-Json).best_lr
Write-Host ("==> muP scaling runs at lr={0}" -f $best)

$sizes = @("tiny", "small", "medium", "large", "xl")
foreach ($s in $sizes) {
    $name = "mup_$s"
    Write-Host ("--- $name ---")
    python -m train.train ("configs/{0}.yaml" -f $s) `
        --parameterization mup `
        --lr $best `
        --run_name $name `
        --save_checkpoint
}

Write-Host "==> fit SP vs muP comparison"
$sp_dirs  = $sizes | ForEach-Object { "results/sp_$_" }
$mup_dirs = $sizes | ForEach-Object { "results/mup_$_" }
python -m eval.fit_scaling --sp_runs @sp_dirs --mup_runs @mup_dirs `
    --out_plot report/figures/sp_vs_mup_scaling.png `
    --out_summary results/scaling_summary.json `
    --curves_plot report/figures/training_curves.png
