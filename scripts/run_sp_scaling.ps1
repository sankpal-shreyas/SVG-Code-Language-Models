$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

if (-not (Test-Path "results/sp_best_lr.json")) {
    Write-Error "results/sp_best_lr.json missing. Run scripts/run_sp_sweep.ps1 first."
    exit 1
}
$best = (Get-Content results/sp_best_lr.json | ConvertFrom-Json).best_lr
Write-Host ("==> SP scaling runs at lr={0}" -f $best)

$sizes = @("tiny", "small", "medium", "large", "xl")
foreach ($s in $sizes) {
    $name = "sp_$s"
    Write-Host ("--- $name ---")
    python -m train.train ("configs/{0}.yaml" -f $s) `
        --parameterization sp `
        --lr $best `
        --run_name $name `
        --save_checkpoint
}

Write-Host "==> fit scaling power law"
$run_dirs = $sizes | ForEach-Object { "results/sp_$_" }
python -m eval.fit_scaling --sp_runs @run_dirs --out_plot report/figures/sp_scaling.png --out_summary results/sp_scaling.json --curves_plot report/figures/sp_training_curves.png
