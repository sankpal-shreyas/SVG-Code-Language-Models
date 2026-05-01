$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

Write-Host "==> SP learning rate sweep on Tiny"
python -m train.lr_sweep configs/sp_lr_sweep.yaml

$best = (Get-Content results/sp_best_lr.json | ConvertFrom-Json).best_lr
Write-Host ("best SP LR: {0}" -f $best)
