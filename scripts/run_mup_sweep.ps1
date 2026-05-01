$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

Write-Host "==> muP learning rate sweep on Tiny"
python -m train.lr_sweep configs/mup_lr_sweep.yaml

$best = (Get-Content results/mup_best_lr.json | ConvertFrom-Json).best_lr
Write-Host ("best muP LR: {0}" -f $best)
