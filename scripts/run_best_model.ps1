$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

# pick whichever XL run had the lower val_loss_final
function GetVal($p) { (Get-Content $p | ConvertFrom-Json).val_loss_final }
$sp = "results/sp_xl/final.json"
$mup = "results/mup_xl/final.json"
if (-not ((Test-Path $sp) -or (Test-Path $mup))) {
    Write-Error "no XL final.json found, run the scaling studies first"
    exit 1
}
$best_param = "sp"; $best_val = [double]::PositiveInfinity
if (Test-Path $sp)  { $v = GetVal $sp;  if ($v -lt $best_val) { $best_val = $v; $best_param = "sp" } }
if (Test-Path $mup) { $v = GetVal $mup; if ($v -lt $best_val) { $best_val = $v; $best_param = "mup" } }
Write-Host ("==> best XL is {0} (val={1})" -f $best_param, $best_val)

$lr_file = "results/${best_param}_best_lr.json"
$lr = (Get-Content $lr_file | ConvertFrom-Json).best_lr

# warmstart from the scaling-run checkpoint, then train with a fresh cosine schedule
$prior_ckpt = "results/${best_param}_xl/model.pt"
$run_name = "best_xl_extra"
Write-Host ("==> warmstarting $run_name from $prior_ckpt for 100M tokens at lr={0}" -f $lr)
python -m train.train configs/xl.yaml `
    --parameterization $best_param `
    --lr $lr `
    --run_name $run_name `
    --save_checkpoint `
    --warmstart_from $prior_ckpt `
    --max_tokens 100000000

$ckpt = "results/$run_name/model.pt"
Write-Host "==> generating samples"
python -m eval.generate $ckpt --out_dir results/best_samples

Write-Host "==> validity metrics + figures + test perplexity"
python -m eval.validity --samples results/best_samples --ckpt $ckpt --test_bin data/test.bin
