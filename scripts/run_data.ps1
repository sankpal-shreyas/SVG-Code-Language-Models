$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

Write-Host "==> ingest + tokenize + pack"
python -m data.prepare --stage all --target_train_chars 4.0e8 --vocab_size 4096 --max_seq_tokens 2048

Write-Host "==> stats"
python -m data.stats

Write-Host "data ready, see data/*.bin and report/figures/seq_len_hist.png"
