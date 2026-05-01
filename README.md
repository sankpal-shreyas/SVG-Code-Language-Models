# SVG-Code-Language-Models

CS-GY 6923 (NYU, Spring 2026) optional project. Decoder-only transformer language
models trained on SVG code at five widths, with scaling-law fitting and a comparison
of standard parameterization (SP) vs. muP learning-rate transfer.

## Layout

```
configs/   model and LR-sweep YAMLs
data/      download, clean, tokenize, pack
model/     transformer + muP wiring
train/     training loop and LR sweep
eval/      power-law fitting, generation, validity metrics
scripts/   PowerShell launchers
report/    LaTeX report
results/   logs and summaries (generated)
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

RTX 50-series needs the CUDA-compiled PyTorch:

```powershell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Running

Run from the project root:

```powershell
.\scripts\run_data.ps1          # download, clean, tokenize
.\scripts\run_sp_sweep.ps1      # SP learning-rate sweep on Tiny
.\scripts\run_sp_scaling.ps1    # SP scaling runs (5 sizes)
.\scripts\run_mup_sweep.ps1     # muP learning-rate sweep on Tiny
.\scripts\run_mup_scaling.ps1   # muP scaling runs (5 sizes)
.\scripts\run_best_model.ps1    # extra training + sample generation + metrics
```

## Notes

`model/gpt.py` adapts `karpathy/nanoGPT`. The training loop, LR sweep, and
evaluation pipeline are written for this project.
