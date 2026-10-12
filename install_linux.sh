#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v conda >/dev/null || { echo 'Install Miniconda or Miniforge first.' >&2; exit 1; }
eval "$(conda shell.bash hook)"
# Create an independent environment; this command refuses to replace an existing one.
conda env create -f environment.yml
conda activate escc_he_ici
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install 'git+https://github.com/mahmoodlab/TRIDENT.git@091fe0c7e1759ae3bfa35ad9f320ee57aeb966a1' -c requirements.txt
python -m pip check
python -m ipykernel install --user --name escc_he_ici --display-name 'ESCC HE ICI'
printf '\nActivate with: conda activate escc_he_ici\nLaunch with: jupyter lab notebooks/00_run_selected.ipynb\n'
