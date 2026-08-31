#!/bin/bash --login
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -t 0-12
#SBATCH -J enwik9_classical
#SBATCH -o logs/enwik9_classical_%j.out
#SBATCH -e logs/enwik9_classical_%j.err

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1
source activate llm-compression

cd "${HOME}/scratch/llm-compression"
mkdir -p logs results/classical_archives

python -u code/run_classical_baselines.py \
  --input data/enwik9 \
  --output-dir results/classical_archives \
  --result results/enwik9_classical_baselines.json
