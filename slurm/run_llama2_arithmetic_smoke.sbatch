#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH --gpus=1
#SBATCH -n 12
#SBATCH -t 0-4
#SBATCH -J llama2_ac_smoke
#SBATCH -o logs/llama2_ac_smoke_%j.out
#SBATCH -e logs/llama2_ac_smoke_%j.err

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1
source activate llm-compression
export OMP_NUM_THREADS="${SLURM_NTASKS}"
export HF_HOME="${HOME}/scratch/huggingface"

cd "${HOME}/scratch/llm-compression"
mkdir -p logs results/arithmetic

python -u code/run_llama2_arithmetic_roundtrip.py \
  --input data/enwik9 \
  --archive results/arithmetic/llama2_enwik9_4kb.lmac \
  --result results/arithmetic/llama2_enwik9_4kb.json \
  --limit-bytes 4096 \
  --chunk-size 2048 \
  --top-k 100 \
  --progress-every-tokens 100
