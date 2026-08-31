#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH --gpus=1
#SBATCH -n 12
#SBATCH -t 2-0
#SBATCH -J llama2_ac_1mb
#SBATCH -o logs/llama2_ac_1mb_%j.out
#SBATCH -e logs/llama2_ac_1mb_%j.err

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
  --archive results/arithmetic/llama2_enwik9_1mb.lmac \
  --result results/arithmetic/llama2_enwik9_1mb.json \
  --limit-bytes 1024000 \
  --chunk-size 2048 \
  --top-k 100 \
  --progress-every-tokens 1000
