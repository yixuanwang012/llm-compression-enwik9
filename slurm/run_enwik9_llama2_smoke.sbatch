#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH --gpus=1
#SBATCH -n 12
#SBATCH -t 0-2
#SBATCH -J enwik9_llama2_smoke
#SBATCH -o logs/enwik9_llama2_smoke_%j.out
#SBATCH -e logs/enwik9_llama2_smoke_%j.err

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1
source activate llm-compression

export OMP_NUM_THREADS="${SLURM_NTASKS}"
export HF_HOME="${HOME}/scratch/huggingface"

cd "${HOME}/scratch/llm-compression"
mkdir -p logs results "${HF_HOME}"

python code/run_enwik9_full.py \
  --input data/enwik9 \
  --output results/llama2_7b_enwik9_smoke.json \
  --checkpoint results/llama2_7b_enwik9_smoke_checkpoint.json \
  --model meta-llama/Llama-2-7b-hf \
  --chunk-size 2048 \
  --max-model-tokens 4096 \
  --top-k 100 \
  --limit-bytes 1024000 \
  --fail-on-truncation \
  --device cuda \
  --dtype float16 \
  --checkpoint-every-chunks 50 \
  --progress-every-chunks 10
