# Language Modelling Is Compression: Experiment Repository

This repository contains the code, selected outputs, execution records and
reproducibility materials for a dissertation that evaluates Llama 2 7B as a
lossless text-compression model on the enwik9 dataset.

The project separates two forms of evidence:

1. a model-based coding-length estimate on the complete 1,000,000,000-byte
   enwik9 dataset; and
2. a finite-precision range-coding round trip on 1,024,000 bytes that produced
   a decodable archive and verified byte-exact reconstruction.

The complete 1 GB Llama 2 result is an estimate derived from model
log-probabilities. It is not the size of a generated 1 GB archive.

## Verified results

| Experiment | Scope | Result | Verification |
|---|---:|---:|---|
| Llama 2 7B model evaluation | 1,000,000,000 bytes | 8.8419% model-only; 8.8967% paper-comparable estimate | 488,282 chunks; no skipped or truncated chunks |
| Llama 2 7B smoke test | 1,024,000 bytes | 8.8705% model-only estimate | 500 chunks completed |
| gzip level 9 | 1,000,000,000 raw bytes | 32.2592% actual archive rate | Decompressed SHA-256 matched input |
| bzip2 level 9 | 1,000,000,000 raw bytes | 25.3978% actual archive rate | Decompressed SHA-256 matched input |
| LZMA2/XZ preset 9 | 1,000,000,000 raw bytes | 21.3371% actual archive rate | Decompressed SHA-256 matched input |
| Llama 2 7B range-coding round trip | 1,024,000 bytes | 10.5355% actual archive rate | Byte-exact reconstruction verified |

For the complete Llama 2 evaluation, the 8.8967% value adds one bit for each
byte changed by ASCII mapping, following the comparison procedure used in the
reference study. It does not include a positional restoration stream. The
separate 1,024,000-byte experiment stored the information required for decoding,
including an escape-token stream, framing metadata and a compressed high-bit
mask.

## Repository contents

- `code/`: Python programs for model scoring, conventional baselines and the
  practical range-coding round trip.
- `slurm/`: SLURM submission scripts used on the University of Manchester CSF.
- `results/`: final JSON outputs for the full evaluation, smoke test,
  conventional baselines, 4 KB validation and 1 MB practical experiment.
- `logs/`: SLURM standard-output and standard-error records for the four
  successful model-scoring and range-coding jobs listed below.
- `environment/`: Python, PyTorch, Transformers, CUDA and Conda environment
  records.
- `checksums/`: SHA-256 records for enwik9 and the experiment scripts.
- `notebooks/`: notebook used to generate the experimental pipeline figure.
- `figures/`: PNG and PDF versions of the experimental pipeline.
- `Experiment_Summary.pdf`: concise summary of the completed experiments and
  retained evidence.

The filenames containing `arithmetic` were retained from the development
workflow. The practical implementation uses the `constriction` finite-precision
range coder; range coding follows the same interval-coding principle as
arithmetic coding.

## Software environment

- Python 3.10.20
- PyTorch 2.13.0+cu130
- Transformers 5.14.1
- CUDA 13.0
- constriction 0.4.2
- Model: `meta-llama/Llama-2-7b-hf`

The complete package list is recorded in `environment/conda_environment.txt`.

## Reproduction requirements

1. Obtain access to `meta-llama/Llama-2-7b-hf` through Hugging Face.
2. Download the enwik9 dataset and place it at `data/enwik9`.
3. Verify the dataset against `checksums/enwik9_sha256.txt`.
4. Create a Python environment using the versions recorded in `environment/`.
5. Place the Python files in `code/` and submit the required script from
   `slurm/`.

The SLURM files assume the project is located at
`${HOME}/scratch/llm-compression` and that the Conda environment is named
`llm-compression`. These values should be changed for another system. The main
commands are:

```bash
sbatch slurm/run_enwik9_llama2_smoke.sbatch
sbatch slurm/run_enwik9_llama2_gpu.sbatch
sbatch slurm/run_enwik9_classical_cpu.sbatch
sbatch slurm/run_llama2_arithmetic_smoke.sbatch
sbatch slurm/run_llama2_arithmetic_1mb.sbatch
```

The smoke tests should be completed before submitting the full jobs. The
practical range-coding decoder regenerates model probabilities sequentially, so
the 1,024,000-byte encode-decode run required approximately 4.12 hours.

## Logs and provenance

The successful jobs are:

- `17762596`: 1 MB model-scoring smoke test;
- `18075134`: complete 1 GB Llama 2 model evaluation;
- `18102015`: corrected 4 KB range-coding round trip; and
- `18102816`: final 1 MB range-coding round trip.

An `.err` file does not necessarily indicate failure: the successful jobs wrote
model-loading progress and non-fatal warnings to standard error.

The original classical-compression timing log was not retained. The reported
archive sizes were read from the retained archives, and each archive was
decompressed again for SHA-256 verification. This provenance is recorded in
`results/enwik9_classical_baselines_reconstructed.json`.

## Files not distributed

The following files are intentionally excluded because of size, licensing or
limited usefulness to the marker:

- the 1 GB enwik9 input file;
- Llama 2 model weights;
- the large gzip, bzip2 and XZ archives;
- intermediate checkpoints; and
- the custom `.lmac` archive files.

The retained JSON results, checksums, logs and scripts provide the evidence used
in the dissertation without redistributing the source dataset or model weights.
