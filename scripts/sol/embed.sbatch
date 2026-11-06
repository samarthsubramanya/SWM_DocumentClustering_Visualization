#!/bin/bash
# Chunked SBERT/BGE document embeddings on a Sol GPU; no Postgres needed (ingest reads the archive).
# Copy data/cache/<model>.npy, .ids.npy and .meta.json back to the laptop's data/cache/.
#SBATCH --job-name=swm-embed
#SBATCH --account=class_cse57382398fall2026
#SBATCH --partition=general
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load mamba/latest
source activate swm
export SWM_CHUNK=1
python -c "import torch; print('cuda', torch.cuda.is_available())"
python - <<'PY'
from swm import ingest, represent
docs = ingest.load_20ng()
for m in represent.SBERT_MODELS:
    print(m, represent.embed(docs, m, force=True).shape)
PY
