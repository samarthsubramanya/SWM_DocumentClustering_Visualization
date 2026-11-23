#!/bin/bash
# KG extraction on ASU Sol: serves llama3.1:8b with Ollama on one GPU and fills
# data/cache/kg/ from an exported doc list. No Postgres/Neo4j needed on Sol.
#
#   local:  python scripts/05_kg.py --all-docs --export-docs data/kg_docs.jsonl
#           rsync -a data/kg_docs.jsonl <sol>:<repo>/data/
#   Sol:    mkdir -p logs && sbatch scripts/sol/extract_kg.sbatch
#   local:  rsync -a <sol>:<repo>/data/cache/kg/ data/cache/kg/
#           python scripts/05_kg.py --all-docs          # all cache hits -> Neo4j + labels
#
# Resumable: if the job hits its time limit, submit it again; cached docs are skipped.
#
#SBATCH --job-name=swm-kg
#SBATCH --partition=public          # TODO(sol): your partition
#SBATCH --qos=public                # TODO(sol): your QOS
#SBATCH --gres=gpu:a100:1           # TODO(sol): GPU type available to you
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

WORKERS=${WORKERS:-16}
DOCS=${DOCS:-data/kg_docs.jsonl}
MODEL=${OLLAMA_MODEL:-llama3.1:8b}

# --- Python env -----------------------------------------------------------------
# TODO(sol): activate an env with requirements.txt installed, e.g.
#   module load mamba && source activate swm      or      source .venv/bin/activate

# --- Ollama server on this node ----------------------------------------------------
# TODO(sol): make `ollama` available: `module load ollama` if Sol provides it, otherwise
# install the Linux binary under $HOME (https://ollama.com/download/linux).
export OLLAMA_HOST=127.0.0.1:${OLLAMA_PORT:-11434}
export OLLAMA_MODELS=${OLLAMA_MODELS:-/scratch/$USER/ollama-models}  # 5 GB model off $HOME
export OLLAMA_NUM_PARALLEL=$WORKERS      # one GPU slot per concurrent request
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0         # ~half the KV-cache memory per slot
export OLLAMA_KEEP_ALIVE=-1
ollama serve > "logs/ollama-${SLURM_JOB_ID}.log" 2>&1 &
trap 'kill %1 2>/dev/null || true' EXIT
until curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null; do sleep 2; done
ollama pull "$MODEL"

# --- Extraction -----------------------------------------------------------------------
export OLLAMA_BASE_URL="http://$OLLAMA_HOST"
export OLLAMA_MODEL=$MODEL
export OLLAMA_NUM_CTX=4096               # ~1.2k prompt + 1k output; smaller KV per slot
python scripts/05_kg.py --docs-file "$DOCS" --workers "$WORKERS"
