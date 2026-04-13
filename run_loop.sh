#!/usr/bin/env bash
set -euo pipefail

export OLLAMA_API_BASE="http://localhost:11434"
rm -f .autoresearch_done

echo "=== Starting autonomous research loop ==="
touch experiments.tsv

while [ ! -f .autoresearch_done ]; do
  echo "--- Launching Aider for one experiment turn ---"
  aider --model ollama/deepseek-r1:32b \
    --no-gitignore \
    --yes-always \
    --auto-commits \
    --aiderignore .aiderignore \
    --read program.md \
    --read prepare.py \
    train.py \
    experiments.tsv \
    --message "Perform exactly ONE autonomous experiment turn by following the protocol in program.md. Do not deviate."
  echo "--- Aider turn finished ---"
  echo
done

echo "=== Autonomous research complete ==="
rm -f .autoresearch_done