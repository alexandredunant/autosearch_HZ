#!/usr/bin/env bash
# run_loop.sh – Minimal autonomous research loop.
# Bash only acts as a repeater; all decisions are made by the LLM via program.md.

set -euo pipefail

# --- Ollama configuration ---
export OLLAMA_API_BASE="http://localhost:11434"

# Remove any leftover stop marker from a previous run
rm -f .autoresearch_done

echo "=== Starting autonomous research loop ==="
echo "Press Ctrl+C to stop manually (the LLM will otherwise stop when plateau is reached)."
echo

while [ ! -f .autoresearch_done ]; do
  echo "--- Launching Aider for one experiment turn ---"
  aider --model ollama/deepseek-r1:32b \
    --yes-always \
    --auto-commits \
    --aiderignore .aiderignore \
    --read program.md \
    --read prepare.py \
    --read experiment.tsv \
    train.py \
    experiments.tsv \
    --message "Perform exactly ONE autonomous experiment turn by following the protocol in program.md. Do not deviate."
  echo "--- Aider turn finished ---"
  echo
done

echo "=== Autonomous research complete (stopping condition met) ==="
rm -f .autoresearch_done