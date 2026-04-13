#!/usr/bin/env bash
set -euo pipefail

export OLLAMA_API_BASE="http://localhost:11434"

rm -f .autoresearch_done

# Ensure experiments.tsv exists with proper header
if [[ ! -f experiments.tsv ]]; then
    printf "commit_hash\tval_loglik\tstatus\tdescription\n" > experiments.tsv
fi

# Track best score
BEST_SCORE_FILE=".best_score.txt"
if [[ ! -f "$BEST_SCORE_FILE" ]]; then
    echo "-999999.0" > "$BEST_SCORE_FILE"
fi

echo "=== Starting autonomous research loop ==="

while [[ ! -f .autoresearch_done ]]; do
    echo "--- Asking LLM to propose next feature set ---"

    # LLM edits train.py (FEATURE_NAMES only)
    aider --model ollama/deepseek-r1:32b \
        --no-gitignore \
        --yes-always \
        --auto-commits \
        --aiderignore .aiderignore \
        --read program.md \
        --read prepare.py \
        --read experiments.tsv \
        --read "$BEST_SCORE_FILE" \
        train.py \
        --message "Edit FEATURE_NAMES in train.py to try a new feature combination (add one feature to the current best set, or start with one if none). Do not edit any other file. Do not run any commands."

    echo "--- Running experiment... ---"

    # Run training and capture score
    VAL_LOG=$(python train.py 2>&1 | tail -n 1)
    echo "val_loglik = $VAL_LOG"

    COMMIT_HASH=$(git rev-parse --short HEAD)
    BEST_SCORE=$(cat "$BEST_SCORE_FILE")
    FEATURE_LIST=$(grep '^FEATURE_NAMES' train.py | sed "s/.*=\s*//")

    # Deterministic decision per protocol
    if (( $(echo "$VAL_LOG > $BEST_SCORE" | bc -l) )); then
        STATUS="keep"
        echo "$VAL_LOG" > "$BEST_SCORE_FILE"
        echo "✅ New best score: $VAL_LOG"
    else
        STATUS="discard"
        git reset --hard HEAD~1
        echo "❌ No improvement. Reverted train.py."
    fi

    # Append to experiments.tsv
    printf "%s\t%s\t%s\t%s\n" "$COMMIT_HASH" "$VAL_LOG" "$STATUS" "$FEATURE_LIST" >> experiments.tsv

    # Optional stopping condition: 20 consecutive discards
    # (You can implement later if needed)

    echo "--- Turn complete ---"
    echo
done

echo "=== Autonomous research complete ==="
rm -f .autoresearch_done