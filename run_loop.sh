#!/usr/bin/env bash
set -euo pipefail

export OLLAMA_API_BASE="http://localhost:11434"

rm -f .autoresearch_done

# --- Fix header and newline for experiments.tsv ---
if [[ ! -f experiments.tsv ]]; then
    printf "commit_hash\tval_loglik\tstatus\tdescription\n" > experiments.tsv
else
    if [[ -s experiments.tsv ]]; then
        tail -c1 experiments.tsv | grep -q $'\n' || echo "" >> experiments.tsv
    fi
fi

# --- Track best score ---
BEST_SCORE_FILE=".best_score.txt"
if [[ ! -f "$BEST_SCORE_FILE" ]]; then
    echo "-999999.0" > "$BEST_SCORE_FILE"
fi

echo "=== Starting autonomous research loop ==="

while [[ ! -f .autoresearch_done ]]; do
    echo "--- Asking LLM to propose next feature set ---"

    # Extract current FEATURE_NAMES line exactly as it appears in train.py
    CURRENT_LINE=$(grep '^FEATURE_NAMES' train.py | head -n1)

    # Build a summary of already tried features from experiments.tsv (last 10 lines for brevity)
    TRIED_SUMMARY=$(tail -n 10 experiments.tsv 2>/dev/null | awk -F'\t' 'NR>1 {print $4}' | sort -u | paste -sd ', ')

    # Construct the prompt with exact line and context
    PROMPT="Edit FEATURE_NAMES in train.py to try a new feature combination. 
Current line: '$CURRENT_LINE'
Features already tried (from experiments.tsv): ${TRIED_SUMMARY:-none}
Add exactly ONE new feature (e.g., one raster) to the list. Do NOT add features that have already been tried unless all have been tried. Do not edit any other file. Do not run commands."

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
        --message "$PROMPT"

    echo "--- Running experiment... ---"

    VAL_LOG=$(python train.py 2>&1 | tail -n 1)
    echo "val_loglik = $VAL_LOG"

    COMMIT_HASH=$(git rev-parse --short HEAD)
    BEST_SCORE=$(cat "$BEST_SCORE_FILE")
    FEATURE_LIST=$(grep '^FEATURE_NAMES' train.py | sed -E 's/.*=\s*//; s/\s+$//; s/^[[:space:]]+//')

    # Decision
    if (( $(echo "$VAL_LOG > $BEST_SCORE" | bc -l) )); then
        STATUS="keep"
        echo "$VAL_LOG" > "$BEST_SCORE_FILE"
        echo "✅ New best score: $VAL_LOG"
    else
        STATUS="discard"
        git reset --hard HEAD~1
        echo "❌ No improvement. Reverted train.py."
    fi

    printf "%s\t%s\t%s\t%s\n" "$COMMIT_HASH" "$VAL_LOG" "$STATUS" "$FEATURE_LIST" >> experiments.tsv

    echo "--- Turn complete ---"
    echo
done

echo "=== Autonomous research complete ==="
rm -f .autoresearch_done