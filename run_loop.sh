#!/usr/bin/env bash
set -euo pipefail

export OLLAMA_API_BASE="http://localhost:11434"

# Clean up old stop marker
rm -f .autoresearch_done

# Ensure experiments.tsv exists with a proper header
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

    # Step 1: LLM edits train.py (FEATURE_NAMES only)
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

    # Step 2: Run train.py and capture val_loglik
    VAL_LOG=$(python train.py 2>&1 | tail -n 1)
    echo "val_loglik = $VAL_LOG"

    COMMIT_HASH=$(git rev-parse --short HEAD)
    BEST_SCORE=$(cat "$BEST_SCORE_FILE")
    FEATURE_LIST=$(grep '^FEATURE_NAMES' train.py | sed "s/.*=\s*//")

    # Step 3: Ask LLM to decide keep/discard
    DECISION=$(aider --model ollama/deepseek-r1:32b \
        --no-gitignore \
        --yes-always \
        --message "The experiment with features $FEATURE_LIST achieved val_loglik = $VAL_LOG. The current best score is $BEST_SCORE. Based on the protocol in program.md, should this experiment be 'keep' or 'discard'? Respond with exactly one word: keep or discard." \
        --dry-run 2>&1 | grep -iE "^(keep|discard)$" | head -n 1 | tr '[:upper:]' '[:lower:]')

    # Fallback if LLM doesn't respond cleanly
    if [[ -z "$DECISION" ]]; then
        # Default to numeric comparison as fallback
        if (( $(echo "$VAL_LOG > $BEST_SCORE" | bc -l) )); then
            DECISION="keep"
        else
            DECISION="discard"
        fi
        echo "LLM decision unclear; using numeric fallback: $DECISION"
    else
        echo "LLM decision: $DECISION"
    fi

    # Step 4: Record result and handle git
    if [[ "$DECISION" == "keep" ]]; then
        STATUS="keep"
        echo "$VAL_LOG" > "$BEST_SCORE_FILE"
        echo "New best score: $VAL_LOG"
    else
        STATUS="discard"
        # Revert the commit (undo FEATURE_NAMES change)
        git reset --hard HEAD~1
        echo "Reverted train.py to previous best."
    fi

    # Append to experiments.tsv with a real tab
    printf "%s\t%s\t%s\t%s\n" "$COMMIT_HASH" "$VAL_LOG" "$STATUS" "$FEATURE_LIST" >> experiments.tsv

    echo "--- Turn complete ---"
    echo
done

echo "=== Autonomous research complete ==="
rm -f .autoresearch_done