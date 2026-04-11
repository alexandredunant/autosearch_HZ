#!/bin/bash
# Autonomous research loop runner.
# This script only orchestrates the loop. Bookkeeping lives in loop_state.py.

MAX_PLATEAU=20
TRAIN_LOG="run.log"
MODE="${1:-}"

python loop_state.py ensure-schema
python loop_state.py sync-current
BEST_LOGLIK=$(python loop_state.py best)
PLATEAU_COUNT=$(python loop_state.py plateau)

if [ "$MODE" = "--sync-only" ]; then
    echo "Sync complete."
    if [ -z "$BEST_LOGLIK" ]; then
        echo "Best val_loglik: none yet"
    else
        echo "Best val_loglik: $BEST_LOGLIK"
    fi
    echo "Plateau count: $PLATEAU_COUNT"
    exit 0
fi

echo "Starting autonomous research loop..."
if [ -z "$BEST_LOGLIK" ]; then
    echo "Initial BEST_LOGLIK: none yet (next successful run becomes baseline)"
else
    echo "Initial BEST_LOGLIK: $BEST_LOGLIK"
fi

while [ $PLATEAU_COUNT -lt $MAX_PLATEAU ]; do
    echo "--------------------------------------------------"
    if [ -z "$BEST_LOGLIK" ]; then
        echo "Plateau: $PLATEAU_COUNT/$MAX_PLATEAU | Best: none yet"
    else
        echo "Plateau: $PLATEAU_COUNT/$MAX_PLATEAU | Best: $BEST_LOGLIK"
    fi
    
    # Save current features to compare after Aider runs
    OLD_FEATURES=$(python loop_state.py current-features)
    
    # Call Aider for the NEXT experiment
    export OLLAMA_API_BASE=http://localhost:11434
    aider --model ollama/deepseek-r1:32b \
          --yes-always \
          --auto-commits \
          --read program.md \
          --read prepare.py \
          --read results.tsv \
          --read loop_state.py \
          --read latest_model_summary.txt \
          train.py \
          --message "According to program.md, perform exactly ONE change (add or remove) to STATIC_FEATURE_NAMES in train.py. 
          CRITICAL: Do not repeat any feature combination already listed in results.tsv. 
          Current features are: $OLD_FEATURES. 
          Check the 'features' column in results.tsv to see what has been tried. 
          Do not include shell commands in your answer, only the code change."

    # Get NEW features for logging and duplicate checking
    NEW_FEATURES=$(python loop_state.py current-features)
    EXPERIMENT_COMMIT_HASH=$(git rev-parse --short HEAD)
    EXPERIMENT_DESCRIPTION=$(git log -1 --pretty=%B | tr '\n' ' ' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    # Check if this combination has already been tried
    if python loop_state.py has-features --features "$NEW_FEATURES"; then
        echo "WARNING: Aider suggested a combination already tried ($NEW_FEATURES). Skipping run."
        git reset --hard HEAD~1
        PLATEAU_COUNT=$((PLATEAU_COUNT + 1))
        continue
    fi
    
    # Run the experiment
    conda run -n hazard_agent python train.py > "$TRAIN_LOG" 2>&1
    
    # Extract val_loglik
    VAL_LOGLIK=$(grep "^val_loglik:" "$TRAIN_LOG" | awk '{print $2}')
    
    # Handle crash
    if [ -z "$VAL_LOGLIK" ]; then
        echo "Experiment CRASHED!"
        tail -n 20 "$TRAIN_LOG"
        VAL_LOGLIK="-1e18"
        STATUS="crash"
        git reset --hard HEAD~1
        PLATEAU_COUNT=$((PLATEAU_COUNT + 1))
    else
        echo "Result: $VAL_LOGLIK"
        # Compare with best
        if [ -z "$BEST_LOGLIK" ]; then
            echo "IMPROVEMENT found!"
            BEST_LOGLIK="$VAL_LOGLIK"
            PLATEAU_COUNT=0
            STATUS="keep"
        else
            IS_BETTER=$(echo "$VAL_LOGLIK > $BEST_LOGLIK" | bc -l)
            if [ "$IS_BETTER" -eq 1 ]; then
                echo "IMPROVEMENT found!"
                BEST_LOGLIK="$VAL_LOGLIK"
                PLATEAU_COUNT=0
                STATUS="keep"
            else
                echo "No improvement."
                STATUS="discard"
                # results.tsv is now untracked, so we can safely reset to clean HEAD
                git reset --hard HEAD~1
                PLATEAU_COUNT=$((PLATEAU_COUNT + 1))
            fi
        fi
    fi
    
    # Log to results.tsv
    python loop_state.py log \
        --commit "$EXPERIMENT_COMMIT_HASH" \
        --val-loglik "$VAL_LOGLIK" \
        --status "$STATUS" \
        --features "$NEW_FEATURES" \
        --description "$EXPERIMENT_DESCRIPTION"
    
done

echo "=== PLATEAU REACHED ($MAX_PLATEAU consecutive non-improving attempts) ==="
echo "Best val_loglik: $BEST_LOGLIK"
echo "Final branch: $(git branch --show-current)"
echo "Final commit: $(git rev-parse HEAD)"
