
# Autonomous Feature Selection Protocol

You are an autonomous research agent performing feature selection for a Bayesian logistic regression model.

## Environment & Constraints

- You are running **inside an Aider session**.
- You have read‑only access to `program.md`, `prepare.py`, and `experiments.tsv`.
- You may **only edit** the list `STATIC_FEATURE_NAMES` inside `train.py`.
- Do **not** modify any other part of `train.py` or any other file.
- Do **not** commit `experiments.tsv` to Git.

## File Roles

| File | Purpose | Editable? |
|------|---------|-----------|
| `prepare.py` | Data generation (run once) and immutable data loading. | **NO** |
| `train.py` | Model definition and training loop. Contains `STATIC_FEATURE_NAMES`. Prints **only** the validation log‑likelihood to stdout. | **YES** – Only the `STATIC_FEATURE_NAMES` list. |
| `program.md` | This protocol document. | **NO** |
| `experiments.tsv` | Full experiment history. **You must append one line after each experiment.** Read this to know what's been tried and the best score so far. | Append only; never commit. |

## Experiment Loop (one turn per Aider invocation)

**Important:** You are called once per experiment. You will be called again by the Bash loop. Do **not** try to run multiple experiments in one turn.

For each turn, perform exactly these steps in order:

### 1. Read Current State
- Read `experiments.tsv` completely.
- Identify which feature combinations have already been tried and which are **locked** (marked `status=keep`).
- Determine the **best validation log‑likelihood so far** from rows with `status=keep`. If none exists, the best is `-inf`.

### 2. Decide Next Combination
- Choose **one** feature to add or remove from the current `STATIC_FEATURE_NAMES` list.
- **Never** repeat a combination that already appears in `experiments.tsv`.
- **Never** remove a feature that is part of a locked (successful) combination unless you have a very strong reason (and you must explain it in the commit message).
- Aim for a balance between exploration (trying new features) and exploitation (refining near the current best).

### 3. Edit `train.py`
- Modify **only** the `STATIC_FEATURE_NAMES` list. Example:
  ```python
  STATIC_FEATURE_NAMES = ['feature_a', 'feature_c', 'feature_e']
  ```
- Do **not** change anything else.

### 4. Commit the Change
- Aider will auto‑commit with your commit message.
- Write a short, informative commit message describing what feature was added/removed and why.
- Record the **commit hash** for later use (you can get it with `git rev-parse --short HEAD`).

### 5. Run the Experiment
- Execute: `python train.py`
- Capture the **stdout** output. It will be a single floating‑point number (the validation log‑likelihood).
- If the script fails (non‑zero exit code or no numeric output), treat as **crash**.

### 6. Determine Status
- Compare the new `val_loglik` to the best so far (from step 1).
- **Status**:
  - `keep` if `val_loglik > best_so_far`
  - `discard` if `val_loglik <= best_so_far`
  - `crash` if the run failed

### 7. Record in `experiments.tsv`
- Append **exactly one TAB‑separated line** to `experiments.tsv` in this format:
  ```
  commit_hash<TAB>val_loglik<TAB>status<TAB>description
  ```
  - `commit_hash`: from step 4 (use `git rev-parse --short HEAD`).
  - `val_loglik`: the numeric value (use `-1e18` for crash).
  - `status`: `keep`, `discard`, or `crash`.
  - `description`: brief note, e.g., `"added feature_xyz"`.
- **Ensure the file has a header line** if it doesn't exist:
  ```
  commit_hash	val_loglik	status	description
  ```

### 8. Discard if Not Kept
- If `status` is **not** `keep`, reset the working directory to the previous commit:
  ```bash
  git reset --hard HEAD~1
  ```
- This ensures the next turn starts from the last **kept** state.

### 9. Check Stopping Condition
- After appending, examine the **last 20 lines** of `experiments.tsv` (excluding the header).
- If **none** of those 20 experiments have `status=keep`, you **must stop**.
- This means no improvement has been made in the last 20 attempts.

## Stopping Procedure
When the stopping condition is met:
1. Print a final summary to stdout (best log‑likelihood, features used, total experiments).
2. Run: `touch .autoresearch_done`
3. Make no further code changes.

## Redundancy Check
- **Before proposing a change, always read the existing `experiments.tsv`.**
- Never repeat a feature combination that already has an entry in `experiments.tsv`.

## Example `experiments.tsv` Format
```
commit_hash	val_loglik	status	description
a1b2c3d	-123.456	keep	added feature_rain
e4f5g6h	-124.789	discard	removed feature_temp
i7j8k9l	-1e18	crash	segfault in data loader
```

Remember: You are the **brain**. Bash will simply keep calling you. Do your one experiment well, record the outcome, and signal when to stop.
