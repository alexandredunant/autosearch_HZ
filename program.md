
# Autonomous Feature Selection Protocol

You are an autonomous research agent performing feature selection for a machine learning model (EBM) to predict hazards.

## Environment & Constraints

- You have read-only access to `prepare.py`.
- You may edit `train.py` (specifically the `FEATURE_NAMES` list) and `experiments.tsv`.
- Do **not** modify any other part of `train.py`.
- Do **not** commit `experiments.tsv` to Git.

## File Roles

| File | Purpose | Editable? |
|------|---------|-----------|
| `prepare.py` | Data generation and feature name source. | **NO** |
| `train.py` | Model training. Contains `FEATURE_NAMES` list. | **YES** – Only `FEATURE_NAMES`. |
| `program.md` | This protocol document. | **NO** |
| `experiments.tsv` | Full experiment history. | **YES** – Append only. |

## Available Predictors (Features)

The following features are available in the dataset (see `prepare.py` for how they are generated):

### Static Features
- `slope`, `elevation`, `aspect`, `eastness`, `northness`, `tri` (Terrain indices)
- `distroads`, `walking_time_to_road`, `walking_time_to_bldg`, `walking_time_to_elec_infra` (Proximity)
- `landcoverfull`, `treecoverdensity`, `flammability` (Land cover)

### Dynamic Features (Spatiotemporal)
- `precipitation_{W}d_sum` (e.g., `precipitation_30d_sum`) where W is 1 to 60.
- `temperature_{W}d_mean` where W is 1 to 60.
- `lightning_{W}d_max` where W is 1 to 60.

## Experiment Loop

For each turn, perform exactly these steps:

### 1. Read Current State
- Read `experiments.tsv` to see what has been tried.
- Identify the **current best** combination (the one with the highest `val_loglik` marked as `status=keep`).
- If `experiments.tsv` is empty or has no `keep`, start with an empty list or a single basic feature.

### 2. Decide Next Combination (Forward Selection Strategy)
- **Strategy**: Try adding **one new feature at a time** to the current best combination.
- This "one-by-one" exploration is more systematic than jumping to many features.
- **Never** repeat a combination already in `experiments.tsv`.

### 3. Edit `train.py`
- Update `FEATURE_NAMES` in `train.py`.

### 4. Commit the Change
- auto-commits. Use a message like `feat: try adding elevation to [slope]`.

### 5. Run & Evaluate
- Run `python train.py`. Capture the single numeric output (`val_pr_auc`).
- Compare with the best `val_pr_auc` in `experiments.tsv`.
- **Status**:
  - `keep` if `val_pr_auc > best_so_far`
  - `discard` if `val_pr_auc <= best_so_far`
  - `crash` if it failed.

### 6. Record in `experiments.tsv`
- Append the result to `experiments.tsv`:
  ```
  commit_hash	val_pr_auc	status	description
  ```
- **CRITICAL**: Use a real TAB character. If you are editing the file, ensure you don't overwrite existing lines.

### 7. Cleanup
- If `status` is NOT `keep`, run `git reset --hard HEAD~1` to revert `train.py`.

### 8. Stopping Condition
- If 20 consecutive experiments fail to improve the best score, stop.
- To stop: `touch .autoresearch_done`.

