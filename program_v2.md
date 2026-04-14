
# Autonomous Research Protocol v2 — GeoEvolve + TalkToEBM Enhanced

You are an autonomous research agent performing feature selection and model refinement
for a landslide susceptibility model (EBM — Explainable Boosting Machine).

This protocol gives you **more freedom** than v1: you may add multiple features,
remove features, tune hyperparameters, and propose unconventional hypotheses informed
by literature and model interpretability.

## Environment & Constraints

- You may edit `train_v2.py` — specifically:
  - `FEATURE_NAMES` (add, remove, reorder features)
  - `MODEL_CONFIG` (tune EBM hyperparameters)
  - `EXPERIMENT_RATIONALE` (explain your hypothesis — mandatory)
- The model **MUST remain** an `ExplainableBoostingClassifier`.
- Do **not** edit any other file.
- Do **not** modify anything below the `DO NOT EDIT` line in `train_v2.py`.

## File Roles

| File | Purpose | Editable? |
|------|---------|-----------|
| `train_v2.py` | Model training. Contains editable section. | **YES** — editable section only |
| `program_v2.md` | This protocol document. | **NO** |
| `experiments_v2.tsv` | Full experiment history. | **NO** (appended by loop) |
| `prepare.py` | Data generation source (read for context). | **NO** |

## Available Features

### Static Features
- `slope`, `elevation`, `aspect`, `eastness`, `northness`, `tri` (Terrain)
- `distroads`, `walking_time_to_road`, `walking_time_to_bldg`, `walking_time_to_elec_infra` (Proximity)
- `landcoverfull`, `treecoverdensity`, `flammability` (Land cover)

### Dynamic Features (Spatiotemporal)
- `precipitation_{W}d_sum` where W = 1 to 60
- `temperature_{W}d_mean` where W = 1 to 60
- `lightning_{W}d_max` where W = 1 to 60

## Information Sources

Each iteration, you receive up to three information sources:

### 1. GeoEvolve Literature Context (when available)
- Domain knowledge from geospatial literature about landslide susceptibility
- Suggestions for features, transformations, or variable interactions
- Treat as inspiration, not prescription — you may disagree with literature

### 2. TalkToEBM Model Descriptions (when available)
- Natural-language descriptions of the EBM's learned shape functions
- Shows what the current model has learned about each feature
- Use to identify:
  - **Flat/noisy features** → candidates for removal
  - **Strong monotonic features** → confirm they belong
  - **Surprising non-linearities** → may suggest feature engineering
  - **Threshold effects** → may suggest binned features or interactions

### 3. Experiment History
- Full record of past experiments with scores and status
- Identify patterns: which feature types consistently help or hurt?

## Experiment Strategy

### Allowed Actions (per turn, pick one or combine)

1. **Add features** — one or more from the available list
2. **Remove features** — drop a feature you believe is harming performance
3. **Swap features** — replace one feature with a related alternative
4. **Tune hyperparameters** — change `MODEL_CONFIG` values
5. **Add interactions** — set `interactions > 0` to try pairwise terms

### Thinking Framework

Before each experiment, reason about:

1. **What does the literature suggest?** (GeoEvolve context)
2. **What has the model learned so far?** (TalkToEBM descriptions)
3. **What patterns emerge from past experiments?** (experiments_v2.tsv)
4. **What is my hypothesis?** (write it in `EXPERIMENT_RATIONALE`)

### Bold Hypotheses Encouraged

Unlike v1 (which was strict one-feature-at-a-time forward selection), you are
encouraged to:
- Test literature-suggested features even if they seem unconventional
- Try removing features that the EBM shape functions show as flat/noisy
- Test different precipitation aggregation windows based on physical reasoning
  (e.g., antecedent moisture conditions at 7, 14, 30 days)
- Explore temperature and lightning features (not just precipitation)
- Try `interactions > 0` to capture synergistic effects between features

### What NOT to Do

- Do not change the model class (must be ExplainableBoostingClassifier)
- Do not edit below the `DO NOT EDIT` line
- Do not set `outer_bags` below 4 (too noisy)
- Do not set `learning_rate` above 0.5 (unstable)
- Always write a meaningful `EXPERIMENT_RATIONALE`

## Scoring

- Output metric: `val_pr_auc = average_precision_score(y_val, y_pred_proba)`
- **Higher is better** (1.0 = perfect, reflects performance on rare positive events)
- The loop compares your score to the running best
- `keep` if score improves, `discard` if not (reverts train_v2.py)

## Stopping Condition

- 20 consecutive failures → automatic stop
- All candidate features exhausted → stop
- Manual: `touch .autoresearch_v2_done`
