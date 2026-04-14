#!/usr/bin/env python3
"""
Hazard-agnostic EBM trainer — v2 (GeoEvolve + TalkToEBM aware).

Differences from train.py:
  - Supports multiple features added/removed per iteration
  - Exports the trained EBM model for t2ebm graph description
  - Outputs structured JSON (score + feature importances) instead of bare float
  - Supports optional interaction terms
  - Can load multiple raster sources via RASTER_SOURCES config

Edit FEATURE_NAMES and MODEL_CONFIG to experiment.
"""

# ============================================================
# EDITABLE SECTION — LLM MAY CHANGE THIS BLOCK
# ============================================================
FEATURE_NAMES: list[str] = [
    # Terrain features
    "slope",
    "elevation", 
    "aspect",
    "tri",
    
    # Proximity features
    "distroads",
    
    # Land cover features
    "landcoverfull", 
    "treecoverdensity"
]

# Model hyperparameters the LLM may tune (must remain EBM)
MODEL_CONFIG: dict = {
    "interactions": 10,          # Enable pairwise interactions for better model flexibility
    "max_bins": 64,
    "learning_rate": 0.03,
    "outer_bags": 8,
    "validation_size": 0.15,
    "early_stopping_rounds": 50,
}

# Optional: rationale for this experiment (written by LLM)
EXPERIMENT_RATIONALE: str = """
Baseline model using key terrain features:
- slope, elevation, aspect, tri
Plus proximity to roads
And land cover characteristics
Including interaction terms to capture combined effects
"""
# ============================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================

import json
import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from interpret.glassbox import ExplainableBoostingClassifier

ROOT = Path(__file__).resolve().parent / "autosearch_data"
HAZARD_TYPE = "landslide"
DATA_FILE = ROOT / f"prepared_clean_{HAZARD_TYPE}_ebm.npz"
MODEL_OUT = Path(".last_ebm_model.pkl")


def load_and_filter_features():
    """Load NPZ and select only the features listed in FEATURE_NAMES."""
    data = np.load(DATA_FILE, allow_pickle=True)
    X_full = data['X']
    y = data['y']
    all_feature_names = data['feature_names'].tolist()

    if not FEATURE_NAMES:
        print(json.dumps({"error": "No features selected"}))
        sys.exit(1)

    selected_idx = []
    selected_names = []
    for name in FEATURE_NAMES:
        if name not in all_feature_names:
            print(json.dumps({"error": f"Feature '{name}' not found in data"}))
            sys.exit(1)
        idx = all_feature_names.index(name)
        selected_idx.append(idx)
        selected_names.append(name)

    X = X_full[:, selected_idx]
    return X, y, selected_names


def main():
    # 1. Load and filter
    try:
        X, y, feature_names = load_and_filter_features()
    except Exception as e:
        print(json.dumps({"error": f"Data loading crashed: {e}"}))
        sys.exit(1)

    # 2. Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. Train EBM
    ebm = ExplainableBoostingClassifier(
        feature_names=feature_names,
        interactions=MODEL_CONFIG["interactions"],
        max_bins=MODEL_CONFIG["max_bins"],
        learning_rate=MODEL_CONFIG["learning_rate"],
        outer_bags=MODEL_CONFIG["outer_bags"],
        validation_size=MODEL_CONFIG["validation_size"],
        early_stopping_rounds=MODEL_CONFIG["early_stopping_rounds"],
        random_state=42,
        n_jobs=-1,
    )

    try:
        ebm.fit(X_train, y_train)
    except Exception as e:
        print(json.dumps({"error": f"EBM training crashed: {e}"}))
        sys.exit(1)

    # 4. Evaluate
    y_pred_proba = ebm.predict_proba(X_val)[:, 1]
    val_pr_auc = average_precision_score(y_val, y_pred_proba)

    # 5. Feature importances (EBM term importances)
    importances = {}
    global_exp = ebm.explain_global()
    for i, name in enumerate(global_exp.data()["names"]):
        importances[name] = float(global_exp.data()["scores"][i])

    # 6. Save model for t2ebm graph description
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(ebm, f)

    # 7. Output structured JSON
    output = {
        "val_pr_auc": round(val_pr_auc, 6),
        "n_features": len(feature_names),
        "features": feature_names,
        "importances": importances,
        "rationale": EXPERIMENT_RATIONALE,
        "model_config": MODEL_CONFIG,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
