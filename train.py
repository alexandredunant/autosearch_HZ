#!/usr/bin/env python3
"""
Hazard-agnostic point-process model using Explainable Boosting Machine (EBM).
Edit FEATURE_NAMES to add/remove features.
"""

# ============================================================
# EDITABLE CONSTANTS – LLM MAY CHANGE THIS SECTION ONLY
# ============================================================
FEATURE_NAMES: list[str] = ["slope", "elevation"]
# ============================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================

import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from interpret.glassbox import ExplainableBoostingClassifier

# Load data
ROOT = Path(__file__).resolve().parent / "autosearch_data"
HAZARD_TYPE = "landslide"
DATA_FILE = ROOT / f"prepared_clean_{HAZARD_TYPE}_ebm.npz"

def load_and_filter_features():
    """Load NPZ and select only the features listed in FEATURE_NAMES."""
    data = np.load(DATA_FILE, allow_pickle=True)
    X_full = data['X']
    y = data['y']
    all_feature_names = data['feature_names'].tolist()

    if not FEATURE_NAMES:
        # If no features selected, use all static features (fallback)
        # For a true baseline, you could return an empty X, but we'll use all.
        selected_idx = list(range(len(all_feature_names)))
        selected_names = all_feature_names
    else:
        selected_idx = []
        selected_names = []
        for name in FEATURE_NAMES:
            if name not in all_feature_names:
                print(f"Error: Feature '{name}' not found in data.", file=sys.stderr)
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
        print(f"Data loading crashed: {e}", file=sys.stderr)
        sys.exit(1)

    if X.shape[1] == 0:
        print("Error: No features selected.", file=sys.stderr)
        sys.exit(1)

    # 2. Split into train/validation
    # Use spatial fold if available? For simplicity, random stratified split.
    # We'll preserve the original event_fold for more rigorous CV if desired.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. Train EBM
    ebm = ExplainableBoostingClassifier(
        interactions=0,               # 0 = no pairwise interactions (main effects only)
        max_bins=256,
        learning_rate=0.01,
        outer_bags=16,
        validation_size=0.15,         # uses part of training for early stopping
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )

    try:
        ebm.fit(X_train, y_train)
    except Exception as e:
        print(f"EBM training crashed: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Evaluate on validation set
    y_pred_proba = ebm.predict_proba(X_val)[:, 1]  # probability of class 1 (event)
    val_loglik = -log_loss(y_val, y_pred_proba, labels=[0, 1])

    # Output only the numeric value (as required by the loop)
    print(f"{val_loglik:.6f}")

if __name__ == "__main__":
    main()
