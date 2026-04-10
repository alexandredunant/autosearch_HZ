#!/usr/bin/env python3
"""
Hazard-agnostic point-process model.
Edit STATIC_FEATURE_NAMES to add/remove features.
"""

# ============================================================
# EDITABLE CONSTANTS – LLM MAY CHANGE THIS SECTION ONLY
# ============================================================
STATIC_FEATURE_NAMES: list[str] = [
    # Add or remove feature names from prepare.static_feature_names
]
# ============================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================

import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from prepare import build_point_process_splits, static_feature_names as ALL_STATIC

OUTPUT_DIR = Path("output")
SUMMARY_PATH = Path("latest_model_summary.txt")

def train():
    train_split, val_split, feature_names, mu, sd = build_point_process_splits(
        selected_static=STATIC_FEATURE_NAMES,
        val_fold=3
    )

    for name in STATIC_FEATURE_NAMES:
        if name not in feature_names:
            print(f"Error: '{name}' not in available static features.", file=sys.stderr)
            sys.exit(1)

    def neg_log_posterior(beta):
        log_prior = 0.5 * np.sum(beta ** 2)
        event_eta = train_split.event_x @ beta + train_split.event_offset
        support_eta = np.tensordot(train_split.support_x, beta, axes=([2], [0])) + train_split.support_offset[None, :]
        term1 = np.sum(event_eta)
        log_int = np.log(np.sum(np.exp(support_eta + train_split.support_log_weights[None, :]), axis=1))
        term2 = np.sum(log_int)
        return -(term1 - term2 + log_prior)

    init_beta = np.zeros(len(feature_names))
    res = minimize(neg_log_posterior, init_beta, method='L-BFGS-B')
    beta_map = res.x

    def calc_loglik(split, b):
        e_eta = split.event_x @ b + split.event_offset
        s_eta = np.tensordot(split.support_x, b, axes=([2], [0])) + split.support_offset[None, :]
        t1 = np.sum(e_eta)
        t2 = np.sum(np.log(np.sum(np.exp(s_eta + split.support_log_weights[None, :]), axis=1)))
        return t1 - t2

    val_loglik = calc_loglik(val_split, beta_map)

    SUMMARY_PATH.parent.mkdir(exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        f.write("=== MODEL SUMMARY ===\n")
        f.write(f"val_loglik: {val_loglik:.4f}\n\n")
        f.write("Coefficients:\n")
        for name, val in zip(feature_names, beta_map):
            f.write(f"  {name:20s}: {val:8.4f}\n")

    print(f"val_loglik: {val_loglik:.4f}")

if __name__ == "__main__":
    train()
