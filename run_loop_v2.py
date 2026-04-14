#!/usr/bin/env python3
"""
Autonomous research loop v2 — GeoEvolve + TalkToEBM enhanced.

Key differences from run_loop.py (v1):
  - Pre-training literature check via GeoEvolve knowledge base
  - t2ebm graph descriptions fed back to LLM for model understanding
  - Multi-feature proposals (add/remove multiple features per turn)
  - LLM can also tune MODEL_CONFIG (hyperparameters), not just features
  - Structured JSON output from train_v2.py
  - Designed to be run by Gemini (or any LLM with code-editing ability)
  - Same keep/discard + git mechanism (with the v1 bug fixed)

Usage:
    python run_loop_v2.py

Requires:
    pip install t2ebm geoevolve interpret
"""
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
# LLM backend — set to your Gemini or other endpoint
os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")

TRAIN_PY     = Path("train_v2.py")
EXPERIMENTS  = Path("experiments_v2.tsv")
BEST_F       = Path(".best_score_v2.txt")
DONE_FLAG    = Path(".autoresearch_v2_done")
DATA_FILE    = Path("autosearch_data/prepared_clean_landslide_ebm.npz")
MODEL_FILE   = Path(".last_ebm_model.pkl")
PROGRAM_FILE = Path("program_v2.md")
KNOWLEDGE_DIR = Path("geoevolve_knowledge")

# Aider model config — change for Gemini
AIDER_MODEL = "ollama/deepseek-r1:32b"

# GeoEvolve config
GEOEVOLVE_ENABLED = True
GEOEVOLVE_PERSIST = "geoevolve_storage"

# TalkToEBM config
T2EBM_ENABLED = True
T2EBM_LLM_MODEL = "gpt-4-turbo-2024-04-09"  # for graph descriptions
T2EBM_TOP_N = 5  # describe top N features by importance

MAX_CONSECUTIVE_FAILURES = 20

ALL_FEATURES = np.load(DATA_FILE, allow_pickle=True)["feature_names"].tolist()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def current_features() -> list[str]:
    """Parse FEATURE_NAMES list from train_v2.py."""
    m = re.search(r"^FEATURE_NAMES.*?=\s*(\[.*?\])", TRAIN_PY.read_text(),
                  re.M | re.S)
    return re.findall(r"'(\w+)'", m.group(1)) if m else []


def current_config() -> dict:
    """Parse MODEL_CONFIG dict from train_v2.py."""
    m = re.search(r"^MODEL_CONFIG.*?=\s*(\{.*?\})", TRAIN_PY.read_text(),
                  re.M | re.S)
    if m:
        try:
            return json.loads(m.group(1).replace("'", '"'))
        except json.JSONDecodeError:
            pass
    return {}


def experiment_history() -> list[dict]:
    """Parse experiments_v2.tsv into a list of dicts."""
    if not EXPERIMENTS.exists():
        return []
    rows = []
    for line in EXPERIMENTS.read_text().splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        rows.append({
            "experiment": len(rows) + 1,
            "commit": cols[0],
            "val_pr_auc": float(cols[1]),
            "status": cols[2],
            "features": cols[3],
            "rationale": cols[4] if len(cols) > 4 else "",
        })
    return rows


def discarded_features() -> set[str]:
    """Return features individually tried and discarded."""
    history = experiment_history()
    discarded = set()
    last_keep_set: set[str] = set()
    for exp in history:
        feats = set(re.findall(r"'(\w+)'", exp["features"]))
        added = feats - last_keep_set
        if exp["status"] == "discard":
            discarded |= added
        elif exp["status"] == "keep":
            last_keep_set = feats
    return discarded


def best_score() -> float:
    """Read best score from file."""
    try:
        return float(BEST_F.read_text().strip())
    except (FileNotFoundError, ValueError):
        return -999999.0


def revert_train_py():
    """Revert only train_v2.py (preserve .best_score_v2.txt)."""
    subprocess.run(["git", "checkout", "HEAD~1", "--", str(TRAIN_PY)],
                   check=False)
    subprocess.run(["git", "reset", "HEAD", "--", str(TRAIN_PY)],
                   check=False)


# ============================================================
# GEOEVOLVE — Literature-informed hypothesis generation
# ============================================================
def query_geoevolve(current_feats: list[str],
                    history_summary: str) -> Optional[str]:
    """Query GeoEvolve knowledge base for literature-informed suggestions.

    Returns a text block of suggestions to inject into the LLM prompt.
    """
    if not GEOEVOLVE_ENABLED:
        return None

    try:
        from geoevolve import initialize_or_get_geo_know_db
    except ImportError:
        print("  [GeoEvolve] Not installed, skipping literature check.")
        return None

    try:
        geokg = initialize_or_get_geo_know_db(
            persist_dir=GEOEVOLVE_PERSIST,
            embedding_model_name="text-embedding-3-large",
            llm_model_name="gpt-4.1",
        )

        query = (
            f"For landslide susceptibility modeling using EBM "
            f"(Explainable Boosting Machine), the current features are: "
            f"{current_feats}. "
            f"Recent experiment history: {history_summary}. "
            f"What additional geospatial features or variable transformations "
            f"does the literature suggest? Consider precipitation antecedent "
            f"conditions, soil moisture proxies, geological factors, "
            f"land-use change indicators, and novel remote sensing indices."
        )

        results = geokg.query(query)
        if results and hasattr(results, "response"):
            return str(results.response)
    except Exception as e:
        print(f"  [GeoEvolve] Query failed: {e}")

    return None


# ============================================================
# TalkToEBM — Model understanding via graph descriptions
# ============================================================
def describe_ebm_graphs(top_n: int = T2EBM_TOP_N) -> Optional[str]:
    """Use t2ebm to describe the top-N EBM shape functions.

    Returns a text block the LLM can use to understand what the model learned.
    """
    if not T2EBM_ENABLED:
        return None

    if not MODEL_FILE.exists():
        return None

    try:
        import t2ebm
        import t2ebm.graphs as graphs
    except ImportError:
        print("  [TalkToEBM] Not installed, skipping graph description.")
        return None

    try:
        with open(MODEL_FILE, "rb") as f:
            ebm = pickle.load(f)

        # Get feature importances to pick top-N
        global_exp = ebm.explain_global()
        names = global_exp.data()["names"]
        scores = global_exp.data()["scores"]
        ranked = sorted(zip(names, scores, range(len(names))),
                        key=lambda x: abs(x[1]), reverse=True)

        descriptions = []
        for name, importance, idx in ranked[:top_n]:
            try:
                graph = graphs.extract_graph(ebm, idx)
                simplified = graphs.simplify_graph(
                    graph, min_variation_per_cent=0.05
                )
                graph_text = graphs.graph_to_text(simplified, max_tokens=500)
                prompt = t2ebm.prompts.describe_graph(
                    graph_text,
                    graph_description="log-odds contribution to landslide probability",
                    dataset_description=(
                        "Landslide susceptibility dataset with terrain, "
                        "proximity, land-cover, and precipitation features"
                    ),
                    task_description=(
                        "Describe the relationship this feature has with "
                        "landslide probability. Note thresholds, non-linearities, "
                        "and whether the effect is physically plausible."
                    ),
                )
                desc = t2ebm.describe_graph(T2EBM_LLM_MODEL, ebm, idx)
                descriptions.append(
                    f"## {name} (importance={importance:.4f})\n{desc}"
                )
            except Exception as e:
                descriptions.append(f"## {name}: description failed ({e})")

        return "\n\n".join(descriptions) if descriptions else None
    except Exception as e:
        print(f"  [TalkToEBM] Failed: {e}")
        return None


# ============================================================
# MAIN LOOP
# ============================================================
def build_prompt(cur_feats: list[str], remaining: list[str],
                 literature: Optional[str],
                 graph_descriptions: Optional[str],
                 history: list[dict]) -> str:
    """Build the full prompt for the LLM."""
    # Recent history summary (last 10)
    recent = history[-10:] if history else []
    history_block = "\n".join(
        f"  #{e['experiment']}: {e['status']} val_pr_auc={e['val_pr_auc']} "
        f"({e['rationale']})"
        for e in recent
    )

    prompt_parts = [
        f"Follow program_v2.md for one turn.",
        f"",
        f"## Current State",
        f"FEATURE_NAMES: {cur_feats}",
        f"MODEL_CONFIG: {current_config()}",
        f"Best score so far: {best_score()}",
        f"",
        f"## Available features not yet tried:",
        f"{remaining}",
        f"",
        f"## Recent experiment history:",
        f"{history_block or '(none yet)'}",
    ]

    if literature:
        prompt_parts += [
            f"",
            f"## Literature suggestions (from GeoEvolve knowledge base):",
            f"{literature}",
            f"",
            f"Consider these suggestions but also feel free to propose "
            f"unconventional combinations or test removal of features.",
        ]

    if graph_descriptions:
        prompt_parts += [
            f"",
            f"## Current model understanding (EBM shape functions):",
            f"{graph_descriptions}",
            f"",
            f"Use these descriptions to understand what the model has learned. "
            f"Features with flat or noisy shapes may be candidates for removal. "
            f"Features with strong, physically plausible effects should be kept.",
        ]

    prompt_parts += [
        f"",
        f"## Your task:",
        f"Propose ONE experiment. You may:",
        f"  1. Add one or more features from the available list",
        f"  2. Remove a feature you think is harming performance",
        f"  3. Change MODEL_CONFIG (interactions, learning_rate, etc.)",
        f"  4. Combine any of the above",
        f"",
        f"Edit train_v2.py: update FEATURE_NAMES, MODEL_CONFIG, and "
        f"EXPERIMENT_RATIONALE (explain your hypothesis).",
        f"The model MUST remain an ExplainableBoostingClassifier.",
    ]

    return "\n".join(prompt_parts)


def run_train() -> Optional[dict]:
    """Run train_v2.py and parse JSON output."""
    result = subprocess.run(
        [sys.executable, str(TRAIN_PY)],
        capture_output=True, text=True
    )
    try:
        # train_v2.py outputs a single JSON line
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
    except (json.JSONDecodeError, IndexError):
        pass

    if result.stderr:
        print(f"  [train_v2] stderr: {result.stderr[:500]}")
    return None


def main():
    DONE_FLAG.unlink(missing_ok=True)
    if not EXPERIMENTS.exists():
        EXPERIMENTS.write_text(
            "commit_hash\tval_pr_auc\tstatus\tfeatures\trationale\n"
        )

    consec_fail = 0
    print("=== Starting autonomous research loop v2 ===")
    print(f"  GeoEvolve: {'enabled' if GEOEVOLVE_ENABLED else 'disabled'}")
    print(f"  TalkToEBM: {'enabled' if T2EBM_ENABLED else 'disabled'}")
    print(f"  LLM model: {AIDER_MODEL}")
    print()

    while not DONE_FLAG.exists():
        iteration = len(experiment_history()) + 1
        print(f"--- Iteration {iteration} ---")

        cur = current_features()
        skip = discarded_features()
        remaining = [f for f in ALL_FEATURES if f not in cur and f not in skip]
        history = experiment_history()

        if not remaining and consec_fail >= 5:
            print("All candidate features exhausted and no recent improvement.")
            break

        # Phase 1: Query GeoEvolve for literature context
        print("  [1/4] Querying literature (GeoEvolve)...")
        history_summary = "; ".join(
            f"{e['status']}({e['val_pr_auc']})" for e in history[-5:]
        )
        literature = query_geoevolve(cur, history_summary)
        if literature:
            print(f"  [GeoEvolve] Got {len(literature)} chars of suggestions")

        # Phase 2: Describe current model via t2ebm
        print("  [2/4] Describing model (TalkToEBM)...")
        graph_desc = describe_ebm_graphs()
        if graph_desc:
            print(f"  [TalkToEBM] Got {len(graph_desc)} chars of descriptions")

        # Phase 3: LLM proposes changes
        print("  [3/4] LLM proposing experiment...")
        prompt = build_prompt(cur, remaining, literature, graph_desc, history)

        aider_read_files = [str(PROGRAM_FILE)]
        if EXPERIMENTS.exists():
            aider_read_files.append(str(EXPERIMENTS))

        subprocess.run([
            "aider", "--model", AIDER_MODEL,
            "--no-gitignore", "--yes-always", "--auto-commits",
            "--no-show-model-warnings",
            "--aiderignore", ".aiderignore",
            *[arg for f in aider_read_files for arg in ("--read", f)],
            str(TRAIN_PY), "--message", prompt,
        ], check=False)

        # Phase 4: Train and evaluate
        print("  [4/4] Training EBM...")
        train_result = run_train()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()

        if train_result is None:
            val = None
            rationale = "crash"
            feats_str = str(current_features())
        else:
            val = train_result["val_pr_auc"]
            rationale = train_result.get("rationale", "")
            feats_str = str(train_result["features"])

        best = best_score()
        print(f"  val_pr_auc={val}  best={best}")

        # Decision
        if val is None:
            status = "crash"
            revert_train_py()
            consec_fail += 1
            print("  [CRASH] Reverted.")
        elif val > best:
            status = "keep"
            BEST_F.write_text(f"{val}\n")
            consec_fail = 0
            print(f"  [KEEP] New best: {val}")
        else:
            status = "discard"
            revert_train_py()
            consec_fail += 1
            print(f"  [DISCARD] Reverted train_v2.py (best_score preserved).")

        # Log
        safe_rationale = rationale.replace("\t", " ").replace("\n", " ")[:200]
        with EXPERIMENTS.open("a") as f:
            f.write(
                f"{commit}\t{val or 0:.6f}\t{status}\t"
                f"{feats_str}\t{safe_rationale}\n"
            )

        if consec_fail >= MAX_CONSECUTIVE_FAILURES:
            print(f"{MAX_CONSECUTIVE_FAILURES} consecutive failures. Stopping.")
            break

        print()

    print("=== Done ===")


if __name__ == "__main__":
    main()
