#!/usr/bin/env python3
import os, re, subprocess, sys
import numpy as np
from pathlib import Path

os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

TRAIN_PY    = Path("train.py")
EXPERIMENTS = Path("experiments.tsv")
BEST_F      = Path(".best_score.txt")
DONE_FLAG   = Path(".autoresearch_done")
DATA_FILE   = Path("autosearch_data/prepared_clean_landslide_ebm.npz")

ALL_FEATURES = np.load(DATA_FILE, allow_pickle=True)["feature_names"].tolist()

def current_features():
    m = re.search(r"^FEATURE_NAMES.*?=\s*(\[.*?\])", TRAIN_PY.read_text(), re.M)
    return re.findall(r"'(\w+)'", m.group(1)) if m else []

def tried_features():
    if not EXPERIMENTS.exists(): return set()
    lines = EXPERIMENTS.read_text().splitlines()[1:]
    return {f for l in lines for f in re.findall(r"'(\w+)'", l.split("\t")[3]) if len(l.split("\t")) >= 4}

def best_score():
    return float(BEST_F.read_text()) if BEST_F.exists() else -999999.0

DONE_FLAG.unlink(missing_ok=True)
if not EXPERIMENTS.exists():
    EXPERIMENTS.write_text("commit_hash\tval_loglik\tstatus\tdescription\n")

consec_fail = 0
print("=== Starting autonomous research loop ===")

while not DONE_FLAG.exists():
    cur = current_features()
    remaining = [f for f in ALL_FEATURES if f not in cur and f not in tried_features()]
    if not remaining:
        print("All features tried. Stopping."); break

    prompt = (
        f"Follow program.md exactly for one turn.\n"
        f"Current FEATURE_NAMES: {cur}\n"
        f"Features not yet tried individually: {remaining}\n"
        f"Pick ONE feature from the list above and add it to FEATURE_NAMES in train.py."
    )

    subprocess.run([
        "aider", "--model", "ollama/deepseek-r1:32b",
        "--no-gitignore", "--yes-always", "--auto-commits",
        "--no-show-model-warnings",
        "--aiderignore", ".aiderignore",
        "--read", "program.md",
        "--read", str(EXPERIMENTS),
        "--read", str(BEST_F),
        "train.py", "--message", prompt,
    ], check=False)

    result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True)
    try:
        val = float(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        val = None

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    feats  = current_features()
    best   = best_score()
    print(f"val_loglik={val}  best={best}  features={feats}")

    if val is None:
        status = "crash"; subprocess.run(["git", "reset", "--hard", "HEAD~1"]); consec_fail += 1
    elif val > best:
        status = "keep"; BEST_F.write_text(f"{val}\n"); consec_fail = 0; print(f"✅ New best: {val}")
    else:
        status = "discard"; subprocess.run(["git", "reset", "--hard", "HEAD~1"]); consec_fail += 1; print(f"❌ Reverted.")

    desc = str(feats)
    with EXPERIMENTS.open("a") as f:
        f.write(f"{commit}\t{val or 0:.6f}\t{status}\t{desc}\n")

    if consec_fail >= 20:
        print("20 consecutive failures. Stopping."); break

print("=== Done ===")
