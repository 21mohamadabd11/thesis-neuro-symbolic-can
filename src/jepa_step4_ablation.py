"""Phase 3, Section E — Step 4 ablation: MOMENT + JEPA prediction error only.

Reuses the exact per-window prediction-error inference from jepa_inference.py (same 8
fixed masks per branch), builds the Step-4 anomaly score as the mean of the three branch
errors, and reports AUC-ROC per attack. This is the "JEPA prediction error only" rung of
the six-step ablation — no concept layer, no rules.

Labelling (as specified for this step): the score of every window in an attack file is
labelled 1, the normal validation windows labelled 0; AUC is computed per attack over
[normal val scores | attack file scores].
  NOTE: Step 1 in ablation_results.csv used the attack files' OWN per-window labels
  (within-file). This step uses normal-val-vs-whole-attack-file instead — a slightly
  different protocol (flagged; conclusion is unaffected given Section D's null result).

Manual run:
    py -3.14 src/jepa_step4_ablation.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jepa_module import MultiResolutionJEPA, make_block_mask, N_PATCHES, MOMENT_DIM  # noqa: E402
from jepa_inference import (  # noqa: E402  (reuse the identical inference machinery)
    per_window_errors, N_MASKS, SEED, SPLIT_FILES, BRANCHES, ATTACKS, DEFAULT_EMB_DIR,
)
import eval_utils  # noqa: E402

STEP_NAME = "Step4_JEPA_pred_error"


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 4 ablation — JEPA prediction-error AUC.")
    ap.add_argument("--emb-dir", default=DEFAULT_EMB_DIR)
    ap.add_argument("--checkpoint", default=None, help="default: experiments/jepa_checkpoint.pt")
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb_dir = Path(args.emb_dir)
    proj = Path(__file__).resolve().parent.parent
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (proj / "experiments" / "jepa_checkpoint.pt")
    csv_path = proj / "experiments" / "ablation_results.csv"

    print("=" * 66)
    print("Phase 3E — Step 4 ablation (JEPA prediction error only, no concept layer)")
    print(f"  checkpoint={ckpt_path.name}  emb_dir={emb_dir}  device={device}")
    print("=" * 66)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model = MultiResolutionJEPA(n_patches=N_PATCHES, moment_dim=MOMENT_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    branches = dict(model.named_branches())
    mask_sizes = {name: b.n_mask_patches for name, b in branches.items()}
    print(f"[model] loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
          f"val {ckpt.get('val_loss', float('nan')):.4f})")

    # identical fixed masks as jepa_inference (same seed + order) -> identical errors
    torch.manual_seed(SEED)
    branch_masks = {name: [make_block_mask(N_PATCHES, mask_sizes[name]).to(device)
                           for _ in range(N_MASKS)] for name in BRANCHES}

    # per-window anomaly score = mean of the 3 branch errors
    scores: dict[str, np.ndarray] = {}
    for split, fname in SPLIT_FILES.items():
        path = emb_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"embedding file not found: {path}")
        emb = torch.from_numpy(np.load(str(path))).float()
        e = {b: per_window_errors(branches[b], emb, branch_masks[b], args.batch_size, device)
             for b in BRANCHES}
        scores[split] = (e["short"] + e["medium"] + e["long"]) / 3.0
        print(f"[score] {split:<11} n={len(scores[split]):>4}  mean={scores[split].mean():.6f}")
        del emb

    # AUC per attack: normal val (label 0) vs attack file (label 1)
    normal = scores["normal"]
    print("\nStep 4 — JEPA prediction error only (no concept layer)")
    results: dict[str, float] = {}
    for atk in ATTACKS:
        s = np.concatenate([normal, scores[atk]])
        y = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(scores[atk]), dtype=int)])
        auc = eval_utils.compute_auc_roc(s, y)
        results[atk] = float(auc)
        print(f"  {atk + ':':<12}AUC = {auc:.3f}")

    # upsert the Step-4 row into experiments/ablation_results.csv
    row = {"step": STEP_NAME, **{k: round(results[k], 4) for k in ATTACKS}}
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["step"] != STEP_NAME]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nupserted '{STEP_NAME}' -> {csv_path}")
    print("done.")


if __name__ == "__main__":
    main()
