"""Phase 3, Section D — JEPA inference + prediction-error visualisation.

Loads the trained MultiResolutionJEPA checkpoint and measures per-branch prediction
error on normal vs. attack windows (in MOMENT embedding space). This is the Step-4
signal: does the JEPA, trained only on NORMAL traffic, predict attack patches WORSE
than normal ones?

  * checkpoint : experiments/jepa_checkpoint.pt (eval mode, no gradients).
  * data       : val_embeddings.npy (normal) + the 5 attack test embedding files.
  * per-window error: for each window we run the branch, mask a contiguous block, and take
    the mean squared prediction error over the masked patches -> one scalar per window,
    per branch. To make normal vs. attack comparable and low-noise, each window's error is
    averaged over N_MASKS fixed masks (the SAME masks are reused across every split).

    (`MultiResolutionJEPA.compute_prediction_error` returns an aggregate scalar per branch;
     the mean±std table and the KDE plots need per-window distributions, so we compute those
     here directly — the aggregate is just the mean of these per-window errors.)

Outputs: a verification table (✓ if attack mean error > normal mean error) and
figures/prediction_error_curves.png (3 stacked subplots, one per branch, overlapping KDEs).

Manual run:
    py -3.14 src/jepa_inference.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jepa_module import MultiResolutionJEPA, make_block_mask, N_PATCHES, MOMENT_DIM  # noqa: E402

try:  # so the ✓/✗ marks print cleanly on Windows
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

DEFAULT_EMB_DIR = r"C:\Users\moea0\ThesisData\embeddings"
SEED = 42
N_MASKS = 8                      # masks averaged per window (reused across all splits)
BRANCHES = ["short", "medium", "long"]
ATTACKS = ["plateau", "suppress", "flooding", "continuous", "playback"]
# split display name -> embedding filename
SPLIT_FILES = {
    "normal": "val_embeddings.npy",
    "plateau": "test_plateau_emb.npy",
    "suppress": "test_suppress_emb.npy",
    "flooding": "test_flooding_emb.npy",
    "continuous": "test_continuous_emb.npy",
    "playback": "test_playback_emb.npy",
}
SPLITS = list(SPLIT_FILES)       # normal + 5 attacks
COLORS = {
    "normal": "#1f77b4", "plateau": "#ff7f0e", "suppress": "#d62728",
    "flooding": "#2ca02c", "continuous": "#9467bd", "playback": "#8c564b",
}


@torch.no_grad()
def per_window_errors(branch, emb: torch.Tensor, masks, batch_size: int, device) -> np.ndarray:
    """Per-window mean prediction error [N] for one branch, averaged over `masks`."""
    n = emb.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, batch_size):
        xb = emb[i:i + batch_size].to(device, non_blocking=True)      # [b, 64, 1024]
        acc = torch.zeros(xb.shape[0], device=device)
        for m in masks:
            pred = branch(xb, m)                                      # [b, n_masked, 1024]
            tgt = xb[:, m, :]                                         # [b, n_masked, 1024]
            acc += ((pred - tgt) ** 2).mean(dim=(1, 2))              # [b]
        out[i:i + batch_size] = (acc / len(masks)).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="JEPA inference: per-branch error normal vs attacks.")
    ap.add_argument("--emb-dir", default=DEFAULT_EMB_DIR)
    ap.add_argument("--checkpoint", default=None, help="default: experiments/jepa_checkpoint.pt")
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb_dir = Path(args.emb_dir)
    proj = Path(__file__).resolve().parent.parent
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (proj / "experiments" / "jepa_checkpoint.pt")
    out_png = proj / "figures" / "prediction_error_curves.png"

    print("=" * 70)
    print("Phase 3D — JEPA inference (prediction error: normal vs. attacks)")
    print(f"  checkpoint={ckpt_path.name}  emb_dir={emb_dir}  device={device}  masks/window={N_MASKS}")
    print("=" * 70)

    # ---- model ----
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model = MultiResolutionJEPA(n_patches=N_PATCHES, moment_dim=MOMENT_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[model] loaded checkpoint from epoch {ckpt.get('epoch','?')} "
          f"(val_loss {ckpt.get('val_loss', float('nan')):.4f})")

    branches = dict(model.named_branches())
    mask_sizes = {name: b.n_mask_patches for name, b in branches.items()}

    # fixed masks per branch, generated once, reused for every split (fair comparison)
    torch.manual_seed(SEED)
    branch_masks = {name: [make_block_mask(N_PATCHES, mask_sizes[name]).to(device)
                           for _ in range(N_MASKS)] for name in BRANCHES}

    # ---- per-window errors for every split ----
    errors: dict[str, dict[str, np.ndarray]] = {}
    for split, fname in SPLIT_FILES.items():
        path = emb_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"embedding file not found: {path}")
        emb = torch.from_numpy(np.load(str(path))).float()           # [N, 64, 1024]
        errors[split] = {b: per_window_errors(branches[b], emb, branch_masks[b],
                                              args.batch_size, device) for b in BRANCHES}
        print(f"[data] {split:<11} {tuple(emb.shape)}  scored")
        del emb

    # ---- verification table ----
    print("\n=== per-branch prediction error  (mean ± std;  ✓ = attack mean > normal mean) ===")
    header = f"{'Branch':<8}" + "".join(f"{s:<17}" for s in ["normal"] + ATTACKS)
    print(header)
    print("-" * len(header))
    n_sep = 0
    for b in BRANCHES:
        n_mean = errors["normal"][b].mean()
        n_std = errors["normal"][b].std()
        row = f"{b:<8}" + f"{f'{n_mean:.4f}±{n_std:.4f}':<17}"
        for atk in ATTACKS:
            a_mean, a_std = errors[atk][b].mean(), errors[atk][b].std()
            mark = "✓" if a_mean > n_mean else "✗"
            n_sep += a_mean > n_mean
            row += f"{f'{a_mean:.4f}±{a_std:.4f}{mark}':<17}"
        print(row)
    print("-" * len(header))
    print(f"{n_sep}/{len(BRANCHES) * len(ATTACKS)} branch×attack combinations separate "
          "(attack error > normal error).")

    # ---- figure: 3 stacked KDE subplots ----
    print("\n[plot] prediction_error_curves.png ...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    for ax, b in zip(axes, BRANCHES):
        pooled = np.concatenate([errors[s][b] for s in SPLITS])
        xmax = float(np.percentile(pooled, 99.5))
        xgrid = np.linspace(0.0, max(xmax, 1e-9), 400)
        for s in SPLITS:
            e = errors[s][b]
            try:
                y = gaussian_kde(e)(xgrid)
            except Exception:
                continue
            ax.plot(xgrid, y, color=COLORS[s], lw=1.8, label=s)
            ax.fill_between(xgrid, y, color=COLORS[s], alpha=0.08)
        ax.set_title(f"{b} branch — contiguous mask = {mask_sizes[b]} patches")
        ax.set_xlabel("JEPA prediction error (MSE in MOMENT embedding space)")
        ax.set_ylabel("density")
        ax.set_xlim(0.0, xmax)
        ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    fig.suptitle("Multi-resolution JEPA prediction error — normal vs. attacks (SynCAN)", y=1.005)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      saved -> {out_png}")
    print("\ndone.")


if __name__ == "__main__":
    main()
