"""CANet-style baseline for SynCAN, evaluated at WINDOW level (Phase 1 baseline).

  >>> Protocol note: <<<
  The CANet paper (Hanselmann 2020) reports plateau AUC ~0.982 / suppress ~0.743 on
  the *raw per-message event stream*.  Thesis evaluates everything at the
  window level the MOMENT pipeline consumes: 512-step windows on a 15 ms zero-order-
  hold grid, per-signal z-scored. Those two protocols are NOT comparable. A
  systematic study (memory/canet-stopcheck-findings) showed that at window level the
  attack difficulty even inverts: suppress is easy (>0.74) while plateau caps ~0.83,
  because the 15 ms ZOH grid already holds normal slow signals flat between updates,
  masking a plateau freeze, and 7.68 s windows dilute the ~46%-attack content.
  I will use a COMMON window-level protocol so CANet and the
  MOMENT method are compared apples-to-apples. The paper numbers below are shown for
  reference only; the STOP-CHECK bar is window-level.

Detector (window-level, CANet-style)
------------------------------------
Per CAN ID, a JointLSTMPredictor models that ID's signals' normal dynamics (CANet's
mechanism). Each signal is then scored by the max of three standardized, complementary
per-window anomaly features (all fit on the normal validation set):
  * z_pred  : LSTM one-step prediction error  -> abnormal *dynamics* (flooding/continuous)
  * z_var   : |z of the window's temporal std| -> abnormal *variability* (either direction)
  * z_still : sub-window stillness (min of 8 sub-window stds) -> abnormal *freeze* (plateau/suppress)
A window's score is the max over its 20 signals. Threshold = 95th percentile on the
normal validation set. Metrics via eval_utils, for all 5 attack types.

Models are saved to experiments/canet_models/ (one .pt per ID + meta.json).
MOMENT NOT used here, this is the classical baseline to beat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from syncan_dataloader import (  # noqa: E402
    load_syncan, N_SIGNALS, N_IDS, SIGNALS_PER_ID, CHANNEL_OFFSET,
)
import eval_utils  # noqa: E402

ATTACK_TYPES = ["plateau", "suppress", "flooding", "continuous", "playback"]
PAPER_AUC = {"plateau": 0.982, "suppress": 0.743}  # per-message protocol — reference only
ID_RANGES = [(CHANNEL_OFFSET[i], SIGNALS_PER_ID[i]) for i in range(N_IDS)]
N_SUBWIN = 8


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class JointLSTMPredictor(nn.Module):
    """Autoregressive one-step-ahead LSTM over one CAN ID's k signals jointly."""

    def __init__(self, n_sig: int, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_sig, hidden, num_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_sig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, k]
        out, _ = self.lstm(x)
        return self.head(out)


# --------------------------------------------------------------------------- #
# Feature helpers
# --------------------------------------------------------------------------- #
def _id_batch(id_windows: np.ndarray, idx, device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(id_windows[idx])).float()  # [B, k, T]
    return x.transpose(1, 2).contiguous().to(device)                     # [B, T, k]


def per_id_pred_error(model, id_windows, batch_size, device, time_agg="max") -> np.ndarray:
    """Per-window, per-signal one-step prediction error for one ID -> [N, k]."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, id_windows.shape[0], batch_size):
            x = _id_batch(id_windows, slice(i, i + batch_size), device)
            se = (model(x[:, :-1, :]) - x[:, 1:, :]) ** 2                  # [B, T-1, k]
            agg = se.amax(dim=1) if time_agg == "max" else se.mean(dim=1)
            out.append(agg.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, id_windows.shape[1]))


def min_subwin_std(windows: np.ndarray, n_sub: int = N_SUBWIN) -> np.ndarray:
    """Smallest sub-window temporal std per signal -> [N, 20] (low == frozen segment)."""
    n, c, t = windows.shape
    sub = windows[:, :, : (t // n_sub) * n_sub].reshape(n, c, n_sub, t // n_sub)
    return sub.std(axis=3).min(axis=2)


def raw_features(models, windows, batch_size, device, time_agg):
    """Return the three raw per-signal window features: (pred_err, win_std, submin_std)."""
    n = windows.shape[0]
    pred = np.zeros((n, N_SIGNALS))
    for i, (start, k) in enumerate(ID_RANGES):
        pred[:, start:start + k] = per_id_pred_error(
            models[i], windows[:, start:start + k, :], batch_size, device, time_agg)
    win_std = windows.std(axis=2)                     # [N, 20]
    submin = min_subwin_std(windows)                  # [N, 20]
    return pred, win_std, submin


def fit_stats(val_feats):
    """Per-signal mean/std for each feature, with robust floors, from normal val."""
    pred, wstd, sub = val_feats
    s = {
        "pred_mean": pred.mean(0),
        "pred_std": np.maximum(pred.std(0), 0.1 * np.median(pred.std(0)) + 1e-9),
        "wstd_mean": wstd.mean(0), "wstd_std": wstd.std(0) + 1e-9,
        "sub_mean": sub.mean(0), "sub_std": sub.std(0) + 1e-9,
    }
    return {k: v.tolist() for k, v in s.items()}


def combined_score(feats, stats) -> np.ndarray:
    """Combine the three standardized features -> [N] window anomaly scores."""
    pred, wstd, sub = feats
    z_pred = (pred - np.asarray(stats["pred_mean"])) / np.asarray(stats["pred_std"])
    z_var = np.abs((wstd - np.asarray(stats["wstd_mean"])) / np.asarray(stats["wstd_std"]))
    z_still = -(sub - np.asarray(stats["sub_mean"])) / np.asarray(stats["sub_std"])
    # clip to tame heavy-tailed prediction errors on hard-to-predict signals
    sig = np.maximum.reduce([np.clip(z_pred, None, 8.0), z_var, z_still])   # [N, 20]
    return sig.max(axis=1)                                                  # [N]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_one_id(id_idx, k, id_train, id_val, args, device) -> nn.Module:
    torch.manual_seed(args.seed + id_idx)
    model = JointLSTMPredictor(k, args.hidden, args.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    n = id_train.shape[0]
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, args.batch_size):
            x = _id_batch(id_train, perm[i:i + args.batch_size], device)
            opt.zero_grad()
            loss = loss_fn(model(x[:, :-1, :]), x[:, 1:, :])
            loss.backward()
            opt.step()
    ve = per_id_pred_error(model, id_val, args.batch_size, device, args.time_agg).mean()
    print(f"    [{id_idx + 1:>2}/{N_IDS}] id{id_idx + 1:<2} ({k} sig) trained "
          f"({time.time() - t0:.1f}s, val_pred_err={ve:.4f})")
    return model


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Window-level CANet-style baseline.")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--time-agg", choices=["max", "mean"], default="max")
    ap.add_argument("--max-train-windows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent.parent / "experiments" / "canet_models")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"CANet baseline (WINDOW-LEVEL protocol) — {N_IDS} per-ID joint LSTMs + variance features")
    print(f"device={device}  epochs={args.epochs}  batch={args.batch_size}  hidden={args.hidden}")
    print("=" * 70)

    print("[1/4] loading SynCAN windows...")
    data = load_syncan(data_dir=args.data_dir, verbose=args.verbose)
    train, val = data.train_windows, data.val_windows
    if args.max_train_windows and train.shape[0] > args.max_train_windows:
        sel = np.random.choice(train.shape[0], args.max_train_windows, replace=False)
        train = train[sel]
        print(f"      (subsampled train to {train.shape[0]} windows)")
    print(f"      train {train.shape}  val {val.shape}")

    print(f"[2/4] training {N_IDS} per-ID joint LSTMs -> {out_dir}")
    models = [None] * N_IDS
    t_train = time.time()
    for i, (start, k) in enumerate(ID_RANGES):
        models[i] = train_one_id(
            i, k, np.ascontiguousarray(train[:, start:start + k, :]),
            np.ascontiguousarray(val[:, start:start + k, :]), args, device)
        torch.save(models[i].state_dict(), out_dir / f"id_{i + 1:02d}_k{k}.pt")
    print(f"      trained in {time.time() - t_train:.1f}s")

    print("[3/4] fitting features on normal validation set + calibrating threshold...")
    val_feats = raw_features(models, val, args.batch_size, device, args.time_agg)
    stats = fit_stats(val_feats)
    threshold = float(np.percentile(combined_score(val_feats, stats), 95))
    meta = {
        "protocol": "window_level", "detector": "per_id_joint_lstm + variance features",
        "paper_auc_per_message_reference": PAPER_AUC, "threshold_p95": threshold,
        "stats": stats, "id_ranges": ID_RANGES, "signal_names": data.signal_names,
        "dt_ms": data.dt_ms, "window": data.window, "config": vars(args),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"      threshold (95th pct of val score) = {threshold:.4f}   saved meta.json")

    print("[4/4] evaluating on all 5 attack types (window-level)...\n")
    header = (f"{'attack':<12}{'AUC-ROC':>9}{'AP':>9}{'F1':>8}{'F1_thr':>9}{'delay':>9}"
              f"{'paper*':>9}")
    print(header)
    print("-" * len(header))
    results = {}
    for atk in ATTACK_TYPES:
        w, y = data.test_windows[atk], data.test_labels[atk]
        scores = combined_score(raw_features(models, w, args.batch_size, device, args.time_agg), stats)
        auc = eval_utils.compute_auc_roc(scores, y)
        ap = eval_utils.compute_average_precision(scores, y)
        f1, f1_thr = eval_utils.compute_f1_at_best_threshold(scores, y)
        delay = eval_utils.compute_detection_delay(scores, y, threshold)
        results[atk] = {"auc": auc, "ap": ap, "f1": f1, "delay": delay}
        ref = f"{PAPER_AUC[atk]:.3f}" if atk in PAPER_AUC else "-"
        print(f"{atk:<12}{auc:>9.3f}{ap:>9.3f}{f1:>8.3f}{f1_thr:>9.3f}{delay:>9.1f}{ref:>9}")
    print("-" * len(header))
    print("* paper AUC is on the raw per-message stream (different protocol) — reference only.")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved metrics -> {out_dir / 'results.json'}")
    print("done.")


if __name__ == "__main__":
    main()
