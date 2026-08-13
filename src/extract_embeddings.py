"""Phase 2, Section B — extract frozen-MOMENT embeddings for every window, plus the
Step-1 ablation (MOMENT-only anomaly score = distance from the normal centroid).

For each split we run the frozen MOMENT encoder over all windows in batches and save
the full patch embeddings [N, 64, 1024] to data/syncan/embeddings/*.npy. The files are
large (~5 GB total; train alone ~3.2 GB), so each is written incrementally to a
memory-mapped .npy. The mean-over-patches vector
[N, 1024] is accumulated on the fly for the ablation, so we never re-read the big files.

Step 1 (Six-step ablation, lower bound):
  centroid = mean over training windows of their mean-pooled embedding   -> [1024]
  score(window) = L2 distance of its mean-pooled embedding from centroid
  AUC (eval_utils) vs labels, per attack -> written as the Step-1 row of
  experiments/ablation_results.csv.


"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moment_encoder import MOMENTEncoder, N_PATCHES, D_MODEL, N_CHANNELS, W  # noqa: E402
from syncan_dataloader import load_syncan  # noqa: E402
import eval_utils  # noqa: E402

# test files, in save order; keys match syncan_dataloader.test_windows
TEST_KEYS = ["normal", "plateau", "suppress", "flooding", "continuous", "playback"]
ATTACK_KEYS = ["plateau", "suppress", "flooding", "continuous", "playback"]  # scored in Step 1
# window-level CANet baseline (for a reference column in the printout only)
CANET_REF = {"plateau": 0.854, "suppress": 0.790, "flooding": 0.840,
             "continuous": 0.680, "playback": 0.654}


def extract_and_save(encoder: MOMENTEncoder, windows: np.ndarray, path: Path,
                     batch_size: int) -> np.ndarray:
    """Encode all `windows` [N, 20, 512] -> save full patch embeddings [N, 64, 1024] to
    `path` (memmapped .npy). Returns the mean-over-patches array [N, 1024]."""
    n = windows.shape[0]
    print(f"      extracting {path.name:<26} ({n} windows, batch {batch_size})...")
    mm = np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float32,
                                   shape=(n, N_PATCHES, D_MODEL))
    pooled = np.empty((n, D_MODEL), dtype=np.float32)
    t0 = time.time()
    n_batches = (n + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, n, batch_size)):
        xb = torch.from_numpy(np.ascontiguousarray(windows[i:i + batch_size])).float()
        zb = encoder(xb).cpu().numpy()                 # [b, 64, 1024]
        mm[i:i + batch_size] = zb
        pooled[i:i + batch_size] = zb.mean(axis=1)     # [b, 1024]
        if n_batches > 20 and bi % 50 == 0 and bi > 0:
            done = min(i + batch_size, n)
            print(f"        {done:>6}/{n}  ({done / n:4.0%})   {time.time() - t0:.0f}s")
    mm.flush()
    del mm
    print(f"      saved   {path.name:<26} shape [{n}, {N_PATCHES}, {D_MODEL}]  "
          f"({time.time() - t0:.1f}s)")
    return pooled


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract MOMENT embeddings + Step-1 ablation.")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--emb-dir", default=None, help="default: <data-dir>/embeddings")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=None,
                    help="cap windows per file (plumbing smoke) -> writes to embeddings_smoke/")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = None if args.device == "auto" else args.device

    proj = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else (proj / "data" / "syncan")
    if args.emb_dir:
        emb_dir = Path(args.emb_dir)
    else:
        emb_dir = data_dir / ("embeddings_smoke" if args.limit else "embeddings")
    emb_dir.mkdir(parents=True, exist_ok=True)

    def cap(a):
        return a[: args.limit] if args.limit else a

    print("=" * 68)
    print("Phase 2B — MOMENT embedding extraction + Step-1 ablation")
    print(f"  emb_dir = {emb_dir}")
    print("=" * 68)

    print("[1/4] loading SynCAN windows...")
    data = load_syncan(data_dir=args.data_dir, verbose=False)

    print("[2/4] loading frozen MOMENT-1-large...")
    enc = MOMENTEncoder(device=device)
    assert enc.trainable_param_count() == 0, "MOMENT not frozen"
    print(f"      device={enc.device}, trainable params=0")

    print("[3/4] extracting embeddings -> .npy ...")
    t_all = time.time()
    train_pooled = extract_and_save(enc, cap(data.train_windows),
                                    emb_dir / "train_embeddings.npy", args.batch_size)
    extract_and_save(enc, cap(data.val_windows),
                     emb_dir / "val_embeddings.npy", args.batch_size)
    test_pooled = {}
    for key in TEST_KEYS:
        test_pooled[key] = extract_and_save(
            enc, cap(data.test_windows[key]), emb_dir / f"test_{key}_emb.npy", args.batch_size)
    print(f"      all embeddings written in {time.time() - t_all:.1f}s")

    print("\n[4/4] Step 1 ablation — distance from normal centroid (mean-pooled)...")
    centroid = train_pooled.mean(axis=0)               # [1024]
    header = f"{'attack':<12}{'AUC-ROC':>9}{'CANet*':>9}"
    print(header)
    print("-" * len(header))
    results = {}
    for atk in ATTACK_KEYS:
        scores = np.linalg.norm(test_pooled[atk] - centroid, axis=1)   # [N]
        labels = cap(data.test_labels[atk])
        auc = eval_utils.compute_auc_roc(scores, labels)
        results[atk] = auc
        print(f"{atk:<12}{auc:>9.3f}{CANET_REF[atk]:>9.3f}")
    print("-" * len(header))
    print("* CANet = window-level baseline (reference). Step 1 is the MOMENT-only lower bound.")

    if args.limit:
        print("\n(smoke run: --limit set, skipping ablation_results.csv write)")
    else:
        csv_path = proj / "experiments" / "ablation_results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"step": "Step1_MOMENT_centroid_L2", **{k: round(results[k], 4) for k in ATTACK_KEYS}}
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df[df["step"] != row["step"]]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(csv_path, index=False)
        print(f"\nsaved Step-1 row -> {csv_path}")

    print("\nSECTION B done: 8 embedding files + Step-1 ablation.")


if __name__ == "__main__":
    main()
