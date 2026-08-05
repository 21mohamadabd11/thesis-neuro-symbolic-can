"""Phase 3, Section C — Stage-1 JEPA training loop.

Trains the (trainable) MultiResolutionJEPA to predict masked MOMENT patch embeddings
from context, on NORMAL traffic only (CLAUDE.md Stage 1). MOMENT is NOT loaded here — we
train directly on the pre-extracted embeddings, so this is fast and MOMENT stays frozen
by construction.

  * data   : pre-extracted embeddings (train_1-3 -> train, train_4 -> val), all normal.
  * model  : MultiResolutionJEPA (3 branches: short/medium/long) from jepa_module.py.
  * loss   : sum of the three branches' MSE (in 1024-d MOMENT embedding space).
  * optim  : Adam, lr=1e-3, weight_decay=1e-4.
  * output : best (lowest-val-loss) checkpoint -> experiments/jepa_checkpoint.pt.

Embeddings live OUTSIDE the project (see memory/phase2-progress); --emb-dir defaults there.

Manual run (takes a while — 30 epochs over 12k windows on the GPU):
    py -3.14 src/jepa_trainer.py --epochs 30
    py -3.14 src/jepa_trainer.py --epochs 30 --batch-size 32 --lr 5e-4
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jepa_module import MultiResolutionJEPA, N_PATCHES, MOMENT_DIM  # noqa: E402

DEFAULT_EMB_DIR = r"C:\Users\moea0\ThesisData\embeddings"
WEIGHT_DECAY = 1e-4
SEED = 42


class EmbeddingDataset(Dataset):
    """Wraps a memmapped [N, 64, 1024] .npy; yields one [64, 1024] float tensor per item."""

    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"embedding file not found: {path}")
        self.arr = np.load(str(path), mmap_mode="r")          # lazy — not all in RAM
        if self.arr.ndim != 3 or self.arr.shape[1:] != (N_PATCHES, MOMENT_DIM):
            raise ValueError(f"{path.name}: expected [N, {N_PATCHES}, {MOMENT_DIM}], "
                             f"got {self.arr.shape}")

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.arr[i], dtype=np.float32))     # [64, 1024], writable copy


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    """One pass. Loss = sum of the 3 branches' MSE. Returns window-averaged loss."""
    model.train(train)
    total, count = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:                                  # [B, 64, 1024]
            batch = batch.to(device, non_blocking=True)
            outputs = model(batch)                            # [(preds, targets, mask)] x3
            loss = sum(F.mse_loss(preds, targets) for preds, targets, _ in outputs)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = batch.shape[0]
            total += float(loss.item()) * bs
            count += bs
    return total / count


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1 JEPA training on normal MOMENT embeddings.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb-dir", default=DEFAULT_EMB_DIR)
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb_dir = Path(args.emb_dir)
    proj = Path(__file__).resolve().parent.parent
    ckpt_path = proj / "experiments" / "jepa_checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("Phase 3C — Stage-1 JEPA training (normal traffic only)")
    print(f"  emb_dir={emb_dir}\n  device={device}  epochs={args.epochs}  "
          f"batch={args.batch_size}  lr={args.lr}  weight_decay={WEIGHT_DECAY}")
    print("=" * 66)

    print("[load] train/val embeddings (memmapped)...")
    train_ds = EmbeddingDataset(emb_dir / "train_embeddings.npy")
    val_ds = EmbeddingDataset(emb_dir / "val_embeddings.npy")
    print(f"       train {len(train_ds)} windows  |  val {len(val_ds)} windows")
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin, drop_last=False)

    model = MultiResolutionJEPA(n_patches=N_PATCHES, moment_dim=MOMENT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    print(f"[model] MultiResolutionJEPA — {n_params:,} trainable params (3 branches)\n")

    best_val, best_epoch = float("inf"), -1
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, None, device, train=False)
        tag = ""
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "args": vars(args)}, ckpt_path)
            tag = "  <- best, saved"
        print(f"epoch {epoch:>3}/{args.epochs}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"({time.time() - t_ep:.1f}s){tag}")

    total_s = time.time() - t_start
    print("\n" + "=" * 66)
    print(f"DONE. best val loss = {best_val:.4f} at epoch {best_epoch}  "
          f"({best_epoch}/{args.epochs})")
    print(f"checkpoint -> {ckpt_path}")
    print(f"total training time: {total_s:.1f}s  ({total_s / 60:.1f} min)")
    print("=" * 66)


if __name__ == "__main__":
    main()
