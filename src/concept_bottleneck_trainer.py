"""Phase 4, Section B — Stage-2 concept bottleneck training loop.

Trains the K prototype vectors of the concept bottleneck (from concept_bottleneck.py) on
NORMAL-traffic latents (Stage 2). Everything upstream is FROZEN:
  * MOMENT is never touched in this stage.
  * the PCA is FIXED — loaded from experiments/concept_pca.pkl (fitted in Section A).
  * only the K prototype vectors are trainable.

Pipeline (Section-A pivot): raw MOMENT embeddings -> mean-pool 64 patches -> [N,1024] ->
fixed PCA -> [N,256] -> PrototypeBottleneck. concept_rarity = 1 - max_k s_k is the Step-5
anomaly signal.

Loss (exactly as specified):
    L_recon     = MSE(z, sum_k s_k p_k)      # reconstruct z as the soft weighted sum of prototypes
    L_diversity = -mean(||p_i - p_j||^2)      # repulsion — prevents prototype collapse
    L_total     = L_recon + 0.1 * L_diversity

Unsupervised: NO attack labels in training. After training, concept_rarity is measured on the
normal val set and all 5 attack test sets as a PREVIEW of Step-5 separation, and the six
per-window rarity arrays are saved for the Section-F Step-5 ablation.

"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concept_bottleneck import (  # noqa: E402
    PrototypeBottleneck, extract_moment_latents, DEFAULT_EMB_DIR,
)

ATTACKS = ["plateau", "suppress", "flooding", "continuous", "playback"]
SEED = 42


def resolve_attack_file(emb_dir: Path, attack: str) -> Path:
    """Embedding file for `attack`. Tries the Section-B spec name first, then the name the
    Phase-3 inference code used (known to exist). Raises if neither is present."""
    candidates = [f"{attack}_embeddings.npy", f"test_{attack}_emb.npy"]
    for c in candidates:
        if (emb_dir / c).exists():
            return emb_dir / c
    raise FileNotFoundError(f"no embedding file for '{attack}' — tried {candidates} in {emb_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 concept bottleneck training (normal only).")
    ap.add_argument("--emb-dir", default=DEFAULT_EMB_DIR)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.045)
    ap.add_argument("--lambda-div", type=float, default=0.001)
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    args = ap.parse_args()

    try:  # so any non-ASCII prints cleanly on the Windows console
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb_dir = Path(args.emb_dir)
    proj = Path(__file__).resolve().parent.parent
    pca_path = proj / "experiments" / "concept_pca.pkl"
    ckpt_path = proj / "experiments" / "concept_bottleneck_checkpoint.pt"
    rarity_dir = proj / "experiments" / "concept_rarity_arrays"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("Phase 4B — Stage-2 concept bottleneck training (normal traffic only)")
    print(f"  emb_dir={emb_dir}  device={device}")
    print(f"  K={args.K}  tau={args.tau}  epochs={args.epochs}  batch={args.batch_size}  "
          f"lr={args.lr}  wd={args.weight_decay}  lambda_div={args.lambda_div}")
    print("=" * 74)

    # ---- data: fixed PCA -> normal train/val latents (unsupervised, normal only) ----
    if not pca_path.exists():
        raise FileNotFoundError(f"fitted PCA not found: {pca_path} (run Section A first)")
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    print("[data] applying the FIXED Section-A PCA to normal train/val embeddings...")
    train_latents, _ = extract_moment_latents(emb_dir, emb_file="train_embeddings.npy", pca=pca)
    val_latents, _ = extract_moment_latents(emb_dir, emb_file="val_embeddings.npy", pca=pca)
    print(f"       train latents {train_latents.shape}  |  val latents {val_latents.shape}")

    train_t = torch.from_numpy(train_latents).to(device)          # [N, 256]
    val_t = torch.from_numpy(val_latents).to(device)              # [M, 256]
    n = train_t.shape[0]

    # ---- model: bottleneck, k-means++ init (same as Section A); prototypes are the only params ----
    bottleneck = PrototypeBottleneck(K=args.K, d_model=train_t.shape[1], tau=args.tau).to(device)
    print(f"[model] initialising prototypes with k-means++ on {n} train latents...")
    bottleneck.init_from_kmeans(train_latents)
    optimizer = torch.optim.Adam(bottleneck.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_trainable = sum(p.numel() for p in bottleneck.parameters() if p.requires_grad)
    print(f"        trainable params = {n_trainable:,} (prototype vectors only; PCA + MOMENT frozen)\n")

    def recon_loss(z: torch.Tensor) -> torch.Tensor:
        s, _ = bottleneck(z)                                     # [b, K]
        z_recon = s @ bottleneck.prototypes                     # [b, K] @ [K, d] -> [b, d]
        return F.mse_loss(z, z_recon)

    # ---- training ----
    best_val_recon, best_epoch = float("inf"), -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        bottleneck.train()
        perm = torch.randperm(n, device=device)
        tr_recon = tr_div = tr_total = 0.0
        nb = 0
        for i in range(0, n, args.batch_size):
            z = train_t[perm[i:i + args.batch_size]]
            L_recon = recon_loss(z)
            L_div = bottleneck.L_diversity()
            L_total = L_recon + args.lambda_div * L_div
            optimizer.zero_grad(set_to_none=True)
            L_total.backward()
            optimizer.step()
            tr_recon += L_recon.item(); tr_div += L_div.item(); tr_total += L_total.item(); nb += 1
        tr_recon /= nb; tr_div /= nb; tr_total /= nb

        bottleneck.eval()
        with torch.no_grad():
            s_val, rarity_val = bottleneck(val_t)
            val_recon = F.mse_loss(val_t, s_val @ bottleneck.prototypes).item()
            vr_mean = rarity_val.mean().item(); vr_std = rarity_val.std().item()

        tag = ""
        if val_recon < best_val_recon:
            best_val_recon, best_epoch = val_recon, epoch
            torch.save({"prototypes": bottleneck.prototypes.detach().cpu(),
                        "tau": bottleneck.tau, "K": bottleneck.K, "d_model": bottleneck.d_model,
                        "epoch": epoch, "val_recon": val_recon}, ckpt_path)
            tag = "  <- best, saved"
        print(f"epoch {epoch:>3}/{args.epochs}  train: L_recon {tr_recon:.5f}  L_div {tr_div:.4f}  "
              f"L_total {tr_total:.5f}  | val: L_recon {val_recon:.5f}  "
              f"rarity {vr_mean:.4f}±{vr_std:.4f}{tag}")

    print(f"\n[train] done in {time.time() - t0:.1f}s. best val L_recon {best_val_recon:.5f} "
          f"at epoch {best_epoch}")
    print(f"        checkpoint -> {ckpt_path}")

    # ---- reload best checkpoint (prototypes + tau + K) ----
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    best = PrototypeBottleneck(K=ck["K"], d_model=ck["d_model"], tau=ck["tau"]).to(device)
    with torch.no_grad():
        best.prototypes.copy_(ck["prototypes"].to(device))
    best.eval()
    print(f"\n[eval] loaded best checkpoint (epoch {ck['epoch']}, val L_recon {ck['val_recon']:.5f})")

    @torch.no_grad()
    def rarity_of(latents_np: np.ndarray) -> np.ndarray:
        _, r = best(torch.from_numpy(latents_np).to(device))
        return r.cpu().numpy()

    # normal val
    rr_val = rarity_of(val_latents)
    print(f"[eval] val (normal): mean {rr_val.mean():.4f}  std {rr_val.std():.4f}  "
          f"p95 {np.percentile(rr_val, 95):.4f}")

    # 5 attack test sets (same fixed PCA; unsupervised — labels never used, just measured)
    rarity_arrays = {"normal_val": rr_val}
    for atk in ATTACKS:
        path = resolve_attack_file(emb_dir, atk)
        atk_lat, _ = extract_moment_latents(emb_dir, emb_file=path.name, pca=pca)
        rarity_arrays[atk] = rarity_of(atk_lat)
        print(f"       {atk:<11} <- {path.name}  ({atk_lat.shape[0]} windows)")

    # ---- save the six per-window rarity arrays for the Step-5 ablation (Section F) ----
    rarity_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in rarity_arrays.items():
        np.save(rarity_dir / f"{name}.npy", arr)
    print(f"\n[save] 6 concept_rarity arrays -> {rarity_dir}")

    # ---- summary table: Step-5 preview ----
    normal_mean = float(rarity_arrays["normal_val"].mean())
    print("\n" + "=" * 58)
    print("STEP-5 PREVIEW — mean concept_rarity per split")
    print("-" * 58)
    print(f"{'split':<14}{'mean':>9}{'std':>9}{'delta vs normal':>18}")
    for name in ["normal_val"] + ATTACKS:
        a = rarity_arrays[name]
        print(f"{name:<14}{a.mean():>9.4f}{a.std():>9.4f}{a.mean() - normal_mean:>+18.4f}")
    print("-" * 58)
    print("(attacks with mean rarity ABOVE normal -> Step-5 will carry signal)")
    print("=" * 58)

    print("\nPhase 4B COMPLETE")


if __name__ == "__main__":
    main()
