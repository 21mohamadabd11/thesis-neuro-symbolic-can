"""Phase 4, Section A — soft prototype (concept) bottleneck + initialisation.

Pipeline position (CLAUDE.md): the bottleneck clusters the normal-traffic latent space into
K human-inspectable regimes. Each window is soft-assigned to the K prototypes;
``concept_rarity = 1 - max_k s_k`` is the Step-5 anomaly signal (0 = matches a known normal
regime, ->1 = matches none).

Why this exists / what Phase 3 proved: JEPA *prediction error* was a null anomaly signal
(Step-4 AUC ~= 0.50) because MOMENT patches are locally smooth and interpolate equally well
on normal and attack windows. The hypothesis for Phase 4 is that the anomaly lives in the
embedding VALUES, not their local predictability — so we cluster the embedding VALUES and
score how far a window falls from every learned normal prototype.

Contents
--------
* ``PrototypeBottleneck``   — K prototypes, soft assignment, concept rarity, diversity loss.
* ``extract_moment_latents`` — window latents for clustering: mean-pool the raw MOMENT patch
  embeddings [N, 64, 1024] -> [N, 1024], then PCA -> [N, 256]. The fitted PCA is returned
  (and optionally pickled) so the SAME projection is reused on val/test at inference.
* ``calibrate_tau``         — data-driven softmax temperature (see design notes).

Design notes
------------
* **We cluster PCA-reduced MOMENT embeddings, NOT JEPA latents.** Section A first tried the
  JEPA medium-branch context latents (mean-pool): they COLLAPSED — all 12126 windows mapped
  to ~one point, so the 16 prototypes were identical and concept_rarity was a constant 0.9375
  (= 1 - 1/K). Raw MOMENT embeddings, by contrast, demonstrably vary across windows (Phase 2
  t-SNE + non-trivial Step-1 AUC). So the concept layer clusters mean-pooled MOMENT
  embeddings, dimensionality-reduced by PCA to 256. **NOTE:** this deviates from the CLAUDE.md
  design (concept bottleneck over *JEPA* latents) — flagged for the thesis narrative and the
  Step-5 ablation (JEPA no longer feeds the concept layer).
* **PCA is fitted on training only and pickled** (``pca_path``); reuse it via ``pca=`` on
  val/test so the projection is identical (no leakage).
* **tau is calibrated to the data**, not fixed: tau = 25th percentile of pairwise squared
  distances on a random 1000-window subset. This scales the softmax to the actual latent
  geometry so assignments are peaked-but-not-saturated (a fixed tau=1.0 over 256-D distances
  either saturates or flattens concept_rarity).

Manual run (loads train embeddings, mean-pool + PCA, fits k-means, checks non-degeneracy):
    py -3.14 src/concept_bottleneck.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_EMB_DIR = r"C:\Users\moea0\ThesisData\embeddings"
KMEANS_SEED = 42
N_PATCHES = 64          # patches per window (W=512 / patch_len=8)
MOMENT_DIM = 1024       # MOMENT-1-large embedding dim
# (the concept layer is independent of the JEPA model now — no jepa_module import)


# --------------------------------------------------------------------------- #
# 1. Soft prototype bottleneck
# --------------------------------------------------------------------------- #
class PrototypeBottleneck(nn.Module):
    """K learnable prototype vectors in the concept latent space with soft assignment.

    forward(z [B, d_model]) -> (s, concept_rarity):
      * s              [B, K] : soft assignment, ``softmax(-||z - p_k||^2 / tau)`` (rows sum to 1)
      * concept_rarity [B]    : ``1 - max_k s_k`` in [0, 1)  (0 = strong match, ->1 = no match)
    """

    def __init__(self, K: int, d_model: int = 256, tau: float = 1.0):
        super().__init__()
        self.K = int(K)
        self.d_model = int(d_model)
        self.tau = float(tau)
        # small random init so the module is usable before init_from_kmeans overwrites it
        self.prototypes = nn.Parameter(torch.randn(self.K, self.d_model) * 0.02)  # [K, d_model]

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # squared Euclidean distance to each prototype: ||z - p_k||^2  -> [B, K]
        # (z: [B, 1, d] - p: [1, K, d]) ** 2 summed over d
        dist_sq = ((z.unsqueeze(1) - self.prototypes.unsqueeze(0)) ** 2).sum(dim=2)   # [B, K]
        s = F.softmax(-dist_sq / self.tau, dim=1)                                     # [B, K]
        concept_rarity = 1.0 - s.max(dim=1).values                                    # [B]
        return s, concept_rarity

    @torch.no_grad()
    def init_from_kmeans(self, embeddings_np: np.ndarray):
        """Initialise the K prototypes from k-means++ centroids of `embeddings_np` [N, d_model].
        Returns the fitted sklearn KMeans (for inspecting inertia / cluster sizes)."""
        from sklearn.cluster import KMeans

        emb = np.asarray(embeddings_np, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.d_model:
            raise ValueError(f"expected [N, {self.d_model}] latents, got {emb.shape}")

        km = KMeans(n_clusters=self.K, init="k-means++", n_init=10, random_state=KMEANS_SEED)
        km.fit(emb)
        centroids = torch.from_numpy(km.cluster_centers_).float().to(self.prototypes.device)
        self.prototypes.copy_(centroids)                     # [K, d_model]
        return km

    def L_diversity(self) -> torch.Tensor:
        """Repulsion loss ``-mean(||p_i - p_j||^2 for i != j)``. Minimising it (more negative)
        pushes the prototypes apart, preventing collapse (CLAUDE.md Concept Bottleneck Loss)."""
        p = self.prototypes                                                          # [K, d]
        dist_sq = ((p.unsqueeze(1) - p.unsqueeze(0)) ** 2).sum(dim=2)                 # [K, K], diag = 0
        if self.K < 2:
            return p.sum() * 0.0                                                      # no pairs -> 0
        mean_offdiag = dist_sq.sum() / (self.K * (self.K - 1))                        # diag contributes 0
        return -mean_offdiag


# --------------------------------------------------------------------------- #
# 2. Data-driven softmax temperature
# --------------------------------------------------------------------------- #
def calibrate_tau(latents: np.ndarray, n_sample: int = 1000, percentile: float = 25.0,
                  seed: int = 0) -> float:
    """Softmax temperature = `percentile`-th percentile of pairwise SQUARED distances on a
    random `n_sample`-window subset of `latents`. Scales the softmax to the latent geometry so
    concept assignments are peaked but not saturated."""
    from scipy.spatial.distance import pdist

    rng = np.random.RandomState(seed)
    m = min(int(n_sample), len(latents))
    sub = np.asarray(latents[rng.choice(len(latents), size=m, replace=False)], dtype=np.float64)
    d2 = pdist(sub, metric="sqeuclidean")                    # condensed [m*(m-1)/2]
    return float(max(np.percentile(d2, percentile), 1e-8))   # floor guards against a degenerate 0


# --------------------------------------------------------------------------- #
# 3. Concept latents from raw MOMENT embeddings (mean-pool + PCA) — no JEPA
# --------------------------------------------------------------------------- #
def extract_moment_latents(emb_dir=DEFAULT_EMB_DIR, n_components: int = 256,
                           emb_file: str = "train_embeddings.npy", batch_size: int = 512,
                           pca=None, pca_path=None):
    """Window latents for the concept layer, straight from RAW MOMENT embeddings (no JEPA):

      1. load `emb_file` [N, 64, 1024] (memmapped) from `emb_dir`,
      2. mean-pool the 64 patches -> [N, 1024],
      3. PCA -> [N, n_components].

    If `pca` is None a new PCA is fitted on this data (training) and, when `pca_path` is
    given, pickled there so the SAME projection can be reused at inference (pass it back via
    `pca=`). If `pca` is provided, it is only applied (transform) — use this for val/test.

    Returns ``(latents [N, n_components] float32, fitted-or-reused PCA)``.
    """
    from sklearn.decomposition import PCA

    emb_path = Path(emb_dir) / emb_file
    if not emb_path.exists():
        raise FileNotFoundError(f"embedding file not found: {emb_path}")
    arr = np.load(str(emb_path), mmap_mode="r")              # [N, 64, 1024], lazy
    if arr.ndim != 3 or arr.shape[1:] != (N_PATCHES, MOMENT_DIM):
        raise ValueError(f"{emb_path.name}: expected [N, {N_PATCHES}, {MOMENT_DIM}], got {arr.shape}")
    n = arr.shape[0]

    # (2) mean-pool patches -> [N, 1024], in batches (avoid loading the whole 3+ GB file)
    pooled = np.empty((n, MOMENT_DIM), dtype=np.float32)
    for i in range(0, n, batch_size):
        pooled[i:i + batch_size] = np.asarray(arr[i:i + batch_size], dtype=np.float32).mean(axis=1)

    # (3) PCA -> [N, n_components]  (fit on training if none supplied, else reuse)
    if pca is None:
        pca = PCA(n_components=n_components, random_state=42)
        latents = pca.fit_transform(pooled).astype(np.float32)
        if pca_path is not None:
            Path(pca_path).parent.mkdir(parents=True, exist_ok=True)
            with open(pca_path, "wb") as f:
                pickle.dump(pca, f)
    else:
        latents = pca.transform(pooled).astype(np.float32)
    return latents, pca


# --------------------------------------------------------------------------- #
# 4. Smoke test:  py -3.14 src/concept_bottleneck.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    try:  # so any non-ASCII prints cleanly on the Windows console
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proj = Path(__file__).resolve().parent.parent
    emb_dir = DEFAULT_EMB_DIR
    pca_path = proj / "experiments" / "concept_pca.pkl"

    print("=" * 72)
    print("Phase 4A — prototype bottleneck: MOMENT+PCA latents + k-means++ init")
    print(f"  emb_dir={emb_dir}  device={device}")
    print("=" * 72)

    # (1) MOMENT+PCA window latents -> [N, 256]
    print("[1] extracting MOMENT latents (mean-pool 64 patches -> [N,1024]) + PCA -> [N,256]...")
    latents, pca = extract_moment_latents(emb_dir=emb_dir, n_components=256, pca_path=pca_path)
    D = latents.shape[1]
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"    latents shape = {latents.shape}  (expect [12126, 256])")
    print(f"    latent stats: mean={latents.mean():.4f}  std={latents.std():.4f}")
    print(f"    PCA explained variance (256 comps) = {evr:.3f}   PCA pickled -> {pca_path.name}")
    print(f"    between-window std (mean over dims) = {latents.std(axis=0).mean():.4f}  "
          f"(mean-pooled JEPA latents were ~0.0002 -> collapse)")
    assert latents.ndim == 2 and D == 256, f"unexpected latent shape {latents.shape}"

    # (2) calibrate tau = 25th pct of pairwise squared distances on 1000 random windows
    tau = calibrate_tau(latents, n_sample=1000, percentile=25.0, seed=0)
    print(f"\n[2] tau (25th pct pairwise sq-dist on 1000-window subset) = {tau:.4f}")

    # (3) fit k-means++ K=16 and initialise the bottleneck
    K = 16
    print(f"\n[3] fitting k-means++ (K={K}) and initialising PrototypeBottleneck(d_model={D}, tau={tau:.3f})...")
    bottleneck = PrototypeBottleneck(K=K, d_model=D, tau=tau).to(device)
    km = bottleneck.init_from_kmeans(latents)
    sizes = np.bincount(km.labels_, minlength=K)
    total_ss = float(latents.var(axis=0).sum() * len(latents))
    print(f"    k-means inertia = {km.inertia_:.1f}   (total between-window SS = {total_ss:.1f};  "
          f"ratio {km.inertia_ / max(total_ss, 1e-9):.3f})")
    print(f"    cluster sizes   = {sizes.tolist()}  (sum={int(sizes.sum())})")

    # (4) forward pass on a random batch -> shapes of s and concept_rarity
    print(f"\n[4] forward pass on a random batch [8, {D}]...")
    z_rand = torch.randn(8, D, device=device)
    s, rarity = bottleneck(z_rand)
    print(f"    s shape={tuple(s.shape)} (expect (8, {K}))   "
          f"concept_rarity shape={tuple(rarity.shape)} (expect (8,))")
    print(f"    s row sums (~1.0) = {np.round(s.sum(dim=1).detach().cpu().numpy(), 3).tolist()}")

    # (5) concept_rarity over 2000 REAL windows — must NOT be all identical (the whole point)
    idx = np.random.RandomState(0).choice(len(latents), size=2000, replace=False)
    _, rarity_real = bottleneck(torch.from_numpy(latents[idx]).to(device))
    rr = rarity_real.detach().cpu().numpy()
    print(f"\n[5] concept_rarity over 2000 real windows: "
          f"min={rr.min():.4f}  mean={rr.mean():.4f}  max={rr.max():.4f}  std={rr.std():.4f}")
    assert rr.std() > 0.01, (f"FAIL: concept_rarity is ~constant (std={rr.std():.5f}) — the "
                             f"bottleneck is still degenerate")

    # (6) diversity loss + prototype separation
    div = bottleneck.L_diversity()
    protos = bottleneck.prototypes.detach()
    pair = ((protos.unsqueeze(1) - protos.unsqueeze(0)) ** 2).sum(dim=2)
    eye = torch.eye(K, dtype=torch.bool, device=protos.device)
    offdiag = pair[~eye]
    print(f"\n[6] L_diversity = {div.item():.2f}   prototype ||p_i - p_j||^2 (i!=j): "
          f"min={offdiag.min().item():.3f}  mean={offdiag.mean().item():.3f}  max={offdiag.max().item():.3f}")
    assert offdiag.min().item() > 1e-8, "FAIL: two prototypes are identical (k-means collapse)"

    print("\n" + "=" * 72)
    print(f"SMOKE TEST PASSED: MOMENT+PCA latents [{len(latents)}, 256], tau={tau:.3f}, k-means++ init, "
          f"concept_rarity VARIES (std={rr.std():.3f}) -> bottleneck non-degenerate.")
    print("=" * 72)
