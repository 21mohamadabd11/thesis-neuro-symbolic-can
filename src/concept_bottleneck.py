"""Phase 4, Section A — soft prototype (concept) bottleneck + initialisation.

Pipeline position (CLAUDE.md): the bottleneck sits on top of the JEPA latent space and
clusters it into K human-inspectable regimes of NORMAL CAN behaviour. Each window is
soft-assigned to the K prototypes; ``concept_rarity = 1 - max_k s_k`` is the Step-5
anomaly signal (0 = window matches a known normal regime, ->1 = matches none).

Why this exists / what Phase 3 proved: JEPA *prediction error* was a null anomaly signal
(Step-4 AUC ~= 0.50) because MOMENT patches are locally smooth and interpolate equally well
on normal and attack windows. The hypothesis for Phase 4 is that the anomaly lives in the
embedding VALUES, not their local predictability — so we cluster the JEPA latent VALUES and
score how far a window falls from every learned normal prototype.

Contents
--------
* ``PrototypeBottleneck``  — K prototypes, soft assignment, concept rarity, diversity loss.
* ``extract_jepa_latents`` — window-level latents from the trained JEPA (medium branch
  context encoder, no masking, no predictor), used to k-means++ initialise the prototypes.

Design notes
------------
* **Window latent = mean-pool over patches.** The context encoder outputs [B, 64, 256];
  we mean-pool over the 64 patches to a single [B, 256] window representation. Clustering is
  therefore over window-level regimes, matching PrototypeBottleneck.forward's [B, d_model]
  input and the required [N, 256] extraction shape.
* **Medium branch.** Per CLAUDE.md's resolution table the medium branch targets suppress /
  mid-scale structure; it is a reasonable single-branch latent for the first concept pass.
  (Multi-branch concept fusion, if needed, is a later refinement.)
* **tau caveat.** Squared distances are summed over 256 dims, so ``||z - p_k||^2`` can be
  large and ``tau=1.0`` may make the softmax very peaked on real latents. tau is left as a
  tunable knob here (Section A only initialises); calibrate it in Sections B/C.

Manual run (extracts latents from the real checkpoint + embeddings, fits k-means):
    py -3.14 src/concept_bottleneck.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jepa_module import MultiResolutionJEPA, N_PATCHES, MOMENT_DIM  # noqa: E402

DEFAULT_EMB_DIR = r"C:\Users\moea0\ThesisData\embeddings"
KMEANS_SEED = 42


# --------------------------------------------------------------------------- #
# 1. Soft prototype bottleneck
# --------------------------------------------------------------------------- #
class PrototypeBottleneck(nn.Module):
    """K learnable prototype vectors in JEPA latent space with soft (softmax) assignment.

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
# 2. JEPA latent extraction (medium branch context encoder — no mask, no predictor)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_jepa_latents(checkpoint_path, emb_dir, device, batch_size: int = 64) -> np.ndarray:
    """Window-level JEPA latents for every normal training window.

    Loads MultiResolutionJEPA from `checkpoint_path`, then for each window in
    train_embeddings.npy [N, 64, 1024] runs ONLY the medium branch's context-encoder path
    (input_proj + positional encoding + context_encoder over ALL 64 patches — no masking,
    no predictor) and mean-pools over the 64 patches -> one [256] latent per window.

    Returns np.ndarray [N, 256] (float32).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    emb_path = Path(emb_dir) / "train_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"embedding file not found: {emb_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model = MultiResolutionJEPA(n_patches=N_PATCHES, moment_dim=MOMENT_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    branch = model.branch_medium                             # medium resolution branch
    d_model = branch.d_model

    arr = np.load(str(emb_path), mmap_mode="r")              # [N, 64, 1024], lazy
    if arr.ndim != 3 or arr.shape[1:] != (N_PATCHES, MOMENT_DIM):
        raise ValueError(f"{emb_path.name}: expected [N, {N_PATCHES}, {MOMENT_DIM}], got {arr.shape}")
    n = arr.shape[0]
    out = np.empty((n, d_model), dtype=np.float32)

    # positional encoding for all 64 patches, added exactly as JEPABranch.forward does
    pos = branch.pos_enc(torch.arange(N_PATCHES, device=device)).unsqueeze(0)   # [1, 64, d]

    for i in range(0, n, batch_size):
        xb = torch.from_numpy(np.array(arr[i:i + batch_size], dtype=np.float32)).to(device)  # [b, 64, 1024]
        x = branch.input_proj(xb) + pos                      # [b, 64, d]
        context = branch.context_encoder(x)                  # [b, 64, d]  (all patches, unmasked)
        latent = context.mean(dim=1)                         # [b, d]  window-level latent (mean-pool)
        out[i:i + batch_size] = latent.cpu().numpy()

    return out


# --------------------------------------------------------------------------- #
# 3. Smoke test:  py -3.14 src/concept_bottleneck.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proj = Path(__file__).resolve().parent.parent
    ckpt_path = proj / "experiments" / "jepa_checkpoint.pt"
    emb_dir = DEFAULT_EMB_DIR

    print("=" * 66)
    print("Phase 4A — prototype bottleneck: latent extraction + k-means++ init")
    print(f"  checkpoint={ckpt_path.name}  emb_dir={emb_dir}  device={device}")
    print("=" * 66)

    # (1) extract JEPA latents  -> [N, 256]
    print("[1] extracting JEPA latents (medium branch context encoder, no mask)...")
    latents = extract_jepa_latents(ckpt_path, emb_dir, device)
    d_model = latents.shape[1]
    print(f"    latents shape = {latents.shape}  (expect [12126, 256])")
    print(f"    latent stats: mean={latents.mean():.4f}  std={latents.std():.4f}")
    assert latents.ndim == 2 and d_model == 256, f"unexpected latent shape {latents.shape}"

    # (2) fit k-means++ K=16 and initialise the bottleneck
    K = 16
    print(f"\n[2] fitting k-means++ (K={K}) and initialising PrototypeBottleneck...")
    bottleneck = PrototypeBottleneck(K=K, d_model=d_model, tau=1.0).to(device)
    km = bottleneck.init_from_kmeans(latents)
    sizes = np.bincount(km.labels_, minlength=K)
    print(f"    k-means inertia = {km.inertia_:.1f}")
    print(f"    cluster sizes   = {sizes.tolist()}  (sum={int(sizes.sum())})")

    # (3) forward pass on a random batch -> shapes of s and concept_rarity
    print("\n[3] forward pass on a random batch [8, 256]...")
    z_rand = torch.randn(8, d_model, device=device)
    s, rarity = bottleneck(z_rand)
    print(f"    s shape             = {tuple(s.shape)}   (expect (8, {K}))")
    print(f"    concept_rarity shape= {tuple(rarity.shape)}     (expect (8,))")
    print(f"    s row sums (~1.0)   = {np.round(s.sum(dim=1).detach().cpu().numpy(), 4).tolist()}")
    print(f"    concept_rarity      = {np.round(rarity.detach().cpu().numpy(), 4).tolist()}")

    # sanity: on REAL latents (near their own centroids) rarity should be lower than on noise
    idx = np.random.RandomState(0).choice(len(latents), size=8, replace=False)
    z_real = torch.from_numpy(latents[idx]).to(device)
    _, rarity_real = bottleneck(z_real)
    print(f"    rarity on real latents (should trend lower) = "
          f"{np.round(rarity_real.detach().cpu().numpy(), 4).tolist()}")

    # (4) diversity loss value
    div = bottleneck.L_diversity()
    print(f"\n[4] L_diversity = {div.item():.4f}  (negative; more-negative = prototypes further apart)")

    # (5) confirm prototypes are not all identical (no k-means collapse)
    protos = bottleneck.prototypes.detach()
    pair = ((protos.unsqueeze(1) - protos.unsqueeze(0)) ** 2).sum(dim=2)              # [K, K]
    eye = torch.eye(K, dtype=torch.bool, device=protos.device)
    offdiag = pair[~eye]
    print(f"\n[5] prototype separation ||p_i - p_j||^2 (i!=j): "
          f"min={offdiag.min().item():.4f}  mean={offdiag.mean().item():.4f}  max={offdiag.max().item():.4f}")
    assert offdiag.min().item() > 1e-8, "FAIL: two prototypes are identical (k-means collapse)"

    print("\nSMOKE TEST PASSED: latents [N,256] extracted, k-means++ init, soft assignment + "
          "concept rarity + diversity loss all working, prototypes distinct.")
