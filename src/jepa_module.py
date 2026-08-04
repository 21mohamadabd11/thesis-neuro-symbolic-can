"""Multi-resolution JEPA prediction module — Phase 3 (Sections A + B).

A JEPA branch learns what NORMAL CAN traffic looks like in MOMENT embedding space by
predicting masked patch embeddings from the surrounding (context) patches. High
prediction error at inference = anomaly signal. All prediction happens in the 1024-d
MOMENT embedding space — never in raw signal space (CLAUDE.md design decision #2).

Section B: three parallel, independent branches over the same [B, 64, 1024] input, each
masking a contiguous block of a different size to target a different attack timescale:
  * short  (6 patches)  -> flooding  (abrupt, small window)
  * medium (24 patches) -> suppress  (medium absence)
  * long   (38 patches) -> plateau   (long freeze, ~60% of 64)

Contract:
  * input is always MOMENT patch embeddings [B, 64, 1024] (64 patches from W=512).
  * masking is I-JEPA style: one or two CONTIGUOUS blocks, never scattered patches.
  * targets are `.detach()`ed and the module detaches its input, so NO gradient can flow
    back into MOMENT (which is frozen). This file has no MOMENT / dataloader dependency —
    it operates on raw tensors only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MOMENT_DIM = 1024
N_PATCHES = 64
# per-branch contiguous-mask sizes (patches out of 64)
MASK_SHORT = 6      # flooding — abrupt, small window
MASK_MEDIUM = 24    # suppress — medium absence
MASK_LONG = 38      # plateau  — long freeze (~60% of 64)


# --------------------------------------------------------------------------- #
# 1. Masking strategy — contiguous blocks of a requested size (I-JEPA)
# --------------------------------------------------------------------------- #
def make_block_mask(n_patches: int, n_mask_patches: int) -> torch.Tensor:
    """Boolean mask [n_patches] (True = masked) covering exactly `n_mask_patches` patches
    as ONE or TWO contiguous blocks. If two blocks, the budget is split across them.
    Never masks scattered individual patches."""
    n_mask = min(max(1, int(n_mask_patches)), n_patches)
    mask = torch.zeros(n_patches, dtype=torch.bool)
    n_unmask = n_patches - n_mask

    # use two blocks when there's room; else one
    use_two = (n_mask >= 2) and (n_unmask >= 2) and (torch.rand(1).item() < 0.5)
    if not use_two:
        start = int(torch.randint(0, n_unmask + 1, (1,)).item())
        mask[start:start + n_mask] = True
        return mask

    len1 = n_mask // 2
    len2 = n_mask - len1
    # split the unmasked budget into 3 gaps g0/g1/g2 -> layout [g0][len1][g1][len2][g2]
    cuts = sorted(int(c) for c in torch.randint(0, n_unmask + 1, (2,)).tolist())
    g0, g1 = cuts[0], cuts[1] - cuts[0]
    s1 = g0
    s2 = g0 + len1 + g1
    mask[s1:s1 + len1] = True
    mask[s2:s2 + len2] = True
    return mask


# --------------------------------------------------------------------------- #
# Fixed 1D sinusoidal positional encoding
# --------------------------------------------------------------------------- #
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = N_PATCHES):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model], not trained

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pe[positions]       # [len(positions), d_model]


# --------------------------------------------------------------------------- #
# 2. Single JEPA resolution branch
# --------------------------------------------------------------------------- #
class JEPABranch(nn.Module):
    """Context encoder + predictor for one resolution. Owns its mask size
    (`n_mask_patches`) so the wrapper can request the right block size per branch.

    forward(patch_embeddings [B, N, 1024], mask [N] bool) -> predicted embeddings at the
    masked positions [B, n_masked, 1024]. Context = the unmasked patches; the predictor
    fills learnable mask tokens (with target positional encoding) from that context.
    """

    def __init__(self, moment_dim: int = MOMENT_DIM, d_model: int = 256, n_heads: int = 4,
                 dim_feedforward: int = 512, n_context_layers: int = 4,
                 n_predictor_layers: int = 2, n_patches: int = N_PATCHES,
                 n_mask_patches: int = MASK_LONG):
        super().__init__()
        self.d_model = d_model
        self.n_mask_patches = n_mask_patches                 # this branch's block size
        self.input_proj = nn.Linear(moment_dim, d_model)     # 1024 -> 256
        self.output_proj = nn.Linear(d_model, moment_dim)    # 256 -> 1024 (loss in MOMENT space)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=n_patches)

        def _encoder(n_layers: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                activation="gelu", batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, num_layers=n_layers)

        self.context_encoder = _encoder(n_context_layers)    # 4-layer
        self.predictor = _encoder(n_predictor_layers)        # 2-layer
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, patch_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = patch_embeddings.shape
        device = patch_embeddings.device
        ctx_idx = (~mask).nonzero(as_tuple=False).squeeze(1)   # unmasked positions (ascending)
        tgt_idx = mask.nonzero(as_tuple=False).squeeze(1)       # masked positions (ascending)
        n_masked = int(tgt_idx.numel())

        x = self.input_proj(patch_embeddings)                  # [B, N, d]
        x = x + self.pos_enc(torch.arange(N, device=device)).unsqueeze(0)

        context = self.context_encoder(x[:, ctx_idx, :])       # [B, N_ctx, d]

        mask_tokens = self.mask_token.expand(B, n_masked, -1)  # [B, n_masked, d]
        mask_tokens = mask_tokens + self.pos_enc(tgt_idx).unsqueeze(0)   # target positions

        pred_in = torch.cat([context, mask_tokens], dim=1)     # [B, N, d]
        pred_out = self.predictor(pred_in)[:, -n_masked:, :]   # take the mask-token outputs
        return self.output_proj(pred_out)                      # [B, n_masked, 1024]


# --------------------------------------------------------------------------- #
# 3. Multi-resolution wrapper — three independent branches
# --------------------------------------------------------------------------- #
class MultiResolutionJEPA(nn.Module):
    """Three parallel JEPA branches (short/medium/long), each with its own weights and
    its own contiguous-mask size, all consuming the same [B, 64, 1024] input."""

    def __init__(self, n_patches: int = N_PATCHES, moment_dim: int = MOMENT_DIM,
                 d_model: int = 256, n_heads: int = 4, dim_feedforward: int = 512,
                 mask_sizes: tuple[int, int, int] = (MASK_SHORT, MASK_MEDIUM, MASK_LONG)):
        super().__init__()
        self.n_patches = n_patches
        s, m, l = mask_sizes

        def _branch(n_mask: int) -> JEPABranch:
            return JEPABranch(
                moment_dim=moment_dim, d_model=d_model, n_heads=n_heads,
                dim_feedforward=dim_feedforward, n_context_layers=4,
                n_predictor_layers=2, n_patches=n_patches, n_mask_patches=n_mask)

        self.branch_short = _branch(s)
        self.branch_medium = _branch(m)
        self.branch_long = _branch(l)

    def named_branches(self) -> list[tuple[str, JEPABranch]]:
        return [("short", self.branch_short), ("medium", self.branch_medium),
                ("long", self.branch_long)]

    def forward(self, patch_embeddings: torch.Tensor):
        """[B, 64, 1024] -> list of (predictions, targets, mask), one per branch
        (order: short, medium, long). The input is detached (MOMENT is frozen), so no
        gradient can flow back into it; targets are detached too."""
        emb = patch_embeddings.detach()
        outputs = []
        for _, branch in self.named_branches():
            mask = make_block_mask(self.n_patches, branch.n_mask_patches).to(emb.device)
            predictions = branch(emb, mask)                    # [B, n_masked, 1024]
            targets = emb[:, mask, :].detach()                 # [B, n_masked, 1024]
            outputs.append((predictions, targets, mask))
        return outputs

    @torch.no_grad()
    def compute_prediction_error(self, patch_embeddings: torch.Tensor) -> dict[str, float]:
        """Per-branch scalar mean MSE (inference anomaly signal): {'short','medium','long'}."""
        emb = patch_embeddings.detach()
        errors: dict[str, float] = {}
        for name, branch in self.named_branches():
            mask = make_block_mask(self.n_patches, branch.n_mask_patches).to(emb.device)
            errors[name] = F.mse_loss(branch(emb, mask), emb[:, mask, :]).item()
        return errors


# --------------------------------------------------------------------------- #
# 4. Smoke test:  py -3.14 src/jepa_module.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, DIM = 4, N_PATCHES, MOMENT_DIM

    # random tensor simulating a batch of frozen-MOMENT embeddings; requires_grad=True so
    # we can prove no gradient leaks back into it.
    x = torch.randn(B, N, DIM, requires_grad=True)
    jepa = MultiResolutionJEPA(n_patches=N, moment_dim=DIM)

    single = sum(p.numel() for p in jepa.branch_short.parameters())
    total = sum(p.numel() for p in jepa.parameters())
    print(f"[smoke] input {tuple(x.shape)}  |  per-branch {single:,}  x3 = total {total:,}  "
          "(expect ~11M)")

    outputs = jepa(x)  # list of (preds, targets, mask): short, medium, long
    expected = [("short", MASK_SHORT), ("medium", MASK_MEDIUM), ("long", MASK_LONG)]
    total_loss = torch.zeros(())
    for (name, exp_mask), (preds, targets, mask) in zip(expected, outputs):
        n_masked = int(mask.sum())
        n_blocks = int(((mask[1:].int() - mask[:-1].int()) == 1).sum()) + int(mask[0].item())
        bloss = F.mse_loss(preds, targets)
        total_loss = total_loss + bloss
        print(f"[smoke] {name:<7}: mask {n_masked:>2}/{N} ({n_blocks} block(s)), "
              f"preds {tuple(preds.shape)}, MSE {bloss.item():.4f}")
        assert n_masked == exp_mask, f"{name}: masked {n_masked}, expected {exp_mask}"
        assert preds.shape == (B, n_masked, DIM)
        assert 1 <= n_blocks <= 2

    print(f"[smoke] total loss (sum of 3 branches) = {total_loss.item():.4f}")
    total_loss.backward()

    print(f"[smoke] input.grad is None: {x.grad is None}   (MUST be True — MOMENT frozen)")
    for name, br in jepa.named_branches():
        g = br.input_proj.weight.grad
        print(f"[smoke]   {name:<7} branch trained: {g is not None and float(g.abs().sum()) > 0}")

    errs = jepa.compute_prediction_error(x.detach())
    print(f"[smoke] compute_prediction_error() -> "
          f"{{'short': {errs['short']:.4f}, 'medium': {errs['medium']:.4f}, 'long': {errs['long']:.4f}}}")

    assert len(outputs) == 3
    assert set(errs) == {"short", "medium", "long"}
    assert total == 3 * single, "total params should be exactly 3x a single branch"
    assert x.grad is None, "FAIL: gradient leaked into the input (MOMENT not frozen)"
    print("\nSMOKE TEST PASSED: 3-branch multi-resolution JEPA — per-branch masked "
          "prediction + errors, ~11M params, no grad leak into input.")
