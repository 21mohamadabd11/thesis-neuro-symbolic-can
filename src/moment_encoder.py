"""Frozen MOMENT-1-large encoder wrapper (Phase 2, Section A).

Wraps AutonLab/MOMENT-1-large (a pre-trained T5-based time-series transformer) as a
FROZEN feature extractor. All parameters are frozen immediately after loading and a
hard assertion guarantees zero trainable parameters — MOMENT is NEVER trained in this
thesis (see CLAUDE.md key design decision #1).

    encoder = MOMENTEncoder()
    z = encoder(x)          # x: [B, 20, 512]  ->  z: [B, 64, 1024]  (patch embeddings)

Shapes: W=512, patch_len=8 -> N_patches=64; MOMENT-1-large d_model=1024.

Channel handling: MOMENT is channel-independent — a [B, 20, 512] input is processed as
20 univariate series, giving per-channel patch reps [B, 20, 64, 1024]. To produce the
[B, 64, 1024] tensor the JEPA stage consumes (CLAUDE.md core architecture), we mean-pool
over the 20 channels. (Flagged as a design choice — alternatives: keep channels, or
concatenate — but the spec fixes the output at [B, 64, 1024].)

Note: MOMENT-1-large is ~1.5 GB — the first run downloads and caches it (HuggingFace
cache). That is expected and one-time.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from momentfm import MOMENTPipeline  # noqa: E402

W = 512
PATCH_LEN = 8
N_PATCHES = W // PATCH_LEN     # 64
D_MODEL = 1024                 # MOMENT-1-large
N_CHANNELS = 20                # SynCAN signals
MODEL_NAME = "AutonLab/MOMENT-1-large"


class MOMENTEncoder(nn.Module):
    """Frozen MOMENT-1-large. forward([B, 20, 512]) -> patch embeddings [B, 64, 1024]."""

    def __init__(self, device: str | None = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.moment = MOMENTPipeline.from_pretrained(
            MODEL_NAME, model_kwargs={"task_name": "embedding"},
        )
        self.moment.init()  # finalize model setup (per momentfm examples)

        # ---- FREEZE everything immediately, then hard-verify (design decision #1) ----
        for param in self.moment.parameters():
            param.requires_grad = False
        self.moment.eval().to(self.device)
        n_trainable = self.trainable_param_count()
        assert n_trainable == 0, f"MOMENT is not fully frozen ({n_trainable} trainable params)"

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.moment.parameters() if p.requires_grad)

    @torch.no_grad()  # MOMENT is frozen — never need gradients through it
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N_CHANNELS, W] -> patch embeddings [B, N_PATCHES, D_MODEL]."""
        if x.dim() != 3 or x.shape[1] != N_CHANNELS or x.shape[2] != W:
            raise ValueError(f"expected input [B, {N_CHANNELS}, {W}], got {tuple(x.shape)}")
        x = x.to(self.device, dtype=torch.float32)
        emb = self._encode_patches(x)                 # [B, 64, 1024]
        return emb

    def _encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-patch embeddings [B, 64, 1024], collapsing MOMENT's channel dim."""
        B = x.shape[0]
        # The default 'embedding' task pools over patches; request the un-reduced output
        # so the patch dimension survives. Fall back to the default call if unsupported.
        try:
            out = self.moment(x_enc=x, reduction="none")
        except TypeError:
            out = self.moment(x_enc=x)
        emb = getattr(out, "embeddings", None)
        if emb is None:
            raise RuntimeError(f"MOMENT output has no `.embeddings` (type={type(out)})")
        return self._as_patch_embeddings(emb, B)

    @staticmethod
    def _as_patch_embeddings(emb: torch.Tensor, B: int) -> torch.Tensor:
        """Normalize whatever MOMENT returns into [B, N_PATCHES, D_MODEL]."""
        if emb.dim() == 4:                                   # [B, C, P, D]
            emb = emb.mean(dim=1)                            # -> [B, P, D]
        elif emb.dim() == 3 and emb.shape[1] == N_CHANNELS and emb.shape[2] == D_MODEL:
            raise RuntimeError(
                f"MOMENT .embeddings is per-channel pooled {tuple(emb.shape)} (no patch "
                "dim) — per-patch reps not exposed here; encoder-level access needed.")
        elif emb.dim() == 2:                                 # [B, D] fully pooled
            raise RuntimeError(
                f"MOMENT .embeddings is fully pooled {tuple(emb.shape)} (no patch dim) — "
                "per-patch reps needed; encoder-level access needed.")
        if emb.shape != (B, N_PATCHES, D_MODEL):
            raise RuntimeError(
                f"got patch-embedding shape {tuple(emb.shape)}, expected "
                f"{(B, N_PATCHES, D_MODEL)}")
        return emb.contiguous()


# --------------------------------------------------------------------------- #
# Self-check (run manually: py -3.14 src/moment_encoder.py)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from syncan_dataloader import load_syncan

    print("[1/4] loading MOMENT-1-large (frozen)  — first run downloads ~1.5 GB, please wait...")
    enc = MOMENTEncoder()
    print(f"      device = {enc.device}")

    n_trainable = enc.trainable_param_count()
    n_total = sum(p.numel() for p in enc.moment.parameters())
    print(f"[2/4] frozen check: trainable params = {n_trainable}  (MUST be 0)   "
          f"[total params = {n_total:,}]")

    print("[3/4] loading one batch of normal val windows (train_4)...")
    data = load_syncan(verbose=False)
    B = 4
    x = torch.from_numpy(data.val_windows[:B]).float()      # [B, 20, 512]
    print(f"      input batch shape = {tuple(x.shape)}")

    # diagnostic: show the raw MOMENT output shape(s) before our reshaping, so that if the
    # final assertion fails we can see exactly what MOMENT returned.
    with torch.no_grad():
        try:
            raw_none = enc.moment(x_enc=x.to(enc.device), reduction="none")
            rn = getattr(raw_none, "embeddings", None)
            print(f"      raw .embeddings (reduction='none') = "
                  f"{tuple(rn.shape) if rn is not None else 'MISSING'}")
        except Exception as e:  # noqa: BLE001
            print(f"      reduction='none' not usable: {type(e).__name__}: {e}")
        raw_def = enc.moment(x_enc=x.to(enc.device))
        rd = getattr(raw_def, "embeddings", None)
        print(f"      raw .embeddings (default)          = "
              f"{tuple(rd.shape) if rd is not None else 'MISSING'}")

    print("[4/4] forward()...")
    z = enc(x)
    print(f"      output shape = {tuple(z.shape)}   (MUST be [{B}, {N_PATCHES}, {D_MODEL}])")
    print(f"      output dtype = {z.dtype}   all-finite = {bool(torch.isfinite(z).all())}")

    assert n_trainable == 0, "FAIL: MOMENT has trainable parameters"
    assert z.shape == (B, N_PATCHES, D_MODEL), f"FAIL: output shape {tuple(z.shape)}"
    print("\nSECTION A OK: MOMENT frozen (0 trainable), forward -> [B, 64, 1024], no errors.")
