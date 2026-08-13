"""Phase 3, Section F — unit tests for the JEPA prediction module."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# import the module under test from ../src
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import jepa_module  # noqa: E402  (the module itself — used for the no-import structural check)
from jepa_module import (  # noqa: E402
    MOMENT_DIM,
    N_PATCHES,
    MASK_SHORT,
    MASK_MEDIUM,
    MASK_LONG,
    make_block_mask,
    JEPABranch,
    MultiResolutionJEPA,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _SkipTest(Exception):
    """Raised to skip a test when run as a plain script (pytest uses pytest.skip)."""


def _skip(msg: str):
    """Skip the current test. Uses pytest's native skip under pytest, else our runner."""
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise _SkipTest(msg)


def _n_blocks(mask: torch.Tensor) -> int:
    """Number of contiguous True-runs in a 1-D boolean mask (rising edges + leading True).
    Same counting rule as the jepa_module smoke test."""
    m = mask.to(torch.int).cpu()
    rises = int(((m[1:] - m[:-1]) == 1).sum())
    return rises + int(m[0].item())


# --------------------------------------------------------------------------- #
# Group 1 — masking strategy (make_block_mask)
# --------------------------------------------------------------------------- #
def test_mask_exact_count_and_contiguity():
    """Masks exactly `n_mask_patches` patches as ONE or TWO contiguous blocks — never
    scattered — for all three branch sizes and for random sizes."""
    torch.manual_seed(0)
    for size in (MASK_SHORT, MASK_MEDIUM, MASK_LONG):
        for _ in range(300):  # enough iterations to exercise both 1-block and 2-block paths
            mask = make_block_mask(N_PATCHES, size)
            assert mask.dtype == torch.bool and tuple(mask.shape) == (N_PATCHES,)
            assert int(mask.sum()) == size, f"size {size}: masked {int(mask.sum())}"
            nb = _n_blocks(mask)
            assert 1 <= nb <= 2, f"size {size}: {nb} blocks (must be 1-2, never scattered)"

    for _ in range(300):
        size = int(torch.randint(1, N_PATCHES + 1, (1,)).item())
        mask = make_block_mask(N_PATCHES, size)
        assert int(mask.sum()) == size
        assert 1 <= _n_blocks(mask) <= 2


def test_mask_clamps_out_of_range():
    """Requests below 1 clamp to a single masked patch; requests above n_patches clamp to
    a full (single-block) mask."""
    for _ in range(50):
        assert int(make_block_mask(N_PATCHES, 0).sum()) == 1
        assert int(make_block_mask(N_PATCHES, -5).sum()) == 1
        assert int(make_block_mask(N_PATCHES, 999).sum()) == N_PATCHES
    full = make_block_mask(N_PATCHES, N_PATCHES)
    assert bool(full.all()) and _n_blocks(full) == 1


def test_mask_deterministic_with_seed():
    """Same torch seed -> identical mask (needed so inference/ablation masks are reproducible)."""
    torch.manual_seed(123)
    a = make_block_mask(N_PATCHES, MASK_MEDIUM)
    torch.manual_seed(123)
    b = make_block_mask(N_PATCHES, MASK_MEDIUM)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# Group 2 — shape & structure contracts
# --------------------------------------------------------------------------- #
def test_branch_forward_shape():
    """JEPABranch(x[B,64,1024], mask) -> predictions [B, n_masked, 1024]."""
    torch.manual_seed(0)
    B = 2
    branch = JEPABranch(n_mask_patches=MASK_MEDIUM)
    x = torch.randn(B, N_PATCHES, MOMENT_DIM)
    mask = make_block_mask(N_PATCHES, MASK_MEDIUM)
    with torch.no_grad():
        out = branch(x, mask)
    assert out.shape == (B, int(mask.sum()), MOMENT_DIM)


def test_multiresjepa_forward_contract():
    """forward -> list of 3 (pred, target, mask): short/medium/long with masked counts
    6/24/38 and matching pred/target shapes; each mask is 1-2 contiguous blocks."""
    torch.manual_seed(0)
    B = 2
    model = MultiResolutionJEPA()
    x = torch.randn(B, N_PATCHES, MOMENT_DIM)
    with torch.no_grad():
        outs = model(x)
    assert len(outs) == 3
    for (pred, tgt, mask), exp in zip(outs, (MASK_SHORT, MASK_MEDIUM, MASK_LONG)):
        assert int(mask.sum()) == exp
        assert pred.shape == (B, exp, MOMENT_DIM)
        assert tgt.shape == pred.shape
        assert 1 <= _n_blocks(mask) <= 2


def test_compute_prediction_error_keys_finite_deterministic():
    """compute_prediction_error -> {'short','medium','long'} finite floats, reproducible
    under a fixed seed (it draws its own masks internally)."""
    torch.manual_seed(0)
    model = MultiResolutionJEPA()
    x = torch.randn(3, N_PATCHES, MOMENT_DIM)
    torch.manual_seed(123)
    e1 = model.compute_prediction_error(x)
    torch.manual_seed(123)
    e2 = model.compute_prediction_error(x)
    assert set(e1) == {"short", "medium", "long"}
    for k in e1:
        assert isinstance(e1[k], float) and math.isfinite(e1[k])
        assert abs(e1[k] - e2[k]) < 1e-6, f"{k}: non-deterministic ({e1[k]} vs {e2[k]})"


def test_param_structure_three_equal_branches():
    """Three independent, identically-sized branches; total == 3x a single branch, ~11M."""
    model = MultiResolutionJEPA()
    single = sum(p.numel() for p in model.branch_short.parameters())
    medium = sum(p.numel() for p in model.branch_medium.parameters())
    long_ = sum(p.numel() for p in model.branch_long.parameters())
    total = sum(p.numel() for p in model.parameters())
    assert single == medium == long_, "branches must have identical parameter counts"
    assert total == 3 * single, "total params must be exactly 3x a single branch"
    assert 9_000_000 <= total <= 13_000_000, f"expected ~11M params, got {total:,}"


# --------------------------------------------------------------------------- #
# Group 3 — no gradient into MOMENT (CLAUDE.md required check + structural guard)
# --------------------------------------------------------------------------- #
def test_no_gradient_into_moment():
    """A backward pass trains the branches but leaves NO gradient on the input embeddings
    (the module detaches its input) — so gradient can never reach the frozen MOMENT
    encoder that produced them."""
    torch.manual_seed(0)
    x = torch.randn(2, N_PATCHES, MOMENT_DIM, requires_grad=True)
    model = MultiResolutionJEPA()
    outs = model(x)
    loss = sum(F.mse_loss(pred, tgt) for pred, tgt, _ in outs)
    loss.backward()

    assert x.grad is None, "gradient leaked into the input — MOMENT would not be frozen"
    for name, br in model.named_branches():
        g = br.input_proj.weight.grad
        assert g is not None and float(g.abs().sum()) > 0.0, f"{name} branch got no gradient"


def test_module_has_no_moment_or_dataloader_import():
    """Structural guarantee: jepa_module operates on raw tensors only — it must not import
    MOMENT or the dataloader (otherwise a stray reference could couple it to the encoder)."""
    import inspect
    src = inspect.getsource(jepa_module)
    for forbidden in ("import momentfm", "from momentfm", "MOMENTPipeline",
                      "moment_encoder", "syncan_dataloader"):
        assert forbidden not in src, f"jepa_module must not depend on {forbidden!r}"


# --------------------------------------------------------------------------- #
# 4a — mechanism test (synthetic; validates the prediction-error machinery)
# --------------------------------------------------------------------------- #
def test_mechanism_error_rises_on_unpredictable_content():
    """After training a branch on a smooth/predictable distribution, per-window error is
    (i) lower than before training on clean content, and (ii) much higher when the masked
    content is UNPREDICTABLE from context. This proves the error mechanism is directionally
    correct, independently of MOMENT's smoothness (the real-data null captured by 4b)."""
    torch.manual_seed(0)
    noise_gen = torch.Generator().manual_seed(0)
    N, d, mask_size = N_PATCHES, 32, 16

    def smooth_batch(B: int, seed: int) -> torch.Tensor:
        """Low-frequency sinusoids along the patch axis -> masked blocks are predictable
        from the surrounding context."""
        g = torch.Generator().manual_seed(seed)
        t = torch.linspace(0.0, 1.0, N).view(1, N, 1)          # [1, N, 1]
        amp = torch.randn(B, 1, d, generator=g)
        phase = torch.rand(B, 1, d, generator=g) * (2 * math.pi)
        return amp * torch.sin(2 * math.pi * 2.0 * t + phase)   # [B, N, d], low freq

    branch = JEPABranch(moment_dim=d, d_model=32, n_heads=2, dim_feedforward=64,
                        n_patches=N, n_mask_patches=mask_size)

    # fixed eval mask + held-out clean / corrupted eval sets (context identical; only the
    # masked-position *targets* differ between clean and corrupted)
    torch.manual_seed(999)
    eval_mask = make_block_mask(N, mask_size)
    x_clean = smooth_batch(64, seed=7777)
    x_corrupt = x_clean.clone()
    noise = torch.randn(64, int(eval_mask.sum()), d, generator=noise_gen) * (float(x_clean.std()) * 5.0 + 1.0)
    x_corrupt[:, eval_mask, :] = noise                          # masked content no longer predictable

    def masked_mse(x: torch.Tensor) -> float:
        with torch.no_grad():
            pred = branch(x, eval_mask)
            tgt = x[:, eval_mask, :]
            return float(((pred - tgt) ** 2).mean())

    branch.eval()
    err_clean_untrained = masked_mse(x_clean)

    branch.train()
    opt = torch.optim.Adam(branch.parameters(), lr=1e-3)
    for step in range(300):
        x = smooth_batch(32, seed=1000 + step)
        m = make_block_mask(N, mask_size)                      # random masks during training
        loss = F.mse_loss(branch(x, m), x[:, m, :])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    branch.eval()
    err_clean = masked_mse(x_clean)
    err_corrupt = masked_mse(x_corrupt)

    assert err_clean < err_clean_untrained, (
        f"training did not reduce predictable-content error "
        f"({err_clean:.4f} !< untrained {err_clean_untrained:.4f})")
    assert err_corrupt > 2.0 * err_clean, (
        f"prediction error did not rise on unpredictable content "
        f"(corrupt {err_corrupt:.4f} !> 2x clean {err_clean:.4f})")


# --------------------------------------------------------------------------- #
# 4b — regression lock on the Section D/E null result (needs checkpoint + embeddings)
# --------------------------------------------------------------------------- #
def test_null_result_regression_step4_auc():
    """Reproduce the Step-4 pipeline (Section E) on the real checkpoint + embeddings and
    assert the documented NULL result: JEPA-prediction-error AUC ~= 0.50 (band [0.40, 0.60])
    for every attack. This deliberately does NOT assert error_normal < error_attack, which
    Sections D/E showed to be false. Skips if the checkpoint/embeddings are unavailable."""
    proj = Path(__file__).resolve().parent.parent
    ckpt_path = proj / "experiments" / "jepa_checkpoint.pt"

    # lazy imports: heavier deps + they pull the exact Section D/E machinery/constants
    from jepa_inference import (per_window_errors, N_MASKS, SEED, SPLIT_FILES,
                                BRANCHES, ATTACKS, DEFAULT_EMB_DIR)
    import eval_utils

    emb_dir = Path(DEFAULT_EMB_DIR)
    missing = ([str(ckpt_path)] if not ckpt_path.exists() else [])
    missing += [str(emb_dir / f) for f in SPLIT_FILES.values() if not (emb_dir / f).exists()]
    if missing:
        _skip("real-data regression test needs the trained checkpoint + embeddings; missing:\n    "
              + "\n    ".join(missing))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model = MultiResolutionJEPA(n_patches=N_PATCHES, moment_dim=MOMENT_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    branches = dict(model.named_branches())
    mask_sizes = {name: b.n_mask_patches for name, b in branches.items()}

    # identical fixed masks as jepa_inference / Section E (same seed + draw order)
    torch.manual_seed(SEED)
    branch_masks = {name: [make_block_mask(N_PATCHES, mask_sizes[name]).to(device)
                           for _ in range(N_MASKS)] for name in BRANCHES}

    scores: dict[str, np.ndarray] = {}
    means: dict[str, float] = {}
    for split, fname in SPLIT_FILES.items():
        emb = torch.from_numpy(np.load(str(emb_dir / fname))).float()
        e = {b: per_window_errors(branches[b], emb, branch_masks[b], 64, device) for b in BRANCHES}
        scores[split] = (e["short"] + e["medium"] + e["long"]) / 3.0   # Step-4 score
        means[split] = float(scores[split].mean())
        del emb

    normal = scores["normal"]
    for atk in ATTACKS:
        s = np.concatenate([normal, scores[atk]])
        y = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(scores[atk]), dtype=int)])
        auc = eval_utils.compute_auc_roc(s, y)
        assert 0.40 <= auc <= 0.60, (
            f"{atk}: Step-4 AUC {auc:.3f} left the documented null band [0.40, 0.60] — "
            f"the JEPA-prediction-error null result changed; re-examine before trusting it.")

    # scale sanity: per-window errors stay tiny and non-separating (~6e-4 in Sections D/E)
    for split, mu in means.items():
        assert mu < 0.05, f"{split}: mean per-window error {mu:.5f} >> recorded ~6e-4 (scale regressed)"


# --------------------------------------------------------------------------- #
# standalone runner:  py -3.14 tests/test_jepa.py
# --------------------------------------------------------------------------- #
def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = skipped = failed = 0
    print(f"running {len(tests)} JEPA unit tests (Phase 3, Section F)\n" + "-" * 66)
    for name, fn in tests:
        try:
            fn()
        except _SkipTest as e:
            print(f"SKIP  {name}\n      {e}")
            skipped += 1
        except AssertionError as e:
            print(f"FAIL  {name}\n      {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001 — surface unexpected errors as failures
            import traceback
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
        else:
            print(f"PASS  {name}")
            passed += 1
    print("-" * 66)
    print(f"{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
