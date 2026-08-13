"""Phase 2, Section C — t-SNE of frozen-MOMENT embeddings (Checkpoint 1).

Projects mean-pooled MOMENT embeddings of normal vs. the 5 SynCAN attack types to 2-D
with t-SNE and saves a color-coded scatter to figures/tsne_embeddings.png.
Purpose: visually confirm normal/attack separation.

6 balanced classes, 586 windows each:
  * normal    : 586 windows sampled (seed 42) from train_embeddings.npy   [--normal-source]
  * plateau, suppress, flooding, continuous, playback : the 5 attack test files (586 each)
Each window's [64, 1024] patch embedding is mean-pooled over the 64 patches -> [1024].

Embeddings live OUTSIDE the project; the default --emb-dir
points there. NOTE: Section B saved the test files as `test_<attack>_emb.npy` (not
`_embeddings.npy`); this loader accepts either suffix.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — save to file without a display
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

DEFAULT_EMB_DIR = r"C:\Users\moea0\ThesisData\embeddings"
ATTACKS = ["plateau", "suppress", "flooding", "continuous", "playback"]
CLASSES = ["normal"] + ATTACKS
COLORS = {
    "normal": "#1f77b4", "plateau": "#ff7f0e", "suppress": "#d62728",
    "flooding": "#2ca02c", "continuous": "#9467bd", "playback": "#8c564b",
}
N_PER_CLASS = 586
SEED = 42


def resolve(emb_dir: Path, candidates: list[str]) -> Path:
    """Return the first existing file among `candidates` (handles _emb/_embeddings)."""
    for name in candidates:
        p = emb_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"none of {candidates} found in {emb_dir} — check --emb-dir and filenames")


def pooled_from(path: Path, n: int | None = None, seed: int = SEED) -> np.ndarray:
    """Load [N, 64, 1024] (memmapped), optionally sample `n`, mean-pool patches -> [n, 1024]."""
    arr = np.load(path, mmap_mode="r")                       # [N, 64, 1024]
    total = arr.shape[0]
    if n is not None and total > n:
        idx = np.sort(np.random.RandomState(seed).choice(total, size=n, replace=False))
        sub = np.asarray(arr[idx])                           # materialize only the sample
    else:
        sub = np.asarray(arr)
    return sub.mean(axis=1).astype(np.float32)               # [n, 1024]


def main() -> None:
    ap = argparse.ArgumentParser(description="t-SNE of MOMENT embeddings (normal vs attacks).")
    ap.add_argument("--emb-dir", default=DEFAULT_EMB_DIR)
    ap.add_argument("--out", default=None, help="default: <project>/figures/tsne_embeddings.png")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--normal-source", choices=["train", "test_normal"], default="train")
    args = ap.parse_args()

    proj = Path(__file__).resolve().parent.parent
    emb_dir = Path(args.emb_dir)
    out = Path(args.out) if args.out else (proj / "figures" / "tsne_embeddings.png")
    if not emb_dir.exists():
        raise FileNotFoundError(f"--emb-dir does not exist: {emb_dir}")

    print(f"[1/4] loading mean-pooled embeddings from {emb_dir}")
    if args.normal_source == "train":
        p = resolve(emb_dir, ["train_embeddings.npy", "train_emb.npy"])
        normal = pooled_from(p, n=N_PER_CLASS, seed=SEED)
        print(f"      {'normal':<11} <- {p.name}  (sampled {len(normal)}, seed {SEED})")
    else:
        p = resolve(emb_dir, ["test_normal_emb.npy", "test_normal_embeddings.npy"])
        normal = pooled_from(p)
        print(f"      {'normal':<11} <- {p.name}  ({len(normal)})")

    feats, labels = [normal], ["normal"] * len(normal)
    for atk in ATTACKS:
        p = resolve(emb_dir, [f"test_{atk}_emb.npy", f"test_{atk}_embeddings.npy"])
        pooled = pooled_from(p)                              # all windows (586)
        feats.append(pooled)
        labels += [atk] * len(pooled)
        print(f"      {atk:<11} <- {p.name}  ({len(pooled)})")

    X = np.concatenate(feats, axis=0)
    y = np.array(labels)
    print(f"      combined: X={X.shape}  ({len(CLASSES)} classes)")

    print(f"[2/4] running t-SNE (n_components=2, perplexity={args.perplexity}, "
          f"random_state={SEED}) on {X.shape[0]} points — this takes a minute...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=SEED, init="pca")
    Z = tsne.fit_transform(X)

    print("[3/4] plotting...")
    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in CLASSES:                                      # normal first, attacks over it
        m = y == cls
        ax.scatter(Z[m, 0], Z[m, 1], s=9, alpha=0.6, linewidths=0,
                   color=COLORS[cls], label=f"{cls} (n={int(m.sum())})")
    ax.set_title("t-SNE of frozen MOMENT-1-large embeddings — SynCAN normal vs. attacks",
                 fontsize=12)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="best", fontsize=9, markerscale=2.0, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    print("[4/4] saving...")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"      saved -> {out}")
    print("done.")


if __name__ == "__main__":
    main()
