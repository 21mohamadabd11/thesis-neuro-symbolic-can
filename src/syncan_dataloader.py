"""SynCAN data loader for thesis.

Turns the raw, *asynchronous* SynCAN CSVs (github.com/etas/SynCAN) into regular,
z-scored, windowed multivariate tensors ready for the MOMENT encoder.

Ground truth verified from the real files (see memory/syncan-dataset-facts):
  * 10 message IDs (id1..id10), 20 signals total, per-ID counts (2,3,2,1,2,2,2,1,1,4)
  * long/event CSV format, 7 columns:
        Label, Time, ID, Signal1_of_ID, Signal2_of_ID, Signal3_of_ID, Signal4_of_ID
  * asynchronous bus: per-ID message period 15/30/45 ms (GCD = 15 ms)
  * signals already min-max scaled to [0, 1]
  * files: train_1..4 (all normal) + test_normal + 5 attack files
           (plateau, continuous, playback, suppress, flooding)

Design:
  * resample every signal onto a 15 ms grid with zero-order hold (ZOH)
  * W = 512 timesteps per window (64 patches @ patch_len 8, multiple of 8)
  * per-signal z-score using train_1..3 statistics (saved, reapplied to val/test)
  * split: train = train_1,2,3 | val = train_4 | test = test_normal + 5 attacks
  * stride 256 for train/val (50% overlap), 512 for test (non-overlapping)
  * output as a `SynCANData` dataclass; a lazy `SynCANWindows` Dataset is also
    provided for Stage-1 JEPA training.

Raw resampled grids are cached to ``<data_dir>/.cache/`` as .npz so the expensive
parse/resample runs only once; normalization is cheap and applied on load.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:  # torch only needed for the lazy Dataset; keep the loader importable without it
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _TorchDataset = object  # type: ignore
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# Dataset constants (verified from the real files)
# --------------------------------------------------------------------------- #
SIGNALS_PER_ID: tuple[int, ...] = (2, 3, 2, 1, 2, 2, 2, 1, 1, 4)   # id1..id10
N_IDS = 10
N_SIGNALS = sum(SIGNALS_PER_ID)                                     # == 20
# offset of each ID's first channel in the flat 20-dim signal vector
CHANNEL_OFFSET: tuple[int, ...] = tuple(int(x) for x in np.cumsum((0,) + SIGNALS_PER_ID)[:-1])
# fixed, documented channel order: id1_s1, id1_s2, id2_s1, ...
SIGNAL_NAMES: list[str] = [
    f"id{i + 1}_s{j + 1}" for i in range(N_IDS) for j in range(SIGNALS_PER_ID[i])
]

EXPECTED_COLS = [
    "Label", "Time", "ID",
    "Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID",
]
SIGNAL_COLS = EXPECTED_COLS[3:]  # the four Signal*_of_ID columns

TRAIN_FILES = ("train_1", "train_2", "train_3")
VAL_FILE = "train_4"
# test_normal first (false-positive rate), then the 5 attack types
TEST_FILES = (
    "test_normal", "test_plateau", "test_continuous",
    "test_playback", "test_suppress", "test_flooding",
)

DEFAULT_DT_MS = 15
DEFAULT_WINDOW = 512
DEFAULT_STRIDE_TRAIN = 256   # 50% overlap
DEFAULT_STRIDE_TEST = 512    # non-overlapping
_STD_FLOOR = 1e-8


# --------------------------------------------------------------------------- #
# Output container
# --------------------------------------------------------------------------- #
@dataclass
class SynCANData:
    """Windowed SynCAN tensors, ``[N, 20, W]`` channel-major (MOMENT-ready)."""

    train_windows: np.ndarray                      # [N_tr, 20, W]  (normal)
    val_windows: np.ndarray                        # [N_val, 20, W] (normal, held-out)
    test_windows: dict[str, np.ndarray]            # attack -> [N, 20, W]
    test_labels: dict[str, np.ndarray]             # attack -> [N]     window-level 0/1
    test_step_labels: dict[str, np.ndarray]        # attack -> [N, W]  per-step 0/1
    norm_mean: np.ndarray                          # [20]
    norm_std: np.ndarray                           # [20]
    signal_names: list[str] = field(default_factory=lambda: list(SIGNAL_NAMES))
    dt_ms: int = DEFAULT_DT_MS
    window: int = DEFAULT_WINDOW

    def as_core_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(train_windows, test_windows, test_labels)``.

        Test files are concatenated in ``TEST_FILES`` order.
        """
        keys = [k.replace("test_", "") for k in TEST_FILES]
        test_x = np.concatenate([self.test_windows[k] for k in keys], axis=0)
        test_y = np.concatenate([self.test_labels[k] for k in keys], axis=0)
        return self.train_windows, test_x, test_y

    def summary(self) -> str:
        lines = ["SynCANData summary",
                 f"  dt={self.dt_ms}ms  window={self.window}  channels={len(self.signal_names)}",
                 f"  train_windows {self.train_windows.shape}",
                 f"  val_windows   {self.val_windows.shape}",
                 "  test_windows:"]
        for k, w in self.test_windows.items():
            pos = int(self.test_labels[k].sum())
            n = len(self.test_labels[k])
            rate = (pos / n) if n else 0.0
            lines.append(f"    {k:<11} {str(w.shape):<20} anomaly_windows={pos}/{n} ({rate:.1%})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "syncan"


def _source_path(data_dir: Path, name: str) -> Path:
    """Return the .zip (preferred) or .csv source for ``name``."""
    zip_p = data_dir / f"{name}.zip"
    csv_p = data_dir / f"{name}.csv"
    if zip_p.exists():
        return zip_p
    if csv_p.exists():
        return csv_p
    raise FileNotFoundError(
        f"SynCAN source for '{name}' not found in {data_dir} (looked for {name}.zip/.csv)"
    )


# --------------------------------------------------------------------------- #
# Parsing + resampling
# --------------------------------------------------------------------------- #
def read_syncan_file(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read a SynCAN CSV (directly from its .zip) into a validated, canonicalized frame.

    SynCAN files are inconsistent: the test files and train_1 carry a header row
    (test uses ``Signal*_of_ID``, train_1 uses ``Signal*``) while train_2..4 have
    **no header** and ragged rows (4-7 fields, since IDs carry 1-4 signals). We
    detect the header per file, canonicalize the 7 columns to ``EXPECTED_COLS`` by
    position, and let missing trailing signals become NaN.
    """
    peek = pd.read_csv(path, compression="infer", nrows=1, header=None)
    has_header = str(peek.iloc[0, 0]).strip() == "Label"
    if has_header:
        df = pd.read_csv(path, compression="infer", nrows=nrows, header=0)
        cols = list(df.columns)
        if len(cols) != 7 or cols[:3] != ["Label", "Time", "ID"]:
            raise ValueError(f"{path.name}: unexpected header {cols}")
        df.columns = EXPECTED_COLS  # unify 'Signal1' / 'Signal1_of_ID' -> canonical
    else:
        df = pd.read_csv(path, compression="infer", nrows=nrows, header=None, names=EXPECTED_COLS)
    df["Label"] = df["Label"].astype("int8")
    df["Time"] = df["Time"].astype("float64")
    df["ID"] = df["ID"].astype("category")
    for c in SIGNAL_COLS:
        df[c] = df[c].astype("float32")
    return df


def resample_file(df: pd.DataFrame, dt_ms: int = DEFAULT_DT_MS) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold resample the async messages onto a uniform ``dt_ms`` grid.

    Returns ``(grid[T, 20] float32, step_labels[T] int8)``. Leading grid points
    before a signal's first message are back-filled with its first value.
    """
    df = df.sort_values("Time", kind="stable")
    times = df["Time"].to_numpy(dtype=np.float64)
    if times.size == 0:
        return np.zeros((0, N_SIGNALS), np.float32), np.zeros((0,), np.int8)

    t0, t1 = float(times[0]), float(times[-1])
    grid = np.arange(t0, t1 + dt_ms, dt_ms, dtype=np.float64)
    T = grid.shape[0]
    out = np.zeros((T, N_SIGNALS), dtype=np.float32)

    for gid, g in df.groupby("ID", observed=True, sort=False):
        id_num = int(str(gid)[2:])            # 'id7' -> 7
        k = SIGNALS_PER_ID[id_num - 1]
        off = CHANNEL_OFFSET[id_num - 1]
        st = g["Time"].to_numpy(dtype=np.float64)   # ascending (df is time-sorted)
        idx = np.searchsorted(st, grid, side="right") - 1
        np.clip(idx, 0, st.size - 1, out=idx)       # -1 (pre-first-msg) -> back-fill idx 0
        for j in range(k):
            vals = g[SIGNAL_COLS[j]].to_numpy(dtype=np.float32)
            out[:, off + j] = vals[idx]

    labels = df["Label"].to_numpy(dtype=np.int8)
    idx_l = np.searchsorted(times, grid, side="right") - 1
    np.clip(idx_l, 0, times.size - 1, out=idx_l)
    step_labels = labels[idx_l]
    return out, step_labels


# --------------------------------------------------------------------------- #
# Caching + normalization
# --------------------------------------------------------------------------- #
def _cache_key(name: str, dt_ms: int, nrows: Optional[int]) -> str:
    return f"{name}_dt{dt_ms}" + (f"_n{nrows}" if nrows else "")


def _raw_grid(data_dir: Path, cache_dir: Path, name: str, dt_ms: int,
              nrows: Optional[int], force: bool, verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    """Get the raw (pre-normalization) resampled grid + labels, using the .npz cache."""
    cpath = cache_dir / f"{_cache_key(name, dt_ms, nrows)}.npz"
    if cpath.exists() and not force:
        d = np.load(cpath)
        return d["grid"], d["labels"]
    t = time.time()
    df = read_syncan_file(_source_path(data_dir, name), nrows=nrows)
    grid, labels = resample_file(df, dt_ms)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cpath, grid=grid, labels=labels)
    if verbose:
        print(f"    resampled {name:<15} rows={len(df):>9,} -> grid{grid.shape} "
              f"({time.time() - t:.1f}s, cached)")
    return grid, labels


def _fit_or_load_norm(data_dir: Path, cache_dir: Path, dt_ms: int,
                      nrows: Optional[int], force: bool, verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-signal mean/std from the training files (train_1..3 only). No test leakage."""
    npath = cache_dir / f"norm_{_cache_key('train', dt_ms, nrows)}.json"
    if npath.exists() and not force:
        obj = json.loads(npath.read_text())
        return np.asarray(obj["mean"], np.float32), np.asarray(obj["std"], np.float32)

    s = np.zeros(N_SIGNALS, np.float64)
    ss = np.zeros(N_SIGNALS, np.float64)
    count = 0
    for name in TRAIN_FILES:
        grid, _ = _raw_grid(data_dir, cache_dir, name, dt_ms, nrows, force, verbose)
        g64 = grid.astype(np.float64)
        s += g64.sum(axis=0)
        ss += (g64 * g64).sum(axis=0)
        count += grid.shape[0]
    mean = s / count
    var = np.maximum(ss / count - mean * mean, 0.0)
    std = np.maximum(np.sqrt(var), _STD_FLOOR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npath.write_text(json.dumps(
        {"mean": mean.tolist(), "std": std.tolist(),
         "channels": SIGNAL_NAMES, "count": int(count)}, indent=2))
    if verbose:
        print(f"    fit z-score on {list(TRAIN_FILES)}: {count:,} timesteps")
    return mean.astype(np.float32), std.astype(np.float32)


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def make_windows(grid: np.ndarray, step_labels: np.ndarray, window: int, stride: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a ``[T, 20]`` grid into ``[N, 20, W]`` channel-major windows.

    Returns ``(windows[N,20,W], win_labels[N], step_labels[N,W])``.
    A window is anomalous iff any step within it is anomalous.
    """
    T = grid.shape[0]
    if T < window:
        return (np.empty((0, grid.shape[1], window), np.float32),
                np.empty((0,), np.int8),
                np.empty((0, window), np.int8))
    sw = np.lib.stride_tricks.sliding_window_view(grid, window, axis=0)  # [T-W+1, 20, W]
    windows = np.ascontiguousarray(sw[::stride], dtype=np.float32)
    swl = np.lib.stride_tricks.sliding_window_view(step_labels, window)[::stride]  # [N, W]
    step_out = np.ascontiguousarray(swl, dtype=np.int8)
    win_labels = step_out.max(axis=1).astype(np.int8)
    return windows, win_labels, step_out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_syncan(data_dir: Optional[str | Path] = None,
                window: int = DEFAULT_WINDOW,
                dt_ms: int = DEFAULT_DT_MS,
                stride_train: int = DEFAULT_STRIDE_TRAIN,
                stride_test: int = DEFAULT_STRIDE_TEST,
                stride_val: Optional[int] = None,
                cache_dir: Optional[str | Path] = None,
                nrows: Optional[int] = None,
                force: bool = False,
                verbose: bool = True) -> SynCANData:
    """Build the full windowed SynCAN dataset per the approved design.

    Parameters
    ----------
    data_dir   : folder holding the SynCAN .zip/.csv files (default: ../data/syncan).
    nrows      : if set, read only the first ``nrows`` rows of each file (fast checks).
    force      : ignore cached grids/norm params and rebuild.
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    cache_dir = Path(cache_dir) if cache_dir else (data_dir / ".cache")
    if stride_val is None:
        stride_val = stride_train
    if window % 8 != 0:
        raise ValueError(f"window={window} must be a multiple of MOMENT patch_len=8")

    if verbose:
        print(f"[syncan] data_dir={data_dir}\n[syncan] cache_dir={cache_dir}")
        print("[syncan] fitting normalization...")
    mean, std = _fit_or_load_norm(data_dir, cache_dir, dt_ms, nrows, force, verbose)

    def _norm_windows(name: str, stride: int):
        grid, labels = _raw_grid(data_dir, cache_dir, name, dt_ms, nrows, force, verbose)
        grid = (grid - mean) / std                       # per-signal z-score, float32
        return make_windows(grid.astype(np.float32, copy=False), labels, window, stride)

    if verbose:
        print("[syncan] windowing train / val / test...")
    train_parts = [_norm_windows(n, stride_train)[0] for n in TRAIN_FILES]
    train_windows = (np.concatenate(train_parts, axis=0) if train_parts
                     else np.empty((0, N_SIGNALS, window), np.float32))
    del train_parts

    val_windows = _norm_windows(VAL_FILE, stride_val)[0]

    test_windows, test_labels, test_step = {}, {}, {}
    for name in TEST_FILES:
        w, wl, sl = _norm_windows(name, stride_test)
        key = name.replace("test_", "")
        test_windows[key], test_labels[key], test_step[key] = w, wl, sl

    return SynCANData(
        train_windows=train_windows,
        val_windows=val_windows,
        test_windows=test_windows,
        test_labels=test_labels,
        test_step_labels=test_step,
        norm_mean=mean,
        norm_std=std,
        signal_names=list(SIGNAL_NAMES),
        dt_ms=dt_ms,
        window=window,
    )


# --------------------------------------------------------------------------- #
# Lazy Dataset for Stage-1 JEPA training (Phase 3)
# --------------------------------------------------------------------------- #
class SynCANWindows(_TorchDataset):
    """Memory-light window view over already-normalized SynCAN grids.

    Slices ``[20, W]`` windows on the fly instead of materializing every window,
    which matters for dense strides during JEPA training.
    """

    def __init__(self, grids: list[np.ndarray], window: int = DEFAULT_WINDOW,
                 stride: int = DEFAULT_STRIDE_TRAIN, labels: Optional[list[np.ndarray]] = None):
        if not _HAS_TORCH:  # pragma: no cover
            raise ImportError("PyTorch is required for SynCANWindows")
        self.grids = grids
        self.labels = labels
        self.window = window
        self.stride = stride
        self.index: list[tuple[int, int]] = []
        for gi, g in enumerate(grids):
            for s in range(0, max(0, g.shape[0] - window + 1), stride):
                self.index.append((gi, s))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        gi, s = self.index[i]
        w = self.grids[gi][s:s + self.window].T                  # [20, W]
        x = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
        if self.labels is None:
            return x
        y = int(self.labels[gi][s:s + self.window].max())
        return x, y


def build_training_windows(data_dir: Optional[str | Path] = None, dt_ms: int = DEFAULT_DT_MS,
                           window: int = DEFAULT_WINDOW, stride: int = DEFAULT_STRIDE_TRAIN,
                           cache_dir: Optional[str | Path] = None, nrows: Optional[int] = None,
                           force: bool = False, verbose: bool = False) -> "SynCANWindows":
    """Convenience builder: normalized train_1..3 grids wrapped in a lazy Dataset."""
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    cache_dir = Path(cache_dir) if cache_dir else (data_dir / ".cache")
    mean, std = _fit_or_load_norm(data_dir, cache_dir, dt_ms, nrows, force, verbose)
    grids = []
    for name in TRAIN_FILES:
        g, _ = _raw_grid(data_dir, cache_dir, name, dt_ms, nrows, force, verbose)
        grids.append(((g - mean) / std).astype(np.float32))
    return SynCANWindows(grids, window=window, stride=stride)


# --------------------------------------------------------------------------- #
# Self-checks (embedded tests)
# --------------------------------------------------------------------------- #
def _test_resample_synthetic() -> None:
    """ZOH + back-fill + label correctness on a tiny hand-checked example."""
    rows = [
        (0, 0.0, "id1", 0.1, 0.2, np.nan, np.nan),
        (0, 5.0, "id2", 0.3, 0.4, 0.5, np.nan),
        (0, 10.0, "id1", 0.6, 0.7, np.nan, np.nan),
        (1, 15.0, "id2", 0.8, 0.9, 1.0, np.nan),
        (0, 20.0, "id1", 0.11, 0.12, np.nan, np.nan),
    ]
    df = pd.DataFrame(rows, columns=EXPECTED_COLS)
    df["ID"] = df["ID"].astype("category")
    out, lab = resample_file(df, dt_ms=5)                 # grid = 0,5,10,15,20
    assert np.allclose(out[:, 0], [0.1, 0.1, 0.6, 0.6, 0.11]), out[:, 0]   # id1_s1 (ch 0)
    assert np.allclose(out[:, 2], [0.3, 0.3, 0.3, 0.8, 0.8]), out[:, 2]    # id2_s1 (ch 2), back-filled
    assert list(lab) == [0, 0, 0, 1, 0], lab
    # windowing determinism
    g = np.random.RandomState(0).randn(1000, N_SIGNALS).astype(np.float32)
    z = np.zeros(1000, np.int8)
    a1 = make_windows(g, z, 512, 256)[0]
    a2 = make_windows(g, z, 512, 256)[0]
    assert a1.shape == (2, N_SIGNALS, 512) and np.array_equal(a1, a2), a1.shape
    print("  [ok] synthetic ZOH + label + windowing checks")


def _self_check(data: SynCANData, full: bool) -> None:
    W = data.window
    tw = data.train_windows
    assert tw.ndim == 3 and tw.shape[1] == N_SIGNALS and tw.shape[2] == W, tw.shape
    for name, arr in [("train", data.train_windows), ("val", data.val_windows),
                      *[(f"test.{k}", v) for k, v in data.test_windows.items()]]:
        assert arr.size == 0 or np.isfinite(arr).all(), f"non-finite values in {name}"
    m, s = float(tw.mean()), float(tw.std())
    assert abs(m) < 0.1, f"train mean {m} not ~0"
    assert 0.7 < s < 1.3, f"train std {s} not ~1"
    assert data.test_labels["normal"].max() == 0, "test_normal must contain no anomalies"
    if full:
        for k in ("plateau", "continuous", "playback", "suppress", "flooding"):
            assert data.test_labels[k].sum() > 0, f"attack '{k}' has no anomalous windows"
    # shapes agree
    for k in data.test_windows:
        assert len(data.test_windows[k]) == len(data.test_labels[k]) == len(data.test_step_labels[k])
    print(f"  [ok] shape / finiteness / z-score(mean={m:+.3f},std={s:.3f}) / label checks"
          + (" (full)" if full else " (quick)"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build + self-check the SynCAN dataset.")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--dt-ms", type=int, default=DEFAULT_DT_MS)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--stride-train", type=int, default=DEFAULT_STRIDE_TRAIN)
    ap.add_argument("--stride-test", type=int, default=DEFAULT_STRIDE_TEST)
    ap.add_argument("--quick", action="store_true", help="truncate reads for a fast check")
    ap.add_argument("--force", action="store_true", help="rebuild caches")
    args = ap.parse_args()

    print("=== unit checks ===")
    _test_resample_synthetic()

    nrows = 500_000 if args.quick else None
    print(f"\n=== building SynCAN ({'quick/truncated' if nrows else 'full'}) ===")
    t = time.time()
    data = load_syncan(data_dir=args.data_dir, window=args.window, dt_ms=args.dt_ms,
                       stride_train=args.stride_train, stride_test=args.stride_test,
                       nrows=nrows, force=args.force, verbose=True)
    print("\n" + data.summary())
    print("\n=== self-check ===")
    _self_check(data, full=(nrows is None))
    print(f"\nALL CHECKS PASSED  ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
