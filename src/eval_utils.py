"""Evaluation metrics for CAN-bus anomaly detection (Phase 1).

Four metrics used throughout the thesis evaluation (see CLAUDE.md § Evaluation Metrics):

    compute_auc_roc(scores, labels)              -> float
    compute_average_precision(scores, labels)    -> float
    compute_f1_at_best_threshold(scores, labels) -> (best_f1, best_threshold)
    compute_detection_delay(scores, labels, threshold) -> float

Convention
----------
* ``scores``  : anomaly scores, higher == more anomalous (1-D array-like).
* ``labels``  : binary ground truth, 1 == anomaly/attack, 0 == normal (1-D array-like).
* For ``compute_detection_delay`` the arrays must be in **temporal order**.

Ranking metrics use scikit-learn; detection delay is a custom temporal metric
("frames from attack start to first alert", CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

_EPS = 1e-12


def _as_arrays(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce inputs to 1-D float scores and int labels."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(np.int64)
    if s.shape[0] != y.shape[0]:
        raise ValueError(f"scores/labels length mismatch: {s.shape[0]} vs {y.shape[0]}")
    if s.size == 0:
        raise ValueError("scores/labels are empty")
    return s, y


def compute_auc_roc(scores, labels) -> float:
    """Area under the ROC curve (threshold-independent ranking quality).

    Returns NaN if only one class is present (AUC undefined).
    """
    s, y = _as_arrays(scores, labels)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def compute_average_precision(scores, labels) -> float:
    """Average precision (area under the precision-recall curve).

    Returns NaN if there are no positive (anomaly) labels.
    """
    s, y = _as_arrays(scores, labels)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def compute_f1_at_best_threshold(scores, labels) -> tuple[float, float]:
    """Best F1 over all thresholds and the threshold that achieves it.

    A frame is flagged anomalous when ``score >= threshold``.
    Returns ``(best_f1, best_threshold)``. If there are no positive labels,
    returns ``(0.0, inf)`` (nothing should be flagged).
    """
    s, y = _as_arrays(scores, labels)
    if y.sum() == 0:
        return 0.0, float("inf")
    precision, recall, thresholds = precision_recall_curve(y, s)
    # precision/recall have length len(thresholds)+1; the trailing point
    # (precision=1, recall=0) has no threshold -> drop it before aligning.
    p, r = precision[:-1], recall[:-1]
    f1 = 2.0 * p * r / (p + r + _EPS)
    best = int(np.argmax(f1))
    return float(f1[best]), float(thresholds[best])


def _positive_segments(y: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end_inclusive] index ranges where ``y == 1``."""
    segments: list[tuple[int, int]] = []
    n = y.shape[0]
    i = 0
    while i < n:
        if y[i] == 1:
            j = i
            while j + 1 < n and y[j + 1] == 1:
                j += 1
            segments.append((i, j))
            i = j + 1
        else:
            i += 1
    return segments


def compute_detection_delay(scores, labels, threshold) -> float:
    """Mean detection delay in frames from attack onset to first alert.

    For each contiguous attack episode (a run of ``label == 1``), the delay is
    the offset from the episode's first frame to the first frame within it whose
    ``score >= threshold``. An episode that is never alerted is treated as missed
    and penalised with its full length. Returns the mean over all episodes, or
    NaN if there are no attack episodes.
    """
    s, y = _as_arrays(scores, labels)
    segments = _positive_segments(y)
    if not segments:
        return float("nan")
    delays: list[int] = []
    for start, end in segments:
        seg = s[start:end + 1]
        alerts = np.nonzero(seg >= threshold)[0]
        if alerts.size > 0:
            delays.append(int(alerts[0]))          # frames after onset until first alert
        else:
            delays.append(int(end - start + 1))    # missed -> penalise by episode length
    return float(np.mean(delays))


# --------------------------------------------------------------------------- #
# Dummy-data self-tests
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== eval_utils self-test ===")

    # 1) perfect separation: anomalies score higher than every normal.
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.array([0.10, 0.20, 0.15, 0.30, 0.25, 0.80, 0.90, 0.85, 0.95, 0.70])
    auc = compute_auc_roc(scores, labels)
    ap = compute_average_precision(scores, labels)
    f1, thr = compute_f1_at_best_threshold(scores, labels)
    print(f"[perfect]  AUC={auc:.3f}  AP={ap:.3f}  F1={f1:.3f} @ thr={thr:.3f}")
    assert np.isclose(auc, 1.0), auc
    assert np.isclose(ap, 1.0), ap
    assert np.isclose(f1, 1.0), f1

    # 2) uninformative scores (all equal): AUC=0.5, AP=positive-rate, F1=2PR/(P+R).
    labels2 = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    scores2 = np.full(8, 0.5)
    auc2 = compute_auc_roc(scores2, labels2)
    ap2 = compute_average_precision(scores2, labels2)
    f12, thr2 = compute_f1_at_best_threshold(scores2, labels2)
    print(f"[random]   AUC={auc2:.3f}  AP={ap2:.3f}  F1={f12:.3f} @ thr={thr2:.3f}")
    assert np.isclose(auc2, 0.5), auc2
    assert np.isclose(ap2, 0.5), ap2
    assert np.isclose(f12, 2.0 / 3.0, atol=1e-6), f12   # P=0.5, R=1.0

    # 3) detection delay across two attack episodes (onset->first alert).
    #                idx:  0    1    2    3    4    5    6    7    8    9    10   11
    labels3 = np.array([0,   0,   1,   1,   1,   1,   0,   0,   1,   1,   1,   0])
    scores3 = np.array([0.1, 0.1, 0.2, 0.2, 0.9, 0.9, 0.1, 0.1, 0.1, 0.95, 0.1, 0.1])
    d3 = compute_detection_delay(scores3, labels3, threshold=0.5)
    # episode [2..5]: first alert at offset 2; episode [8..10]: first alert at offset 1 -> mean 1.5
    print(f"[delay x2] mean_delay={d3:.3f} frames  (expect 1.5)")
    assert np.isclose(d3, 1.5), d3

    # 4) missed episode is penalised by its length.
    labels4 = np.array([0, 1, 1, 0])
    scores4 = np.array([0.1, 0.1, 0.1, 0.1])
    d4 = compute_detection_delay(scores4, labels4, threshold=0.5)
    print(f"[missed]   mean_delay={d4:.3f} frames  (expect 2.0)")
    assert np.isclose(d4, 2.0), d4

    # 5) degenerate-input guards.
    auc_one = compute_auc_roc([0.1, 0.2, 0.3], [0, 0, 0])           # one class -> NaN
    ap_none = compute_average_precision([0.1, 0.2, 0.3], [0, 0, 0])  # no pos -> NaN
    f1_none, thr_none = compute_f1_at_best_threshold([0.1, 0.2], [0, 0])  # -> (0.0, inf)
    dd_none = compute_detection_delay([0.1, 0.2], [0, 0], 0.5)       # no attack -> NaN
    print(f"[guards]   AUC(1-class)={auc_one}  AP(no-pos)={ap_none}  "
          f"F1(no-pos)=({f1_none}, {thr_none})  delay(no-attack)={dd_none}")
    assert np.isnan(auc_one) and np.isnan(ap_none) and np.isnan(dd_none)
    assert f1_none == 0.0 and np.isinf(thr_none)

    print("\nALL eval_utils TESTS PASSED")
