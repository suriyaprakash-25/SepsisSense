"""
Path-signature features for SepsisSense (experimental, see src/experiments_v4.py).

Morrill et al. (PhysioNet/CinC 2019 winner) fed truncated path signatures of short
look-back windows to gradient-boosted trees. Level-1 signature terms are plain
increments, which v3 already has (`__diff6`). The new information sits in level 2:
the Levy area of each pair of channels, which describes *how* two signals move
together over the window (e.g. heart rate rising before blood pressure falls, or a
vital sign accelerating rather than drifting).

For a path X over window [s, e] with increments dX_k = X_k - X_{k-1}:

    S^{ij} = sum_{k=s+1..e} (X^i_{k-1} - X^i_s) dX^j_k + 1/2 dX^i_k dX^j_k
    Levy area A^{ij} = 1/2 (S^{ij} - S^{ji})

Using a per-row running sum C^{ij}_k = sum X^i_{k-1} dX^j_k, every window is O(1):

    A^{ij} = 1/2 [ (C^{ij}_e - C^{ij}_s) - (C^{ji}_e - C^{ji}_s) - X^i_s (X^j_e - X^j_s) + X^j_s (X^i_e - X^i_s) ]

All terms use rows <= the current hour only (causal). Channels are the forward-filled
vitals plus partial SOFA, standardised with fixed constants (so training and the app
compute identical values); values before the first measurement count as average (0).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

import features_v3 as f3

# fixed standardisation (mean, sd) measured on the v3 training split
CHANNELS: dict[str, tuple[float, float]] = {
    "HR": (84.28, 17.34),
    "MAP": (82.62, 16.11),
    "Resp": (18.71, 5.01),
    "O2Sat": (97.17, 2.96),
    "Temp": (36.87, 0.71),
    "score__partial_sofa": (1.15, 1.55),
}
WINDOWS = (6, 12)
_SHORT = {"score__partial_sofa": "SOFA"}


def _pairs() -> list[tuple[str, str]]:
    names = ["time"] + list(CHANNELS)
    return list(combinations(names, 2))


SIG_FEATURES: list[str] = [
    f"sig__{_SHORT.get(a, a)}_{_SHORT.get(b, b)}__{w}h" for w in WINDOWS for a, b in _pairs()
]


def signature_block(cols: dict[str, np.ndarray], patients: np.ndarray) -> np.ndarray:
    """Levy areas for every channel pair and window. `cols` maps channel name -> per-row values
    (rows contiguous per patient, in time order). Returns float32 array (n_rows, len(SIG_FEATURES))."""
    patients = np.asarray(patients)
    n = len(patients)
    first = np.r_[True, patients[1:] != patients[:-1]]
    p0 = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    idx = np.arange(n)

    paths = {"time": (idx - p0).astype(np.float64) / 6.0}
    for name, (mu, sd) in CHANNELS.items():
        v = (np.asarray(cols[name], dtype=np.float64) - mu) / sd
        paths[name] = np.where(np.isfinite(v), v, 0.0)

    prev = {k: np.where(first, v, np.r_[v[:1], v[:-1]]) for k, v in paths.items()}
    d = {k: paths[k] - prev[k] for k in paths}  # 0 on each patient's first row

    out = np.empty((n, len(SIG_FEATURES)), dtype=np.float32)
    col = 0
    for w in WINDOWS:
        s = np.maximum(idx - w, p0)
        for a, b in _pairs():
            cab = np.cumsum(prev[a] * d[b])
            cba = np.cumsum(prev[b] * d[a])
            xa, xb = paths[a], paths[b]
            area = 0.5 * ((cab - cab[s]) - (cba - cba[s]) - xa[s] * (xb - xb[s]) + xb[s] * (xa - xa[s]))
            out[:, col] = area
            col += 1
    return out


FEATURES: list[str] = list(f3.FEATURES) + SIG_FEATURES


def engineer(df: pd.DataFrame, patient_col: str = "Patient") -> pd.DataFrame:
    """v3 features plus signature features, one row per input row."""
    base = f3.engineer(df, patient_col=patient_col)
    sig = signature_block({c: base[c].to_numpy() for c in CHANNELS}, df[patient_col].to_numpy())
    return pd.concat([base.reset_index(drop=True), pd.DataFrame(sig, columns=SIG_FEATURES)], axis=1)
