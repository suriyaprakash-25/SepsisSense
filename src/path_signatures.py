"""
SepsisSense Path Signatures & Irregular Vitals Sampling Module
==============================================================

Implements Rough Path Signatures (iterated integrals & Lévy area) for multi-channel
physiological time-series in critical care, based on the Oxford University winning method
from the PhysioNet/CinC 2019 Challenge (Morrill, Lyons et al.):

1. Level-1 Path Signatures: Net directional increments (time-invariance)
2. Level-2 Path Signatures: Iterated integrals & Coordinate Lévy Area (phase-space curvature)
3. Hemodynamic Coupling Loops:
   - (Heart Rate, MAP): Circulatory shock decompensation loop
   - (Heart Rate, Respiration): Cardiorespiratory coupling
   - (Temperature, Heart Rate): Febrile septic response vs benign tachycardia
   - (MAP, O2Sat): Perfusion-oxygenation degradation
4. Shock Index Velocity & Trajectory Volatility (total path variation)

Author: SepsisSense Machine Learning & Clinical Data Science Team
License: MIT
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("PathSignatures")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _f = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    _h.setFormatter(_f)
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


def compute_iterated_integral(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[float, float, float]:
    r"""
    Compute Level-2 iterated integrals of a 2D continuous path (X, Y) using the trapezoidal rule:
    
    S_{1,2} = \int_0^T (X(t) - X(0)) dY(t)
    S_{2,1} = \int_0^T (Y(t) - Y(0)) dX(t)
    Lévy Area A_{1,2} = 0.5 * (S_{1,2} - S_{2,1})  [Signed enclosed geometric area]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0

    # Handle NaNs via linear interpolation / forward fill
    if np.isnan(x).any():
        nans = np.isnan(x)
        x[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), x[~nans]) if (~nans).any() else 0.0
    if np.isnan(y).any():
        nans = np.isnan(y)
        y[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), y[~nans]) if (~nans).any() else 0.0

    # Center at start
    x_centered = x - x[0]
    y_centered = y - y[0]

    # Increments
    dx = np.diff(x_centered)
    dy = np.diff(y_centered)

    # Midpoints for Stratonovich/trapezoidal integration
    x_mid = 0.5 * (x_centered[:-1] + x_centered[1:])
    y_mid = 0.5 * (y_centered[:-1] + y_centered[1:])

    s12 = float(np.sum(x_mid * dy))
    s21 = float(np.sum(y_mid * dx))
    levy_area = 0.5 * (s12 - s21)

    return s12, s21, levy_area


class VitalSignSignatureExtractor:
    """
    Extracts rough path signature features across rolling patient vital signs windows.
    Robust to irregular sampling intervals, missing time-points, and noise.
    """

    CANONICAL_CHANNELS = ["HR", "MAP", "Resp", "Temp", "O2Sat"]

    def __init__(self, window_hours: int = 6):
        self.window_hours = window_hours

    def extract_trajectory_signatures(
        self,
        df_trajectory: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Extract path signatures and geometric phase features from a single patient trajectory window.
        
        Parameters
        ----------
        df_trajectory : pd.DataFrame
            Patient time series containing vital signs columns (HR, MAP, Resp, Temp, O2Sat).
            
        Returns
        -------
        Dict[str, float] with Level 1 increments, Level 2 Lévy areas, and path variation.
        """
        features: Dict[str, float] = {}

        if len(df_trajectory) == 0:
            return self._get_empty_signatures()

        # Slice to window
        sub = df_trajectory.tail(self.window_hours).copy()
        n = len(sub)

        # ── 1. Level-1 Signatures: Net Displacements & Path Volatility ────────
        for col in self.CANONICAL_CHANNELS:
            if col in sub.columns and sub[col].notna().sum() >= 2:
                vals = sub[col].dropna().values
                features[f"sig1_net_disp_{col}"] = float(vals[-1] - vals[0])
                # Total variation (path length / volatility)
                features[f"sig1_total_variation_{col}"] = float(np.sum(np.abs(np.diff(vals))))
            else:
                features[f"sig1_net_disp_{col}"] = 0.0
                features[f"sig1_total_variation_{col}"] = 0.0

        # ── 2. Level-2 Signatures: Geometric Lévy Areas ───────────────────────
        pairs = [
            ("HR", "MAP", "shock_loop"),          # Tachycardia + hypotension phase area
            ("HR", "Resp", "cardiorespiratory"),   # SIRS tachypnea + tachycardia coupling
            ("Temp", "HR", "febrile_dissoc"),     # Temperature vs heart rate trajectory
            ("MAP", "O2Sat", "perfusion_hypox")    # Perfusion collapse vs hypoxemia
        ]

        for c1, c2, name in pairs:
            if c1 in sub.columns and c2 in sub.columns and n >= 2:
                v1 = sub[c1].ffill().bfill().values
                v2 = sub[c2].ffill().bfill().values
                if len(v1) >= 2 and len(v2) >= 2 and not (np.isnan(v1).all() or np.isnan(v2).all()):
                    # Standardize scales to make geometric area scale-invariant
                    std1 = np.nanstd(v1) + 1e-4
                    std2 = np.nanstd(v2) + 1e-4
                    norm_v1 = (v1 - np.nanmean(v1)) / std1
                    norm_v2 = (v2 - np.nanmean(v2)) / std2
                    _, _, levy_area = compute_iterated_integral(norm_v1, norm_v2)
                    features[f"sig2_levy_area_{name}"] = float(levy_area)
                else:
                    features[f"sig2_levy_area_{name}"] = 0.0
            else:
                features[f"sig2_levy_area_{name}"] = 0.0

        # ── 3. Shock Index Dynamic Velocity ──────────────────────────────────
        if "HR" in sub.columns and "SBP" in sub.columns and n >= 2:
            hr_vals = sub["HR"].ffill().bfill().values
            sbp_vals = sub["SBP"].ffill().bfill().values
            si = np.where(sbp_vals > 20.0, hr_vals / sbp_vals, 0.70)
            features["sig_shock_index_velocity"] = float(si[-1] - si[0])
            features["sig_shock_index_max"] = float(np.max(si))
        elif "HR" in sub.columns and "MAP" in sub.columns and n >= 2:
            hr_vals = sub["HR"].ffill().bfill().values
            map_vals = sub["MAP"].ffill().bfill().values
            # Modified Shock Index = HR / MAP
            msi = np.where(map_vals > 15.0, hr_vals / map_vals, 1.0)
            features["sig_shock_index_velocity"] = float(msi[-1] - msi[0])
            features["sig_shock_index_max"] = float(np.max(msi))
        else:
            features["sig_shock_index_velocity"] = 0.0
            features["sig_shock_index_max"] = 0.70

        return features

    def _get_empty_signatures(self) -> Dict[str, float]:
        empty = {}
        for col in self.CANONICAL_CHANNELS:
            empty[f"sig1_net_disp_{col}"] = 0.0
            empty[f"sig1_total_variation_{col}"] = 0.0
        for name in ["shock_loop", "cardiorespiratory", "febrile_dissoc", "perfusion_hypox"]:
            empty[f"sig2_levy_area_{name}"] = 0.0
        empty["sig_shock_index_velocity"] = 0.0
        empty["sig_shock_index_max"] = 0.70
        return empty

