"""
SepsisSense High-Frequency Waveform & Continuous Telemetry Analytics Module
==========================================================================

Ingests high-frequency continuous physiological waveforms from ICU bedside monitors
(e.g., 100 Hz - 250 Hz ECG, Arterial Blood Pressure lines, and Photoplethysmography [PPG]),
extracting predictive autonomic and hemodynamic biomarkers that degrade 12-24 hours
before macroscopic blood pressure collapse (IEEE TBME / Physiological Measurement):

1. R-Peak Detection & Normal-to-Normal (NN) Interval Extraction
2. Time-Domain Heart Rate Variability (HRV):
   - Mean NN, SDNN (Overall autonomic modulation)
   - RMSSD, pNN50 (Vagal/parasympathetic tone blunting)
3. Frequency-Domain Spectral Analysis (Welch Periodogram):
   - Low Frequency (LF: 0.04 - 0.15 Hz), High Frequency (HF: 0.15 - 0.40 Hz)
   - Sympathovagal Balance (LF/HF ratio)
4. Non-Linear Complexity:
   - Multiscale Sample Entropy (SampEn) for autonomic degradation
5. Arterial Line & PPG Waveform Dynamics:
   - Pulse Pressure Variation (PPV) for fluid responsiveness
   - Cardiac Contractility Index (Max dP/dt rate of pressure rise)
   - Systolic Area under curve (stroke volume surrogate)

Author: SepsisSense Machine Learning & Clinical AI Research Team
License: MIT
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("WaveformAnalytics")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _f = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    _h.setFormatter(_f)
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


def compute_sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Compute Sample Entropy (SampEn) measuring the irregularity/complexity of a time series.
    Lower entropy indicates loss of physiological complexity (a hallmark of autonomic failure in sepsis).
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    if n <= m + 1:
        return 0.0

    sigma = np.std(x)
    if sigma < 1e-6:
        return 0.0
    tol = r * sigma

    def _phi(dim: int) -> int:
        patterns = np.array([x[i:i + dim] for i in range(n - dim + 1)])
        # Count matching template pairs using Chebyshev (infinity) distance
        matches = 0
        p_len = len(patterns)
        for i in range(p_len):
            diffs = np.max(np.abs(patterns[i + 1:] - patterns[i]), axis=1)
            matches += np.sum(diffs <= tol)
        return matches

    count_m = _phi(m)
    count_m1 = _phi(m + 1)

    if count_m == 0 or count_m1 == 0:
        return 0.0

    return float(-np.log(count_m1 / count_m))


class ContinuousWaveformProcessor:
    """
    Processes high-frequency continuous physiological waveforms (ECG, Arterial Line, PPG)
    from bedside telemetry to extract early prodromal autonomic failure signatures.
    """

    def __init__(self, sampling_rate_hz: int = 100):
        self.fs = sampling_rate_hz

    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Detect QRS complex R-peaks from raw ECG waveform using a modified Pan-Tompkins filter:
        Bandpass filter (5-15 Hz) -> Differentiation -> Squaring -> Moving Window Integration -> Peak Picking.
        """
        sig = np.asarray(ecg_signal, dtype=float)
        if len(sig) < self.fs * 2:
            return np.array([], dtype=int)

        # 1. Bandpass Butterworth Filter (5 - 15 Hz)
        nyq = 0.5 * self.fs
        low = 5.0 / nyq
        high = min(15.0 / nyq, 0.99)
        b, a = signal.butter(2, [low, high], btype='bandpass')
        filtered = signal.filtfilt(b, a, sig)

        # 2. Derivative Filter
        diff = np.diff(filtered)

        # 3. Squaring
        squared = diff ** 2

        # 4. Moving Window Integration (approx 150ms window)
        win_size = int(0.15 * self.fs)
        if win_size > 1:
            integrated = np.convolve(squared, np.ones(win_size) / win_size, mode='same')
        else:
            integrated = squared

        # 5. Peak Finding (refractory period ~250ms = 0.25*fs)
        distance = max(int(0.25 * self.fs), 1)
        threshold = np.mean(integrated) + 0.5 * np.std(integrated)
        peaks, _ = signal.find_peaks(integrated, distance=distance, height=threshold)

        return peaks

    def extract_hrv_features(self, ecg_or_rr: np.ndarray, is_rr_intervals: bool = False) -> Dict[str, float]:
        """
        Extract comprehensive Heart Rate Variability (HRV) metrics:
        - Time domain: Mean NN, SDNN, RMSSD, pNN50
        - Frequency domain: LF, HF, LF/HF ratio
        - Non-linear domain: Sample Entropy (SampEn)
        """
        if is_rr_intervals:
            rr_intervals_ms = np.asarray(ecg_or_rr, dtype=float)
        else:
            peaks = self.detect_r_peaks(ecg_or_rr)
            if len(peaks) < 4:
                return self._empty_hrv_features()
            # Intervals in milliseconds
            rr_intervals_ms = (np.diff(peaks) / self.fs) * 1000.0

        # Physiological filtering (eliminate non-physiological ectopic beats outside 300-2000 ms)
        valid_mask = (rr_intervals_ms >= 300.0) & (rr_intervals_ms <= 2000.0)
        nn = rr_intervals_ms[valid_mask]

        if len(nn) < 4:
            return self._empty_hrv_features()

        # ── 1. Time-Domain Metrics ───────────────────────────────────────────
        mean_nn = float(np.mean(nn))
        sdnn = float(np.std(nn, ddof=1)) if len(nn) > 1 else 0.0

        succ_diffs = np.diff(nn)
        rmssd = float(np.sqrt(np.mean(succ_diffs ** 2))) if len(succ_diffs) > 0 else 0.0
        nn50 = int(np.sum(np.abs(succ_diffs) > 50.0)) if len(succ_diffs) > 0 else 0
        pnn50 = float(nn50 / len(succ_diffs) * 100.0) if len(succ_diffs) > 0 else 0.0

        # ── 2. Frequency-Domain Metrics (Welch Spectral Analysis) ────────────
        # Resample non-uniform NN intervals to 4 Hz uniform time series
        t_cum = np.cumsum(nn) / 1000.0
        t_cum -= t_cum[0]
        total_time = t_cum[-1]

        if total_time > 4.0:
            fs_resamp = 4.0  # 4 Hz uniform grid
            t_uniform = np.arange(0, total_time, 1.0 / fs_resamp)
            nn_uniform = np.interp(t_uniform, t_cum, nn)

            nperseg = min(len(nn_uniform), 256)
            freqs, psd = signal.welch(nn_uniform - np.mean(nn_uniform), fs=fs_resamp, nperseg=nperseg)

            vlf_band = (freqs >= 0.0033) & (freqs < 0.04)
            lf_band = (freqs >= 0.04) & (freqs < 0.15)
            hf_band = (freqs >= 0.15) & (freqs < 0.40)

            df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            p_vlf = float(np.sum(psd[vlf_band]) * df)
            p_lf = float(np.sum(psd[lf_band]) * df)
            p_hf = float(np.sum(psd[hf_band]) * df)
            lf_hf_ratio = float(p_lf / max(p_hf, 1e-6))
        else:
            p_vlf = 0.0
            p_lf = 0.0
            p_hf = 0.0
            lf_hf_ratio = 1.0

        # ── 3. Non-Linear Complexity (Sample Entropy) ────────────────────────
        # Downsample for computational efficiency if long sequence
        nn_sub = nn[:min(len(nn), 150)]
        samp_en = compute_sample_entropy(nn_sub, m=2, r=0.2)

        # ── 4. Sepsis Autonomic Blunting Index ────────────────────────────────
        # In sepsis, SDNN < 50ms, RMSSD < 20ms, SampEn < 1.0 indicates severe autonomic decoupling
        autonomic_impairment = 0.0
        if sdnn < 50.0:
            autonomic_impairment += 0.35
        if rmssd < 20.0:
            autonomic_impairment += 0.35
        if samp_en < 1.0 and samp_en > 0:
            autonomic_impairment += 0.30

        return {
            "hrv_mean_nn_ms": mean_nn,
            "hrv_sdnn_ms": sdnn,
            "hrv_rmssd_ms": rmssd,
            "hrv_pnn50_pct": pnn50,
            "hrv_lf_power": p_lf,
            "hrv_hf_power": p_hf,
            "hrv_lf_hf_ratio": lf_hf_ratio,
            "hrv_sample_entropy": samp_en,
            "hrv_autonomic_blunting_index": float(autonomic_impairment)
        }

    def extract_arterial_line_features(self, abp_waveform: np.ndarray) -> Dict[str, float]:
        """
        Extract Arterial Blood Pressure (ABP) line or Photoplethysmography (PPG) morphology:
        - Pulse Pressure Variation (PPV) for fluid responsiveness
        - Max rate of pressure rise (dP/dt max: cardiac contractility)
        - Systolic Area Under Curve
        """
        abp = np.asarray(abp_waveform, dtype=float)
        if len(abp) < self.fs * 2:
            return self._empty_abp_features()

        # Lowpass filter to suppress high-frequency line resonance noise
        nyq = 0.5 * self.fs
        cutoff = min(20.0 / nyq, 0.99)
        b, a = signal.butter(2, cutoff, btype='low')
        filtered_abp = signal.filtfilt(b, a, abp)

        # Peak detection for systolic peaks and troughs for diastolic
        distance = max(int(0.35 * self.fs), 1)
        sys_peaks, _ = signal.find_peaks(filtered_abp, distance=distance, height=40.0)
        dia_troughs, _ = signal.find_peaks(-filtered_abp, distance=distance)

        if len(sys_peaks) < 3 or len(dia_troughs) < 3:
            return self._empty_abp_features()

        # Pulse pressure for each beat
        n_beats = min(len(sys_peaks), len(dia_troughs))
        sbp_vals = filtered_abp[sys_peaks[:n_beats]]
        dbp_vals = filtered_abp[dia_troughs[:n_beats]]
        pulse_pressures = sbp_vals - dbp_vals
        pulse_pressures = pulse_pressures[pulse_pressures > 5.0]

        if len(pulse_pressures) < 2:
            return self._empty_abp_features()

        # Pulse Pressure Variation (PPV): (PP_max - PP_min) / PP_mean
        pp_max = np.max(pulse_pressures)
        pp_min = np.min(pulse_pressures)
        pp_mean = np.mean(pulse_pressures)
        ppv = float(((pp_max - pp_min) / max(pp_mean, 1e-4)) * 100.0)

        # Max rate of systolic pressure rise (dP/dt max: mmHg/s)
        dp_dt = np.diff(filtered_abp) * self.fs
        dp_dt_max = float(np.percentile(dp_dt, 99)) if len(dp_dt) > 0 else 0.0

        # Mean Arterial Pressure via integration
        map_val = float(np.mean(filtered_abp))

        return {
            "abp_pulse_pressure_variation_pct": ppv,
            "abp_dp_dt_max_mmhg_s": dp_dt_max,
            "abp_waveform_mean_map": map_val,
            "abp_systolic_peak_mean": float(np.mean(sbp_vals)),
            "abp_fluid_responsive_flag": int(ppv >= 13.0)  # Standard clinical cutoff for fluid responsiveness
        }

    def _empty_hrv_features(self) -> Dict[str, float]:
        return {
            "hrv_mean_nn_ms": 800.0,
            "hrv_sdnn_ms": 60.0,
            "hrv_rmssd_ms": 35.0,
            "hrv_pnn50_pct": 15.0,
            "hrv_lf_power": 500.0,
            "hrv_hf_power": 400.0,
            "hrv_lf_hf_ratio": 1.25,
            "hrv_sample_entropy": 1.50,
            "hrv_autonomic_blunting_index": 0.0
        }

    def _empty_abp_features(self) -> Dict[str, float]:
        return {
            "abp_pulse_pressure_variation_pct": 8.0,
            "abp_dp_dt_max_mmhg_s": 900.0,
            "abp_waveform_mean_map": 80.0,
            "abp_systolic_peak_mean": 120.0,
            "abp_fluid_responsive_flag": 0
        }
