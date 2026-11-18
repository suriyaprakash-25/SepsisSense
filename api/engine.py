"""
SepsisSense inference engine for the clinician web UI (model v2).

Replays real, de-identified ICU stays from the public PhysioNet/CinC 2019
Challenge through the trained ensemble.

* Only the held-out TEST split from training (models/data_split.json) is used:
  these patients were never used for fitting, calibration or threshold choice.
* Features come from src/features_v2.py, the same code used in training, and
  missing values are imputed with the exact training medians
  (models/feature_medians.json).
* Alerts: calibrated risk >= the saved threshold (chosen on the same calibrated
  scale), with a causal 2-hour persistence filter.
* Ward composition: the first 7 eligible septic and first 9 eligible non-septic
  test stays (ID order), plus up to 2 stays chosen *because the model alerted*
  (one septic, one not) to demonstrate the alert workflow. Every patient carries
  its selection reason. Pool statistics cover ALL test patients.
* Risk drivers are TreeSHAP contributions from the XGBoost and LightGBM members
  (log-odds, averaged); they approximate *why*, not an exact decomposition.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

import features_v2 as fv  # noqa: E402
from data_harmonization import UniversalDataHarmonizer  # noqa: E402

from clinical import FEATURE_CONCEPT, concept_for, feature_label  # noqa: E402

log = logging.getLogger("sepsissense.engine")

DATA = ROOT / "physionet.org" / "files" / "challenge-2019" / "1.0.0" / "training"
SET_A = DATA / "training_setA"
MODEL_PATH = ROOT / "models" / "sepsissense_precision_ensemble.pkl"
METRICS_PATH = ROOT / "models" / "precision_metrics.json"
MEDIANS_PATH = ROOT / "models" / "feature_medians.json"
SPLIT_PATH = ROOT / "models" / "data_split.json"
CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "cohort_v5.pkl"

VITALS = ["HR", "MAP", "SBP", "DBP", "Resp", "Temp", "O2Sat", "EtCO2"]
LABS = [
    "Lactate", "WBC", "Platelets", "Creatinine", "BUN", "Bilirubin_total", "pH",
    "HCO3", "BaseExcess", "PaCO2", "FiO2", "SaO2", "Glucose", "Potassium",
    "Chloride", "Calcium", "Magnesium", "Phosphate", "Hgb", "Hct", "PTT",
    "Fibrinogen", "AST", "Alkalinephos", "Bilirubin_direct", "TroponinI",
]

N_SEPTIC = 7
N_NON_SEPTIC = 9


@dataclass
class PatientRecord:
    pid: str
    bed: str
    unit: str
    age: float
    sex: str
    hosp_adm_hours: float
    n_hours: int
    base_hour: int
    sepsis_label_hour: Optional[int]  # first hour with SepsisLabel = 1 (onset - 6 h); research view only
    selection_reason: str
    raw: pd.DataFrame  # harmonized measurements; NaN = not charted that hour
    risk: np.ndarray  # calibrated probability (alert scale)
    raw_score: np.ndarray  # uncalibrated ensemble score
    member_risk: Dict[str, np.ndarray]
    alert: np.ndarray
    contrib: np.ndarray  # (hours, n_features) log-odds contributions
    feature_values: pd.DataFrame  # model inputs per hour (after imputation)


@dataclass
class Cohort:
    patients: Dict[str, PatientRecord] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    threshold: float = 0.5
    base_rate: float = 0.02
    risk_scale_max: float = 0.4
    pool_stats: Dict[str, Any] = field(default_factory=dict)


def stay_path(pid: str) -> Path:
    return DATA / ("training_setB" if pid.startswith("p1") else "training_setA") / f"{pid}.psv"


def _read_psv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    df["Patient"] = path.stem
    df["DatasetSource"] = "B" if path.stem.startswith("p1") else "A"
    return df


def load_model() -> Any:
    ens = joblib.load(MODEL_PATH)
    for m in ens.models.values():
        try:
            m.set_params(device="cpu")
        except Exception:
            pass
    return ens


def model_score(ens: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = ens._raw_predict_proba(X)[:, 1]
    return raw, np.clip(ens.optimizer.calibrate(raw), 0.0, 1.0)


def tree_contributions(ens: Any, X: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    parts = []
    if "xgboost" in ens.models:
        booster = ens.models["xgboost"].get_booster()
        c = booster.predict(xgb.DMatrix(X, feature_names=booster.feature_names), pred_contribs=True)
        parts.append(c[:, :-1])
    if "lightgbm" in ens.models:
        c = ens.models["lightgbm"].predict(X, pred_contrib=True)
        parts.append(np.asarray(c)[:, :-1])
    if not parts:
        return np.zeros_like(X)
    return sum(parts) / len(parts)


def _pool_stats(y: np.ndarray, patients: np.ndarray, sets: np.ndarray, risk: np.ndarray, alert: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    df = pd.DataFrame({"p": patients, "y": y, "a": alert, "s": sets})
    per = df.groupby("p").agg(septic=("y", "max"), alerted=("a", "max"))
    tp = int(((alert == 1) & (y == 1)).sum())
    fp = int(((alert == 1) & (y == 0)).sum())
    u_pos, u_neg, best, inaction = fv.utility_terms(y, patients)
    return {
        "patients": int(len(per)),
        "hours": int(len(y)),
        "septic_patients": int(per["septic"].sum()),
        "hour_roc_auc": float(roc_auc_score(y, risk)),
        "hour_pr_auc": float(average_precision_score(y, risk)),
        "positive_hour_rate": float(y.mean()),
        "alert_hours": int(alert.sum()),
        "alert_precision": float(tp / (tp + fp)) if tp + fp else None,
        "alert_recall": float(tp / max(1, int(y.sum()))),
        "septic_patients_alerted": int(per.loc[per["septic"] == 1, "alerted"].sum()),
        "non_septic_patients_alerted": int(per.loc[per["septic"] == 0, "alerted"].sum()),
        "physionet_utility": fv.normalized_utility(alert, u_pos, u_neg, best, inaction),
    }


def build_cohort() -> Cohort:
    if not SPLIT_PATH.exists() or not MEDIANS_PATH.exists():
        raise FileNotFoundError("models/data_split.json or feature_medians.json missing: run python src/train_full.py")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    test_ids = sorted(split["test"])
    medians = fv.load_medians(MEDIANS_PATH)

    log.info("Reading %d held-out test stays...", len(test_ids))
    paths = [stay_path(p) for p in test_ids]
    paths = [p for p in paths if p.exists()]
    with ThreadPoolExecutor(max_workers=8) as ex:
        dfs = list(ex.map(_read_psv, paths))
    raw_all = UniversalDataHarmonizer(cache_dir=str(CACHE_PATH.parent)).harmonize_dataframe(
        pd.concat(dfs, ignore_index=True), source_name="PhysioNet"
    ).reset_index(drop=True)
    del dfs

    log.info("Engineering features...")
    feats = fv.impute(fv.engineer(raw_all), medians)
    ens = load_model()
    fnames = list(ens.feature_names)
    t = float(ens.optimizer.optimal_threshold)
    X = feats[fnames].to_numpy(dtype=np.float32)

    log.info("Scoring %d hours...", len(X))
    raw_score, risk = model_score(ens, X)
    pid_arr = raw_all["Patient"].to_numpy()
    labels_all = raw_all["SepsisLabel"].fillna(0).to_numpy(dtype=int)
    alert = fv.causal_alerts(risk, pid_arr, t).astype(int)
    order = list(dict.fromkeys(pid_arr))
    groups: Dict[str, np.ndarray] = {}
    start = 0
    for i in range(1, len(pid_arr) + 1):
        if i == len(pid_arr) or pid_arr[i] != pid_arr[start]:
            groups[pid_arr[start]] = np.arange(start, i)
            start = i

    # ---- ward selection (deterministic, ID order) ----
    chosen: Dict[str, str] = {}
    n_s = n_n = 0
    for pid in order:
        labels = labels_all[groups[pid]]
        n = len(labels)
        if labels.max() == 1:
            first = int(labels.argmax())
            if n_s < N_SEPTIC and 12 <= first <= 70 and n >= first + 8:
                chosen[pid] = "consecutive_septic"
                n_s += 1
        elif n_n < N_NON_SEPTIC and 36 <= n <= 120:
            chosen[pid] = "consecutive_non_septic"
            n_n += 1
    want = {"septic": True, "non_septic": True}
    for pid in order:
        if pid in chosen or not any(want.values()):
            continue
        idx = groups[pid]
        if alert[idx].max() == 0 or len(idx) < 24 or int(np.flatnonzero(alert[idx])[0]) < 8:
            continue
        key = "septic" if labels_all[idx].max() == 1 else "non_septic"
        if want[key]:
            chosen[pid] = "alert_example"
            want[key] = False

    members_all = {k: m.predict_proba(X)[:, 1] for k, m in ens.models.items()}
    sel_rows = np.concatenate([groups[p] for p in chosen])
    contrib_sel = tree_contributions(ens, X[sel_rows])
    contrib_lookup, off = {}, 0
    for p in chosen:
        k = len(groups[p])
        contrib_lookup[p] = contrib_sel[off:off + k]
        off += k

    rng = np.random.default_rng(7)
    septic_offsets = [-8, -5, -2, 0, 2, 4, 6]
    s_i = 0
    recs: List[dict] = []
    for pid, reason in chosen.items():
        idx = groups[pid]
        raw_p = raw_all.iloc[idx].reset_index(drop=True)
        labels = labels_all[idx]
        n = len(idx)
        label_hour = int(labels.argmax()) if labels.max() == 1 else None
        a = alert[idx]
        if reason == "alert_example":
            # Start at the last hour of the first alert episode, so the alert is active on the ward.
            base = int(np.flatnonzero(a)[0])
            while base + 1 < n and a[base + 1]:
                base += 1
        elif label_hour is not None:
            base = label_hour + septic_offsets[s_i % len(septic_offsets)]
            s_i += 1
        else:
            base = int(rng.integers(12, max(13, n - 14)))
        unit = "SICU" if raw_p["Unit2"].fillna(0).max() == 1 else "MICU"
        adm = raw_p["HospAdmTime"].iloc[0]
        recs.append(dict(
            pid=pid, bed="", unit=unit,
            age=float(raw_p["Age"].iloc[0]),
            sex="M" if raw_p["Gender"].iloc[0] == 1 else "F",
            hosp_adm_hours=float(-adm) if pd.notna(adm) else 0.0,
            n_hours=n,
            base_hour=int(np.clip(base, 6, n - 1)),
            sepsis_label_hour=label_hour,
            selection_reason=reason,
            raw=raw_p,
            risk=risk[idx],
            raw_score=raw_score[idx],
            member_risk={k: v[idx] for k, v in members_all.items()},
            alert=a,
            contrib=contrib_lookup[pid],
            feature_values=feats.iloc[idx][fnames].reset_index(drop=True),
        ))
    # Beds numbered per unit in a stable order
    counters = {"MICU": 0, "SICU": 0}
    for r in sorted(recs, key=lambda r: (r["unit"], r["pid"])):
        counters[r["unit"]] += 1
        r["bed"] = f"{r['unit']}-{counters[r['unit']]:02d}"

    cohort = Cohort(feature_names=fnames, threshold=t)
    cohort.base_rate = float(labels_all.mean())
    cal_max = float(np.max(getattr(ens.optimizer.calibrator, "y_thresholds_", [0.4])))
    cohort.risk_scale_max = float(min(1.0, max(0.2, math.ceil(cal_max * 10) / 10)))
    cohort.pool_stats = _pool_stats(labels_all, pid_arr, raw_all["DatasetSource"].to_numpy(), risk, alert)
    for r in sorted(recs, key=lambda r: (r["unit"] != "MICU", r["bed"])):
        cohort.patients[r["pid"]] = PatientRecord(**r)
    return cohort


def load_cohort(force: bool = False) -> Cohort:
    if CACHE_PATH.exists() and not force:
        stale = MODEL_PATH.exists() and MODEL_PATH.stat().st_mtime > CACHE_PATH.stat().st_mtime
        if not stale:
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        log.info("Model is newer than the cohort cache; rebuilding.")
    cohort = build_cohort()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cohort, f)
    return cohort


# ------------------------------------------------------------- driver grouping

def grouped_drivers(rec: PatientRecord, hour: int, feature_names: List[str], top: int = 6) -> List[Dict[str, Any]]:
    c = rec.contrib[hour]
    fv_row = rec.feature_values.iloc[hour]
    groups: Dict[str, float] = {}
    members: Dict[str, List[Dict[str, Any]]] = {}
    for j, f in enumerate(feature_names):
        concept = concept_for(f)
        groups[concept] = groups.get(concept, 0.0) + float(c[j])
        members.setdefault(concept, []).append({
            "feature": f,
            "label": feature_label(f),
            "value": float(fv_row[f]),
            "contribution": float(c[j]),
        })
    ranked = sorted(groups.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top]
    out = []
    for concept, val in ranked:
        feats = sorted(members[concept], key=lambda m: abs(m["contribution"]), reverse=True)[:4]
        out.append({
            "concept": concept,
            "label": FEATURE_CONCEPT[concept]["label"],
            "primary": FEATURE_CONCEPT[concept]["primary"],
            "contribution": val,
            "features": feats,
        })
    return out
