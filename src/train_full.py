"""
SepsisSense full-data training pipeline (v2), GPU-accelerated.

    python src/train_full.py                 # all PhysioNet 2019 patients, GPU
    python src/train_full.py --device cpu    # no GPU

What it does
------------
1. Loads every PhysioNet/CinC 2019 training stay (sets A and B).
2. Splits by PATIENT, stratified by hospital system and sepsis status:
   70% train / 10% calibration / 20% held-out test. The test patients are
   never used for fitting, calibration or threshold choice; their IDs are saved
   so the web app replays only these unseen stays.
3. Builds the 67 v2 features (src/features_v2.py) in memory-friendly chunks,
   imputes with medians from the training split only.
4. Training set only: 6 h label dilation and trajectory-preserving hard negative
   mining (all septic + all SIRS-like non-septic + 6x quiescent controls).
5. Fits XGBoost (CUDA), LightGBM (GPU) and Random Forest (CPU), combined with
   fixed soft-voting weights 0.45 / 0.45 / 0.10.
6. Calibrates with isotonic regression on the calibration split, then chooses
   the alert threshold on that SAME calibrated scale: the best official PhysioNet
   2019 utility that alerts at most 15% of non-septic patients (alarm-burden cap),
   using a causal 2-hour persistence filter.
7. Reports everything on the held-out test split, overall and per hospital system.

Fixes relative to v1 (src/train_precision_model.py): threshold chosen and applied
on one scale; causal persistence filter; lab-availability features computed before
forward fill; calibration fitted on its own split; official utility definition.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import features_v2 as fv  # noqa: E402
import features_v3 as f3  # noqa: E402
from data_harmonization import UniversalDataHarmonizer  # noqa: E402
from precision_ensemble import SepsisPrecisionEnsemble  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_full")

DATA = ROOT / "physionet.org" / "files" / "challenge-2019" / "1.0.0" / "training"
MODELS = ROOT / "models"
WEIGHTS = {"lightgbm": 0.45, "xgboost": 0.45, "random_forest": 0.10}
CHUNK = 2500


# ------------------------------------------------------------------ loading

def read_stay(path: Path, set_name: str) -> pd.DataFrame:
    d = pd.read_csv(path, sep="|", dtype=np.float32)
    d.insert(0, "Patient", path.stem)
    d["DatasetSource"] = set_name
    return d


def list_stays() -> list[tuple[Path, str]]:
    out = []
    for s in ("A", "B"):
        folder = DATA / f"training_set{s}"
        out += [(folder / e.name, s) for e in sorted(os.scandir(folder), key=lambda e: e.name) if e.name.endswith(".psv")]
    return out


def load_chunk(items: list[tuple[Path, str]]) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=8) as ex:
        dfs = list(ex.map(lambda it: read_stay(*it), items))
    return pd.concat(dfs, ignore_index=True)


# ----------------------------------------------------------------- helpers

def dilate_labels(labels: np.ndarray, patients: np.ndarray, hours: int = 6) -> np.ndarray:
    """Mark the `hours` before each patient's first positive label as positive (vectorised)."""
    y = labels.astype(np.int8).copy()
    change = np.r_[True, patients[1:] != patients[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(y)]
    for s, e in zip(starts, ends):
        seg = labels[s:e]
        if seg.any():
            f = int(np.argmax(seg))
            y[s + max(0, f - hours): s + f] = 1
    return y


def mine_patients(meta: pd.DataFrame, raw_max: pd.DataFrame, ratio: float, seed: int) -> set:
    """All septic + all SIRS-like non-septic + `ratio` x (number septic) quiescent controls."""
    septic = meta.index[meta["septic"] == 1]
    non = meta.index[meta["septic"] == 0]
    rm = raw_max.reindex(non)
    hard = rm.index[(rm["HR"] > 95) | (rm["Temp"] > 38.3) | (rm["Resp"] > 22) | (rm["WBC"] > 12)]
    quiet = non.difference(hard)
    k = min(len(quiet), int(len(septic) * ratio))
    rng = np.random.RandomState(seed)
    sampled = rng.choice(np.asarray(quiet), size=k, replace=False) if k < len(quiet) else np.asarray(quiet)
    log.info("Mining: %d septic | %d hard negatives | %d of %d quiescent controls", len(septic), len(hard), len(sampled), len(quiet))
    return set(septic) | set(hard) | set(sampled)


def ensemble_raw(models: dict, X: np.ndarray) -> np.ndarray:
    total = sum(WEIGHTS[k] for k in models)
    return sum(WEIGHTS[k] * m.predict_proba(X)[:, 1] for k, m in models.items()) / total


def evaluate(y: np.ndarray, risk: np.ndarray, patients: np.ndarray, threshold: float) -> dict:
    alert = fv.causal_alerts(risk, patients, threshold)
    u_pos, u_neg, best, inaction = fv.utility_terms(y, patients)
    tp = int((alert & (y == 1)).sum()); fp = int((alert & (y == 0)).sum())
    fn = int((~alert & (y == 1)).sum()); tn = int((~alert & (y == 0)).sum())
    per = pd.DataFrame({"p": patients, "y": y, "a": alert}).groupby("p").agg(septic=("y", "max"), alerted=("a", "max"))
    # lead time: hours from first alert to first positive label (+6 = onset), septic patients alerted before label
    leads = []
    for pid, grp in pd.DataFrame({"p": patients, "y": y, "a": alert}).groupby("p", sort=False):
        if grp["y"].max() == 1 and grp["a"].any():
            first_lab = int(np.argmax(grp["y"].to_numpy()))
            first_alert = int(np.argmax(grp["a"].to_numpy()))
            leads.append(first_lab + 6 - first_alert)
    return {
        "patients": int(len(per)),
        "hours": int(len(y)),
        "septic_patients": int(per["septic"].sum()),
        "positive_hour_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, risk)),
        "pr_auc": float(average_precision_score(y, risk)),
        "brier_score": float(brier_score_loss(y, risk)),
        "alert_precision": tp / (tp + fp) if tp + fp else None,
        "alert_recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "septic_patients_alerted": int(per.loc[per["septic"] == 1, "alerted"].sum()),
        "non_septic_patients_alerted": int(per.loc[per["septic"] == 0, "alerted"].sum()),
        "median_hours_alert_before_onset": float(np.median(leads)) if leads else None,
        "physionet_utility": fv.normalized_utility(alert, u_pos, u_neg, best, inaction),
    }


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--neg-ratio", type=float, default=6.0, help="v2 recipe only")
    ap.add_argument("--recipe", choices=["v3", "v2"], default="v3",
                    help="v3: 185 features, native missing values, utility-weighted loss, tuned XGBoost + LightGBM "
                         "(chosen in src/experiments_v3.py). v2: previous recipe.")
    ap.add_argument("--xgb-trees", type=int, default=760, help="chosen in experiments_v3.py sweep (664 at 85%% data, x1.15)")
    ap.add_argument("--xgb-depth", type=int, default=4)
    ap.add_argument("--lgb-trees", type=int, default=210, help="chosen in experiments_v3.py sweep (183 at 85%% data, x1.15)")
    ap.add_argument("--lgb-leaves", type=int, default=127)
    ap.add_argument("--lgb-min-child", type=int, default=100)
    ap.add_argument("--xgb-weight", type=float, default=0.5, help="Soft-voting weight of XGBoost (LightGBM gets the rest)")
    ap.add_argument("--max-non-septic-alert-rate", type=float, default=0.15,
                    help="Alarm-burden cap: at most this share of non-septic calibration patients may be alerted "
                         "(threshold = best PhysioNet utility under the cap; 1.0 = pure utility maximum)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="Use only N stays (smoke test)")
    ap.add_argument("--out", type=Path, default=MODELS, help="Output folder (default: models/)")
    args = ap.parse_args()
    t_start = time.time()
    timings = {}

    stays = list_stays()
    if args.limit:
        rng = np.random.RandomState(args.seed)
        stays = [stays[i] for i in sorted(rng.choice(len(stays), args.limit, replace=False))]
    log.info("Stays found: %d (set A %d, set B %d)", len(stays), sum(s == "A" for _, s in stays), sum(s == "B" for _, s in stays))

    # ---- 1. features in chunks ----
    t0 = time.time()
    harmonizer = UniversalDataHarmonizer(cache_dir=str(ROOT / "data" / "processed"))
    feat_parts, info_parts, rawmax_parts = [], [], []
    for i in range(0, len(stays), CHUNK):
        raw = harmonizer.harmonize_dataframe(load_chunk(stays[i:i + CHUNK]), source_name="PhysioNet")
        feat_parts.append((f3 if args.recipe == "v3" else fv).engineer(raw))
        info_parts.append(pd.DataFrame({
            "Patient": raw["Patient"].to_numpy(),
            "Set": raw["DatasetSource"].to_numpy(),
            "y": raw["SepsisLabel"].fillna(0).to_numpy(dtype=np.int8),
        }))
        rawmax_parts.append(raw.groupby("Patient", sort=False)[["HR", "Temp", "Resp", "WBC"]].max())
        del raw
        gc.collect()
        log.info("  features: %d / %d stays", min(i + CHUNK, len(stays)), len(stays))
    feats = pd.concat(feat_parts, ignore_index=True); del feat_parts
    info = pd.concat(info_parts, ignore_index=True); del info_parts
    raw_max = pd.concat(rawmax_parts); del rawmax_parts
    gc.collect()
    timings["features_s"] = round(time.time() - t0, 1)
    log.info("Feature matrix: %s rows x %d features (%.0f MB)", f"{len(feats):,}", feats.shape[1], feats.memory_usage().sum() / 2**20)

    # ---- 2. patient-level split ----
    meta = info.groupby("Patient", sort=False).agg(septic=("y", "max"), Set=("Set", "first"))
    strata = meta["Set"] + "_" + meta["septic"].astype(str)
    train_p, rest_p = train_test_split(meta.index, test_size=0.30, random_state=args.seed, stratify=strata)
    cal_p, test_p = train_test_split(rest_p, test_size=2 / 3, random_state=args.seed, stratify=strata.loc[rest_p])
    split = {"train": sorted(train_p), "calibration": sorted(cal_p), "test": sorted(test_p)}
    role = pd.Series("train", index=meta.index)
    role.loc[list(cal_p)] = "calibration"
    role.loc[list(test_p)] = "test"
    info["role"] = info["Patient"].map(role).to_numpy()
    for r in ("train", "calibration", "test"):
        m = meta.loc[split[r]]
        log.info("  %-11s %6d patients (%d septic) | set A %d, set B %d", r, len(m), int(m["septic"].sum()), int((m["Set"] == "A").sum()), int((m["Set"] == "B").sum()))

    # ---- 3. impute with training medians ----
    is_train = (info["role"] == "train").to_numpy()
    feature_names = list(feats.columns)
    medians = {c: (float(v) if np.isfinite(v) else 0.0)
               for c, v in zip(feature_names, np.nanmedian(feats.loc[is_train].to_numpy(np.float64), axis=0))}
    impute = args.recipe == "v2"
    if impute:  # v3 lets the trees use missing values natively (medians still saved for reference)
        feats = feats.fillna(medians)

    # ---- 4. training rows: dilation + mining ----
    t0 = time.time()
    tr_info = info.loc[is_train].reset_index(drop=True)
    X_tr_all = feats.loc[is_train].to_numpy(dtype=np.float32)
    w_tr = None
    if args.recipe == "v3":
        # Utility-weighted loss (Hammoud et al. 2019): label = utility-optimal action, weight = |U(1) - U(0)|.
        u_pos, u_neg, _, _ = fv.utility_terms(tr_info["y"].to_numpy(), tr_info["Patient"].to_numpy())
        X_tr = X_tr_all
        y_tr = (u_pos > u_neg).astype(np.int8)
        w_tr = np.abs(u_pos - u_neg).astype(np.float32)
        spw = 1.0
        log.info("Utility-weighted training: %s rows, %s positive targets, weight range %.2f-%.2f",
                 f"{len(y_tr):,}", f"{int(y_tr.sum()):,}", float(w_tr.min()), float(w_tr.max()))
    else:
        y_tr_all = dilate_labels(tr_info["y"].to_numpy(), tr_info["Patient"].to_numpy(), hours=6)
        keep_p = mine_patients(meta.loc[split["train"]], raw_max, args.neg_ratio, args.seed)
        keep = tr_info["Patient"].isin(keep_p).to_numpy()
        X_tr, y_tr = X_tr_all[keep], y_tr_all[keep]
        del y_tr_all
        spw = min(int((y_tr == 0).sum()) / max(int(y_tr.sum()), 1), 6.0)
        log.info("Training rows after mining: %s (%s positive, scale_pos_weight %.2f)", f"{len(y_tr):,}", f"{int(y_tr.sum()):,}", spw)
    gc.collect()
    pos, neg = int(y_tr.sum()), int(len(y_tr) - y_tr.sum())

    # ---- 5. fit models ----
    gpu = args.device == "gpu"
    models = {}
    t1 = time.time()
    if args.recipe == "v3":
        xgb_params = dict(n_estimators=args.xgb_trees, learning_rate=0.03, max_depth=args.xgb_depth, min_child_weight=20,
                          subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0)
    else:
        xgb_params = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
                          scale_pos_weight=spw)
    models["xgboost"] = xgb.XGBClassifier(
        **xgb_params, tree_method="hist", device="cuda" if gpu else "cpu", random_state=args.seed, n_jobs=-1, verbosity=0,
    ).fit(X_tr, y_tr, sample_weight=w_tr)
    timings["xgboost_s"] = round(time.time() - t1, 1)
    log.info("  XGBoost (%s) trained in %.1fs", "CUDA" if gpu else "CPU", timings["xgboost_s"])

    t1 = time.time()
    if args.recipe == "v3":
        lgb_params = dict(n_estimators=args.lgb_trees, learning_rate=0.03, num_leaves=args.lgb_leaves,
                          min_child_samples=args.lgb_min_child, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                          reg_lambda=2.0, random_state=args.seed, n_jobs=-1, verbosity=-1)
        lgb_try = ["cpu"]  # CPU histogram LightGBM matched the experiments and is fast on 12 threads
    else:
        lgb_params = dict(n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05, subsample=0.85,
                          subsample_freq=1, colsample_bytree=0.85, scale_pos_weight=spw, random_state=args.seed,
                          n_jobs=-1, verbosity=-1)
        lgb_try = ["gpu", "cpu"] if gpu else ["cpu"]
    lgb_device = None
    for dev in lgb_try:
        try:
            models["lightgbm"] = lgb.LGBMClassifier(device=dev, **lgb_params).fit(X_tr, y_tr, sample_weight=w_tr)
            lgb_device = dev.upper()
            break
        except Exception as e:  # OpenCL GPU unavailable: fall back cleanly
            log.warning("  LightGBM %s failed (%s)", dev, e)
    timings["lightgbm_s"] = round(time.time() - t1, 1)
    log.info("  LightGBM (%s) trained in %.1fs", lgb_device, timings["lightgbm_s"])

    if args.recipe == "v2":
        t1 = time.time()
        models["random_forest"] = RandomForestClassifier(
            n_estimators=80, max_depth=8, max_features="sqrt", class_weight="balanced_subsample",
            random_state=args.seed, n_jobs=6,
        ).fit(X_tr, y_tr)
        timings["random_forest_s"] = round(time.time() - t1, 1)
        log.info("  Random forest (CPU) trained in %.1fs", timings["random_forest_s"])
        WEIGHTS.update({"lightgbm": 0.45, "xgboost": 0.45, "random_forest": 0.10})
    else:
        WEIGHTS.clear()
        WEIGHTS.update({"xgboost": args.xgb_weight, "lightgbm": 1.0 - args.xgb_weight})
    if gpu:  # predict on CPU later so the web app never needs CUDA
        models["xgboost"].set_params(device="cpu")
    del X_tr, y_tr
    gc.collect()
    timings["training_rows_s"] = round(time.time() - t0, 1)

    # ---- 6. calibrate + choose threshold on the calibration split ----
    is_cal = (info["role"] == "calibration").to_numpy()
    X_cal = feats.loc[is_cal].to_numpy(dtype=np.float32)
    cal_info = info.loc[is_cal].reset_index(drop=True)
    raw_cal = ensemble_raw(models, X_cal)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_cal, cal_info["y"].to_numpy())
    risk_cal = iso.predict(raw_cal)
    y_cal, p_cal = cal_info["y"].to_numpy(), cal_info["Patient"].to_numpy()
    u_pos, u_neg, best, inaction = fv.utility_terms(y_cal, p_cal)
    candidates = np.unique(np.round(risk_cal, 4))
    candidates = candidates[(candidates > 0.005) & (candidates < 0.95)]
    cal_codes, cal_uniques = pd.factorize(p_cal)
    cal_septic = np.zeros(len(cal_uniques), dtype=bool)
    np.logical_or.at(cal_septic, cal_codes, y_cal == 1)
    scan = []
    for t in candidates:
        a = fv.causal_alerts(risk_cal, p_cal, float(t))
        tp = int((a & (y_cal == 1)).sum()); fp = int((a & (y_cal == 0)).sum())
        pt_alerted = np.zeros(len(cal_uniques), dtype=bool)
        np.logical_or.at(pt_alerted, cal_codes, a)
        scan.append({
            "threshold": float(t),
            "utility": fv.normalized_utility(a, u_pos, u_neg, best, inaction),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / max(1, int(y_cal.sum())),
            "alert_rate": float(a.mean()),
            "septic_patients_alerted_rate": float(pt_alerted[cal_septic].mean()),
            "non_septic_patients_alerted_rate": float(pt_alerted[~cal_septic].mean()),
        })
    scan_df = pd.DataFrame(scan)
    utility_opt = scan_df.loc[scan_df["utility"].idxmax()]
    capped = scan_df[scan_df["non_septic_patients_alerted_rate"] <= args.max_non_septic_alert_rate]
    best_row = capped.loc[capped["utility"].idxmax()] if len(capped) else utility_opt
    hp = scan_df[scan_df["precision"] >= 0.30]
    high_precision = hp.loc[hp["recall"].idxmax()].to_dict() if len(hp) else None
    threshold = float(best_row["threshold"])
    log.info("Alert threshold (calibrated risk) = %.4f -> calibration utility %.3f, septic patients alerted %.0f%%, "
             "non-septic alerted %.0f%% (cap %.0f%%; unconstrained utility max %.3f at %.4f)",
             threshold, best_row["utility"], 100 * best_row["septic_patients_alerted_rate"],
             100 * best_row["non_septic_patients_alerted_rate"], 100 * args.max_non_septic_alert_rate,
             utility_opt["utility"], utility_opt["threshold"])
    del X_cal
    gc.collect()

    # ---- 7. held-out test ----
    is_test = (info["role"] == "test").to_numpy()
    X_te = feats.loc[is_test].to_numpy(dtype=np.float32)
    te_info = info.loc[is_test].reset_index(drop=True)
    risk_te = iso.predict(ensemble_raw(models, X_te))
    test_metrics = evaluate(te_info["y"].to_numpy(), risk_te, te_info["Patient"].to_numpy(), threshold)
    by_set = {}
    for s in ("A", "B"):
        m = (te_info["Set"] == s).to_numpy()
        by_set[s] = evaluate(te_info["y"].to_numpy()[m], risk_te[m], te_info["Patient"].to_numpy()[m], threshold)
    log.info("TEST: ROC-AUC %.3f | PR-AUC %.3f | utility %.3f | alert precision %s | septic alerted %d/%d | non-septic alerted %d",
             test_metrics["roc_auc"], test_metrics["pr_auc"], test_metrics["physionet_utility"],
             f"{test_metrics['alert_precision']:.3f}" if test_metrics["alert_precision"] is not None else "n/a",
             test_metrics["septic_patients_alerted"], test_metrics["septic_patients"], test_metrics["non_septic_patients_alerted"])
    for s, mm in by_set.items():
        log.info("  set %s: ROC-AUC %.3f | utility %.3f | septic alerted %d/%d", s, mm["roc_auc"], mm["physionet_utility"],
                 mm["septic_patients_alerted"], mm["septic_patients"])

    # ---- 8. save (back up v1 first) ----
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    backup = MODELS / "v1_backup"
    if out == MODELS and not backup.exists():
        backup.mkdir(parents=True)
        for f in ("sepsissense_precision_ensemble.pkl", "precision_metrics.json", "precision_feature_columns.pkl"):
            if (MODELS / f).exists():
                shutil.copy2(MODELS / f, backup / f)
        log.info("Backed up v1 model files to %s", backup)

    ens = SepsisPrecisionEnsemble(target_precision=0.0, random_state=args.seed)
    ens.models = models
    ens.feature_names = feature_names
    ens.member_weights = dict(WEIGHTS)
    ens.impute = impute
    ens.recipe = args.recipe
    ens.optimizer.calibrator = iso
    ens.optimizer.optimal_threshold = threshold
    ens.optimal_threshold = threshold
    ens.is_fitted = True
    joblib.dump(ens, out / "sepsissense_precision_ensemble.pkl", compress=3)
    joblib.dump(feature_names, out / "precision_feature_columns.pkl")
    (out / "feature_medians.json").write_text(json.dumps(medians, indent=1), encoding="utf-8")
    (out / "data_split.json").write_text(json.dumps(split), encoding="utf-8")

    # Scores for re-evaluating other operating points without retraining (derived data; git-ignored).
    np.savez_compressed(
        out / "eval_scores.npz",
        cal_risk=risk_cal, cal_y=y_cal, cal_patient=p_cal,
        test_risk=risk_te, test_y=te_info["y"].to_numpy(), test_patient=te_info["Patient"].to_numpy(),
        test_set=te_info["Set"].to_numpy(),
    )
    timings["total_s"] = round(time.time() - t_start, 1)
    metrics = {
        "version": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_sources": ["PhysioNet_SetA", "PhysioNet_SetB"],
        "total_patients": int(len(meta)),
        "total_records": int(len(info)),
        "sepsis_patients": int(meta["septic"].sum()),
        "split": {r: {"patients": len(split[r]), "septic": int(meta.loc[split[r], "septic"].sum())} for r in split},
        "training_rows_after_mining": int(pos + neg),
        "recipe": args.recipe,
        "n_features": len(feature_names),
        "member_weights": dict(WEIGHTS),
        "hyperparameters": {"xgboost": xgb_params, "lightgbm": {k: v for k, v in lgb_params.items() if k not in ("verbosity", "n_jobs")}},
        "devices": {"xgboost": "CUDA" if gpu else "CPU", "lightgbm": lgb_device,
                    **({"random_forest": "CPU"} if "random_forest" in models else {})},
        "threshold": {
            "value": threshold,
            "scale": "calibrated probability",
            "rule": (f"maximise official PhysioNet 2019 utility on the calibration split while alerting at most "
                     f"{args.max_non_septic_alert_rate:.0%} of non-septic patients; causal 2 h persistence"),
            "max_non_septic_alert_rate": args.max_non_septic_alert_rate,
            "calibration_utility": float(best_row["utility"]),
            "calibration_precision": float(best_row["precision"]),
            "calibration_recall": float(best_row["recall"]),
            "calibration_septic_patients_alerted_rate": float(best_row["septic_patients_alerted_rate"]),
            "calibration_non_septic_patients_alerted_rate": float(best_row["non_septic_patients_alerted_rate"]),
            "utility_optimal_alternative": utility_opt.to_dict(),
            "high_precision_alternative": high_precision,
        },
        "calibrator_max": float(np.max(iso.y_thresholds_)),
        "validation_metrics": test_metrics,  # held-out TEST split
        "test_by_set": by_set,
        "threshold_scan": scan_df.round(4).to_dict(orient="records")[:: max(1, len(scan_df) // 60)],
        "timings_s": timings,
    }
    (out / "precision_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    log.info("Saved model and metrics. Total time %.1f min", timings["total_s"] / 60)


if __name__ == "__main__":
    main()
