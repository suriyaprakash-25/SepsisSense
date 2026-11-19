"""
Model-improvement experiments for SepsisSense (research log, reproducible).

    python src/experiments_v3.py            # builds the feature cache on first run (~5 min), then runs all variants

Protocol (no test-set peeking):
  * Uses the SAME patient split as model v2 (models/data_split.json).
  * Training patients are split again, 85 / 15, into inner-train and inner-validation
    (patient-level, stratified by hospital system and sepsis status).
  * Every variant is trained on inner-train and compared on inner-validation.
    The calibration and test patients are not touched here.

Variants are motivated by PhysioNet/CinC 2019 literature:
  E0  v2 baseline: 67 features, median-imputed, XGB+LGB+RF 300 trees, class weight 6
  E1  + v3 features (185): measurement counts/recency, vital dynamics, partial SOFA,
      SIRS, NEWS, BUN/Cr   (Du et al. 2019; Yang et al. 2019; Morrill et al. 2019)
  E2  E1 but trees see missing values natively (no imputation), RF dropped
  E3  E2 + tuned boosting: more trees, lower learning rate, early stopping
  E4  E3 + utility-weighted loss: label = utility-optimal action, weight = |U(1) - U(0)|
      (Hammoud et al. 2019)
  E5  E3 without time-in-hospital features (ICULOS, HospAdmTime): shortcut ablation
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import features_v2 as fv  # noqa: E402
import features_v3 as f3  # noqa: E402
from data_harmonization import UniversalDataHarmonizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("experiments")
logging.getLogger("data_harmonization").setLevel(logging.WARNING)

DATA = ROOT / "physionet.org" / "files" / "challenge-2019" / "1.0.0" / "training"
CACHE = ROOT / "data" / "processed" / "features_v3"
OUT = ROOT / "models" / "experiments_v3.json"
SEED = 42
LOS_FEATURES = ["ICULOS", "HospAdmTime"]


# ------------------------------------------------------------- feature cache

def build_cache() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    stays = []
    for s in ("A", "B"):
        folder = DATA / f"training_set{s}"
        stays += [(folder / e.name, s) for e in sorted(os.scandir(folder), key=lambda e: e.name) if e.name.endswith(".psv")]
    log.info("Building v3 feature cache for %d stays...", len(stays))
    harm = UniversalDataHarmonizer(cache_dir=str(CACHE))

    def read(item):
        path, s = item
        d = pd.read_csv(path, sep="|", dtype=np.float32)
        d.insert(0, "Patient", path.stem)
        d["DatasetSource"] = s
        return d

    X_parts, info_parts = [], []
    for i in range(0, len(stays), 2500):
        with ThreadPoolExecutor(max_workers=8) as ex:
            raw = pd.concat(list(ex.map(read, stays[i:i + 2500])), ignore_index=True)
        raw = harm.harmonize_dataframe(raw, source_name="PhysioNet").reset_index(drop=True)
        X_parts.append(f3.engineer(raw).to_numpy(dtype=np.float32))
        info_parts.append(pd.DataFrame({
            "Patient": raw["Patient"].to_numpy(), "Set": raw["DatasetSource"].to_numpy(),
            "y": raw["SepsisLabel"].fillna(0).to_numpy(dtype=np.int8),
        }))
        del raw
        gc.collect()
        log.info("  %d / %d", min(i + 2500, len(stays)), len(stays))
    np.save(CACHE / "X.npy", np.concatenate(X_parts))
    pd.concat(info_parts, ignore_index=True).to_parquet(CACHE / "info.parquet")
    (CACHE / "features.json").write_text(json.dumps(f3.FEATURES), encoding="utf-8")


def load_cache():
    if not (CACHE / "X.npy").exists() or json.loads((CACHE / "features.json").read_text()) != f3.FEATURES:
        build_cache()
    X = np.load(CACHE / "X.npy", mmap_mode="r")
    info = pd.read_parquet(CACHE / "info.parquet")
    return X, info, list(f3.FEATURES)


# ------------------------------------------------------------------ helpers

def dilate(y: np.ndarray, pts: np.ndarray, hours: int = 6) -> np.ndarray:
    out = y.astype(np.int8).copy()
    starts = np.flatnonzero(np.r_[True, pts[1:] != pts[:-1]])
    ends = np.r_[starts[1:], len(y)]
    for s, e in zip(starts, ends):
        if y[s:e].any():
            f = int(np.argmax(y[s:e]))
            out[s + max(0, f - hours): s + f] = 1
    return out


def operating_metrics(score: np.ndarray, y: np.ndarray, pts: np.ndarray, sets: np.ndarray) -> dict:
    """Threshold-free metrics plus the SepsisSense operating rule (best utility with <= 15% non-septic alerted)."""
    u_pos, u_neg, best, ina = fv.utility_terms(y, pts)
    codes, uniq = pd.factorize(pts)
    septic = np.zeros(len(uniq), bool)
    np.logical_or.at(septic, codes, y == 1)
    qs = np.unique(np.quantile(score, np.linspace(0.5, 0.999, 220)))
    rows = []
    for t in qs:
        a = fv.causal_alerts(score, pts, float(t))
        pa = np.zeros(len(uniq), bool)
        np.logical_or.at(pa, codes, a)
        tp = int((a & (y == 1)).sum()); fp = int((a & (y == 0)).sum())
        rows.append((float(t), fv.normalized_utility(a, u_pos, u_neg, best, ina), pa[septic].mean(), pa[~septic].mean(),
                     tp / (tp + fp) if tp + fp else 0.0))
    scan = pd.DataFrame(rows, columns=["t", "utility", "septic_alerted", "non_septic_alerted", "hour_precision"])
    capped = scan[scan.non_septic_alerted <= 0.15]
    op = capped.loc[capped.utility.idxmax()] if len(capped) else scan.loc[scan.utility.idxmax()]
    by_set = {s: (float(roc_auc_score(y[sets == s], score[sets == s])) if (sets == s).any() else None) for s in ("A", "B")}
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "max_utility": float(scan.utility.max()),
        "op_utility": float(op.utility),
        "op_septic_alerted": float(op.septic_alerted),
        "op_non_septic_alerted": float(op.non_septic_alerted),
        "op_hour_precision": float(op.hour_precision),
        "roc_auc_set_A": by_set["A"], "roc_auc_set_B": by_set["B"],
    }


# ------------------------------------------------------------------ models

def fit_variant(name, Xtr, ytr, wtr, Xva, yva, cfg) -> tuple[np.ndarray, dict]:
    t0 = time.time()
    preds, info = {}, {}
    spw = cfg.get("spw")
    if cfg["tuned"]:
        xgbm = xgb.XGBClassifier(
            n_estimators=3000, learning_rate=0.03, max_depth=6, min_child_weight=20, subsample=0.8,
            colsample_bytree=0.6, reg_lambda=2.0, scale_pos_weight=spw or 1.0, tree_method="hist", device="cuda",
            eval_metric="aucpr", early_stopping_rounds=150, random_state=SEED, verbosity=0, max_bin=256,
        )
        xgbm.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
        info["xgb_trees"] = int(xgbm.best_iteration) + 1
        lgbm = lgb.LGBMClassifier(
            n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=100, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0, scale_pos_weight=spw or 1.0,
            random_state=SEED, n_jobs=10, verbosity=-1,
        )
        lgbm.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], eval_metric="average_precision",
                 callbacks=[lgb.early_stopping(150, verbose=False)])
        info["lgb_trees"] = int(lgbm.best_iteration_)
    else:
        xgbm = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist", device="cuda", random_state=SEED, verbosity=0,
        ).fit(Xtr, ytr)
        lgbm = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.85, scale_pos_weight=spw, random_state=SEED, n_jobs=10, verbosity=-1,
        ).fit(Xtr, ytr)
    xgbm.set_params(device="cpu")
    preds["xgboost"] = xgbm.predict_proba(Xva)[:, 1]
    preds["lightgbm"] = lgbm.predict_proba(Xva)[:, 1]
    weights = {"xgboost": 0.5, "lightgbm": 0.5}
    if cfg.get("rf"):
        rf = RandomForestClassifier(n_estimators=80, max_depth=8, max_features="sqrt", class_weight="balanced_subsample",
                                    random_state=SEED, n_jobs=6).fit(Xtr, ytr)
        preds["random_forest"] = rf.predict_proba(Xva)[:, 1]
        weights = {"xgboost": 0.45, "lightgbm": 0.45, "random_forest": 0.10}
    score = sum(weights[k] * preds[k] for k in preds)
    info["fit_seconds"] = round(time.time() - t0, 1)
    info["member_roc_auc"] = {k: round(float(roc_auc_score(yva, v)), 4) for k, v in preds.items()}
    return score, info


def main() -> None:
    X, info, feats = load_cache()
    split = json.loads((ROOT / "models" / "data_split.json").read_text())
    train_p = np.asarray(split["train"])
    meta = info.groupby("Patient", sort=False).agg(septic=("y", "max"), Set=("Set", "first")).loc[train_p]
    strata = meta["Set"] + "_" + meta["septic"].astype(str)
    in_tr, in_va = train_p, None
    in_tr, in_va = train_test_split(meta.index, test_size=0.15, random_state=SEED, stratify=strata)
    in_tr, in_va = set(in_tr), set(in_va)
    p_all = info["Patient"].to_numpy()
    m_tr = np.fromiter((p in in_tr for p in p_all), bool, len(p_all))
    m_va = np.fromiter((p in in_va for p in p_all), bool, len(p_all))
    log.info("Inner split: %d train patients (%d rows) | %d validation patients (%d rows)",
             len(in_tr), m_tr.sum(), len(in_va), m_va.sum())

    y_all = info["y"].to_numpy()
    tr_pts, va_pts = p_all[m_tr], p_all[m_va]
    y_tr_raw, y_va = y_all[m_tr], y_all[m_va]
    y_tr_dil = dilate(y_tr_raw, tr_pts)
    va_sets = info["Set"].to_numpy()[m_va]

    # utility-optimal labels and weights for E4 (Hammoud et al.)
    u_pos, u_neg, _, _ = fv.utility_terms(y_tr_raw, tr_pts)
    y_tr_util = (u_pos > u_neg).astype(np.int8)
    w_tr_util = np.abs(u_pos - u_neg).astype(np.float32)

    idx_v2 = [feats.index(f) for f in fv.FEATURES]
    idx_v3 = list(range(len(feats)))
    idx_nolos = [i for i, f in enumerate(feats) if f not in LOS_FEATURES]

    Xtr_full = np.asarray(X[m_tr]); Xva_full = np.asarray(X[m_va])
    med = np.nanmedian(Xtr_full, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)

    def take(Xm, cols, impute):
        A = Xm[:, cols]
        if impute:
            A = np.where(np.isnan(A), med[cols], A)
        return np.ascontiguousarray(A, dtype=np.float32)

    variants = [
        ("E0 v2 baseline (67 feat, imputed, XGB+LGB+RF)", idx_v2, True, dict(tuned=False, spw=6.0, rf=True), "dil"),
        ("E1 + v3 features (185)", idx_v3, True, dict(tuned=False, spw=6.0, rf=True), "dil"),
        ("E2 E1, native missing values, no RF", idx_v3, False, dict(tuned=False, spw=6.0, rf=False), "dil"),
        ("E3 E2 + tuned boosting, early stopping", idx_v3, False, dict(tuned=True, spw=6.0), "dil"),
        ("E4 E3 + utility-weighted loss", idx_v3, False, dict(tuned=True, spw=None), "util"),
        ("E5 E3 without ICULOS / HospAdmTime", idx_nolos, False, dict(tuned=True, spw=6.0), "dil"),
    ]
    only = set(sys.argv[1:])
    results = json.loads(OUT.read_text()) if OUT.exists() else {}

    if "CROSS" in only:
        # E7: cross-hospital generalisation. Train on one hospital system's training patients,
        # test on the OTHER system's held-out test patients. v2 recipe vs v3 recipe.
        test_p = set(split["test"])
        sets_all = info["Set"].to_numpy()
        m_trainsplit = np.isin(p_all, train_p)
        m_testsplit = np.fromiter((p in test_p for p in p_all), bool, len(p_all))
        u_pos_a, u_neg_a, _, _ = fv.utility_terms(y_all, p_all)
        for src, dst in (("A", "B"), ("B", "A")):
            mtr = m_trainsplit & (sets_all == src)
            mte = m_testsplit & (sets_all == dst)
            Xs = np.asarray(X[mtr]); Xt = np.asarray(X[mte])
            yt, pt = y_all[mte], p_all[mte]
            ytr_d = dilate(y_all[mtr], p_all[mtr])
            med_s = np.nanmedian(Xs, axis=0); med_s = np.where(np.isfinite(med_s), med_s, 0.0).astype(np.float32)
            imp = lambda A, c: np.ascontiguousarray(np.where(np.isnan(A[:, c]), med_s[c], A[:, c]), dtype=np.float32)  # noqa: E731
            # v2 recipe
            t0 = time.time()
            Xa, Xb = imp(Xs, idx_v2), imp(Xt, idx_v2)
            sc2, _ = fit_variant("v2", Xa, ytr_d, None, Xb, yt, dict(tuned=False, spw=6.0, rf=True))
            m2 = operating_metrics(sc2, yt, pt, sets_all[mte])
            # v3 recipe (final hyperparameters)
            Xa, Xb = np.ascontiguousarray(Xs), np.ascontiguousarray(Xt)
            ytr_u = (u_pos_a[mtr] > u_neg_a[mtr]).astype(np.int8); wtr_u = np.abs(u_pos_a[mtr] - u_neg_a[mtr]).astype(np.float32)
            xm = xgb.XGBClassifier(n_estimators=760, learning_rate=0.03, max_depth=4, min_child_weight=20, subsample=0.8,
                                   colsample_bytree=0.6, reg_lambda=2.0, tree_method="hist", device="cuda",
                                   random_state=SEED, verbosity=0).fit(Xa, ytr_u, sample_weight=wtr_u)
            xm.set_params(device="cpu")
            lm = lgb.LGBMClassifier(n_estimators=210, learning_rate=0.03, num_leaves=127, min_child_samples=100,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0,
                                    random_state=SEED, n_jobs=10, verbosity=-1).fit(Xa, ytr_u, sample_weight=wtr_u)
            sc3 = 0.5 * xm.predict_proba(Xb)[:, 1] + 0.5 * lm.predict_proba(Xb)[:, 1]
            m3 = operating_metrics(sc3, yt, pt, sets_all[mte])
            for tag, m in (("v2", m2), ("v3", m3)):
                results[f"E7 {src}->{dst} {tag}"] = {"name": f"E7 train set {src} -> test set {dst}, {tag} recipe", **m}
                log.info("  train %s -> test %s | %s recipe | ROC-AUC %.4f | PR-AUC %.4f | cap utility %.3f | septic %.1f%% | non-septic %.1f%%",
                         src, dst, tag, m["roc_auc"], m["pr_auc"], m["op_utility"], 100 * m["op_septic_alerted"],
                         100 * m["op_non_septic_alerted"])
            log.info("  (%.0fs)", time.time() - t0)
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
            del Xs, Xt, Xa, Xb
            gc.collect()
        return

    if "SWEEP" in only:
        # E6: tune the E4 setup (v3 features, native missing values, utility-weighted loss).
        # Early stopping on ROC-AUC with long patience (PR-AUC stopped LightGBM after 2 trees in E3/E5).
        Xtr = take(Xtr_full, idx_v3, False); Xva = take(Xva_full, idx_v3, False)
        member_scores = {}
        trials = [("xgb", d) for d in (4, 6, 8)] + [("lgb", (l, c)) for l in (31, 63, 127) for c in (100, 400)]
        for kind, hp in trials:
            t0 = time.time()
            if kind == "xgb":
                m = xgb.XGBClassifier(
                    n_estimators=4000, learning_rate=0.03, max_depth=hp, min_child_weight=20, subsample=0.8,
                    colsample_bytree=0.6, reg_lambda=2.0, tree_method="hist", device="cuda", eval_metric="auc",
                    early_stopping_rounds=300, random_state=SEED, verbosity=0,
                ).fit(Xtr, y_tr_util, sample_weight=w_tr_util, eval_set=[(Xva, y_va)], verbose=False)
                trees = int(m.best_iteration) + 1
                m.set_params(device="cpu")
                name = f"xgb_depth{hp}"
            else:
                leaves, mcs = hp
                m = lgb.LGBMClassifier(
                    n_estimators=4000, learning_rate=0.03, num_leaves=leaves, min_child_samples=mcs, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0, random_state=SEED, n_jobs=10, verbosity=-1,
                ).fit(Xtr, y_tr_util, sample_weight=w_tr_util, eval_set=[(Xva, y_va)], eval_metric="auc",
                      callbacks=[lgb.early_stopping(300, first_metric_only=True, verbose=False)])
                trees = int(m.best_iteration_)
                name = f"lgb_leaves{leaves}_mcs{mcs}"
            sc = m.predict_proba(Xva)[:, 1]
            member_scores[name] = sc
            met = operating_metrics(sc, y_va, va_pts, va_sets)
            results[f"E6 {name}"] = {"name": f"E6 sweep {name}", "trees": trees, "fit_seconds": round(time.time() - t0, 1), **met}
            log.info("  %-22s trees %4d | ROC-AUC %.4f | PR-AUC %.4f | cap utility %.3f | septic %.1f%% | %.0fs", name, trees,
                     met["roc_auc"], met["pr_auc"], met["op_utility"], 100 * met["op_septic_alerted"], time.time() - t0)
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
        best_x = max((k for k in member_scores if k.startswith("xgb")), key=lambda k: results[f"E6 {k}"]["pr_auc"])
        best_l = max((k for k in member_scores if k.startswith("lgb")), key=lambda k: results[f"E6 {k}"]["pr_auc"])
        for wx in (0.3, 0.5, 0.7):
            sc = wx * member_scores[best_x] + (1 - wx) * member_scores[best_l]
            met = operating_metrics(sc, y_va, va_pts, va_sets)
            results[f"E6 ensemble wx{wx}"] = {"name": f"E6 ensemble {best_x}*{wx} + {best_l}*{1 - wx:.1f}", **met,
                                              "members": [best_x, best_l], "xgb_weight": wx}
            log.info("  ensemble %s*%.1f + %s*%.1f | ROC-AUC %.4f | PR-AUC %.4f | cap utility %.3f | septic %.1f%%",
                     best_x, wx, best_l, 1 - wx, met["roc_auc"], met["pr_auc"], met["op_utility"], 100 * met["op_septic_alerted"])
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
        return
    for name, cols, impute, cfg, label in variants:
        key = name.split()[0]
        if only and key not in only:
            continue
        Xtr = take(Xtr_full, cols, impute); Xva = take(Xva_full, cols, impute)
        ytr, wtr = (y_tr_util, w_tr_util) if label == "util" else (y_tr_dil, None)
        log.info("%s: %d features ...", name, len(cols))
        score, fit_info = fit_variant(name, Xtr, ytr, wtr, Xva, y_va, cfg)
        m = operating_metrics(score, y_va, va_pts, va_sets)
        results[key] = {"name": name, "n_features": len(cols), **m, **fit_info}
        log.info("  ROC-AUC %.4f | PR-AUC %.4f | max utility %.3f | at <=15%% cap: utility %.3f, septic %.1f%%, non-septic %.1f%%, "
                 "hour precision %.1f%% | %.0fs", m["roc_auc"], m["pr_auc"], m["max_utility"], m["op_utility"],
                 100 * m["op_septic_alerted"], 100 * m["op_non_septic_alerted"], 100 * m["op_hour_precision"],
                 fit_info["fit_seconds"])
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
        del Xtr, Xva
        gc.collect()


if __name__ == "__main__":
    main()
