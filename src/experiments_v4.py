"""
Precision / recall experiments after model v3 (research log, reproducible).

    python src/experiments_v4.py MODEL    # model variants (GPU): signatures, seed bagging, CatBoost, Optuna, ensembles
    python src/experiments_v4.py RULES    # operating rules: per-hospital thresholds, alert-rule variants

Same protocol as experiments_v3.py: training patients only, split 85 / 15 into
inner-train and inner-validation; calibration and test patients are not touched.

Model variants (all use v3 features, native missing values, utility-weighted loss):
  B0  v3 recipe: XGB depth 4 (760 trees) + LGB 127 leaves (210 trees), 50/50
  S0  B0 + path-signature features (Levy areas of vitals / partial SOFA, 6 h and 12 h)   (Morrill et al. 2019)
  M1  seed bagging: 5 seeds of each member, on the better feature set from B0 / S0
  M2  CatBoost on GPU, early stopping on inner-validation
  M3  XGBoost hyperparameters searched with Optuna (TPE), PR-AUC objective
  ENS combinations of the above

Operating-rule variants are judged by 2-fold cross-fitting inside inner-validation,
repeated 5 times: isotonic calibration and thresholds are chosen on one half and
measured on the other, so a rule cannot win by fitting its own evaluation data.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from itertools import combinations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import experiments_v3 as e3
import features_v2 as fv
import features_v4 as f4

log = e3.log
ROOT, SEED = e3.ROOT, e3.SEED
OUT = ROOT / "models" / "experiments_v4.json"
SCORES = e3.CACHE / "v4_inner_val_scores.npz"
MATS = e3.CACHE / "v4_mats"
KEEP = 0.4  # share of non-septic patients' training hours kept (weighted 1 / KEEP)
CAP = 0.15

XGB_P = dict(n_estimators=760, learning_rate=0.03, max_depth=4, min_child_weight=20, subsample=0.8,
             colsample_bytree=0.6, reg_lambda=2.0, tree_method="hist", device="cuda", verbosity=0)
LGB_P = dict(n_estimators=210, learning_rate=0.03, num_leaves=127, min_child_samples=100, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0, n_jobs=10, verbosity=-1)


# ------------------------------------------------------------------ data

def isin_ids(values, ids) -> np.ndarray:
    """Hash-based membership for string patient IDs (np.isin on object arrays compares pair by pair:
    1.5 M rows x 24 k IDs took ~15 minutes)."""
    return pd.Index(pd.unique(np.asarray(list(ids), dtype=object))).get_indexer(np.asarray(values, dtype=object)) >= 0


def setup():
    X, info, feats = e3.load_cache()
    split = json.loads((ROOT / "models" / "data_split.json").read_text())
    train_p = np.asarray(split["train"])
    meta = info.groupby("Patient", sort=False).agg(septic=("y", "max"), Set=("Set", "first")).loc[train_p]
    strata = meta["Set"] + "_" + meta["septic"].astype(str)
    in_tr, in_va = train_test_split(meta.index, test_size=0.15, random_state=SEED, stratify=strata)
    p_all = info["Patient"].to_numpy()
    m_tr = isin_ids(p_all, in_tr)
    m_va = isin_ids(p_all, in_va)

    y_all = info["y"].to_numpy()
    nb, ns = X.shape[1], len(f4.SIG_FEATURES)
    sig_path = MATS / "sig_all.npy"
    if sig_path.exists() and np.load(sig_path, mmap_mode="r").shape == (len(p_all), ns):
        sig = np.load(sig_path, mmap_mode="r")  # reading columns of the 1.1 GB cache is slow under memory pressure
    else:
        t0 = time.time()
        sig = f4.signature_block({c: np.asarray(X[:, feats.index(c)]) for c in f4.CHANNELS}, p_all)
        MATS.mkdir(parents=True, exist_ok=True)
        np.save(sig_path, sig)
        log.info("Signature features: %d columns in %.0fs", sig.shape[1], time.time() - t0)
    MATS.mkdir(parents=True, exist_ok=True)

    # utility labels / weights need each training patient's full sequence
    u_pos, u_neg, _, _ = fv.utility_terms(y_all[m_tr], p_all[m_tr])
    ytr_full = (u_pos > u_neg).astype(np.int8)
    wtr_full = np.abs(u_pos - u_neg).astype(np.float32)
    # Memory: keep every hour of septic patients and a random KEEP share of non-septic patients' hours,
    # re-weighted by 1 / KEEP (unbiased loss). All variants share the same rows, so comparisons stay fair.
    septic_p = set(meta.index[meta["septic"] == 1])
    tr_rows = np.flatnonzero(m_tr)
    is_sep = isin_ids(p_all[tr_rows], list(septic_p))
    keep = is_sep | (np.random.default_rng(SEED).random(len(tr_rows)) < KEEP)
    m_tr = np.zeros_like(m_tr)
    m_tr[tr_rows[keep]] = True
    wtr_full = np.where(is_sep, wtr_full, wtr_full / KEEP).astype(np.float32)
    ytr, wtr = ytr_full[keep], wtr_full[keep]

    def to_disk(name, mask, with_sig):
        # Training matrices live on disk as memory-mapped .npy files: file-backed pages do not count
        # against Windows commit memory, which is nearly exhausted on this 16 GB laptop.
        path = MATS / f"{name}.npy"
        ncol = nb + (ns if with_sig else 0)
        rows = np.flatnonzero(mask)
        if not path.exists() or np.load(path, mmap_mode="r").shape != (len(rows), ncol):
            out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), ncol))
            for i in range(0, len(rows), 100_000):
                r = rows[i:i + 100_000]
                out[i:i + len(r), :nb] = X[r]
                if with_sig:
                    out[i:i + len(r), nb:] = sig[r]
            out.flush()
            del out
        return np.load(path, mmap_mode="r")

    d = dict(
        files={(False, "tr"): ("tr_v3", m_tr), (False, "va"): ("va_v3", m_va),
               (True, "tr"): ("tr_v3sig", m_tr), (True, "va"): ("va_v3sig", m_va)},
        y_va=y_all[m_va], va_pts=p_all[m_va], va_sets=info["Set"].to_numpy()[m_va],
    )
    for (with_sig, _), (name, mask) in d["files"].items():
        to_disk(name, mask, with_sig)
    d["ytr"], d["wtr"] = ytr, wtr
    log.info("Inner split: %d train patients (%d of %d rows kept: all septic-patient hours + %.0f%% of others) | "
             "%d validation patients (%d rows)", len(in_tr), m_tr.sum(), len(tr_rows), 100 * KEEP, len(in_va), m_va.sum())
    del sig, X
    gc.collect()
    return d


def feature_set(d, use_sig):
    """(train, validation) matrices, memory-mapped from disk."""
    return tuple(np.load(MATS / f"{d['files'][(use_sig, part)][0]}.npy", mmap_mode="r") for part in ("tr", "va"))


# ------------------------------------------------------------------ models

class _Batches(xgb.DataIter):
    """Feed XGBoost in row batches so only the compressed histogram matrix reaches the GPU.
    (A full float copy on the device fails on Windows/WDDM when system commit memory is low.)"""

    def __init__(self, X, y, w, batch=150_000):
        self.X, self.y, self.w, self.batch, self.i = X, y, w, batch, 0
        super().__init__()

    def next(self, input_data):
        if self.i >= len(self.X):
            return False
        j = min(self.i + self.batch, len(self.X))
        input_data(data=self.X[self.i:j], label=self.y[self.i:j], weight=None if self.w is None else self.w[self.i:j])
        self.i = j
        return True

    def reset(self):
        self.i = 0


_DTRAIN: dict = {}


def xgb_train(Xtr, y, w, p: dict, rounds: int, evals=None, early=None):
    """Native XGBoost training on a batched QuantileDMatrix; GPU first, CPU if the GPU cannot allocate."""
    key = (id(Xtr), Xtr.shape, p.get("max_bin", 256))
    if _DTRAIN.get("key") != key:
        _DTRAIN.clear()
        gc.collect()
        _DTRAIN.update(key=key, d=xgb.QuantileDMatrix(_Batches(Xtr, y, w), max_bin=p.get("max_bin", 256)))
    dtr = _DTRAIN["d"]
    ev = ([(xgb.QuantileDMatrix(evals[0], evals[1], ref=dtr, max_bin=p.get("max_bin", 256)), "va")]
          if evals is not None else None)
    params = {"objective": "binary:logistic", "tree_method": "hist", "eval_metric": "auc", **p}
    for device in ("cuda", "cpu"):
        try:
            bst = xgb.train({**params, "device": device}, dtr, rounds, evals=ev or (),
                            early_stopping_rounds=early, verbose_eval=False)
            break
        except xgb.core.XGBoostError as exc:
            if device == "cpu" or "memory" not in str(exc).lower():
                raise
            log.warning("  GPU out of memory, retrying XGBoost on CPU")
            gc.collect()
    bst.set_param({"device": "cpu"})
    return bst


def _xgb_params(seed, over):
    q = {**XGB_P, **over}
    return q.pop("n_estimators"), {
        "eta": q["learning_rate"], "max_depth": q["max_depth"], "min_child_weight": q["min_child_weight"],
        "subsample": q["subsample"], "colsample_bytree": q["colsample_bytree"], "lambda": q["reg_lambda"],
        "alpha": q.get("reg_alpha", 0.0), "gamma": q.get("gamma", 0.0), "max_bin": q.get("max_bin", 256), "seed": seed,
    }


def fit_xgb(Xtr, y, w, Xva, seed=SEED, **over):
    rounds, p = _xgb_params(seed, over)
    return xgb_train(Xtr, y, w, p, rounds).inplace_predict(Xva)


def fit_lgb(Xtr, y, w, Xva, seed=SEED, **over):
    m = lgb.LGBMClassifier(**{**LGB_P, **over}, random_state=seed).fit(Xtr, y, sample_weight=w)
    return m.predict_proba(Xva)[:, 1]


def rank01(s):
    """Rank-normalise a score so members on different scales average fairly."""
    return pd.Series(s).rank(pct=True).to_numpy()


def paired_bootstrap(y, pts, base, cand, n=100):
    """Patient-level bootstrap of the difference (cand - base) in PR-AUC and ROC-AUC.
    Resampling patients with replacement == weighting each row by its patient's draw count."""
    rng = np.random.default_rng(SEED)
    codes, uniq = pd.factorize(pts)
    dpr, dauc = [], []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes].astype(np.float64)
        m = w > 0
        yy, ww = y[m], w[m]
        dpr.append(average_precision_score(yy, cand[m], sample_weight=ww) - average_precision_score(yy, base[m], sample_weight=ww))
        dauc.append(roc_auc_score(yy, cand[m], sample_weight=ww) - roc_auc_score(yy, base[m], sample_weight=ww))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]  # noqa: E731
    return {"d_pr_auc_ci95": q(dpr), "d_roc_auc_ci95": q(dauc)}


def run_models() -> None:
    d = setup()
    y_va, va_pts, va_sets = d["y_va"], d["va_pts"], d["va_sets"]
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    scores = dict(np.load(SCORES)) if SCORES.exists() else {}

    def record(key, name, score, extra=None, t0=None):
        scores[key] = score.astype(np.float32)
        m = e3.operating_metrics(score, y_va, va_pts, va_sets)
        if key != "B0" and "B0" in scores:
            m.update(paired_bootstrap(y_va, va_pts, scores["B0"], score))
        results[key] = {"name": name, **m, **(extra or {})}
        if t0:
            results[key]["fit_seconds"] = round(time.time() - t0, 1)
        log.info("%-46s ROC-AUC %.4f | PR-AUC %.4f | cap utility %.3f | septic %.1f%% | non-septic %.1f%% | hour prec %.1f%%%s",
                 name, m["roc_auc"], m["pr_auc"], m["op_utility"], 100 * m["op_septic_alerted"],
                 100 * m["op_non_septic_alerted"], 100 * m["op_hour_precision"],
                 f" | dPR-AUC CI {m['d_pr_auc_ci95']}" if "d_pr_auc_ci95" in m else "")
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
        np.savez(SCORES, **scores)

    ytr, wtr = d["ytr"], d["wtr"]

    # B0 / S0 -------------------------------------------------------------
    for key, use_sig in (("S0", True), ("B0", False)):  # S0 first: the v3-only view frees the combined block
        if key in results:
            continue
        t0 = time.time()
        Xtr, Xva = feature_set(d, use_sig)
        sx = fit_xgb(Xtr, ytr, wtr, Xva)
        sl = fit_lgb(Xtr, ytr, wtr, Xva)
        scores[f"{key}_xgb"], scores[f"{key}_lgb"] = sx.astype(np.float32), sl.astype(np.float32)
        record(key, f"{key} v3 recipe" + (" + signatures" if use_sig else ""), 0.5 * sx + 0.5 * sl,
               {"n_features": Xtr.shape[1]}, t0)

    if "d_pr_auc_ci95" not in results["S0"]:
        results["S0"].update(paired_bootstrap(y_va, va_pts, scores["B0"], scores["S0"]))
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    b0, s0 = results["B0"], results["S0"]
    use_sig = s0["pr_auc"] > b0["pr_auc"] and s0["roc_auc"] >= b0["roc_auc"] - 0.001
    results["use_signatures"] = bool(use_sig)
    log.info("Signature features %s (PR-AUC %.4f vs %.4f)", "KEPT" if use_sig else "dropped", s0["pr_auc"], b0["pr_auc"])
    Xtr, Xva = feature_set(d, use_sig)
    tag = "S" if use_sig else "B"
    base_key = "S0" if use_sig else "B0"

    # M1 seed bagging -----------------------------------------------------
    if "M1" not in results:
        t0 = time.time()
        seeds = [SEED, 7, 123, 2024, 31337]
        xs = [scores[f"{base_key}_xgb"]] + [fit_xgb(Xtr, ytr, wtr, Xva, seed=s) for s in seeds[1:]]
        ls = [scores[f"{base_key}_lgb"]] + [fit_lgb(Xtr, ytr, wtr, Xva, seed=s) for s in seeds[1:]]
        scores["M1_xgb"], scores["M1_lgb"] = np.mean(xs, 0).astype(np.float32), np.mean(ls, 0).astype(np.float32)
        record("M1", f"M1 {tag}0 x 5 seeds", 0.5 * scores["M1_xgb"] + 0.5 * scores["M1_lgb"], {"seeds": seeds}, t0)

    # M2 CatBoost (CPU) ---------------------------------------------------
    if "M2" not in results:
        # CatBoost on the GPU runs in a child process: if the CUDA allocation fails, CatBoost aborts the
        # whole process (C++ terminate, not a Python exception), and this run must survive that.
        import subprocess
        t0 = time.time()
        _DTRAIN.clear()
        gc.collect()
        np.save(MATS / "ytr.npy", ytr); np.save(MATS / "wtr.npy", wtr); np.save(MATS / "yva.npy", y_va)
        out_scores, out_info = MATS / "cat_scores.npy", MATS / "cat_info.json"
        for f in (out_scores, out_info):
            f.unlink(missing_ok=True)
        tr_name, va_name = (d["files"][(use_sig, part)][0] for part in ("tr", "va"))
        sc = None
        for device in ("GPU", "CPU"):
            r = subprocess.run([sys.executable, "-u", __file__, "CATBOOST", device, tr_name, va_name], timeout=5400)
            if r.returncode == 0 and out_scores.exists():
                sc = np.load(out_scores)
                cat_info = json.loads(out_info.read_text())
                break
            log.warning("CatBoost on %s failed (exit code %s)%s", device, r.returncode,
                        ", retrying on CPU" if device == "GPU" else "")
        if sc is not None:
            scores["M2_cat"] = sc.astype(np.float32)
            record("M2", f"M2 CatBoost ({cat_info['device']})", sc, cat_info, t0)
        else:
            results["M2"] = {"name": "M2 CatBoost", "error": "failed on GPU and CPU"}

    # M3 Optuna-tuned XGBoost (GPU) --------------------------------------
    if "M3" not in results:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        t0 = time.time()
        best = {"pr": -1.0}

        def objective(trial):
            p = dict(
                max_depth=trial.suggest_int("max_depth", 3, 8),
                min_child_weight=trial.suggest_float("min_child_weight", 1, 200, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 0.9),
                reg_lambda=trial.suggest_float("reg_lambda", 0.1, 30, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                gamma=trial.suggest_float("gamma", 1e-3, 5, log=True),
                max_bin=trial.suggest_categorical("max_bin", [128, 256, 512]),
            )
            _, xp = _xgb_params(SEED, {**p, "learning_rate": 0.05})
            bst = xgb_train(Xtr, ytr, wtr, xp, 4000, evals=(Xva, y_va), early=200)
            trees = int(bst.best_iteration) + 1
            sc = bst.inplace_predict(Xva, iteration_range=(0, trees))
            pr = average_precision_score(y_va, sc)
            if pr > best["pr"]:
                best.update(pr=pr, score=sc, params={**p, "trees": trees, "learning_rate": 0.05})
            log.info("  optuna trial %2d: PR-AUC %.4f | trees %d | %s", trial.number, pr, trees,
                     {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()})
            return pr

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.enqueue_trial(dict(max_depth=4, min_child_weight=20, subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0,
                                 reg_alpha=1e-3, gamma=1e-3, max_bin=256))
        study.optimize(objective, n_trials=24, timeout=1500)
        scores["M3_xgb"] = best["score"].astype(np.float32)
        record("M3", "M3 Optuna XGBoost (GPU)", best["score"], {"params": best["params"], "trials": len(study.trials)}, t0)

    # Ensembles -----------------------------------------------------------
    members = {k: scores[v] for k, v in (("xgb", "M1_xgb"), ("lgb", "M1_lgb"), ("cat", "M2_cat"), ("xgb_opt", "M3_xgb"))
               if v in scores}
    for r in range(2, len(members) + 1):
        for combo in combinations(members, r):
            key = "ENS " + "+".join(combo)
            sc = np.mean([members[c] for c in combo], axis=0)
            record(key, key + " (mean prob)", sc)
            if r >= 3:
                rk = np.mean([rank01(members[c]) for c in combo], axis=0)
                record(key + " rank", key + " (mean rank)", rk)

    cands = {k: v for k, v in results.items() if isinstance(v, dict) and k != "B0" and "pr_auc" in v}
    best_key = max(cands, key=lambda k: (cands[k]["pr_auc"] + cands[k]["op_utility"]))
    results["best_model_key"] = best_key
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    log.info("Best by PR-AUC + capped utility: %s", best_key)


# ------------------------------------------------------------------ rules

def _prev(s, first, k=1):
    out = s.copy()
    for _ in range(k):
        out = np.r_[-np.inf, out[:-1]]
        out[first] = -np.inf
        first = first | np.r_[False, first[:-1]]
    return out


def rule_current(s, pts, t):
    return fv.causal_alerts(s, pts, t)


def rule_strict2(s, pts, t):
    first = np.r_[True, pts[1:] != pts[:-1]]
    return (s >= t) & (_prev(s, first) >= t)


def rule_3h(s, pts, t):
    first = np.r_[True, pts[1:] != pts[:-1]]
    return ((s >= t) & (_prev(s, first) >= 0.85 * t) & (_prev(s, first, 2) >= 0.85 * t)) | (s >= min(1.3 * t, 0.9))


def rule_smooth3(s, pts, t):
    sm = pd.Series(s).groupby(pts, sort=False).rolling(3, min_periods=1).mean().to_numpy()
    return fv.causal_alerts(sm, pts, t)


def rule_latch(s, pts, t):
    """Start as current rule; stay on while risk >= 0.7 t (hysteresis, fewer on/off flickers)."""
    start = fv.causal_alerts(s, pts, t)
    first = np.r_[True, pts[1:] != pts[:-1]]
    keep = s >= 0.7 * t
    n = len(s)
    idx = np.arange(n)
    boundary = first | ~keep | np.r_[True, ~keep[:-1]]
    run_start = np.maximum.accumulate(np.where(boundary, idx, 0))
    last_start = np.maximum.accumulate(np.where(start, idx, -1))
    return start | (keep & (last_start >= run_start))


def rule_rising(s, pts, t):
    first = np.r_[True, pts[1:] != pts[:-1]]
    return fv.causal_alerts(s, pts, t) & (s >= _prev(s, first, 3))


RULES = {"current (2 h)": rule_current, "strict 2 h": rule_strict2, "3 h persistence": rule_3h,
         "3 h smoothed": rule_smooth3, "latch (hysteresis)": rule_latch, "rising only": rule_rising}


def run_rules() -> None:
    results = json.loads(OUT.read_text())
    scores = dict(np.load(SCORES))
    key = sys.argv[2] if len(sys.argv) > 2 else results.get("best_model_key", "B0")
    keys = [key] + (["B0"] if key != "B0" else [])
    X, info, feats = e3.load_cache()
    split = json.loads((ROOT / "models" / "data_split.json").read_text())
    train_p = np.asarray(split["train"])
    meta = info.groupby("Patient", sort=False).agg(septic=("y", "max"), Set=("Set", "first")).loc[train_p]
    strata = meta["Set"] + "_" + meta["septic"].astype(str)
    _, in_va = train_test_split(meta.index, test_size=0.15, random_state=SEED, stratify=strata)
    p_all = info["Patient"].to_numpy()
    m_va = isin_ids(p_all, in_va)
    y, pts, sets = info["y"].to_numpy()[m_va], p_all[m_va], info["Set"].to_numpy()[m_va]
    u_pos, u_neg, best, ina = fv.utility_terms(y, pts)
    codes, uniq = pd.factorize(pts)
    septic = np.zeros(len(uniq), bool); np.logical_or.at(septic, codes, y == 1)
    pset = pd.Series(sets).groupby(codes).first().to_numpy()
    va_meta = meta.loc[uniq]

    def pat_alert(a):
        pa = np.zeros(len(uniq), bool); np.logical_or.at(pa, codes, a); return pa

    def choose(score_c, rule, rows, pat_mask):
        """Best summed utility on `rows` with <= CAP of non-septic patients (in pat_mask) alerted."""
        grid = np.unique(np.quantile(score_c[rows], np.linspace(0.6, 0.999, 120)))
        best_t, best_u = grid[-1], -np.inf
        sub_pts = pts[rows]
        for t in grid:
            a = np.zeros(len(y), bool)
            a[rows] = rule(score_c[rows], sub_pts, float(t))
            pa = pat_alert(a)
            ns = pa[pat_mask & ~septic].mean() if (pat_mask & ~septic).any() else 0
            if ns > CAP:
                continue
            u = float(np.where(a[rows], u_pos[rows], u_neg[rows]).sum())
            if u > best_u:
                best_u, best_t = u, float(t)
        return best_t

    out = {}
    for skey in keys:
        raw = scores[skey].astype(np.float64)
        for rname, rule in RULES.items():
            for strat in ("global", "per-hospital"):
                reps = []
                for rep in range(5):
                    f1, f2 = train_test_split(va_meta.index, test_size=0.5, random_state=100 + rep,
                                              stratify=(va_meta["Set"] + "_" + va_meta["septic"].astype(str)))
                    alerts = np.zeros(len(y), bool)
                    thr_log = []
                    for sel, ev in ((f1, f2), (f2, f1)):
                        sel_rows, ev_rows = isin_ids(pts, sel), isin_ids(pts, ev)
                        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw[sel_rows], y[sel_rows])
                        sc = iso.predict(raw)
                        sel_pat = isin_ids(uniq, sel)
                        if strat == "global":
                            t = choose(sc, rule, sel_rows, sel_pat)
                            alerts[ev_rows] = rule(sc[ev_rows], pts[ev_rows], t)
                            thr_log.append({"all": t})
                        else:
                            tl = {}
                            for s_ in ("A", "B"):
                                t = choose(sc, rule, sel_rows & (sets == s_), sel_pat & (pset == s_))
                                r_ = ev_rows & (sets == s_)
                                alerts[r_] = rule(sc[r_], pts[r_], t)
                                tl[s_] = t
                            thr_log.append(tl)
                    pa = pat_alert(alerts)
                    tp = int((alerts & (y == 1)).sum()); fp = int((alerts & (y == 0)).sum())
                    reps.append({
                        "utility": fv.normalized_utility(alerts, u_pos, u_neg, best, ina),
                        "septic_alerted": float(pa[septic].mean()),
                        "non_septic_alerted": float(pa[~septic].mean()),
                        "patient_precision": float(pa[septic].sum() / max(1, pa.sum())),
                        "hour_precision": tp / max(1, tp + fp),
                        "hour_recall": tp / max(1, int((y == 1).sum())),
                        "non_septic_alerted_A": float(pa[~septic & (pset == "A")].mean()),
                        "non_septic_alerted_B": float(pa[~septic & (pset == "B")].mean()),
                        "septic_alerted_A": float(pa[septic & (pset == "A")].mean()),
                        "septic_alerted_B": float(pa[septic & (pset == "B")].mean()),
                    })
                avg = {k: round(float(np.mean([r[k] for r in reps])), 4) for k in reps[0]}
                avg["utility_sd"] = round(float(np.std([r["utility"] for r in reps])), 4)
                out[f"{skey} | {rname} | {strat}"] = avg
                log.info("%-6s %-20s %-13s utility %.3f±%.3f | septic %.1f%% | non-septic %.1f%% (A %.1f / B %.1f) | patient prec %.1f%% | hour prec %.1f%% | hour rec %.1f%%",
                         skey, rname, strat, avg["utility"], avg["utility_sd"], 100 * avg["septic_alerted"],
                         100 * avg["non_septic_alerted"], 100 * avg["non_septic_alerted_A"], 100 * avg["non_septic_alerted_B"],
                         100 * avg["patient_precision"], 100 * avg["hour_precision"], 100 * avg["hour_recall"])
                results["RULES"] = out
                OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")


def run_confirm() -> None:
    """Re-check the leading candidates on the CALIBRATION patients (never used for training, tuning or picking
    variants). Members are retrained on the same inner-train rows with fixed tree counts; the test split stays untouched."""
    import subprocess
    results = json.loads(OUT.read_text())
    d = setup()
    X, info, feats = e3.load_cache()
    split = json.loads((ROOT / "models" / "data_split.json").read_text())
    p_all = info["Patient"].to_numpy()
    m_cal = isin_ids(p_all, split["calibration"])
    path = MATS / "cal_v3.npy"
    rows = np.flatnonzero(m_cal)
    if not path.exists() or np.load(path, mmap_mode="r").shape != (len(rows), X.shape[1]):
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), X.shape[1]))
        for i in range(0, len(rows), 100_000):
            out[i:i + 100_000] = X[rows[i:i + 100_000]]
        out.flush(); del out
    Xcal = np.load(path, mmap_mode="r")
    y, pts, sets = info["y"].to_numpy()[m_cal], p_all[m_cal], info["Set"].to_numpy()[m_cal]
    Xtr, _ = feature_set(d, False)
    ytr, wtr = d["ytr"], d["wtr"]
    log.info("Confirmation on %d calibration patients (%d rows)", len(np.unique(pts)), len(y))

    seeds = [SEED, 7, 123, 2024, 31337]
    xs = [fit_xgb(Xtr, ytr, wtr, Xcal, seed=s_) for s_ in seeds]
    ls = [fit_lgb(Xtr, ytr, wtr, Xcal, seed=s_) for s_ in seeds]
    opt = dict(results["M3"]["params"]); trees = opt.pop("trees")
    xo = fit_xgb(Xtr, ytr, wtr, Xcal, n_estimators=trees, **opt)
    _DTRAIN.clear(); gc.collect()
    np.save(MATS / "ytr.npy", ytr); np.save(MATS / "wtr.npy", wtr); np.save(MATS / "yva.npy", y)
    (MATS / "cat_scores.npy").unlink(missing_ok=True)
    cat = None
    for device in ("GPU", "CPU"):
        r = subprocess.run([sys.executable, "-u", __file__, "CATBOOST", device, "tr_v3", "cal_v3",
                            str(results["M2"]["trees"])], timeout=5400)
        if r.returncode == 0 and (MATS / "cat_scores.npy").exists():
            cat = np.load(MATS / "cat_scores.npy").astype(np.float64)
            break
    xb, lb = np.mean(xs, 0), np.mean(ls, 0)
    cands = {
        "B0 v3 recipe": 0.5 * xs[0] + 0.5 * ls[0],
        "M1 5-seed bagging": 0.5 * xb + 0.5 * lb,
        "M3 Optuna XGBoost": xo,
        "LGB(bag) + Optuna XGB": 0.5 * lb + 0.5 * xo,
        "XGB(bag) + LGB(bag) + Optuna XGB": (xb + lb + xo) / 3,
    }
    if cat is not None:
        cands["All 4 (rank mean)"] = np.mean([rank01(v) for v in (xb, lb, cat, xo)], axis=0)
        cands["XGB + LGB + CatBoost (bagged, prob)"] = (xb + lb + cat) / 3
    conf = {}
    for name, sc in cands.items():
        m = e3.operating_metrics(sc, y, pts, sets)
        if name != "B0 v3 recipe":
            m.update(paired_bootstrap(y, pts, cands["B0 v3 recipe"], sc))
        conf[name] = m
        log.info("%-38s ROC-AUC %.4f | PR-AUC %.4f | cap utility %.3f | septic %.1f%% | non-septic %.1f%% | hour prec %.1f%%%s",
                 name, m["roc_auc"], m["pr_auc"], m["op_utility"], 100 * m["op_septic_alerted"],
                 100 * m["op_non_septic_alerted"], 100 * m["op_hour_precision"],
                 f" | dPR-AUC CI {m['d_pr_auc_ci95']} | dROC CI {m['d_roc_auc_ci95']}" if "d_pr_auc_ci95" in m else "")
        results["CONFIRM_calibration_split"] = conf
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")


def run_catboost(device: str, tr_name: str, va_name: str, trees: str | None = None) -> None:
    """Child process for M2 (see run_models). With `trees`, train a fixed number of trees and only predict on va."""
    from catboost import CatBoostClassifier, Pool
    Xtr = np.load(MATS / f"{tr_name}.npy", mmap_mode="r"); Xva = np.load(MATS / f"{va_name}.npy", mmap_mode="r")
    ytr, wtr, yva = np.load(MATS / "ytr.npy"), np.load(MATS / "wtr.npy"), np.load(MATS / "yva.npy")
    hw = (dict(task_type="GPU", devices="0", gpu_ram_part=0.35, pinned_memory_size="512MB")
          if device == "GPU" else dict(task_type="CPU", thread_count=10))
    common = dict(learning_rate=0.06, depth=7, l2_leaf_reg=3.0, border_count=128, loss_function="Logloss",
                  random_seed=SEED, verbose=500, train_dir=str(MATS / "catboost_info"), **hw)
    if trees:
        cb = CatBoostClassifier(iterations=int(trees), **common)
        cb.fit(Pool(Xtr, ytr, weight=wtr))
    else:
        cb = CatBoostClassifier(iterations=4000, eval_metric="AUC", metric_period=25, od_type="Iter", od_wait=400,
                                use_best_model=True, **common)
        cb.fit(Pool(Xtr, ytr, weight=wtr), eval_set=Pool(Xva, yva))
    np.save(MATS / "cat_scores.npy", cb.predict_proba(Xva)[:, 1].astype(np.float32))
    best = int(trees) if trees else int(cb.get_best_iteration()) + 1
    (MATS / "cat_info.json").write_text(json.dumps({"device": device, "trees": best}))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "MODEL"
    if stage == "CATBOOST":
        run_catboost(*sys.argv[2:6])
    elif stage == "CONFIRM":
        run_confirm()
    else:
        run_models() if stage == "MODEL" else run_rules()
