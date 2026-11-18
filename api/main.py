"""
SepsisSense clinician API.

    uvicorn api.main:app --port 8000        (from the repository root)

Serves JSON under /api and, when built, the React UI from frontend/dist.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

API_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(API_DIR))

import clinical  # noqa: E402
import engine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sepsissense.api")

ELEVATED_CUTOFF = 0.08  # display band only (~3.5x the hourly base rate); not validated
MAX_OFFSET = 24
STALE_VITAL_HOURS = 4

app = FastAPI(title="SepsisSense API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

COHORT: Optional[engine.Cohort] = None


@app.on_event("startup")
def _startup() -> None:
    global COHORT
    log.info("Loading cohort (first run builds the cache, ~1 min)...")
    COHORT = engine.load_cohort()
    log.info("Cohort ready: %d patients", len(COHORT.patients))


# ------------------------------------------------------------------ helpers

def _f(v: Any, dp: Optional[int] = None) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, dp) if dp is not None else x


def _cohort() -> engine.Cohort:
    if COHORT is None:
        raise HTTPException(503, "Cohort still loading")
    return COHORT


def _now_hour(rec: engine.PatientRecord, offset: int) -> int:
    return int(min(rec.base_hour + max(0, offset), rec.n_hours - 1))


def _latest(rec: engine.PatientRecord, name: str, h: int) -> Dict[str, Any]:
    col = rec.raw[name].iloc[: h + 1] if name in rec.raw.columns else pd.Series(dtype=float)
    valid = col.dropna()
    if valid.empty:
        return {"value": None, "hours_ago": None, "flag": None}
    idx = int(valid.index[-1])
    val = float(valid.iloc[-1])
    prev = float(valid.iloc[-2]) if len(valid) > 1 else None
    dp = clinical.MEASUREMENTS.get(name, {}).get("dp", 1)
    return {
        "value": round(val, dp),
        "previous": round(prev, dp) if prev is not None else None,
        "hours_ago": h - idx,
        "flag": clinical.flag(name, val),
    }


def _tier(risk: float, alert: bool) -> str:
    if alert:
        return "alert"
    if risk >= ELEVATED_CUTOFF:
        return "elevated"
    return "low"


def _alert_since(rec: engine.PatientRecord, h: int) -> Optional[int]:
    if not rec.alert[h]:
        return None
    i = h
    while i > 0 and rec.alert[i - 1]:
        i -= 1
    return i


def _agreement(rec: engine.PatientRecord, h: int) -> Dict[str, Any]:
    probs = {k: float(v[h]) for k, v in rec.member_risk.items()}
    sd = float(np.std(list(probs.values())))
    level = "high" if sd < 0.05 else "moderate" if sd < 0.12 else "low"
    return {"level": level, "sd": round(sd, 3), "members": {k: round(v, 3) for k, v in probs.items()}}


def _icu_hour(rec: engine.PatientRecord, h: int) -> int:
    v = _f(rec.raw["ICULOS"].iloc[h])
    return int(v) if v is not None else h + 1


def _summary(rec: engine.PatientRecord, h: int, now: int) -> Dict[str, Any]:
    c = _cohort()
    vitals = {v: _latest(rec, v, h) for v in engine.VITALS}
    for v, d in vitals.items():
        d["stale"] = d["hours_ago"] is not None and d["hours_ago"] > STALE_VITAL_HOURS
    lact = _latest(rec, "Lactate", h)
    wbc = _latest(rec, "WBC", h)
    risk = float(rec.risk[h])
    alert = bool(rec.alert[h])
    lo = max(0, h - 11)
    sirs = clinical.sirs_criteria(
        vitals["HR"]["value"], vitals["Resp"]["value"], vitals["Temp"]["value"],
        wbc["value"] if (wbc["hours_ago"] is not None and wbc["hours_ago"] <= 24) else None,
        _latest(rec, "PaCO2", h)["value"],
    )
    qsofa = clinical.qsofa_criteria(vitals["Resp"]["value"], vitals["SBP"]["value"])
    since = _alert_since(rec, h)
    return {
        "pid": rec.pid,
        "bed": rec.bed,
        "unit": rec.unit,
        "age": int(rec.age),
        "sex": rec.sex,
        "hour_index": h,
        "now_index": now,
        "icu_hour": _icu_hour(rec, h),
        "hosp_to_icu_hours": round(rec.hosp_adm_hours, 1),
        "record_hours": rec.n_hours,
        "record_ended": now >= rec.n_hours - 1,
        "risk": round(risk, 4),
        "risk_6h_ago": round(float(rec.risk[max(0, h - 6)]), 4),
        "risk_peak_24h": round(float(rec.risk[max(0, h - 23): h + 1].max()), 4),
        "relative_risk": round(risk / c.base_rate, 1) if c.base_rate else None,
        "raw_score": round(float(rec.raw_score[h]), 4),
        "spark": [round(float(x), 4) for x in rec.risk[lo: h + 1]],
        "alert": alert,
        "alert_since_icu_hour": _icu_hour(rec, since) if since is not None else None,
        "alert_hours": (h - since + 1) if since is not None else 0,
        "tier": _tier(risk, alert),
        "agreement": _agreement(rec, h),
        "vitals": vitals,
        "lactate": lact,
        "sirs": sirs,
        "qsofa": qsofa,
        "selection_reason": rec.selection_reason,
        "research": {
            "sepsis_label_icu_hour": _icu_hour(rec, rec.sepsis_label_hour) if rec.sepsis_label_hour is not None else None,
            "sepsis_onset_icu_hour": (_icu_hour(rec, rec.sepsis_label_hour) + 6) if rec.sepsis_label_hour is not None else None,
            "septic_stay": rec.sepsis_label_hour is not None,
        },
    }


# ------------------------------------------------------------------- routes

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": COHORT is not None}


@app.get("/api/meta")
def meta() -> Dict[str, Any]:
    c = _cohort()
    internal = {}
    if engine.METRICS_PATH.exists():
        internal = json.loads(engine.METRICS_PATH.read_text(encoding="utf-8"))
    return {
        "threshold": c.threshold,
        "elevated_cutoff": ELEVATED_CUTOFF,
        "base_rate": round(c.base_rate, 4),
        "max_offset": MAX_OFFSET,
        "n_features": len(c.feature_names),
        "pool": c.pool_stats,
        "internal_validation": internal,
        "ward_size": len(c.patients),
        "selection_counts": {
            k: sum(1 for p in c.patients.values() if p.selection_reason == k)
            for k in ("consecutive_septic", "consecutive_non_septic", "alert_example")
        },
    }


@app.get("/api/ward")
def ward(offset: int = Query(0, ge=0, le=MAX_OFFSET)) -> Dict[str, Any]:
    c = _cohort()
    rows = []
    for rec in c.patients.values():
        now = _now_hour(rec, offset)
        rows.append(_summary(rec, now, now))
    return {"offset": offset, "patients": rows}


@app.get("/api/patients/{pid}")
def patient(
    pid: str,
    offset: int = Query(0, ge=0, le=MAX_OFFSET),
    hour: Optional[int] = Query(None, ge=0, description="View the chart as of this hour index (≤ now)"),
) -> Dict[str, Any]:
    c = _cohort()
    rec = c.patients.get(pid)
    if rec is None:
        raise HTTPException(404, f"Unknown patient {pid}")
    now = _now_hour(rec, offset)
    h = now if hour is None else int(min(hour, now))

    series: List[Dict[str, Any]] = []
    cols = engine.VITALS + engine.LABS
    for i in range(now + 1):
        row = {
            "i": i,
            "icu_hour": _icu_hour(rec, i),
            "risk": round(float(rec.risk[i]), 4),
            "raw_score": round(float(rec.raw_score[i]), 4),
            "alert": int(rec.alert[i]),
        }
        for col in cols:
            row[col] = _f(rec.raw[col].iloc[i], 3) if col in rec.raw.columns else None
        series.append(row)

    labs = []
    for name in engine.LABS:
        latest = _latest(rec, name, h)
        if latest["value"] is None:
            continue
        m = clinical.MEASUREMENTS[name]
        col = rec.raw[name].iloc[: h + 1].dropna()
        labs.append({
            "name": name,
            "label": m["label"],
            "unit": m["unit"],
            "group": m.get("group", "Other"),
            "low": m["low"],
            "high": m["high"],
            "dp": m["dp"],
            **latest,
            "count": int(len(col)),
            "history": [{"i": int(k), "icu_hour": _icu_hour(rec, int(k)), "value": round(float(v), m["dp"])} for k, v in col.tail(8).items()],
        })

    summary = _summary(rec, h, now)
    latest_all = {n: _latest(rec, n, h) for n in ("Lactate", "MAP", "SBP")}
    return {
        "summary": summary,
        "series": series,
        "drivers": engine.grouped_drivers(rec, h, c.feature_names),
        "labs": labs,
        "bundle": clinical.bundle_prompts(latest_all, summary["risk"], c.threshold, summary["alert"]),
        "measurements": {k: {kk: v[kk] for kk in ("label", "unit", "low", "high", "dp")} for k, v in clinical.MEASUREMENTS.items()},
    }


# ------------------------------------------------------------- static UI

DIST = API_DIR.parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        target = DIST / full_path
        if full_path and target.is_file():
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
