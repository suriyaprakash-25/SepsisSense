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
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

API_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(API_DIR))

import assess  # noqa: E402
import auth  # noqa: E402
import clinical  # noqa: E402
import engine  # noqa: E402
import state  # noqa: E402
from auth import current_user, require_clinician  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sepsissense.api")

ELEVATED_CUTOFF = 0.08  # display band only (~3.5x the hourly base rate); not validated
MAX_OFFSET = 24
STALE_VITAL_HOURS = 4

app = FastAPI(title="SepsisSense API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.middleware("http")
async def csrf_guard(request: Request, call_next):
    """State-changing API calls must carry a custom header that cross-site forms cannot set."""
    if request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD", "OPTIONS"):
        if request.headers.get("x-sepsissense") != "1":
            return JSONResponse({"detail": "Missing X-SepsisSense header"}, status_code=403)
    return await call_next(request)

COHORT: Optional[engine.Cohort] = None
ASSESSOR: Optional[assess.Assessor] = None


@app.on_event("startup")
def _startup() -> None:
    global COHORT, ASSESSOR
    log.info("Loading cohort (first run builds the cache, ~1 min)...")
    COHORT = engine.load_cohort()
    log.info("Cohort ready: %d patients", len(COHORT.patients))
    auth.init_users(list(COHORT.patients))
    ASSESSOR = assess.Assessor.load()
    log.info("Assessment model ready")


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
        "ack": state.get_ack(rec.pid, _icu_hour(rec, since)) if since is not None else None,
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


def _labs_table(rec: engine.PatientRecord, h: int) -> List[Dict[str, Any]]:
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
    return labs


# ------------------------------------------------------------------- routes

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": COHORT is not None}


@app.get("/api/meta")
def meta(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
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
def ward(user: Dict[str, Any] = Depends(require_clinician)) -> Dict[str, Any]:
    c = _cohort()
    offset = state.get_offset()
    rows = []
    for rec in c.patients.values():
        now = _now_hour(rec, offset)
        rows.append(_summary(rec, now, now))
    return {"offset": offset, "patients": rows}


@app.get("/api/patients/{pid}")
def patient(
    pid: str,
    hour: Optional[int] = Query(None, ge=0, description="View the chart as of this hour index (≤ now)"),
    user: Dict[str, Any] = Depends(current_user),
) -> Dict[str, Any]:
    c = _cohort()
    if user["role"] == "patient" and user.get("pid") != pid:
        raise HTTPException(403, "Patients can only view their own record")
    rec = c.patients.get(pid)
    if rec is None:
        raise HTTPException(404, f"Unknown patient {pid}")
    offset = state.get_offset()
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

    labs = _labs_table(rec, h)

    summary = _summary(rec, h, now)
    if user["role"] == "patient":
        # Retrospective labels and cohort-selection details are for research use only.
        summary["research"] = {"sepsis_label_icu_hour": None, "sepsis_onset_icu_hour": None, "septic_stay": False}
        summary["selection_reason"] = None
    latest_all = {n: _latest(rec, n, h) for n in ("Lactate", "MAP", "SBP")}
    since_now = _alert_since(rec, now)
    now_ack = state.get_ack(rec.pid, _icu_hour(rec, since_now)) if since_now is not None else None
    if user["role"] == "patient":
        # Patients see that their care team reviewed the alert, not the clinical reason or note.
        def _patient_view(a: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            return None if a is None else {
                "alert_since_icu_hour": a["alert_since_icu_hour"], "by": a["by"], "at": a["at"],
                "reason": None, "reason_label": None, "note": "", "by_username": None,
            }
        summary["ack"] = _patient_view(summary.get("ack"))
        now_ack = _patient_view(now_ack)
    return {
        "summary": summary,
        "now_ack": now_ack,
        "series": series,
        "drivers": engine.grouped_drivers(rec, h, c.feature_names),
        "labs": labs,
        "bundle": clinical.bundle_prompts(latest_all, summary["risk"], c.threshold, summary["alert"]),
        "measurements": {k: {kk: v[kk] for kk in ("label", "unit", "low", "high", "dp")} for k, v in clinical.MEASUREMENTS.items()},
    }


# ----------------------------------------------------------- assessment

class Reading(BaseModel):
    hours_ago: int = Field(0, ge=0, le=assess.MAX_HOURS_BACK)
    values: Dict[str, Optional[float]] = Field(default_factory=dict)


class AssessRequest(BaseModel):
    age: float = Field(..., ge=18, le=105)
    sex: str = Field("M", pattern="^(M|F)$")
    unit: str = Field("MICU", pattern="^(MICU|SICU)$")
    icu_hours: int = Field(12, ge=1, le=2000, description="Hours since ICU admission, now")
    hosp_to_icu_hours: float = Field(0, ge=0, le=5000)
    readings: List[Reading] = Field(..., min_length=1, max_length=8)


@app.get("/api/assess/fields")
def assess_fields(user: Dict[str, Any] = Depends(require_clinician)) -> Dict[str, Any]:
    out = []
    for name, (lo, hi) in assess.INPUT_FIELDS.items():
        m = clinical.MEASUREMENTS[name]
        out.append({
            "name": name,
            "label": m["label"],
            "short": m.get("short", m["label"]),
            "unit": m["unit"],
            "kind": m["kind"],
            "group": "Vital signs" if m["kind"] == "vital" else m.get("group", "Other"),
            "ref_low": m["low"],
            "ref_high": m["high"],
            "min": lo,
            "max": hi,
            "dp": m["dp"],
        })
    return {"fields": out, "max_hours_back": assess.MAX_HOURS_BACK}


@app.post("/api/assess")
def assess_patient(req: AssessRequest, user: Dict[str, Any] = Depends(require_clinician)) -> Dict[str, Any]:
    c = _cohort()
    if ASSESSOR is None:
        raise HTTPException(503, "Assessment model still loading")

    # Validate entries
    errors: List[str] = []
    seen_hours = set()
    for r in req.readings:
        if r.hours_ago in seen_hours:
            errors.append(f"Two readings are both {r.hours_ago} h ago; merge them into one column.")
        seen_hours.add(r.hours_ago)
        for name, val in r.values.items():
            if val is None:
                continue
            if name not in assess.INPUT_FIELDS:
                errors.append(f"Unknown measurement '{name}'.")
                continue
            lo, hi = assess.INPUT_FIELDS[name]
            if not (lo <= val <= hi):
                label = clinical.MEASUREMENTS[name]["label"]
                errors.append(f"{label} {val:g} is outside the accepted range {lo:g}–{hi:g} {clinical.MEASUREMENTS[name]['unit']}.")
    if not any(v is not None for r in req.readings for v in r.values.values()):
        errors.append("Enter at least one vital sign or lab value.")
    if errors:
        raise HTTPException(422, {"errors": errors})

    raw, warnings = assess.build_grid(
        [r.model_dump() for r in req.readings], req.age, req.sex, req.unit, req.icu_hours, req.hosp_to_icu_hours
    )
    scored = ASSESSOR.score(raw)
    n = len(raw)
    rec = engine.PatientRecord(
        pid="assessment",
        bed="New assessment",
        unit=req.unit,
        age=req.age,
        sex=req.sex,
        hosp_adm_hours=req.hosp_to_icu_hours,
        n_hours=n,
        base_hour=n - 1,
        sepsis_label_hour=None,
        selection_reason="manual",
        raw=raw.reset_index(drop=True),
        risk=scored["risk"],
        raw_score=scored["raw_score"],
        member_risk=scored["members"],
        alert=engine.causal_persistence_alert(scored["raw_score"], c.threshold),
        contrib=scored["contrib"],
        feature_values=scored["features"],
    )
    h = n - 1
    summary = _summary(rec, h, h)
    entered_hours = sorted({h - r.hours_ago for r in req.readings})
    series = [
        {
            "i": i,
            "hours_ago": h - i,
            "icu_hour": _icu_hour(rec, i),
            "risk": round(float(rec.risk[i]), 4),
            "raw_score": round(float(rec.raw_score[i]), 4),
            "alert": int(rec.alert[i]),
            "entered": i in entered_hours,
        }
        for i in range(n)
    ]
    latest_all = {k: _latest(rec, k, h) for k in ("Lactate", "MAP", "SBP")}
    drivers = engine.grouped_drivers(rec, h, c.feature_names)
    if any(d["concept"] == "time" for d in drivers[:3]):
        warnings.insert(0,
            "Time in ICU is one of the strongest drivers of this score. The model learned that longer stays carry "
            "more risk (an artifact of the PhysioNet data), so identical vitals score higher later in a stay. "
            "Interpret with caution."
        )
    return {
        "summary": summary,
        "series": series,
        "drivers": drivers,
        "labs": _labs_table(rec, h),
        "bundle": clinical.bundle_prompts(latest_all, summary["risk"], c.threshold, summary["alert"]),
        "measurements": {k: {kk: v[kk] for kk in ("label", "unit", "low", "high", "dp")} for k, v in clinical.MEASUREMENTS.items()},
        "warnings": warnings,
    }


# ------------------------------------------------------------------ auth

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


def _me(user: Dict[str, Any]) -> Dict[str, Any]:
    out = auth.public(user)
    if user["role"] == "patient" and user.get("pid") and COHORT is not None:
        rec = COHORT.patients.get(user["pid"])
        out["bed"] = rec.bed if rec else None
    return out


@app.post("/api/auth/login")
def login(req: LoginRequest, request: Request, response: Response) -> Dict[str, Any]:
    user = auth.authenticate(req.username, req.password)
    response.set_cookie(
        auth.COOKIE_NAME,
        auth.issue_token(user),
        max_age=auth.TOKEN_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )
    return {"user": _me(user)}


@app.post("/api/auth/logout")
def logout(response: Response) -> Dict[str, Any]:
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    return {"user": _me(user)}


@app.get("/api/auth/demo-accounts")
def demo_accounts() -> Dict[str, Any]:
    """Demo credentials shown on the sign-in page (local research demo only)."""
    existing = {u["username"] for u in auth.list_users()}
    return {
        "accounts": [
            {"username": a["username"], "password": a["password"], "role": a["role"], "display_name": a["display_name"]}
            for a in auth.DEMO_ACCOUNTS
            if a["username"] in existing and auth.verify_password(a["password"], auth.get_user(a["username"])["password_hash"])
        ]
    }


# ---------------------------------------------------------------- replay

class ReplayRequest(BaseModel):
    offset: int = Field(..., ge=0, le=MAX_OFFSET)


@app.get("/api/replay")
def get_replay(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    return {"offset": state.get_offset(), "max_offset": MAX_OFFSET}


@app.post("/api/replay")
def set_replay(req: ReplayRequest, user: Dict[str, Any] = Depends(require_clinician)) -> Dict[str, Any]:
    return {"offset": state.set_offset(req.offset), "max_offset": MAX_OFFSET}


# ---------------------------------------------------------- alert reviews

class AckRequest(BaseModel):
    reason: str = Field(..., pattern="^(workup|alternative|known|comfort)$")
    note: str = Field("", max_length=280)


@app.post("/api/patients/{pid}/ack")
def ack_alert(pid: str, req: AckRequest, user: Dict[str, Any] = Depends(require_clinician)) -> Dict[str, Any]:
    rec = _cohort().patients.get(pid)
    if rec is None:
        raise HTTPException(404, f"Unknown patient {pid}")
    now = _now_hour(rec, state.get_offset())
    since = _alert_since(rec, now)
    if since is None:
        raise HTTPException(409, "This patient has no active alert to review.")
    return {"ack": state.set_ack(pid, _icu_hour(rec, since), req.reason, req.note.strip(), user)}


@app.delete("/api/patients/{pid}/ack")
def clear_ack(pid: str, user: Dict[str, Any] = Depends(require_clinician)) -> Dict[str, Any]:
    state.clear_ack(pid)
    return {"ok": True}


# ------------------------------------------------------- account admin

class CreateUserRequest(BaseModel):
    username: str = Field(..., pattern=r"^[a-z0-9][a-z0-9._-]{2,31}$")
    display_name: str = Field(..., min_length=1, max_length=80)
    role: str = Field(..., pattern="^(clinician|patient)$")
    pid: Optional[str] = None


@app.get("/api/admin/users")
def admin_users(user: Dict[str, Any] = Depends(require_clinician)) -> Dict[str, Any]:
    c = _cohort()
    beds = {pid: rec.bed for pid, rec in c.patients.items()}
    users = sorted(auth.list_users(), key=lambda u: (u["role"] != "clinician", u["username"]))
    for u in users:
        u["bed"] = beds.get(u.get("pid")) if u.get("pid") else None
    return {
        "users": users,
        "patients": [{"pid": pid, "bed": rec.bed} for pid, rec in c.patients.items()],
        "me": user["username"],
    }


@app.post("/api/admin/users")
def admin_create_user(req: CreateUserRequest, user: Dict[str, Any] = Depends(require_clinician)) -> Dict[str, Any]:
    if req.role == "patient":
        if not req.pid or req.pid not in _cohort().patients:
            raise HTTPException(422, "Choose the patient this account belongs to.")
    password = auth.generate_password()
    created = auth.create_user(req.username, req.display_name.strip(), req.role, req.pid if req.role == "patient" else None, password)
    return {"user": created, "temporary_password": password}


@app.post("/api/admin/users/{username}/reset-password")
def admin_reset_password(username: str, user: Dict[str, Any] = Depends(require_clinician)) -> Dict[str, Any]:
    password = auth.generate_password()
    auth.set_password(username, password)
    return {"username": username, "temporary_password": password}


@app.delete("/api/admin/users/{username}")
def admin_delete_user(username: str, user: Dict[str, Any] = Depends(require_clinician)) -> Dict[str, Any]:
    if username == user["username"]:
        raise HTTPException(409, "You cannot delete your own account while signed in.")
    auth.delete_user(username)
    return {"ok": True}


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
