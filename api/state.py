"""
Shared server-side state, persisted to api/.data/:

  * replay offset - hours the whole ward has advanced. Clinicians control it;
    every signed-in user (including patients) sees the same "now".
  * alert reviews - a clinician's review of an alert episode, visible to the
    care team and to the patient ("your care team reviewed this").
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from auth import DATA_DIR

STATE_PATH = DATA_DIR / "state.json"
_lock = threading.Lock()

ACK_REASONS = {
    "workup": "Sepsis suspected — work-up started",
    "alternative": "Reviewed — alternative explanation",
    "known": "Already being treated for infection",
    "comfort": "Comfort-focused care",
}


def _load() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"offset": 0, "acks": {}}
    data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    data.setdefault("offset", 0)
    data.setdefault("acks", {})
    return data


def _save(data: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(STATE_PATH)


def get_offset() -> int:
    with _lock:
        return int(_load()["offset"])


def set_offset(offset: int) -> int:
    with _lock:
        data = _load()
        data["offset"] = int(offset)
        _save(data)
        return data["offset"]


def get_ack(pid: str, alert_since_icu_hour: Optional[int]) -> Optional[Dict[str, Any]]:
    """Return the review for this alert episode, if any."""
    if alert_since_icu_hour is None:
        return None
    with _lock:
        ack = _load()["acks"].get(pid)
    if ack and int(ack.get("alert_since_icu_hour", -1)) == int(alert_since_icu_hour):
        return ack
    return None


def set_ack(pid: str, alert_since_icu_hour: int, reason: str, note: str, by: Dict[str, Any]) -> Dict[str, Any]:
    ack = {
        "alert_since_icu_hour": int(alert_since_icu_hour),
        "reason": reason,
        "reason_label": ACK_REASONS[reason],
        "note": note,
        "by": by["display_name"],
        "by_username": by["username"],
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with _lock:
        data = _load()
        data["acks"][pid] = ack
        _save(data)
    return ack


def clear_ack(pid: str) -> None:
    with _lock:
        data = _load()
        data["acks"].pop(pid, None)
        _save(data)
