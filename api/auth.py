"""
Authentication and role-based access for the SepsisSense web app.

Two roles:
  * clinician - sees the whole ward, alerts, assessments; manages accounts (admin)
  * patient   - sees only their own record (linked to one replayed stay)

Standard library only:
  * passwords: salted PBKDF2-HMAC-SHA256
  * sessions: HMAC-signed, expiring token in an HttpOnly, SameSite=Lax cookie;
    each user has a token version, bumped on password reset to revoke sessions
  * brute force: per-username lockout after repeated failures

Demo accounts are seeded on first start so the prototype can be explored. They
are for a local research demo only; change or remove them before any wider use.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, Request

DATA_DIR = Path(__file__).resolve().parent / ".data"
USERS_PATH = DATA_DIR / "users.json"
SECRET_PATH = DATA_DIR / "secret.key"

COOKIE_NAME = "ss_session"
TOKEN_TTL_SECONDS = 12 * 3600
PBKDF2_ITERATIONS = 240_000
MAX_FAILURES = 5
LOCKOUT_SECONDS = 60

ROLES = ("clinician", "patient")

DEMO_ACCOUNTS: List[Dict[str, Any]] = [
    {"username": "doctor", "password": "doctor-demo", "display_name": "Dr. Demo", "role": "clinician", "pid": None},
    {"username": "patient", "password": "patient-demo", "display_name": "Patient in MICU-15", "role": "patient", "pid": "p011215"},
    {"username": "patient2", "password": "patient-demo", "display_name": "Patient in MICU-09", "role": "patient", "pid": "p011251"},
]

_lock = threading.Lock()
_failures: Dict[str, List[float]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ hashing

def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS, base64.b64encode(salt).decode(), base64.b64encode(dk).decode()
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), base64.b64decode(salt_b64), int(iters))
        return hmac.compare_digest(dk, base64.b64decode(hash_b64))
    except Exception:
        return False


def generate_password() -> str:
    # Readable temporary password, e.g. "kite-4821-moss"
    words = ["amber", "brook", "cedar", "delta", "ember", "fern", "harbor", "iris", "juniper", "kite",
             "lotus", "maple", "moss", "nova", "olive", "pine", "quartz", "river", "sage", "tidal"]
    return f"{secrets.choice(words)}-{secrets.randbelow(9000) + 1000}-{secrets.choice(words)}"


# ------------------------------------------------------------------- secret

def _secret() -> bytes:
    env = os.environ.get("SEPSISSENSE_SECRET")
    if env:
        return env.encode("utf-8")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SECRET_PATH.exists():
        SECRET_PATH.write_bytes(secrets.token_bytes(32))
    return SECRET_PATH.read_bytes()


_SECRET: Optional[bytes] = None


def _key() -> bytes:
    global _SECRET
    if _SECRET is None:
        _SECRET = _secret()
    return _SECRET


# -------------------------------------------------------------------- users

def _load() -> Dict[str, Dict[str, Any]]:
    if not USERS_PATH.exists():
        return {}
    return json.loads(USERS_PATH.read_text(encoding="utf-8"))


def _save(users: Dict[str, Dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = USERS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(users, indent=1), encoding="utf-8")
    tmp.replace(USERS_PATH)


def init_users(valid_pids: List[str]) -> None:
    """Create the user store with demo accounts on first start."""
    with _lock:
        users = _load()
        if users:
            return
        for acc in DEMO_ACCOUNTS:
            if acc["pid"] is not None and acc["pid"] not in valid_pids:
                continue
            users[acc["username"]] = _new_user(acc["username"], acc["display_name"], acc["role"], acc["pid"], acc["password"])
        _save(users)


def _new_user(username: str, display_name: str, role: str, pid: Optional[str], password: str) -> Dict[str, Any]:
    return {
        "username": username,
        "display_name": display_name,
        "role": role,
        "pid": pid,
        "password_hash": hash_password(password),
        "token_version": 1,
        "created_at": _now_iso(),
        "last_login": None,
    }


def public(user: Dict[str, Any]) -> Dict[str, Any]:
    return {k: user.get(k) for k in ("username", "display_name", "role", "pid", "created_at", "last_login")}


def list_users() -> List[Dict[str, Any]]:
    with _lock:
        return [public(u) for u in _load().values()]


def get_user(username: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _load().get(username)


def create_user(username: str, display_name: str, role: str, pid: Optional[str], password: str) -> Dict[str, Any]:
    with _lock:
        users = _load()
        if username in users:
            raise HTTPException(409, f"Username '{username}' is already taken.")
        if role == "patient" and any(u["role"] == "patient" and u.get("pid") == pid for u in users.values()):
            raise HTTPException(409, "That patient already has an account. Reset its password instead.")
        users[username] = _new_user(username, display_name, role, pid, password)
        _save(users)
        return public(users[username])


def set_password(username: str, password: str) -> None:
    with _lock:
        users = _load()
        if username not in users:
            raise HTTPException(404, "User not found")
        users[username]["password_hash"] = hash_password(password)
        users[username]["token_version"] = int(users[username].get("token_version", 1)) + 1  # revoke sessions
        _save(users)


def delete_user(username: str) -> None:
    with _lock:
        users = _load()
        if username not in users:
            raise HTTPException(404, "User not found")
        if users[username]["role"] == "clinician" and sum(u["role"] == "clinician" for u in users.values()) <= 1:
            raise HTTPException(409, "Cannot delete the last clinician account.")
        del users[username]
        _save(users)


# ------------------------------------------------------------------- login

def authenticate(username: str, password: str) -> Dict[str, Any]:
    key = username.strip().lower()
    now = time.time()
    with _lock:
        recent = [t for t in _failures.get(key, []) if now - t < LOCKOUT_SECONDS]
        _failures[key] = recent
        if len(recent) >= MAX_FAILURES:
            wait = int(LOCKOUT_SECONDS - (now - recent[0])) + 1
            raise HTTPException(429, f"Too many failed attempts. Try again in {wait} s.")
        users = _load()
        user = users.get(key)
        ok = user is not None and verify_password(password, user["password_hash"])
        if not ok:
            if user is None:
                verify_password(password, hash_password("timing-equaliser"))
            _failures[key].append(now)
            raise HTTPException(401, "Incorrect username or password.")
        _failures.pop(key, None)
        user["last_login"] = _now_iso()
        _save(users)
        return user


# ------------------------------------------------------------------- tokens

def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue_token(user: Dict[str, Any]) -> str:
    payload = {"u": user["username"], "v": user.get("token_version", 1), "exp": int(time.time()) + TOKEN_TTL_SECONDS}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64(hmac.new(_key(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def read_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        body, sig = token.split(".")
        expected = _b64(hmac.new(_key(), body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_unb64(body))
        if payload.get("exp", 0) < time.time():
            return None
        user = get_user(payload["u"])
        if user is None or int(user.get("token_version", 1)) != int(payload.get("v", 0)):
            return None
        return user
    except Exception:
        return None


# ------------------------------------------------------------- dependencies

def current_user(request: Request) -> Dict[str, Any]:
    token = request.cookies.get(COOKIE_NAME)
    user = read_token(token) if token else None
    if user is None:
        raise HTTPException(401, "Not signed in")
    return user


def require_clinician(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    if user["role"] != "clinician":
        raise HTTPException(403, "Clinician access required")
    return user
