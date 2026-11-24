"""Minimal per-user accounts: scrypt-hashed passwords in a JSON file, one private library per user."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from pathlib import Path

import config

USERNAME_RE = re.compile(r"^[a-z0-9_]{3,32}$")
MIN_PASSWORD = 8


def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1).hex()


def _load() -> dict:
    try:
        return json.loads(config.USERS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def register(username: str, password: str) -> str | None:
    """Create an account. Returns an error message, or None on success."""
    username = username.strip().lower()
    if not USERNAME_RE.match(username):
        return "Username must be 3-32 characters: letters, digits or underscore."
    if len(password) < MIN_PASSWORD:
        return f"Password must be at least {MIN_PASSWORD} characters."
    users = _load()
    if username in users:
        return "That username is taken."
    salt = os.urandom(16)
    users[username] = {"salt": salt.hex(), "hash": _hash(password, salt)}
    config.USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.USERS_FILE.write_text(json.dumps(users), encoding="utf-8")
    return None


def verify(username: str, password: str) -> bool:
    rec = _load().get(username.strip().lower())
    if rec is None:
        _hash(password, b"\0" * 16)          # keep timing similar for unknown users
        return False
    return hmac.compare_digest(_hash(password, bytes.fromhex(rec["salt"])), rec["hash"])


def user_dirs(username: str) -> tuple[Path, Path]:
    """(documents dir, index dir) private to this user. The name is validated, so no path tricks."""
    if not USERNAME_RE.match(username):
        raise ValueError("invalid username")
    return config.USERS_DIR / username / "documents", config.USERS_DIR / username / "index"
