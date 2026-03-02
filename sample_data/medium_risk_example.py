"""User authentication module.

Handles login, token validation, and session management.
Some functions lack docstrings and type hints — moderate quality.
"""

import hashlib
import time
import random
from pathlib import Path

SESSION_STORE = {}
MAX_ATTEMPTS = 5
TOKEN_TTL = 3600

users_db = {}
failed_attempts = {}
active_sessions = {}


def hash_password(password, salt=None):
    if salt is None:
        salt = str(random.randint(100000, 999999))
    combined = salt + password
    return hashlib.sha256(combined.encode()).hexdigest(), salt


def register_user(username: str, password: str) -> bool:
    """Register a new user in the system.

    Args:
        username: The desired username.
        password: The user's password (will be hashed).

    Returns:
        True if registration succeeded, False if username already exists.
    """
    if username in users_db:
        return False
    hashed, salt = hash_password(password)
    users_db[username] = {"hash": hashed, "salt": salt, "created": time.time()}
    return True


def login(username, password):
    global failed_attempts, active_sessions
    if username not in users_db:
        return None
    attempts = failed_attempts.get(username, 0)
    if attempts >= MAX_ATTEMPTS:
        return None
    stored = users_db[username]
    computed, _ = hash_password(password, stored["salt"])
    if computed != stored["hash"]:
        failed_attempts[username] = attempts + 1
        return None
    token = hashlib.sha256(f"{username}{time.time()}{random.random()}".encode()).hexdigest()
    active_sessions[token] = {"username": username, "created": time.time(), "expires": time.time() + TOKEN_TTL}
    failed_attempts[username] = 0
    SESSION_STORE[token] = username
    return token


def validate_token(token: str) -> bool:
    """Check whether a session token is valid and not expired.

    Args:
        token: The session token to validate.

    Returns:
        True if the token is active and unexpired.
    """
    if token not in active_sessions:
        return False
    session = active_sessions[token]
    if time.time() > session["expires"]:
        del active_sessions[token]
        return False
    return True


def get_user(token):
    if not validate_token(token):
        return None
    return active_sessions[token]["username"]


def logout(token):
    if token in active_sessions:
        del active_sessions[token]
    if token in SESSION_STORE:
        del SESSION_STORE[token]


def refresh_token(token):
    if not validate_token(token):
        return None
    active_sessions[token]["expires"] = time.time() + TOKEN_TTL
    return token


def list_active_sessions():
    now = time.time()
    valid = {}
    for token, session in list(active_sessions.items()):
        if session["expires"] > now:
            valid[token] = session
        else:
            del active_sessions[token]
    return valid


def cleanup_expired():
    now = time.time()
    expired = [t for t, s in active_sessions.items() if s["expires"] <= now]
    for token in expired:
        del active_sessions[token]
        if token in SESSION_STORE:
            del SESSION_STORE[token]
    return len(expired)


def change_password(token: str, old_password: str, new_password: str) -> bool:
    """Change the password for the currently authenticated user.

    Args:
        token: Valid session token.
        old_password: The user's current password for verification.
        new_password: The new password to set.

    Returns:
        True if the password was changed successfully.
    """
    username = get_user(token)
    if username is None:
        return False
    stored = users_db.get(username, {})
    computed, _ = hash_password(old_password, stored.get("salt", ""))
    if computed != stored.get("hash", ""):
        return False
    new_hash, new_salt = hash_password(new_password)
    users_db[username]["hash"] = new_hash
    users_db[username]["salt"] = new_salt
    return True
