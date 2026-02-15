import base64
import json
import time

import jupr_app.auth.session as session


def test_create_and_verify_session_token(monkeypatch):
    monkeypatch.setattr(session.st, "secrets", {"supabase": {"admin_session_secret": "test-secret"}}, raising=False)

    token = session.create_session_token("admin@example.com")
    payload = session.verify_session_token(token)

    assert payload is not None
    assert payload["email"] == "admin@example.com"
    assert payload["exp"] > int(time.time())


def test_verify_rejects_tampered_signature(monkeypatch):
    monkeypatch.setattr(session.st, "secrets", {"supabase": {"admin_session_secret": "test-secret"}}, raising=False)

    token = session.create_session_token("admin@example.com")
    decoded = base64.urlsafe_b64decode(token.encode()).decode()
    raw, _sig = decoded.rsplit("|", 1)

    tampered_payload = json.loads(raw)
    tampered_payload["email"] = "attacker@example.com"
    tampered_raw = json.dumps(tampered_payload, separators=(",", ":"))
    tampered_token = base64.urlsafe_b64encode(f"{tampered_raw}|deadbeef".encode()).decode()

    assert session.verify_session_token(tampered_token) is None


def test_verify_rejects_expired_token(monkeypatch):
    monkeypatch.setattr(session.st, "secrets", {"supabase": {"admin_session_secret": "test-secret"}}, raising=False)
    monkeypatch.setattr(session.time, "time", lambda: 1_000)

    token = session.create_session_token("admin@example.com")

    monkeypatch.setattr(session.time, "time", lambda: 2_500_000)
    assert session.verify_session_token(token) is None
