from __future__ import annotations

import json

from scripts import staging_smoke as smoke


def test_mock_api_success(monkeypatch):
    def fake_get_json(url: str):
        if url.endswith("/health"):
            return 200, {"ok": True}, None
        if url.endswith("/clubs/tres-palapas"):
            return 200, {"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}, None
        return 200, {"club": {"slug": "tres-palapas"}, "leaderboard": []}, None

    monkeypatch.setattr(smoke, "_get_json", fake_get_json)
    rc, summary = smoke.run_smoke("https://api.example.com", None, "tres-palapas")

    assert rc == 0
    assert summary["ok"] is True
    assert len(summary["checks"]) == 3


def test_mock_api_failure(monkeypatch):
    monkeypatch.setattr(smoke, "_get_json", lambda _url: (500, None, "HTTP 500"))

    rc, summary = smoke.run_smoke("https://api.example.com", None, "tres-palapas")
    assert rc == 1
    assert summary["ok"] is False
    assert summary["failures"]


def test_mock_web_success(monkeypatch):
    def fake_get(url: str):
        if url.endswith("/leaderboards"):
            return 200, "<html><body>Leaderboard page</body></html>", None
        if "/clubs/" in url:
            return 200, "<html><body>Tres Palapas club page</body></html>", None
        return 200, "<html><body>Welcome</body></html>", None

    monkeypatch.setattr(smoke, "_get_json", lambda url: (200, {"ok": True}, None) if url.endswith("/health") else (200, {"id":"1","slug":"tres-palapas","name":"Tres"}, None) if url.endswith("/clubs/tres-palapas") else (200, {"club":{},"leaderboard":[]}, None))
    monkeypatch.setattr(smoke, "_http_get", fake_get)

    rc, summary = smoke.run_smoke("https://api.example.com", "https://web.example.com", "tres-palapas")
    assert rc == 0
    assert summary["ok"] is True
    assert len([c for c in summary["checks"] if c["kind"] == "web"]) == 3


def test_no_secrets_are_printed(monkeypatch, capsys):
    monkeypatch.setenv("STAGING_JUPR_API_BASE_URL", "https://user:token@api.example.com")
    monkeypatch.setattr(smoke, "run_smoke", lambda **_kwargs: (0, {"ok": True, "checks": [], "failures": []}))

    rc = smoke.main([])
    out = capsys.readouterr().out

    assert rc == 0
    assert "token" not in out
    assert "user:" not in out


def test_json_summary_shape(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "_get_json", lambda _url: (200, {"ok": True}, None))
    monkeypatch.setattr(smoke, "_check_api", lambda *_args, **_kwargs: ([{"kind": "api", "ok": True, "path": "/health", "status_code": 200}], []))
    monkeypatch.setenv("STAGING_JUPR_API_BASE_URL", "https://api.example.com")

    rc = smoke.main([])
    out = capsys.readouterr().out
    parsed = json.loads(out)

    assert rc == 0
    assert isinstance(parsed, dict)
    assert {"ok", "checks", "failures", "club_slug"}.issubset(parsed.keys())
