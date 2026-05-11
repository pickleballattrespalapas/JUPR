import ast
from pathlib import Path

import pytest

from jupr_app.services.context import ServiceContext
from jupr_app.services.match_service import submit_match_batch
from jupr_app.services.result_types import ServiceResult


ALLOWED_PROCESS_MATCHES_IMPORTERS = {
    Path("jupr_app/services/match_service.py"),
}



def _find_process_matches_importers() -> list[Path]:
    offenders: list[Path] = []
    targets = list(Path("jupr_app/ui/pages").glob("*.py")) + list(Path("services/api").glob("*.py"))
    for file_path in sorted(targets):
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "jupr_app.domain.match_processing":
                continue
            if any(alias.name == "process_matches" for alias in node.names):
                offenders.append(file_path)
                break
    return offenders



def test_ui_and_api_do_not_import_process_matches_directly():
    offenders = [p for p in _find_process_matches_importers() if p not in ALLOWED_PROCESS_MATCHES_IMPORTERS]
    assert not offenders, (
        "UI/API must submit scores via submit_match_batch; direct process_matches imports are forbidden. "
        f"offenders={offenders}"
    )



def test_submit_match_batch_delegates_to_process_matches(monkeypatch):
    captured = {}

    def fake_process_matches(match_list, **kwargs):
        captured["match_list"] = match_list
        captured["kwargs"] = kwargs
        return {"ok": True, "processed": len(match_list)}

    monkeypatch.setattr("jupr_app.services.match_service.process_matches", fake_process_matches)

    ctx = ServiceContext(supabase=object(), club_id="club-123")
    result = submit_match_batch(
        ctx,
        [{"winner": "A"}],
        name_to_id={"A": 1},
        df_players_all="players_df",
        df_leagues="leagues_df",
        df_meta="meta_df",
    )

    assert result.ok is True
    assert result.data == {"ok": True, "processed": 1}
    assert captured["kwargs"]["supabase"] is ctx.supabase
    assert captured["kwargs"]["club_id"] == ctx.club_id
    assert captured["kwargs"]["name_to_id"] == {"A": 1}
    assert captured["kwargs"]["df_players_all"] == "players_df"
    assert captured["kwargs"]["df_leagues"] == "leagues_df"
    assert captured["kwargs"]["df_meta"] == "meta_df"



def test_submit_match_batch_wraps_errors_as_failure(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("jupr_app.services.match_service.process_matches", boom)

    ctx = ServiceContext(supabase=object(), club_id="club-123")
    result = submit_match_batch(
        ctx,
        [{"winner": "A"}],
        name_to_id={"A": 1},
        df_players_all="players_df",
        df_leagues="leagues_df",
        df_meta="meta_df",
    )

    assert result == ServiceResult.failure("kaboom")



def test_fastapi_admin_batch_uses_submit_match_batch(monkeypatch):
    pytest.importorskip("fastapi")
    from services.api import main as api_main

    submit_calls = {}

    class FakeUser:
        email = "admin@example.com"
        user_id = "uid-1"

    class FakeRole:
        role = "admin"

    def fake_submit(ctx, matches, **kwargs):
        submit_calls["ctx"] = ctx
        submit_calls["matches"] = matches
        submit_calls["kwargs"] = kwargs
        return ServiceResult.success(data={"processed": len(matches)})

    monkeypatch.setattr(api_main, "is_next_admin_score_entry_enabled", lambda: True)
    monkeypatch.setattr(api_main, "authenticate_bearer", lambda _auth: FakeUser())
    monkeypatch.setattr(api_main, "get_supabase_client", lambda: object())
    monkeypatch.setattr(api_main, "resolve_admin_role", lambda **_kwargs: FakeRole())
    monkeypatch.setattr(api_main, "has_permission", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(api_main, "load_data", lambda *_args, **_kwargs: ("players", None, "leagues", None, "meta", None, None, None, None, {"A": 1}))
    monkeypatch.setattr(api_main, "write_admin_activity_log", lambda *_args, **_kwargs: ServiceResult.success(data={}))
    monkeypatch.setattr(api_main, "submit_match_batch", fake_submit)

    payload = api_main.MatchBatchRequest(matches=[{"winner": "A"}], source="test")
    out = api_main.submit_admin_match_batch("club-123", payload, authorization="Bearer token")

    assert out["ok"] is True
    assert submit_calls["matches"] == [{"winner": "A"}]
    assert submit_calls["kwargs"]["name_to_id"] == {"A": 1}

    api_source = Path("services/api/main.py").read_text(encoding="utf-8")
    assert "from jupr_app.domain.match_processing import process_matches" not in api_source
