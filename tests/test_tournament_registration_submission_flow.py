from types import SimpleNamespace

from jupr_app.ui.pages import tournament_registration as reg
from jupr_app.ui.pages import tournament_registration_confirmation as confirm
from jupr_app.ui import tournament_registration_confirmation_view as view


def test_submission_result_helpers_store_get_and_clear(monkeypatch):
    state = {}
    monkeypatch.setattr(reg.st, "session_state", state)
    reg._store_submission_result(tournament_id="t1", registration_id="r1", email_status="sent", nav_params={"a": "b"})
    assert reg._get_submission_result("t1") == {"registration_id": "r1", "email_status": "sent", "nav_params": {"a": "b"}}
    state[reg._wizard_key("t1")] = {"current_step": 5}
    reg._clear_registration_wizard_for_new_start("t1")
    assert reg._get_submission_result("t1") == {}
    assert reg._wizard_key("t1") not in state


def test_confirmation_summary_renders_bundle_missing_email_status(monkeypatch):
    calls = []
    monkeypatch.setattr(view.st, "title", lambda *a, **k: calls.append(("title", a)))
    monkeypatch.setattr(view.st, "success", lambda *a, **k: calls.append(("success", a)))
    monkeypatch.setattr(view.st, "warning", lambda *a, **k: calls.append(("warning", a)))
    monkeypatch.setattr(view.st, "info", lambda *a, **k: calls.append(("info", a)))
    monkeypatch.setattr(view.st, "markdown", lambda *a, **k: calls.append(("markdown", a)))
    monkeypatch.setattr(view.st, "table", lambda *a, **k: calls.append(("table", a)))
    monkeypatch.setattr(view.st, "subheader", lambda *a, **k: calls.append(("subheader", a)))
    monkeypatch.setattr(view.st, "write", lambda *a, **k: calls.append(("write", a)))
    monkeypatch.setattr(view.st, "caption", lambda *a, **k: calls.append(("caption", a)))
    bundle = {"registration": {"display_name": "Ada", "email": "ada@example.com"}, "selections": [], "days": [], "event_options": []}
    view.render_registration_confirmation_summary(bundle=bundle, email_status="", sender_status={"ok": True}, show_title=True)
    assert any(call[0] == "success" for call in calls)


def test_session_fallback_registration_id_only_when_query_missing(monkeypatch):
    monkeypatch.setattr(confirm, "get_submission_result", lambda tournament_id: {"registration_id": "session-r", "email_status": "failed"})
    assert confirm._registration_id_from_query_or_session("t1", "query-r")[0] == "query-r"
    rid, submission = confirm._registration_id_from_query_or_session("t1", "")
    assert rid == "session-r"
    assert submission["email_status"] == "failed"
