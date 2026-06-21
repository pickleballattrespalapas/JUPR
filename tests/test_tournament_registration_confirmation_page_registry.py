from jupr_app.ui.page_registry import HIDDEN_PAGE_KEYS, PAGE_KEY_TO_LABEL, PUBLIC_NAV_KEYS, PAGE_DEFINITIONS


def test_tournament_registration_confirmation_registered_public_hidden():
    key = "tournament_registration_confirmation"
    definition = next(page for page in PAGE_DEFINITIONS if page.key == key)
    assert PAGE_KEY_TO_LABEL[key] == "✅ Registration Confirmation"
    assert definition.public is True
    assert key in HIDDEN_PAGE_KEYS
    assert key not in PUBLIC_NAV_KEYS


def test_confirmation_page_shell_is_plain_public_header(monkeypatch):
    from types import SimpleNamespace

    from jupr_app.ui.pages import tournament_registration_confirmation as page

    calls = []

    def fake_page_shell(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(page, "page_shell", fake_page_shell)
    monkeypatch.setattr(page, "registration_feature_available", lambda _supabase: (False, "closed"))
    monkeypatch.setattr(page.st, "error", lambda *_args, **_kwargs: None)

    page.render(SimpleNamespace(supabase=object()))

    assert calls == [
        (
            ("✅ Registration Confirmation", "Your tournament registration details and payment information."),
            {"mode_label": "Public"},
        )
    ]


def test_confirmation_page_does_not_use_page_shell_context_or_public_kwarg():
    from pathlib import Path

    source = Path("jupr_app/ui/pages/tournament_registration_confirmation.py").read_text()
    assert "with page_shell(" not in source
    assert "public=True" not in source


def test_tournament_registration_edit_registered_public_hidden():
    key = "tournament_registration_edit"
    definition = next(page for page in PAGE_DEFINITIONS if page.key == key)
    assert PAGE_KEY_TO_LABEL[key] == "✏️ Edit Registration"
    assert definition.public is True
    assert key in HIDDEN_PAGE_KEYS
    assert key not in PUBLIC_NAV_KEYS


def test_confirmation_page_prefers_query_registration_id(monkeypatch):
    from jupr_app.ui.pages import tournament_registration_confirmation as page

    monkeypatch.setattr(page, "get_submission_result", lambda _tournament_id: {"registration_id": "session-r"})

    registration_id, submission_result = page._registration_id_from_query_or_session("t1", "query-r")

    assert registration_id == "query-r"
    assert submission_result == {}


def test_confirmation_page_uses_session_fallback_registration_id(monkeypatch):
    from jupr_app.ui.pages import tournament_registration_confirmation as page

    monkeypatch.setattr(page, "get_submission_result", lambda _tournament_id: {"registration_id": "session-r", "email_status": "failed"})

    registration_id, submission_result = page._registration_id_from_query_or_session("t1", "")

    assert registration_id == "session-r"
    assert submission_result["email_status"] == "failed"


def test_confirmation_page_missing_registration_id_without_session_is_friendly(monkeypatch):
    from types import SimpleNamespace

    from jupr_app.ui.pages import tournament_registration_confirmation as page

    errors = []
    monkeypatch.setattr(page, "page_shell", lambda *a, **k: None)
    monkeypatch.setattr(page, "registration_feature_available", lambda _supabase: (True, ""))
    monkeypatch.setattr(page, "get_submission_result", lambda _tournament_id: {})
    monkeypatch.setattr(page.st, "query_params", {"tournament_id": "t1"})
    monkeypatch.setattr(page.st, "error", lambda msg, *a, **k: errors.append(msg))
    monkeypatch.setattr(page.st, "button", lambda *a, **k: False)

    page.render(SimpleNamespace(supabase=object(), club_id="club1"))

    assert errors
    assert "could not find" in errors[0].lower()
