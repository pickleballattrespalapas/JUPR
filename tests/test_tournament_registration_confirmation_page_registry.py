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
