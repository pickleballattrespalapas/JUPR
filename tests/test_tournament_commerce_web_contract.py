from pathlib import Path


PUBLIC_CHOOSER = Path(
    "apps/web/app/clubs/[clubSlug]/tournament-registration/"
    "TournamentCommerceChooser.tsx"
)
NEW_FORM = Path(
    "apps/web/app/clubs/[clubSlug]/tournament-registration/"
    "TournamentRegistrationForm.tsx"
)
EDIT_FORM = Path(
    "apps/web/app/clubs/[clubSlug]/tournament-registration/edit/"
    "EditTournamentRegistrationForm.tsx"
)
ADMIN_PANEL = Path(
    "apps/web/app/admin/tournaments/commerce/TournamentCommercePanel.tsx"
)


def test_public_extras_are_responsive_and_explain_separate_payment():
    source = PUBLIC_CHOOSER.read_text(encoding="utf-8")

    assert 'minmax(min(100%, 260px), 1fr)' in source
    assert 'aria-live="polite"' in source
    assert "pay the tournament organizer separately" in source
    assert "Bundle savings" in source
    assert "qualify for a giveaway" in source
    assert "componentLabel(component)" in source
    assert "line.component_snapshot?.length" in source
    assert "component.total_quantity" in source


def test_public_registration_binds_server_quote_and_rotates_changed_key():
    new_source = NEW_FORM.read_text(encoding="utf-8")
    edit_source = EDIT_FORM.read_text(encoding="utf-8")

    assert "commerceQuote?.quote_fingerprint || null" in new_source
    assert "priorFingerprint !== nextFingerprint" in new_source
    assert "setCommerceIdempotencyKey(crypto.randomUUID())" in new_source
    assert "response.current_quote" in new_source
    assert "expected_quote_fingerprint" in new_source

    assert "nextQuote?.quote_fingerprint || null" in edit_source
    assert "commerceQuote?.quote_fingerprint || null" in edit_source
    assert "nextFingerprint !== currentFingerprint" in edit_source
    assert "expected_order_updated_at" in edit_source
    assert "setCommerceIdempotencyKey(crypto.randomUUID())" in edit_source


def test_admin_workspace_has_review_confirm_cancel_fulfillment_and_recovery():
    source = ADMIN_PANEL.read_text(encoding="utf-8")

    assert "An active item needs at least one active option" in source
    assert "<ConfirmAction" in source
    assert 'confirmationText="SAVE"' in source
    assert 'confirmationText="CANCEL"' in source
    assert "at least 8 characters" in source
    assert 'overflowX: "auto"' in source
    assert 'minWidth: "1080px"' in source
    assert "Download CSV" in source
    assert "Authoritative evidence" in source
    assert "Registration and extras" in source
    assert 'recordListValue(order, "lines")' in source


def test_routes_are_installed_on_existing_public_and_admin_apis():
    public_installer = Path(
        "services/api/public_weekly_recap_routes.py"
    ).read_text(encoding="utf-8")
    admin_installer = Path(
        "services/api/admin_operations_routes.py"
    ).read_text(encoding="utf-8")

    assert "install_public_tournament_commerce_routes" in public_installer
    assert "install_admin_tournament_commerce_routes" in admin_installer


def test_disabled_public_route_check_precedes_club_and_database_access():
    source = Path(
        "services/api/public_tournament_commerce_routes.py"
    ).read_text(encoding="utf-8")

    get_start = source.index("def get_tournament_commerce(")
    get_end = source.index("@app.post", get_start)
    get_body = source[get_start:get_end]
    quote_start = source.index("def post_tournament_commerce_quote(")
    quote_body = source[quote_start:]

    assert get_body.index("_require_feature()") < get_body.index(
        "club = get_club"
    )
    assert quote_body.index("_require_feature()") < quote_body.index(
        "club = get_club"
    )
