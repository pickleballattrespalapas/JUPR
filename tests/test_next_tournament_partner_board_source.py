from pathlib import Path


PAGE = Path("apps/web/app/clubs/[clubSlug]/tournament-partner-board/page.tsx")
INTEREST = Path("apps/web/app/clubs/[clubSlug]/tournament-partner-board/PairingInterestPanel.tsx")
REVIEW = Path("apps/web/app/clubs/[clubSlug]/tournament-partner-board/PartnerRequestReviewPanel.tsx")
ROUTES = Path("services/api/public_tournament_pairing_routes.py")


def test_partner_board_uses_consent_projection_and_fastapi_only():
    page = PAGE.read_text(encoding="utf-8")
    interest = INTEREST.read_text(encoding="utf-8")
    review = REVIEW.read_text(encoding="utf-8")

    assert "partner_board_entries" in page
    assert "roster?.players_needing_partners" not in page
    assert "pairing-interest" in interest
    assert "pairing-requests" in review
    combined = "\n".join([page, interest, review]).lower()
    assert "@supabase" not in combined
    assert "createclient(" not in combined
    assert "service_role" not in combined
    assert "requester_selection_id?:" not in review
    assert "target_selection_id?:" not in review


def test_partner_review_exposes_all_owned_transitions_with_confirmation():
    review = REVIEW.read_text(encoding="utf-8")
    routes = ROUTES.read_text(encoding="utf-8")

    assert "/${action}`" in review
    for action in ("accept", "decline", "cancel"):
        assert f'/{action}")' in routes
    assert "confirmAction" in review
    assert "Confirm ${label.toLowerCase()}" in review
    assert "status_code=409" in routes
