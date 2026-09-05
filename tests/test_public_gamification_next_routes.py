from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_badge_codex_next_route_wires_authoritative_buckets_filters_and_paging() -> None:
    page = (
        ROOT / "apps/web/app/clubs/[clubSlug]/badge-codex/page.tsx"
    ).read_text(encoding="utf-8")
    api = (ROOT / "apps/web/lib/badgeApi.ts").read_text(encoding="utf-8")

    for contract in (
        "catalog_buckets",
        'params.set("bucket"',
        'params.set("category"',
        'params.set("scope"',
        "data-badge-bucket",
        "data-badge-id",
        "Share this badge",
        "data-load-more-badges",
        "data-load-more-earners",
        "data-badge-earners-panel",
        "data-trophy-player",
        "badgeAvailabilityLabel",
        "badgeScopeLabel",
        "badgeTimingLabel",
    ):
        assert contract in page

    assert "Complete definitions" not in page

    assert "getClubBadgeEarners" in api
    assert "/earners?" in api
    assert "BadgeTrophyRoomEntry" in api
    for internal_error in ("API error", "Missing JUPR API", "Unable to reach API"):
        assert internal_error not in api


def test_challenge_ladder_next_route_consumes_python_eligibility_and_rulebook() -> None:
    page = (
        ROOT / "apps/web/app/clubs/[clubSlug]/challenge-ladder/page.tsx"
    ).read_text(encoding="utf-8")
    result_component = (
        ROOT / "apps/web/components/ChallengeLadderResultDetails.tsx"
    ).read_text(encoding="utf-8")
    api = (ROOT / "apps/web/lib/challengeLadderApi.ts").read_text(encoding="utf-8")

    assert "eligibleOpponents(" not in page
    assert "canInitiateChallenge(" not in page
    assert "const opponents = eligibility.eligible_opponents" in page
    assert "data-python-eligibility" in page
    assert "data-rulebook-authority" in page
    assert "data-ladder-status" in page
    assert "(data.rulebook ?? []).map" in page
    assert "(data.status_legend ?? []).map" in page
    assert "challenge.completed_at" in page
    assert "player.rank_at_create" in page
    assert "player.current_rank" in page
    assert "player.current_rating_jupr" in page
    assert "playerHref(clubSlug, challenge.winner.player_id)" in page
    assert "ChallengeLadderResultDetails" in page
    assert 'data-result-details="available"' in result_component
    assert "data-result-completeness" in result_component
    assert "Position change:" in result_component
    assert "JUPR before → after" in result_component
    assert "Recorded match" in result_component
    assert "`/clubs/${clubSlug}/matches/${match.match_id}`" in result_component
    assert 'data-result-details="unavailable"' in page
    assert "Match details aren't available for this result" in page
    assert "rank_at_create?: number | null" in api
    assert "current_rank?: number | null" in api
    assert "current_rating_jupr?: number | null" in api
    assert 'completeness: "full" | "partial"' in api
    assert "rating_changes: PublicLadderResultRatingChange[]" in api
    assert "result_details?: PublicLadderResultDetails | null" in api
    assert 'authority: "python"' in api
    assert 'eligibility_authority: "python"' in api
    assert 'cache: "no-store"' in api
    assert "revalidate: 60" not in api


def test_challenge_ladder_admin_derives_match_b_from_two_swing_partner_inputs() -> None:
    panel = (
        ROOT
        / "apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx"
    ).read_text(encoding="utf-8")

    assert "Match A · challenger partner" in panel
    assert "Match A · defender partner" in panel
    assert "Match B swaps the same two partners automatically" in panel
    assert "partner_b_challenger_id: Number(draft.a_def)" in panel
    assert "partner_b_defender_id: Number(draft.a_chal)" in panel
    assert "B challenger partner" not in panel
    assert "B defender partner" not in panel
    assert "b_chal:" not in panel
    assert "b_def:" not in panel
