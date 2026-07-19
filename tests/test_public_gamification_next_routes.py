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
        "Link directly to this badge",
        "data-load-more-badges",
        "data-load-more-earners",
        "data-badge-earners-panel",
        "data-trophy-player",
        "Complete definitions",
    ):
        assert contract in page

    assert "getClubBadgeEarners" in api
    assert "/earners?" in api
    assert "BadgeTrophyRoomEntry" in api


def test_challenge_ladder_next_route_consumes_python_eligibility_and_rulebook() -> None:
    page = (
        ROOT / "apps/web/app/clubs/[clubSlug]/challenge-ladder/page.tsx"
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
    assert 'authority: "python"' in api
    assert 'eligibility_authority: "python"' in api
