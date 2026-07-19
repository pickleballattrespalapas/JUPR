from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_match_explorer_browser_renders_python_projection_without_rating_policy_copy() -> None:
    form = (
        ROOT
        / "apps/web/app/clubs/[clubSlug]/match-explorer/MatchExplorerForm.tsx"
    ).read_text(encoding="utf-8")

    assert "preview.impact_chart.points" in form
    assert "preview.player_impacts" in form
    assert "preview.expected.score_to_11.label" in form
    assert "preview.score.beat_expectation_pp" in form
    assert "deltaYouFromShare" not in form
    assert "MIN_WIN_DELTA_ELO" not in form
    assert "CAP_LOSER_GAIN_ELO" not in form


def test_weekly_recap_browser_has_bounded_paging_and_same_origin_exports() -> None:
    page = (ROOT / "apps/web/app/clubs/[clubSlug]/weekly-recap/page.tsx").read_text(encoding="utf-8")
    api = (ROOT / "apps/web/lib/weeklyRecapApi.ts").read_text(encoding="utf-8")
    proxy = (
        ROOT
        / "apps/web/app/api/clubs/[clubSlug]/weekly-recaps/[weekStart]/pdf/route.ts"
    ).read_text(encoding="utf-8")

    assert 'page_size: "8"' in api
    assert "weekly-recap-previous-page" in page
    assert "weekly-recap-next-page" in page
    assert "Community Events" in page
    assert "New Faces" in page
    assert "`/api/clubs/${encodeURIComponent(clubSlug)}" in api
    assert 'headers.set("Content-Type"' in proxy
    assert 'cache: "no-store"' in proxy


def test_route_specific_explorer_recap_staging_evidence_is_present() -> None:
    spec = (ROOT / "apps/web/e2e/public-explorer-recap.spec.ts").read_text(encoding="utf-8")

    assert "four different players" in spec
    assert "match-explorer-impact-chart" in spec
    assert "weekly-recap-pdf-link" in spec
    assert "data-print-mode" in spec
