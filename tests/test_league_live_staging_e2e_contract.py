from pathlib import Path


SPEC = Path("apps/web/e2e/league-live-session.staging.spec.ts")
FOUR_WEEK_SPEC = Path("apps/web/e2e/league-four-week.staging.spec.ts")
WORKFLOW = Path(".github/workflows/fly_api_staging_deploy.yml")
LEAGUE_LIVE_PAGE = Path("apps/web/app/admin/league-manager/live/page.tsx")


def test_league_live_mutation_e2e_is_explicit_staging_only_and_non_retrying() -> None:
    text = SPEC.read_text(encoding="utf-8")

    assert "JUPR_LEAGUE_LIVE_ALLOW_MUTATION_E2E" in text
    assert "RUN DISPOSABLE STAGING WRITES" in text
    assert "https://juprleagues-api-staging.fly.dev" in text
    assert "production_targets_contacted: false" in text
    assert 'test.describe.configure({ mode: "serial", retries: 0 })' in text
    assert 'confirmation_text: "CREATE LIVE SESSION"' in text
    assert 'source: "staging_league_live_browser_acceptance"' in text
    assert "createDisposableFixtures" in text
    assert "/players/editor/players" in text
    assert "cleanupDisposableFixtures" in text
    assert 'leagueStatus).toBe("archived")' in text
    assert "inactivePlayerCount).toBe(4)" in text
    assert 'const legacyAcceptanceLeagueName = "Acceptance Flex 0822A"' in text
    assert "/match-log/exclude" in text
    assert 'excluded.replay_status).toBe("succeeded")' in text
    assert "legacy_acceptance_matches_excluded" in text
    assert "GITHUB_RUN_ID" in text
    assert "GITHUB_RUN_ATTEMPT" in text
    assert 'test("creates and completes a disposable five-round League Live session"' in text
    assert 'trigger: "Publish reviewed scores"' in text
    assert 'trigger: "Apply movement and continue"' in text
    assert "Unusual score — verify before publish" in text
    assert 'name: "Start next round", exact: true' in text
    assert 'name: "4. Score Entry with Review", exact: true' in text
    assert "retainedOperationId" not in text
    assert 'trigger: "Retry R1"' not in text
    assert 'trigger: "Finish session"' in text
    assert 'liveRoute.searchParams.set("league_id", leagueId)' in text
    assert 'liveRoute.searchParams.set("league_name", expectedLeagueName)' in text
    assert 'liveRoute.searchParams.set("mode", "Individual")' in text
    assert 'article[aria-labelledby="league-live-setup-heading"]' in text
    assert 'name: "Unfinished sessions for this league"' in text
    assert 'hasText: /^Existing sessions/' not in text
    assert "unique_published_matches: new Set(publishedMatchIds).size" in text
    assert 'expect(expectedLeagueName).toMatch(/^League Live E2E /)' in text
    assert "pickleballclubsandwich.com" not in text
    assert "dnoockbwfenunhcibwfn" not in text


def test_fly_workflow_runs_league_live_e2e_once_for_an_explicit_spec_change() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    detect = text.index("- name: Detect explicit League Live browser acceptance")
    handoff = text.index("- name: Build exact-candidate staging handoff")
    run = text.index("- name: Run authorized League Live browser acceptance")
    cleanup = text.index("- name: End League Live browser acceptance auth session")
    upload = text.index("- name: Upload successful staging handoff")

    assert handoff < detect < run < cleanup < upload
    assert "github.event_name == 'push'" in text
    assert "apps/web/e2e/league-live-session.staging.spec.ts" in text
    assert "STAGING_SUPABASE_SERVICE_ROLE_KEY" in text
    assert "STAGING_SUPABASE_ANON_KEY" in text
    assert "VERCEL_AUTOMATION_BYPASS_SECRET" in text
    assert "JUPR_ATTESTED_VERCEL_DEPLOYMENT_ORIGIN" in text
    assert "JUPR_LEAGUE_LIVE_ALLOW_MUTATION_E2E=1" in text
    assert "if: always() && steps.league-live-e2e.outputs.run == 'true'" in text
    assert "prepare_parity_staging_session.py" in text
    assert "            cleanup \\" in text
    assert "JUPR_LEAGUE_LIVE_E2E_SESSION_ID" not in text
    assert "JUPR_LEAGUE_LIVE_E2E_OPERATION_ID" not in text
    assert "JUPR_LEAGUE_LIVE_E2E_LEAGUE_ID" not in text
    assert 'JUPR_LEAGUE_LIVE_E2E_LEAGUE_NAME="League Live E2E ${candidate_suffix}"' in text
    assert "JUPR_LEAGUE_LIVE_E2E_LEAGUE_TYPE" not in text
    assert 'candidate_suffix="${GITHUB_SHA:0:7}-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"' in text
    assert "Acceptance Flex 0822A" not in text
    assert "apps/web/e2e/league-four-week.staging.spec.ts" in text
    assert "spec=e2e/league-four-week.staging.spec.ts" in text
    assert "JUPR_FOUR_WEEK_ALLOW_MUTATION_E2E=1" in text
    assert 'JUPR_FOUR_WEEK_E2E_LEAGUE_NAME="Acceptance Four Week Flex ${candidate_suffix}"' in text
    assert 'JUPR_FOUR_WEEK_E2E_ASPEN_NAME="E2E Created Aspen ${candidate_suffix}"' in text
    assert 'JUPR_FOUR_WEEK_E2E_BIRCH_NAME="E2E Created Birch ${candidate_suffix}"' in text
    assert 'JUPR_FOUR_WEEK_E2E_CLOVER_NAME="E2E Created Clover ${candidate_suffix}"' in text
    assert 'npx playwright test "$acceptance_spec" --retries=0 --forbid-only' in text
    assert "- name: Upload failed League Live browser evidence" in text
    assert "if: failure() && steps.league-live-e2e.outputs.run == 'true'" in text
    assert "dnoockbwfenunhcibwfn" not in text


def test_four_week_acceptance_is_guarded_and_verifies_the_full_story() -> None:
    text = FOUR_WEEK_SPEC.read_text(encoding="utf-8")

    assert "JUPR_FOUR_WEEK_ALLOW_MUTATION_E2E" in text
    assert "JUPR_FOUR_WEEK_E2E_LEAGUE_NAME" in text
    assert "JUPR_FOUR_WEEK_E2E_ASPEN_NAME" in text
    assert "JUPR_FOUR_WEEK_E2E_BIRCH_NAME" in text
    assert "JUPR_FOUR_WEEK_E2E_CLOVER_NAME" in text
    assert "JUPR_FOUR_WEEK_E2E_REPORT_PATH" in text
    assert "RUN DISPOSABLE STAGING WRITES" in text
    assert "https://juprleagues-api-staging.fly.dev" in text
    assert 'test.describe.configure({ mode: "serial", retries: 0 })' in text
    assert "production_targets_contacted: false" in text
    assert "expected_match_count: 36" in text
    assert "expected_session_count: 4" in text
    assert "expected_schedule_weeks: 4" in text
    assert "page.setDefaultTimeout(20_000)" in text
    assert "page.setDefaultNavigationTimeout(45_000)" in text
    assert "/admin/auth/capabilities?club_id=" in text
    assert "capabilities: verifiedCapabilities" in text
    assert 'git_commit_sha: candidateSha' in text
    assert 'await expect(leagueSelect).toHaveValue(leagueName)' in text
    assert 'name: "Weekday", exact: true })).toHaveValue("0")' in text
    assert 'name: "Ladder pod size", exact: true })).toHaveValue("4")' in text
    assert 'name: "Match structure", exact: true })).toHaveValue("one_game")' in text
    assert 'name: "Action", exact: true })).toHaveValue("activate")' in text
    assert "async function chooseOption(select: Locator, value: string)" in text
    assert "element.dispatchEvent(new Event(\"change\", { bubbles: true }))" in text
    assert ".selectOption(" not in text
    assert "991001" in text
    assert "991023" in text
    assert 'trigger: "Freeze and save"' in text
    assert 'trigger: "Mint and verify"' in text
    assert 'trigger: "Archive completed league"' in text
    assert "dnoockbwfenunhcibwfn" not in text


def test_league_live_admin_player_lookup_bypasses_public_cache() -> None:
    text = LEAGUE_LIVE_PAGE.read_text(encoding="utf-8")

    assert 'getClubPlayers(clubSlug, { status: "all", limit: 1000, sort: "name", noStore: true })' in text
