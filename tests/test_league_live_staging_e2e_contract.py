from pathlib import Path


SPEC = Path("apps/web/e2e/league-live-session.staging.spec.ts")
WORKFLOW = Path(".github/workflows/fly_api_staging_deploy.yml")


def test_league_live_mutation_e2e_is_explicit_staging_only_and_non_retrying() -> None:
    text = SPEC.read_text(encoding="utf-8")

    assert "JUPR_LEAGUE_LIVE_ALLOW_MUTATION_E2E" in text
    assert "RUN DISPOSABLE STAGING WRITES" in text
    assert "https://juprleagues-api-staging.fly.dev" in text
    assert "production_targets_contacted: false" in text
    assert 'test.describe.configure({ mode: "serial", retries: 0 })' in text
    assert 'status: "retryable"' in text
    assert "published_match_ids: []" in text
    assert 'trigger: "Retry R1"' in text
    assert 'trigger: "Finish session"' in text
    assert "unique_published_matches: new Set(publishedMatchIds).size" in text
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
    assert "JUPR_LEAGUE_LIVE_E2E_SESSION_ID=d7221db1-db24-52d2-90fc-b667c16e0193" in text
    assert "JUPR_LEAGUE_LIVE_E2E_OPERATION_ID=87a15480-a0d4-4a1d-922f-f200d6e4f259" in text
    assert "dnoockbwfenunhcibwfn" not in text
