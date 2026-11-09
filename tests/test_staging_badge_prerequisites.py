import pytest
from scripts.check_staging_badge_prerequisites import RPCS, STAGING_URL, check


def test_read_only_prerequisites_verify_schema_and_no_processing_jobs():
    calls = []
    def fetch(path, accept="application/json"):
        calls.append(path)
        return {"paths": {"/rpc/" + rpc: {} for rpc in RPCS}} if not path else []
    check(STAGING_URL, "secret", fetch=fetch)
    assert len(calls) == 3
    assert "club_id=eq.tres_palapas" in calls[-1]
    assert all(not path.startswith("rpc/") for path in calls)


@pytest.mark.parametrize("url", ["https://dnoockbwfenunhcibwfn.supabase.co", STAGING_URL + "/other", "http://sijpxjxvdtrehmqvirfi.supabase.co"])
def test_refuses_other_database_before_any_request(url):
    with pytest.raises(ValueError):
        check(url, "secret", fetch=lambda *a: pytest.fail("Unexpected database access"))


def test_refuses_missing_migration_or_processing_job():
    with pytest.raises(ValueError, match="RPCs"):
        check(STAGING_URL, "secret", fetch=lambda *a: {"paths": {}})
    def fetch(path, accept="application/json"):
        if not path:
            return {"paths": {"/rpc/" + rpc: {} for rpc in RPCS}}
        return [{"id": "private-id"}] if "processing" in path else []
    with pytest.raises(ValueError, match="processing"):
        check(STAGING_URL, "secret", fetch=fetch)
