"""A rehearsal must fail closed before it can provision staff or write data."""
import pytest
from scripts.run_interclub_rehearsal import API, AUTH, NoRedirect, validate_environment


def staging():
    return dict(GITHUB_ACTIONS="true",GITHUB_REF="refs/heads/staging",GITHUB_SHA="a"*40,
                STAGING_API_BASE_URL=API,STAGING_SUPABASE_URL=AUTH,
                STAGING_SUPABASE_ANON_KEY="test-anon",STAGING_SUPABASE_SERVICE_ROLE_KEY="test-service")


@pytest.mark.parametrize("change", [dict(GITHUB_ACTIONS="false"),dict(GITHUB_REF="refs/heads/rollback-feb8"),
    dict(GITHUB_SHA="staging"),dict(STAGING_API_BASE_URL="https://api.juprleagues.com"),
    dict(STAGING_SUPABASE_URL="https://dnoockbwfenunhcibwfn.supabase.co"),dict(STAGING_SUPABASE_SERVICE_ROLE_KEY="")])
def test_wrong_environment_is_rejected(change):
    with pytest.raises(RuntimeError):
        validate_environment({**staging(),**change})


def test_canonical_staging_is_allowed():
    validate_environment(staging())


def test_redirects_never_forward_staging_credentials():
    with pytest.raises(RuntimeError,match="redirect"):
        NoRedirect().redirect_request(None,None,302,"",{},"https://other.invalid")


def test_repeat_fixture_date_setup_preserves_old_history_and_pending_approvals(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from scripts import run_interclub_rehearsal as rehearsal
    clock=datetime(2026,9,29,20,tzinfo=timezone.utc)
    monkeypatch.setattr(rehearsal,"now",lambda:clock)
    sid="synthetic-season"
    old=rehearsal.iso(clock-timedelta(days=29))
    recent=rehearsal.iso(clock)
    tables={
        "pcs_interclub_seasons":[{"id":sid,"details":{"name":"Fixture","start_date":"2027-01-01"}}],
        "pcs_interclub_entries":[{"id":"existing","season_id":sid,"entered_at":old},
                                {"id":"new","season_id":sid,"entered_at":recent},
                                {"id":"other","season_id":"another-season","entered_at":recent}],
        "pcs_interclub_pool_members":[{"id":"existing","season_id":sid,"approval_status":"approved","approved_at":old},
                                     {"id":"new","season_id":sid,"approval_status":"approved","approved_at":recent},
                                     {"id":"pending","season_id":sid,"approval_status":"pending","approved_at":None},
                                     {"id":"other","season_id":"another-season","approval_status":"approved","approved_at":recent}],
    }
    updated=[]
    def db(method,table,payload=None,**filters):
        def matches(row):
            for key,value in filters.items():
                if key=="select":continue
                op,expected=value.split(".",1)
                if op=="eq" and row[key]!=expected:return False
                if op=="gt" and (row[key] is None or row[key]<=expected):return False
            return True
        rows=[row for row in tables[table] if matches(row)]
        if method=="PATCH":
            for row in rows:
                updated.append((table,row["id"]))
                row.update(payload)
        return rows
    runner=rehearsal.Rehearsal.__new__(rehearsal.Rehearsal)
    runner.db=db
    runner.open_dates({"id":sid})
    assert updated==[("pcs_interclub_seasons",sid),("pcs_interclub_entries","new"),("pcs_interclub_pool_members","new")]
    assert tables["pcs_interclub_pool_members"][2]["approved_at"] is None
    updated.clear()
    monkeypatch.setattr(rehearsal,"now",lambda:clock+timedelta(minutes=20))
    runner.open_dates({"id":sid})
    assert updated==[], "Preparing the next meet must not rerun existing players' database triggers"
