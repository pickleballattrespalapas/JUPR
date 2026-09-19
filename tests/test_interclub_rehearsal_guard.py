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
