from __future__ import annotations

import os


PUBLIC_INTAKE_WRITE_FLAG = "JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES"
LEAGUE_MANAGER_WRITE_FLAG = "JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES"
MATCH_CANONICAL_NORMALIZE_WRITE_FLAG = (
    "JUPR_ENABLE_STAGING_NEXT_ADMIN_MATCH_CANONICAL_NORMALIZE_WRITES"
)
TRUTHY = {"1", "true", "yes", "y", "on"}
NON_STAGING_WRITE_ENVIRONMENTS = {"local", "test", "development", "dev", "production"}


def staging_public_intake_writes_enabled() -> bool:
    """Return whether public intake writes may run in this environment.

    The extra gate applies only to the isolated staging deployment. Production
    retains its established public flows; staging remains read-only until the
    explicit public-intake wave is selected.
    """

    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if environment in NON_STAGING_WRITE_ENVIRONMENTS:
        return True
    if environment != "staging":
        return False
    return os.getenv(PUBLIC_INTAKE_WRITE_FLAG, "").strip().lower() in TRUTHY


def require_staging_public_intake_writes() -> None:
    if not staging_public_intake_writes_enabled():
        raise PermissionError(
            f"Public intake writes are disabled on staging. Open {PUBLIC_INTAKE_WRITE_FLAG} only for the approved public-intake wave."
        )


def require_staging_league_manager_writes() -> None:
    if staging_league_manager_writes_enabled():
        return
    raise PermissionError(
        f"League Manager writes are disabled on staging. Open {LEAGUE_MANAGER_WRITE_FLAG} only for the approved League Manager wave."
    )


def staging_league_manager_writes_enabled() -> bool:
    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if environment in NON_STAGING_WRITE_ENVIRONMENTS:
        return True
    if environment != "staging":
        return False
    return os.getenv(LEAGUE_MANAGER_WRITE_FLAG, "").strip().lower() in TRUTHY


def require_staging_match_canonical_normalize_writes() -> None:
    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if environment != "staging":
        raise PermissionError("Match Canonical normalization is available only in the isolated staging environment.")
    if os.getenv(MATCH_CANONICAL_NORMALIZE_WRITE_FLAG, "").strip().lower() not in TRUTHY:
        raise PermissionError(
            "Match Canonical normalization writes are disabled on staging. "
            f"Open {MATCH_CANONICAL_NORMALIZE_WRITE_FLAG} only for the approved match-player wave."
        )


def staging_match_canonical_normalize_writes_enabled() -> bool:
    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if environment != "staging":
        return False
    return os.getenv(MATCH_CANONICAL_NORMALIZE_WRITE_FLAG, "").strip().lower() in TRUTHY
