from __future__ import annotations

import os


PUBLIC_INTAKE_WRITE_FLAG = "JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES"
LEAGUE_MANAGER_WRITE_FLAG = "JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES"
TEAM_LEAGUE_ADMIN_WRITE_WAVE = "league-manager"
TEAM_LEAGUE_PUBLIC_WRITE_WAVE = "public-intake-auth"
MATCH_CANONICAL_NORMALIZE_WRITE_FLAG = (
    "JUPR_ENABLE_STAGING_NEXT_ADMIN_MATCH_CANONICAL_NORMALIZE_WRITES"
)
COMMUNICATIONS_MUTATION_FLAG = "JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS"
PERMANENT_OPEN_WRITE_WAVE = "open"
TRUTHY = {"1", "true", "yes", "y", "on"}
NON_STAGING_WRITE_ENVIRONMENTS = {"local", "test", "development", "dev", "production"}


def staging_write_wave_allows(*waves: str) -> bool:
    """Return whether the active staging posture permits a named write surface.

    Named waves remain available for focused diagnosis. The permanent ``open``
    posture is the normal staging state and deliberately includes every reviewed
    named wave. Callers must continue to enforce their independent feature flag
    and environment checks.
    """

    current = os.getenv("JUPR_STAGING_WRITE_WAVE", "").strip()
    allowed = {
        str(wave or "").strip()
        for wave in waves
        if str(wave or "").strip()
    }
    return current == PERMANENT_OPEN_WRITE_WAVE or current in allowed


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


def _local_test_writes_enabled() -> bool:
    return os.getenv("JUPR_ENV", "").strip().lower() in {
        "local",
        "test",
        "development",
        "dev",
    }


def staging_admin_team_league_writes_enabled() -> bool:
    """Keep the unaccepted team-league surface outside production.

    Local/test environments remain usable for deterministic verification. The
    only hosted environment that can write is isolated staging, and it must
    have both the reviewed League Manager wave and its existing write flag.
    """

    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if _local_test_writes_enabled():
        return True
    if environment != "staging":
        return False
    return (
        staging_write_wave_allows(TEAM_LEAGUE_ADMIN_WRITE_WAVE)
        and os.getenv(LEAGUE_MANAGER_WRITE_FLAG, "").strip().lower() in TRUTHY
    )


def require_staging_admin_team_league_writes() -> None:
    if staging_admin_team_league_writes_enabled():
        return
    raise PermissionError(
        "Admin team-league writes are staging-only. Open only the approved "
        f"{TEAM_LEAGUE_ADMIN_WRITE_WAVE} wave with "
        f"{LEAGUE_MANAGER_WRITE_FLAG}=1."
    )


def staging_public_team_league_writes_enabled() -> bool:
    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if _local_test_writes_enabled():
        return True
    if environment != "staging":
        return False
    return (
        staging_write_wave_allows(TEAM_LEAGUE_PUBLIC_WRITE_WAVE)
        and os.getenv(PUBLIC_INTAKE_WRITE_FLAG, "").strip().lower() in TRUTHY
    )


def require_staging_public_team_league_writes() -> None:
    if staging_public_team_league_writes_enabled():
        return
    raise PermissionError(
        "Public team-league writes are staging-only. Open only the approved "
        f"{TEAM_LEAGUE_PUBLIC_WRITE_WAVE} wave with "
        f"{PUBLIC_INTAKE_WRITE_FLAG}=1."
    )


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


def staging_communications_mutations_enabled() -> bool:
    """Keep staging communications reads open without opening their writes."""

    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if environment in {"local", "test", "development", "dev"}:
        return True
    if environment != "staging":
        return False
    return (
        staging_write_wave_allows("communications")
        and os.getenv(COMMUNICATIONS_MUTATION_FLAG, "").strip().lower() in TRUTHY
    )


def require_staging_communications_mutations() -> None:
    if staging_communications_mutations_enabled():
        return
    raise PermissionError(
        "Communications mutations are disabled. Use permanent-open staging or "
        f"the communications wave with {COMMUNICATIONS_MUTATION_FLAG}=1."
    )
