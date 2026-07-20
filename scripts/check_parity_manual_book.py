#!/usr/bin/env python3
"""Validate the structural and completed parity manual-evidence book."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

try:
    from scripts.check_parity_closure_program import MATRIX_PATH, partial_keys
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from check_parity_closure_program import MATRIX_PATH, partial_keys

ROOT = Path(__file__).resolve().parents[1]
BOOK_PATH = ROOT / "docs" / "next_parity_manual_staging_book.md"
EXPECTED_STAGING_PROJECT_REF = "sijpxjxvdtrehmqvirfi"
EXPECTED_STAGING_WEB_ORIGIN = (
    "https://jupr-git-staging-pickleballattrespalapas1.vercel.app"
)
EXPECTED_WAVES = {
    "preflight",
    "public-read",
    "public-intake-auth",
    "admin-read-export",
    "reversible-admin-writes",
    "match-rating-writes",
    "recovery",
}
EXPECTED_MANUAL_MUTATIONS = {
    "support-intake",
    "data-corrections",
    "email-preferences",
    "profile-privacy",
    "verified-updates",
    "tournament-registration",
    "tournament-partner-pairing",
    "weekly-recap",
    "player-updates",
    "subscription-outbox",
    "league-manager",
    "league-awards",
    "tournament-admin",
    "social-moderation",
    "score-entry",
    "match-uploader",
    "match-log",
    "player-editor",
    "league-live",
    "challenge-ladder",
    "moneyball",
    "jupr-live",
    "public-live",
    "tournament-operations",
    "tournament-live-non-score",
}
EXPECTED_FIXTURE_SCOPES = {
    "support-intake",
    "registration-pairing",
    "league-awards-live",
    "match-player-replay",
    "ladder-moneyball-live",
    "tournament-admin-ops-live",
    "recap-subscription-outbox",
    "auth-role-recovery",
}
EXPECTED_FLAG_KEYS = {
    "global",
    "public-intake-auth",
    "admin-read",
    "communications",
    "match-player",
    "league",
    "live-ladder-admin",
    "public-live",
    "tournament-admin",
    "tournament-ops",
    "tournament-live",
    "email-safety",
}
EXPECTED_MIGRATION_KEYS = {
    "baseline-verified-updates",
    "baseline-match-soft-delete",
    "baseline-unsubscribe",
    "baseline-admin-roles",
    "baseline-admin-audit",
    "baseline-replay-jobs",
    "baseline-club-config",
    "baseline-leaderboards",
    "baseline-role-scope",
    "baseline-club-onboarding",
    "baseline-worker-log",
    "baseline-confirmations",
    "baseline-live-sessions",
    "baseline-live-contract",
    "baseline-match-log-resolution",
    "baseline-selection-guards",
    "baseline-selection-locks",
    "baseline-badge-claims",
    "baseline-registration-player",
    "legacy-league-awards-schema",
    "legacy-top-performer-seed",
    "order-02-lockdown",
    "order-02-canonicalize",
    "order-03-registration-edit",
    "order-13-support-intake",
    "order-16-replay-idempotency",
    "order-22-communications",
    "order-20-league-live-domain",
    "order-21-league-live-submit",
    "order-17-player-merge",
    "order-15-partner-pairing",
    "order-24-live-ladder",
    "order-26-tournament-admin",
    "order-23-admin-diagnostics",
    "order-27-tournament-ops",
    "order-28-tournament-live",
    "order-25-public-live",
}
RECOVERY_REQUIRED_KEYS = {
    "match_canonical_audit",
    "data_corrections",
    "email_preferences",
    "profile_privacy",
    "league_manager",
    "match_uploader",
    "match_log",
    "player_editor",
    "admin_tools",
    "challenge_ladder_admin",
    "moneyball",
    "jupr_live",
    "jupr_live_admin",
    "tournaments",
    "tournament_manager",
    "tournament_ops",
    "tournament_live",
    "tournament_registration",
    "tournament_registration_admin",
    "tournament_registration_edit",
    "tournament_partner_board",
    "weekly_recap_admin",
    "player_updates_admin",
    "reset_password",
    "verified_updates_request",
}
REQUIRED_CANDIDATE_FIELDS = {
    "Application candidate Git SHA",
    "Final stacked PR",
    "Vercel preview URL",
    "Vercel deployment ID",
    "Vercel immutable deployment origin",
    "Fly staging image ref",
    "Deployment identity preflight artifact",
    "Staging Supabase project ref",
    "Schema inventory / migration head evidence",
    "Streamlit fallback URL / build",
    "Staging role accounts exercised",
    "Session start / end",
    "Primary operator",
    "Witness / reviewer",
}
RESERVED_COMPLETION_RE = re.compile(
    r"\b(?:pending|blocked|fail|failed|unresolved)\b|\bnot\s+applied\b",
    re.IGNORECASE,
)

BOOK_ROW_RE = re.compile(
    r"^\|\s*`(?P<key>[^`]+)`\s*\|\s*`(?P<result>Pending|Pass|Fail|Blocked)`\s*\|"
    r"\s*(?P<evidence>[^|]*?)\s*\|\s*(?P<recovery>[^|]*?)\s*\|"
    r"\s*(?P<operator>[^|]*?)\s*\|$",
    re.MULTILINE,
)
WAVE_ROW_RE = re.compile(
    r"^\|\s*`(?P<wave>[^`]+)`\s*\|\s*(?P<command>[^|]*?)\s*\|"
    r"\s*(?P<inputs>[^|]*?)\s*\|\s*`(?P<result>Pending|Pass|Fail|Blocked)`\s*\|"
    r"\s*(?P<evidence>[^|]*?)\s*\|\s*(?P<operator>[^|]*?)\s*\|$",
    re.MULTILINE,
)
MANUAL_MUTATION_ROW_RE = re.compile(
    r"^\|\s*`manual:(?P<surface>[^`]+)`\s*\|\s*(?P<route>[^|]*?)\s*\|"
    r"\s*(?P<write>[^|]*?)\s*\|\s*(?P<recovery>[^|]*?)\s*\|"
    r"\s*`(?P<result>Pending|Pass|Fail|Blocked)`\s*\|\s*(?P<operator>[^|]*?)\s*\|$",
    re.MULTILINE,
)
FIXTURE_ROW_RE = re.compile(
    r"^\|\s*`fixture:(?P<scope>[^`]+)`\s*\|\s*(?P<ids>[^|]*?)\s*\|"
    r"\s*(?P<owner>[^|]*?)\s*\|\s*(?P<cleanup>[^|]*?)\s*\|"
    r"\s*`(?P<result>Pending|Pass|Fail|Blocked)`\s*\|$",
    re.MULTILINE,
)
MIGRATION_ROW_RE = re.compile(
    r"^\|\s*`migration:(?P<key>[^`]+)`\s*\|\s*(?P<required>[^|]*?)\s*\|"
    r"\s*(?P<state>[^|]*?)\s*\|\s*(?P<evidence>[^|]*?)\s*\|"
    r"\s*(?P<rollback>[^|]*?)\s*\|\s*(?P<owner>[^|]*?)\s*\|$",
    re.MULTILINE,
)
FLAG_ROW_RE = re.compile(
    r"^\|\s*`flag:(?P<key>[^`]+)`\s*\|\s*(?P<staging>[^|]*?)\s*\|"
    r"\s*(?P<production>[^|]*?)\s*\|\s*(?P<disable>[^|]*?)\s*\|"
    r"\s*(?P<evidence>[^|]*?)\s*\|\s*(?P<owner>[^|]*?)\s*\|$",
    re.MULTILINE,
)
CANDIDATE_ROW_RE = re.compile(
    r"^\|\s*(?P<field>[^|`][^|]*?)\s*\|\s*(?P<value>[^|]*?)\s*\|$",
    re.MULTILINE,
)


def _plain(value: str) -> str:
    return value.strip().strip("`").strip()


def _is_placeholder(value: str) -> bool:
    return _plain(value).lower() in {"", "—", "-", "pending", "tbd", "todo", "unresolved"}


def _is_unresolved(value: str) -> bool:
    normalized = _plain(value).upper()
    return normalized.startswith("UNRESOLVED") or normalized.startswith("BLOCKED")


def _has_reserved_completion_prose(value: str) -> bool:
    return bool(RESERVED_COMPLETION_RE.search(_plain(value)))


def _reject_reserved_fields(
    errors: list[str], label: str, values: dict[str, str]
) -> None:
    for field, value in values.items():
        if _has_reserved_completion_prose(value):
            errors.append(f"{label} contains reserved incomplete/failure prose in {field}.")


def _verified(value: str) -> bool:
    return bool(re.fullmatch(r"Verified:\s*\S(?:.*\S)?", _plain(value)))


def _verified_disabled(value: str) -> bool:
    return bool(re.fullmatch(r"Verified disabled:\s*\S(?:.*\S)?", _plain(value)))


def _canonical_manual_path(value: str) -> bool:
    if (
        not value.startswith("/")
        or value.startswith("//")
        or value.endswith("/")
        or "//" in value
        or any(character in value for character in "%?#\\")
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        return False
    segments = value[1:].split("/")
    return bool(segments) and all(
        segment not in {"", ".", ".."}
        and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:@-]*", segment))
        for segment in segments
    )


def _json_2xx(value: str) -> bool:
    return bool(re.fullmatch(r"2\d\d", value)) and value not in {"204", "205"}


def _positive_projection(value: str) -> bool:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any("=" not in part for part in parts):
        return False
    rejected = {"", "{}", "[]", "false", "null", "none", "n/a"}
    for part in parts:
        field, projection = (piece.strip() for piece in part.split("=", 1))
        if not re.fullmatch(r"[a-z][a-z0-9_.-]*", field):
            return False
        if projection.lower() in rejected:
            return False
    return True


def _manual_route_record(value: str) -> tuple[str, str, str, str] | None:
    match = re.fullmatch(
        r"Verified:\s*method=(POST|PUT|PATCH|DELETE);\s*path=([^;]+);\s*"
        r"resource=([^;]+);\s*prestate=(\S(?:.*\S)?)",
        _plain(value),
    )
    if not match or not _canonical_manual_path(match.group(2)):
        return None
    resource, prestate = match.group(3).strip(), match.group(4).strip()
    if _is_placeholder(resource) or _is_placeholder(prestate):
        return None
    return tuple(part.strip() for part in match.groups())  # type: ignore[return-value]


def _manual_write_record(value: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(
        r"Verified:\s*status=(\d{3});\s*projection=([^;]+);\s*artifact=(\S(?:.*\S)?)",
        _plain(value),
    )
    if (
        not match
        or not _json_2xx(match.group(1))
        or not _positive_projection(match.group(2).strip())
        or _is_placeholder(match.group(3))
    ):
        return None
    return tuple(part.strip() for part in match.groups())  # type: ignore[return-value]


def _manual_recovery_record(value: str) -> tuple[str, str, str, str, str] | None:
    match = re.fullmatch(
        r"Verified:\s*method=(POST|PUT|PATCH|DELETE|RETAIN);\s*path=([^;]+);\s*"
        r"status=([^;]+);\s*projection=([^;]+);\s*artifact=(\S(?:.*\S)?)",
        _plain(value),
    )
    if not match:
        return None
    method, path, status, projection, artifact = (
        part.strip() for part in match.groups()
    )
    if method == "RETAIN":
        route_status_valid = path == "N/A" and status == "N/A"
    else:
        route_status_valid = _canonical_manual_path(path) and _json_2xx(status)
    if (
        not route_status_valid
        or not _positive_projection(projection)
        or _is_placeholder(artifact)
    ):
        return None
    return method, path, status, projection, artifact


def _manual_operators(value: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"operator=([A-Za-z0-9][A-Za-z0-9_.:@-]{2,199});\s*"
        r"witness=([A-Za-z0-9][A-Za-z0-9_.:@-]{2,199})",
        _plain(value),
    )
    if not match or match.group(1) == match.group(2):
        return None
    return match.group(1), match.group(2)


def _immutable_vercel_origin(value: str) -> str | None:
    plain = _plain(value)
    if (
        plain == EXPECTED_STAGING_WEB_ORIGIN
        or not re.fullmatch(
            r"https://[a-z0-9](?:[a-z0-9-]{0,180}[a-z0-9])?-[a-z0-9]{8,64}"
            r"-pickleballattrespalapas1\.vercel\.app",
            plain,
        )
    ):
        return None
    return plain


def _identity_binding(value: str) -> tuple[str, str, str, str] | None:
    match = re.fullmatch(
        r"candidate=([0-9a-fA-F]{40});\s*vercel=([^;]+);\s*fly=([^;]+);\s*artifact=(\S+)",
        _plain(value),
    )
    if not match:
        return None
    return tuple(part.strip() for part in match.groups())  # type: ignore[return-value]


def _section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    if marker not in text:
        return ""
    tail = text.split(marker, 1)[1]
    return tail.split("\n## ", 1)[0]


def _unchecked_checkboxes(text: str, heading: str) -> tuple[int, int]:
    section = _section(text, heading)
    checks = re.findall(r"^- \[(?P<mark>[ xX])\]", section, re.MULTILINE)
    return len(checks), sum(mark.lower() != "x" for mark in checks)


def _manifest_key_errors(
    label: str,
    matches: list[re.Match[str]],
    group: str,
    expected: set[str],
) -> list[str]:
    counts = Counter(match.group(group) for match in matches)
    documented = set(counts)
    errors: list[str] = []
    if missing := sorted(expected - documented):
        errors.append(f"{label} ledger is missing: " + ", ".join(missing))
    if extra := sorted(documented - expected):
        errors.append(f"{label} ledger has unknown entries: " + ", ".join(extra))
    if duplicates := sorted(key for key, count in counts.items() if count != 1):
        errors.append(f"{label} ledger entries must appear exactly once: " + ", ".join(duplicates))
    return errors


def check_book(
    matrix_path: Path = MATRIX_PATH,
    book_path: Path = BOOK_PATH,
) -> list[str]:
    """Check the Pending-safe structural contract used by normal pull-request CI."""

    if not matrix_path.exists():
        return [f"Missing parity matrix: {matrix_path}"]
    if not book_path.exists():
        return [f"Missing manual staging book: {book_path}"]

    expected = partial_keys(matrix_path.read_text(encoding="utf-8"))
    text = book_path.read_text(encoding="utf-8")
    rows = list(BOOK_ROW_RE.finditer(text))
    counts = Counter(match.group("key") for match in rows)
    documented = set(counts)
    errors: list[str] = []

    missing = sorted(expected - documented)
    if missing:
        errors.append("Partial pages missing manual evidence rows: " + ", ".join(missing))

    stale = sorted(documented - expected)
    if stale:
        errors.append("Manual rows no longer marked Partial: " + ", ".join(stale))

    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if duplicates:
        errors.append("Manual evidence rows must appear exactly once: " + ", ".join(duplicates))

    if len(expected) != 45:
        errors.append(
            f"Expected the current manual wave to contain 45 Partial pages, found {len(expected)}."
        )

    wave_counts = Counter(match.group("wave") for match in WAVE_ROW_RE.finditer(text))
    waves = set(wave_counts)
    if missing_waves := sorted(EXPECTED_WAVES - waves):
        errors.append("Manual evidence waves missing from the ledger: " + ", ".join(missing_waves))
    if extra_waves := sorted(waves - EXPECTED_WAVES):
        errors.append("Unknown manual evidence waves: " + ", ".join(extra_waves))
    if duplicate_waves := sorted(key for key, count in wave_counts.items() if count != 1):
        errors.append("Manual evidence waves must appear exactly once: " + ", ".join(duplicate_waves))

    errors.extend(
        _manifest_key_errors(
            "Deferred manual mutation",
            list(MANUAL_MUTATION_ROW_RE.finditer(text)),
            "surface",
            EXPECTED_MANUAL_MUTATIONS,
        )
    )

    errors.extend(
        _manifest_key_errors(
            "Fixture",
            list(FIXTURE_ROW_RE.finditer(text)),
            "scope",
            EXPECTED_FIXTURE_SCOPES,
        )
    )
    errors.extend(
        _manifest_key_errors(
            "Migration",
            list(MIGRATION_ROW_RE.finditer(text)),
            "key",
            EXPECTED_MIGRATION_KEYS,
        )
    )
    errors.extend(
        _manifest_key_errors(
            "Feature-flag",
            list(FLAG_ROW_RE.finditer(text)),
            "key",
            EXPECTED_FLAG_KEYS,
        )
    )

    return errors


def check_book_complete(
    matrix_path: Path = MATRIX_PATH,
    book_path: Path = BOOK_PATH,
    candidate_sha: str | None = None,
    vercel_deployment_id: str | None = None,
    vercel_deployment_origin: str | None = None,
    fly_image_ref: str | None = None,
) -> list[str]:
    """Require fully bound, fail-closed evidence before matrix reconciliation."""

    errors = check_book(matrix_path=matrix_path, book_path=book_path)
    if errors or not book_path.exists():
        return errors

    text = book_path.read_text(encoding="utf-8")
    candidate = {
        match.group("field").strip(): match.group("value").strip()
        for match in CANDIDATE_ROW_RE.finditer(_section(text, "Candidate identity"))
    }
    for field in sorted(REQUIRED_CANDIDATE_FIELDS):
        if field not in candidate or _is_placeholder(candidate[field]):
            errors.append(f"Candidate identity is incomplete: {field}")
    _reject_reserved_fields(errors, "Candidate identity", candidate)

    recorded_ref = _plain(candidate.get("Staging Supabase project ref", ""))
    if recorded_ref and recorded_ref != EXPECTED_STAGING_PROJECT_REF:
        errors.append(
            "Staging Supabase project ref must be "
            f"{EXPECTED_STAGING_PROJECT_REF}, found {recorded_ref}."
        )

    recorded_sha = _plain(candidate.get("Application candidate Git SHA", ""))
    if (
        recorded_sha
        and recorded_sha != "—"
        and not re.fullmatch(r"[0-9a-fA-F]{40}", recorded_sha)
    ):
        errors.append("Recorded application candidate SHA must be 40 hexadecimal characters.")

    if candidate_sha:
        required_sha = candidate_sha.strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", required_sha):
            errors.append("Required candidate SHA must be 40 hexadecimal characters.")
        elif recorded_sha.lower() != required_sha.lower():
            errors.append(
                f"Recorded application candidate SHA {recorded_sha or '(missing)'} "
                f"does not match required SHA {required_sha}."
            )

    recorded_web = _plain(candidate.get("Vercel preview URL", ""))
    if recorded_web and recorded_web != EXPECTED_STAGING_WEB_ORIGIN:
        errors.append(
            f"Vercel preview URL must be the allowlisted staging origin {EXPECTED_STAGING_WEB_ORIGIN}."
        )

    recorded_vercel = _plain(candidate.get("Vercel deployment ID", ""))
    recorded_vercel_origin = _plain(
        candidate.get("Vercel immutable deployment origin", "")
    )
    recorded_fly = _plain(candidate.get("Fly staging image ref", ""))
    if not vercel_deployment_id or _is_placeholder(vercel_deployment_id):
        errors.append("Required Vercel deployment ID is missing.")
    elif recorded_vercel != _plain(vercel_deployment_id):
        errors.append("Recorded Vercel deployment ID does not match the required deployment ID.")
    required_vercel_origin = _immutable_vercel_origin(vercel_deployment_origin or "")
    if required_vercel_origin is None:
        errors.append("Required immutable Vercel deployment origin is missing or invalid.")
    elif _immutable_vercel_origin(recorded_vercel_origin) != required_vercel_origin:
        errors.append(
            "Recorded immutable Vercel deployment origin does not match the required origin."
        )
    if not fly_image_ref or _is_placeholder(fly_image_ref):
        errors.append("Required Fly image ref is missing.")
    elif recorded_fly != _plain(fly_image_ref):
        errors.append("Recorded Fly staging image ref does not match the required image ref.")

    binding = _identity_binding(candidate.get("Deployment identity preflight artifact", ""))
    if binding is None:
        errors.append(
            "Deployment identity preflight artifact must use: "
            "candidate=<sha>; vercel=<id>; fly=<image>; artifact=<run-or-url>."
        )
    else:
        bound_sha, bound_vercel, bound_fly, _artifact = binding
        if bound_sha.lower() != recorded_sha.lower():
            errors.append("Deployment identity artifact is bound to a different candidate SHA.")
        if bound_vercel != recorded_vercel:
            errors.append("Deployment identity artifact is bound to a different Vercel deployment ID.")
        if bound_fly != recorded_fly:
            errors.append("Deployment identity artifact is bound to a different Fly image ref.")

    for match in BOOK_ROW_RE.finditer(text):
        key = match.group("key")
        _reject_reserved_fields(
            errors,
            f"Page {key}",
            {field: match.group(field) for field in ("evidence", "recovery", "operator")},
        )
        if match.group("result") != "Pass":
            errors.append(f"Page evidence is not Pass: {key}")
        if _is_placeholder(match.group("evidence")):
            errors.append(f"Page evidence ID/notes are missing: {key}")
        if _is_placeholder(match.group("operator")):
            errors.append(f"Page operator is missing: {key}")
        recovery = _plain(match.group("recovery"))
        if key in RECOVERY_REQUIRED_KEYS and not _verified(recovery):
            errors.append(f"Recovery proof must be `Verified: <evidence>` for mutating page: {key}")
        if key not in RECOVERY_REQUIRED_KEYS and recovery != "N/A" and not _verified(recovery):
            errors.append(f"Recovery result must be `Verified: <evidence>` or `N/A`: {key}")

    for match in WAVE_ROW_RE.finditer(text):
        wave = match.group("wave")
        _reject_reserved_fields(
            errors,
            f"Automated wave {wave}",
            {field: match.group(field) for field in ("command", "inputs", "evidence", "operator")},
        )
        if match.group("result") != "Pass":
            errors.append(f"Automated wave is not Pass: {wave}")
        for field in ("command", "inputs", "evidence", "operator"):
            if _is_placeholder(match.group(field)):
                errors.append(f"Automated wave {wave} is missing {field} evidence.")

    for match in MANUAL_MUTATION_ROW_RE.finditer(text):
        surface = match.group("surface")
        _reject_reserved_fields(
            errors,
            f"Deferred manual mutation {surface}",
            {
                field: match.group(field)
                for field in ("route", "write", "recovery", "operator")
            },
        )
        if match.group("result") != "Pass":
            errors.append(f"Deferred manual mutation is not Pass: {surface}")
        if _manual_route_record(match.group("route")) is None:
            errors.append(
                f"Deferred manual mutation {surface} route must use: "
                "`Verified: method=<POST|PUT|PATCH|DELETE>; path=<canonical-path>; "
                "resource=<id-or-natural-key>; prestate=<captured-baseline>`."
            )
        if _manual_write_record(match.group("write")) is None:
            errors.append(
                f"Deferred manual mutation {surface} write must use: "
                "`Verified: status=<JSON-2xx>; projection=<field=value[,field=value]>; "
                "artifact=<id-or-url>`."
            )
        if _manual_recovery_record(match.group("recovery")) is None:
            errors.append(
                f"Deferred manual mutation {surface} recovery must use: "
                "`Verified: method=<POST|PUT|PATCH|DELETE|RETAIN>; path=<canonical-path|N/A>; "
                "status=<JSON-2xx|N/A>; projection=<field=value[,field=value]>; "
                "artifact=<id-or-url>`."
            )
        if _manual_operators(match.group("operator")) is None:
            errors.append(
                f"Deferred manual mutation {surface} operator/witness must use distinct identities: "
                "`operator=<identity>; witness=<identity>`."
            )

    for match in FIXTURE_ROW_RE.finditer(text):
        scope = match.group("scope")
        _reject_reserved_fields(
            errors,
            f"Fixture {scope}",
            {field: match.group(field) for field in ("ids", "owner", "cleanup")},
        )
        if match.group("result") != "Pass":
            errors.append(f"Fixture cleanup is not Pass: {scope}")
        for field in ("ids", "owner"):
            if _is_placeholder(match.group(field)):
                errors.append(f"Fixture {scope} is missing {field} evidence.")
        if not _verified(match.group("cleanup")):
            errors.append(f"Fixture {scope} cleanup must be `Verified: <evidence>`.")

    for match in MIGRATION_ROW_RE.finditer(text):
        key = match.group("key")
        _reject_reserved_fields(
            errors,
            f"Migration prerequisite {key}",
            {
                field: match.group(field)
                for field in ("required", "state", "evidence", "rollback", "owner")
            },
        )
        for field in ("required", "evidence", "rollback", "owner"):
            value = match.group(field)
            if _is_placeholder(value) or _is_unresolved(value):
                errors.append(f"Migration prerequisite {key} is unresolved or missing {field} evidence.")
        if _plain(match.group("state")) not in {"Applied", "Verified"}:
            errors.append(f"Migration prerequisite {key} state must be exactly `Applied` or `Verified`.")

    for match in FLAG_ROW_RE.finditer(text):
        key = match.group("key")
        _reject_reserved_fields(
            errors,
            f"Feature-flag prerequisite {key}",
            {
                field: match.group(field)
                for field in ("staging", "production", "disable", "evidence", "owner")
            },
        )
        for field in ("staging", "production", "disable", "owner"):
            value = match.group(field)
            if _is_placeholder(value) or _is_unresolved(value):
                errors.append(f"Feature-flag prerequisite {key} is unresolved or missing {field} evidence.")
        production = _plain(match.group("production")).lower()
        if re.search(r"\b(enabled|on|true)\b|=\s*1\b", production):
            errors.append(f"Feature-flag prerequisite {key} claims production is enabled.")
        if not _verified_disabled(match.group("evidence")):
            errors.append(
                f"Feature-flag prerequisite {key} evidence must be "
                "`Verified disabled: <config evidence>`."
            )

    for heading in ("Fail-closed preflight", "Final reconciliation"):
        count, unchecked = _unchecked_checkboxes(text, heading)
        if not count:
            errors.append(f"{heading} has no completion checkboxes.")
        elif unchecked:
            errors.append(f"{heading} has {unchecked} unchecked item(s).")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require bound candidate identity, complete evidence, recovery, and sign-off.",
    )
    parser.add_argument(
        "--candidate-sha",
        help="When completing the book, require this exact application candidate Git SHA.",
    )
    parser.add_argument(
        "--vercel-deployment-id",
        help="When completing the book, require this exact Vercel deployment ID.",
    )
    parser.add_argument(
        "--vercel-deployment-origin",
        help="When completing the book, require this exact immutable Vercel origin.",
    )
    parser.add_argument(
        "--fly-image-ref",
        help="When completing the book, require this exact Fly image ref.",
    )
    args = parser.parse_args()

    errors = (
        check_book_complete(
            candidate_sha=args.candidate_sha,
            vercel_deployment_id=args.vercel_deployment_id,
            vercel_deployment_origin=args.vercel_deployment_origin,
            fly_image_ref=args.fly_image_ref,
        )
        if args.require_complete
        else check_book()
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    count = len(partial_keys(MATRIX_PATH.read_text(encoding="utf-8")))
    mode = "complete evidence for" if args.require_complete else "structural coverage for"
    print(f"Manual staging book has valid {mode} all {count} Partial page definitions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
