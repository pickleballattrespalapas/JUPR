from __future__ import annotations

import hashlib
import hmac


PUBLIC_TOURNAMENT_REFERENCE_PREFIX = "tr"


def build_public_tournament_reference(
    *,
    tournament_id: str,
    namespace: str,
    source_id: str,
) -> str:
    """Return a stable, opaque reference for a public tournament resource.

    Public roster and partner-board responses must not expose database row IDs.
    The source IDs are UUID-like in production, so a one-way digest is sufficient
    for a public locator; it is not an authorization credential. All writes remain
    protected by a registration edit token and resolve this reference server-side.
    """

    tournament = str(tournament_id or "").strip()
    scope = str(namespace or "").strip().lower()
    source = str(source_id or "").strip()
    if not tournament or not scope or not source:
        return ""
    digest = hashlib.sha256(f"{scope}\x1f{tournament}\x1f{source}".encode("utf-8")).hexdigest()[:24]
    return f"{PUBLIC_TOURNAMENT_REFERENCE_PREFIX}_{digest}"


def public_tournament_reference_matches(
    candidate: str,
    *,
    tournament_id: str,
    namespace: str,
    source_id: str,
) -> bool:
    expected = build_public_tournament_reference(
        tournament_id=tournament_id,
        namespace=namespace,
        source_id=source_id,
    )
    return bool(expected) and hmac.compare_digest(str(candidate or "").strip(), expected)
