# Public player aliases

`players.name` remains the canonical staff/import name. The optional
`players.public_alias` changes public presentation without creating a profile,
changing IDs, replaying ratings, or rewriting match/session history.

Staff can set or clear **Public alias (optional)** in Player Editor. Updates use
the existing audited, idempotent, compare-and-swap editor operation. CourtReserve
import and generator/hub selection use the authenticated staff roster endpoint;
public directory search and sorting use the alias.

The public API boundary projects aliases onto saved names in profiles, matches,
round-robin sessions/results, standings, tournament rosters, league results,
recaps and CSV exports. Recap PDF data is projected before rendering. Interclub
public results resolve the participating clubs. Staff routes and private
registration editing retain canonical names. Public responses remain aliased
even when the caller includes a staff bearer token.

Apply `player_public_aliases` before deploying code. Set
`JUPR_ENABLE_PUBLIC_PLAYER_ALIASES=true` in the API environment to enable the
snapshot/response boundary (enabled in staging). Alias lookups are club-scoped,
paged, cached for at most 15 seconds, and fail closed. Website caches may take
another minute to refresh. Existing PDFs or exports downloaded before an alias
change cannot be recalled.

Acceptance example: canonical **Carol Becker**, alias **Ann Johnson**. A
CourtReserve row for Carol matches the original profile ID; the staff lineup
retains Carol, while public search/profile and result snapshots show Ann.
The staging-only acceptance profile is **Alias acceptance staff**, ID 623,
with public alias **Alias acceptance public**. No production player has been
changed as part of staging acceptance.
