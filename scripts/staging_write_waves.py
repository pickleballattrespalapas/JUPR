from __future__ import annotations

import argparse
from pathlib import Path
import re


NO_WRITE_WAVE = "none"
OPEN_WRITE_WAVE = "open"
ADMIN_WRITE_PILOT_FLAG = "JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT"


def _admin_wave(*flags: str) -> tuple[str, ...]:
    return (ADMIN_WRITE_PILOT_FLAG, *flags)

# Named waves remain available for diagnosis and emergency isolation. The `open`
# wave is the normal staging posture and enables every reviewed staging mutation
# gate. Production-only email and public-live override flags remain disabled.
STAGING_WRITE_WAVES: dict[str, tuple[str, ...]] = {
    NO_WRITE_WAVE: (),
    "public-intake-auth": (
        "JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES",
        "JUPR_ENABLE_STAGING_TOURNAMENT_COMMERCE_WRITES",
    ),
    "communications": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS",
    ),
    "match-player": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY",
        "JUPR_ENABLE_NEXT_ADMIN_REPLAY",
        "JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY",
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER",
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES",
        "JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR",
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_MATCH_CANONICAL_NORMALIZE_WRITES",
    ),
    "match-exclusion-recovery": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY",
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE",
        "JUPR_ENABLE_NEXT_ADMIN_REPLAY",
    ),
    "support-requests": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_SUPPORT_REQUESTS",
    ),
    "league-manager": _admin_wave(
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES",
    ),
    "league-awards": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE",
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES",
    ),
    "league-live-domain": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_PREVIEW",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN",
    ),
    "league-live-submit": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_PREVIEW",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT",
    ),
    "badge-diagnostics": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS",
    ),
    "admin-tools": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_TOOLS",
    ),
    "challenge-ladder": _admin_wave(
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_CHALLENGE_LADDER_WRITES",
    ),
    "moneyball": _admin_wave(
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_MONEYBALL_WRITES",
    ),
    "jupr-live": _admin_wave(
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_JUPR_LIVE_WRITES",
    ),
    "public-live": (
        "JUPR_ENABLE_PUBLIC_LIVE_WRITES",
    ),
    "tournament-mutations": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_MUTATIONS",
    ),
    "tournament-setup": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS",
    ),
    "tournament-registration": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_REGISTRATION_MUTATIONS",
    ),
    "tournament-operations": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
    ),
    "tournament-official-publish": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH",
    ),
    "tournament-email-handoff": _admin_wave(
        "JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS",
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH",
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_EMAIL_HANDOFF",
    ),
    "tournament-live": _admin_wave(
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES",
    ),
    "tournament-live-official-publish": _admin_wave(
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH",
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES",
    ),
    "tournament-commerce-admin": _admin_wave(
        "JUPR_ENABLE_STAGING_TOURNAMENT_COMMERCE_WRITES",
    ),
}

STAGING_WRITE_WAVES[OPEN_WRITE_WAVE] = tuple(
    sorted(
        {
            flag
            for wave, flags in STAGING_WRITE_WAVES.items()
            if wave != NO_WRITE_WAVE
            for flag in flags
        }
    )
)

DORMANT_STAGING_WRITE_FLAGS = (
    # The current import handoff is a GET-only, write_count=0 projection. Keep
    # its reserved future write gate explicitly off until a mutation exists.
    "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_IMPORT_HANDOFF",
)

# Unsafe requests are denied before route execution unless their exact FastAPI
# method/template belongs to the selected wave. Route/service flags remain a
# second, independent authorization layer. Keeping templates (rather than
# prefix regexes) lets static inventory tests reject every future unclassified
# unsafe route.
STAGING_WRITE_WAVE_ROUTES: dict[str, tuple[tuple[str, str], ...]] = {
    NO_WRITE_WAVE: (),
    "public-intake-auth": (
        ("POST", "/clubs/{club_slug}/support/intake"),
        ("POST", "/clubs/{club_slug}/verified-updates/request"),
        ("POST", "/clubs/{club_slug}/tournament-registration"),
        ("POST", "/clubs/{club_slug}/tournament-registration/edit"),
        ("POST", "/clubs/{club_slug}/tournament-registration/edit-link/request"),
        ("POST", "/clubs/{club_slug}/tournament-registration/profile-resolution"),
        ("POST", "/clubs/{club_slug}/tournament-registration/pairing-interest"),
        ("POST", "/clubs/{club_slug}/tournament-registration/pairing-requests/{partner_request_id}/accept"),
        ("POST", "/clubs/{club_slug}/tournament-registration/pairing-requests/{partner_request_id}/decline"),
        ("POST", "/clubs/{club_slug}/tournament-registration/pairing-requests/{partner_request_id}/cancel"),
        ("POST", "/clubs/{club_slug}/team-leagues/{league_name}/registrations"),
        ("POST", "/clubs/{club_slug}/team-leagues/partner-confirmations"),
        ("POST", "/clubs/{club_slug}/tournament-registration/four-player-team"),
        ("POST", "/clubs/{club_slug}/tournament-registration/four-player-team/recover"),
        ("POST", "/clubs/{club_slug}/tournament-team-invitation/resolve"),
        ("POST", "/clubs/{club_slug}/tournament-team-invitation/respond"),
        ("POST", "/clubs/{club_slug}/tournament-commerce/quote"),
        ("POST", "/email-preferences/unsubscribe"),
    ),
    "communications": (
        ("POST", "/admin/clubs/{club_id}/player-updates/send-range"),
        ("POST", "/admin/clubs/{club_id}/player-updates/digests/preview"),
        ("POST", "/admin/clubs/{club_id}/player-updates/digests/queue"),
        ("POST", "/admin/clubs/{club_id}/player-updates/outbox/send"),
        ("POST", "/admin/clubs/{club_id}/player-updates/outbox/retry"),
        ("POST", "/admin/clubs/{club_id}/player-updates/outbox/delete"),
        ("POST", "/admin/clubs/{club_id}/player-updates/subscriptions/{subscription_id}/replace"),
        ("POST", "/admin/clubs/{club_id}/player-updates/subscriptions/{subscription_id}/deactivate"),
        ("PATCH", "/admin/clubs/{club_id}/verified-updates/requests/{subscription_id}"),
        ("POST", "/admin/clubs/{club_id}/weekly-recap/generate"),
        ("PATCH", "/admin/clubs/{club_id}/weekly-recap/recaps/{week_start}"),
        ("POST", "/admin/clubs/{club_id}/weekly-recap/recaps/{week_start}/publish"),
    ),
    "match-player": (
        ("PATCH", "/admin/clubs/{club_id}/match-log/social/{social_match_id}"),
        ("POST", "/admin/clubs/{club_id}/match-log/social/delete"),
        ("PATCH", "/admin/clubs/{club_id}/match-log/edits"),
        ("POST", "/admin/clubs/{club_id}/match-log/edits/{operation_id}/recover"),
        ("POST", "/admin/clubs/{club_id}/match-log/duplicates/resolve"),
        ("POST", "/admin/clubs/{club_id}/replay-history"),
        ("POST", "/admin/clubs/{club_id}/matches/batch"),
        ("POST", "/admin/clubs/{club_id}/match-uploader/round-robin/preview"),
        ("POST", "/admin/clubs/{club_id}/match-uploader/players"),
        ("POST", "/admin/clubs/{club_id}/match-uploader/singles"),
        ("POST", "/admin/clubs/{club_id}/match-uploader/batch"),
        ("POST", "/admin/clubs/{club_id}/players/editor/merge/preview"),
        ("POST", "/admin/clubs/{club_id}/players/editor/merge"),
        ("POST", "/admin/clubs/{club_id}/players/editor/merge/{operation_id}/compensate"),
        ("POST", "/admin/clubs/{club_id}/players/editor/merge/{operation_id}/replay-evidence"),
        ("POST", "/admin/clubs/{club_id}/players/editor/social-identities/auto-link"),
        ("PATCH", "/admin/clubs/{club_id}/players/editor/social-identities/{club_person_id}"),
        ("POST", "/admin/clubs/{club_id}/players/editor/players"),
        ("PATCH", "/admin/clubs/{club_id}/players/editor/players/{player_id}"),
        ("PATCH", "/admin/clubs/{club_id}/players/editor/players/{player_id}/league-ratings/{league_rating_id}"),
        ("POST", "/admin/clubs/{club_id}/match-canonical-audit/run"),
        ("POST", "/admin/clubs/{club_id}/match-canonical-audit/normalize"),
    ),
    "match-exclusion-recovery": (
        ("POST", "/admin/clubs/{club_id}/match-log/duplicates/cleanup"),
        ("POST", "/admin/clubs/{club_id}/match-log/exclude"),
        ("POST", "/admin/clubs/{club_id}/match-log/exclusions/{operation_id}/recover"),
    ),
    "support-requests": (
        ("PATCH", "/admin/clubs/{club_id}/support-requests/{request_id}"),
    ),
    "league-manager": (
        ("POST", "/admin/clubs/{club_id}/league-manager/leagues"),
        ("POST", "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/duplicate"),
        ("POST", "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/lifecycle"),
        ("POST", "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/schedule/preview"),
        ("PATCH", "/admin/clubs/{club_id}/league-manager/leagues/{league_name}"),
        ("PATCH", "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/roster/{player_id}"),
        ("POST", "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/roster/batch"),
        ("PUT", "/admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/settings"),
        ("POST", "/admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/schedule-preview/{phase}"),
        ("POST", "/admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/schedule"),
        ("POST", "/admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/waitlist-actions"),
        ("POST", "/admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/fixtures/{fixture_id}/score"),
        ("POST", "/admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/fixtures/{fixture_id}/reconcile"),
        ("POST", "/admin/clubs/{club_id}/league-manager/team-leagues/operations/{operation_id}/resolve"),
    ),
    "league-awards": (
        ("POST", "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/freeze"),
        ("POST", "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/preview"),
        ("POST", "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/overrides"),
        ("POST", "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/mint"),
        ("POST", "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/archive"),
        ("POST", "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/close"),
        ("PUT", "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/config"),
    ),
    "league-live-domain": (
        ("POST", "/admin/clubs/{club_id}/match-uploader/round-robin/preview"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live/roster-suggestion"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live-sessions"),
        ("PATCH", "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/snapshot"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/plan"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}"),
    ),
    "league-live-submit": (
        ("POST", "/admin/clubs/{club_id}/match-uploader/round-robin/preview"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live/roster-suggestion"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live-sessions"),
        ("PATCH", "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/snapshot"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/plan"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/submit"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/reconcile"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/compensate"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/guests"),
    ),
    "badge-diagnostics": (
        ("PATCH", "/admin/clubs/{club_id}/badges/{badge_id}/state"),
        ("POST", "/admin/clubs/{club_id}/badges/recompute"),
        ("PATCH", "/admin/clubs/{club_id}/badges/revoke"),
    ),
    "admin-tools": (
        ("POST", "/admin/clubs/{club_id}/tools/social-submissions/{event_id}/moderate"),
        ("POST", "/admin/clubs/{club_id}/tools/backfills/tournament-matches/apply"),
        ("PATCH", "/admin/clubs/{club_id}/tools/roles"),
        ("POST", "/admin/clubs/{club_id}/tools/workers/badge-queue"),
        ("POST", "/admin/clubs/{club_id}/tools/workers/badge-recompute"),
        ("POST", "/admin/clubs/{club_id}/tools/backfills/tournament-matches/operations/{operation_key}/recover"),
    ),
    "challenge-ladder": (
        ("POST", "/admin/clubs/{club_id}/challenge-ladder/challenges"),
        ("POST", "/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/start-clock"),
        ("POST", "/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/accept"),
        ("POST", "/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/forfeit"),
        ("POST", "/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/pass"),
        ("POST", "/admin/clubs/{club_id}/challenge-ladder/roster"),
        ("POST", "/admin/clubs/{club_id}/challenge-ladder/roster/{player_id}/move"),
        ("POST", "/admin/clubs/{club_id}/challenge-ladder/roster/replace-tier/preview"),
        ("POST", "/admin/clubs/{club_id}/challenge-ladder/roster/replace-tier"),
        ("PUT", "/admin/clubs/{club_id}/challenge-ladder/roster/{player_id}/overrides"),
        ("PATCH", "/admin/clubs/{club_id}/challenge-ladder/roster/{player_id}/overrides"),
        ("POST", "/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/result"),
        ("POST", "/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/result/preview"),
        ("PATCH", "/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}"),
        ("POST", "/admin/clubs/{club_id}/challenge-ladder/operations/{operation_key}/reconcile"),
    ),
    "moneyball": (
        ("POST", "/admin/clubs/{club_id}/moneyball/preview"),
        ("POST", "/admin/clubs/{club_id}/moneyball/settlement"),
        ("POST", "/admin/clubs/{club_id}/moneyball/submit"),
        ("POST", "/admin/clubs/{club_id}/moneyball/operations/{operation_key}/reconcile"),
    ),
    "jupr-live": (
        ("POST", "/admin/clubs/{club_id}/jupr-live/sessions"),
        ("PATCH", "/admin/clubs/{club_id}/jupr-live/sessions/{session_key}"),
        ("PATCH", "/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/scores"),
        ("POST", "/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/advance"),
        ("POST", "/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/publish"),
        ("POST", "/admin/clubs/{club_id}/jupr-live/operations/{operation_key}/reconcile"),
    ),
    "public-live": (
        ("POST", "/clubs/{club_slug}/live-sessions"),
        ("PATCH", "/clubs/{club_slug}/live-sessions/{session_key}/scores"),
        ("POST", "/clubs/{club_slug}/live-sessions/{session_key}/advance"),
        ("POST", "/clubs/{club_slug}/live-sessions/{session_key}/substitutions"),
        ("POST", "/clubs/{club_slug}/live-sessions/{session_key}/complete"),
    ),
    "tournament-mutations": (
        ("PATCH", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}"),
        ("PATCH", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/status-action"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/delete-draft"),
    ),
    "tournament-setup": (
        ("POST", "/admin/clubs/{club_id}/tournaments/setup/tournaments"),
        ("POST", "/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/impact"),
        ("PATCH", "/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/settings"),
        ("PUT", "/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/draft"),
        ("POST", "/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/publish"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/events/{event_option_id}/config"),
    ),
    "tournament-registration": (
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/broadcast-preview"),
        ("PATCH", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/bulk"),
        ("PATCH", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/{registration_id}"),
        ("PATCH", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/selections/{selection_id}"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/rating-verifications"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/rating-reviews"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/rating-reviews/close"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/teams"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/teams/{team_id}/invitations/reissue"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/teams/{team_id}/roster"),
    ),
    "tournament-operations": (
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws"),
        ("PUT", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams/import-registrations"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams/import-bulk"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/games/round-robin"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/games/playoffs"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/results-import/preview"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/results-import/commit"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/podium"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/podium/awards"),
        ("PATCH", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/games/{game_id}/score"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/draws/{draw_id}/round-robin"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/draws/{draw_id}/playoffs"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/matchups/{matchup_id}/lineups"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/games/{match_game_id}/score"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/games/{match_game_id}/reconcile"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/draws/{draw_id}/podium"),
    ),
    "tournament-official-publish": (
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/matches/publish"),
    ),
    "tournament-email-handoff": (
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/matches/publish"),
    ),
    "tournament-live": (
        ("POST", "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/draws/{draw_id}/commands"),
        ("POST", "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/draws/{draw_id}/operations/{operation_key}/reconcile"),
    ),
    "tournament-live-official-publish": (
        ("POST", "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/draws/{draw_id}/commands"),
        ("POST", "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/draws/{draw_id}/operations/{operation_key}/reconcile"),
        ("POST", "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/matches/publish"),
    ),
    "tournament-commerce-admin": (
        ("POST", "/admin/clubs/{club_id}/tournaments/commerce/tournaments/{tournament_id}/catalog/preview"),
        ("PUT", "/admin/clubs/{club_id}/tournaments/commerce/tournaments/{tournament_id}/catalog"),
        ("PATCH", "/admin/clubs/{club_id}/tournaments/commerce/tournaments/{tournament_id}/orders/{registration_id}/payment"),
        ("POST", "/admin/clubs/{club_id}/tournaments/commerce/tournaments/{tournament_id}/orders/{registration_id}/cancel"),
        ("PATCH", "/admin/clubs/{club_id}/tournaments/commerce/tournaments/{tournament_id}/fulfillment/{fulfillment_id}"),
    ),
}

STAGING_WRITE_WAVE_ROUTES[OPEN_WRITE_WAVE] = tuple(
    sorted(
        {
            route
            for wave, routes in STAGING_WRITE_WAVE_ROUTES.items()
            if wave != NO_WRITE_WAVE
            for route in routes
        }
    )
)


def _route_template_pattern(template: str) -> str:
    parts = re.split(r"(\{[^{}]+\})", str(template))
    return "".join(r"[^/]+" if part.startswith("{") else re.escape(part) for part in parts)


def wave_allows_request(wave: str, method: str, path: str) -> bool:
    routes = STAGING_WRITE_WAVE_ROUTES.get(wave, ())
    clean_method = str(method or "").strip().upper()
    clean_path = str(path or "").strip()
    return any(
        clean_method == allowed_method
        and re.fullmatch(_route_template_pattern(template), clean_path) is not None
        for allowed_method, template in routes
    )

ALL_STAGING_WRITE_FLAGS = tuple(
    sorted(
        {flag for flags in STAGING_WRITE_WAVES.values() for flag in flags}
        | set(DORMANT_STAGING_WRITE_FLAGS)
    )
)

# These are never opened by a staging wave.  The public-live production
# override is intentionally not written into fly.staging.toml at all.
ALWAYS_DISABLED_FLAGS = (
    "JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL",
    "JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION",
)

REGISTRATION_SECRET_WAVES = frozenset(
    {"public-intake-auth", "tournament-registration", OPEN_WRITE_WAVE}
)
PUBLIC_LIVE_SECRET_WAVES = frozenset({"public-live", OPEN_WRITE_WAVE})


def expected_write_flags(wave: str) -> dict[str, bool]:
    if wave not in STAGING_WRITE_WAVES:
        raise ValueError(f"Unknown staging write wave: {wave}")
    enabled = set(STAGING_WRITE_WAVES[wave])
    return {flag: flag in enabled for flag in ALL_STAGING_WRITE_FLAGS}


def configure_fly_staging(path: Path, *, wave: str) -> None:
    expected = expected_write_flags(wave)
    text = path.read_text(encoding="utf-8")
    for name, enabled in expected.items():
        pattern = rf'^(\s*{re.escape(name)}\s*=\s*)"[^"]*"\s*$'
        replacement = rf'\g<1>"{1 if enabled else 0}"'
        text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"fly staging config must define {name} exactly once")
    wave_pattern = r'^(\s*JUPR_STAGING_WRITE_WAVE\s*=\s*)"[^"]*"\s*$'
    text, count = re.subn(
        wave_pattern,
        rf'\g<1>"{wave}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError("fly staging config must define JUPR_STAGING_WRITE_WAVE exactly once")
    path.write_text(text, encoding="utf-8")


def append_github_env(path: Path, *, wave: str) -> None:
    expected = expected_write_flags(wave)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"JUPR_STAGING_WRITE_WAVE={wave}\n")
        for name, enabled in expected.items():
            handle.write(f"{name}={1 if enabled else 0}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure the Fly staging write posture; open is the permanent test default."
    )
    parser.add_argument("--wave", required=True, choices=tuple(STAGING_WRITE_WAVES))
    parser.add_argument("--fly-config", type=Path, required=True)
    parser.add_argument("--github-env", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_fly_staging(args.fly_config, wave=args.wave)
    if args.github_env is not None:
        append_github_env(args.github_env, wave=args.wave)
    enabled = [name for name, value in expected_write_flags(args.wave).items() if value]
    print(f"Configured staging write wave {args.wave!r} with {len(enabled)} enabled gate(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
