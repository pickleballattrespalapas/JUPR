from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

try:
    from services.api.auth import jwt_verification_configured, jwt_verification_mode
except Exception:  # pragma: no cover - imported by non-API contexts too
    def jwt_verification_configured() -> bool:
        return False

    def jwt_verification_mode() -> str:
        return "unavailable"

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
STREAMLIT_FALLBACK_DEFAULT = "https://juprtrespalapas.streamlit.app"

PERMANENT_GUARDRAILS = [
    "No Supabase service-role keys, JWT secrets, or database credentials in Vercel/browser code.",
    "No direct browser writes to rating, match, player, league, badge, or tournament tables.",
    "No JavaScript rewrite of rating, match-processing, badge-evaluation, or replay logic.",
    "Every enabled production write workflow must be club-scoped and server-side through FastAPI/domain services.",
    "Every destructive or rating-adjacent write must have audit attribution and a correction/replay path.",
]

PILOT_GATES = [
    "Closed-club or explicitly approved operational window.",
    "FastAPI endpoint uses Python domain/service authority rather than duplicating logic in Next.js.",
    "Staff-only auth and club-scoped authorization are required before staff-facing writes are broadly enabled.",
    "Audit logging is enabled or the workflow documents why it is read/planning-only.",
    "Streamlit fallback remains available until the specific workflow is proven in staging and production pilot use.",
    "Staging smoke or equivalent contract validation runs before broad operator use.",
]


@dataclass(frozen=True)
class AdminWorkflowDefinition:
    key: str
    label: str
    streamlit_page_key: str
    next_route: str | None
    api_scope: str
    access: str
    risk: str
    env_flag: str
    status_when_disabled: str
    next_action: str
    safety_notes: tuple[str, ...]


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _env_text(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip() or default


WORKFLOWS: tuple[AdminWorkflowDefinition, ...] = (
    AdminWorkflowDefinition(
        key="admin_shell",
        label="Admin Operations Shell",
        streamlit_page_key="admin_login",
        next_route="/admin",
        api_scope="status_only",
        access="staff",
        risk="low",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_SHELL",
        status_when_disabled="available_status_only",
        next_action="Use as the migration cockpit and expose workflow readiness without enabling writes.",
        safety_notes=("Status-only shell; no database mutations.",),
    ),
    AdminWorkflowDefinition(
        key="match_log",
        label="Match Log / Corrections / Replay Planning",
        streamlit_page_key="match_log",
        next_route="/admin/match-log",
        api_scope="match_read_correction_planning",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG",
        status_when_disabled="streamlit_fallback",
        next_action="Use read/duplicate-scan visibility first; enable apply only with JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY after operator review.",
        safety_notes=(
            "Read/scan visibility uses JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG.",
            "Edits and duplicate cleanup require JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY plus Supabase JWT role authorization.",
            "Writes use Python bulk match editor services and audit attribution.",
        ),
    ),
    AdminWorkflowDefinition(
        key="replay_history",
        label="Replay History",
        streamlit_page_key="admin_tools",
        next_route="/admin/replay-history",
        api_scope="replay_snapshots_league_ratings_overall_reset",
        access="super_admin",
        risk="critical",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_REPLAY",
        status_when_disabled="streamlit_fallback",
        next_action="Use after Match Log edits/cleanup to rebuild snapshots and ratings through Python replay_history.",
        safety_notes=(
            "Requires Supabase JWT auth with run_replay permission.",
            "League replay rebuilds snapshots and league_ratings for selected league.",
            "Full reset updates overall player stats and should stay tightly controlled.",
        ),
    ),
    AdminWorkflowDefinition(
        key="score_entry",
        label="Score Entry MVP",
        streamlit_page_key="match_uploader",
        next_route="/clubs/tres-palapas/admin/score-entry",
        api_scope="single_match_batch",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY",
        status_when_disabled="guarded_off",
        next_action="Keep available as a minimal single-match fallback while the full Match Uploader route is piloted.",
        safety_notes=("Existing guarded endpoint requires Supabase JWT auth when enabled.", "Rated match writes must stay server-side through Python services."),
    ),
    AdminWorkflowDefinition(
        key="match_uploader",
        label="Full Match Uploader",
        streamlit_page_key="match_uploader",
        next_route="/admin/match-uploader",
        api_scope="manual_batch_round_robin_score_entry",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER",
        status_when_disabled="guarded_off",
        next_action="Pilot manual/batch, singles input, round-robin scheduling, and new-player creation; player update emails can run after finished batches when enabled.",
        safety_notes=(
            "Manual/batch and round-robin submission require Supabase JWT auth with enter_scores permission.",
            "Writes call FastAPI and the Python process_matches domain path.",
            "Use Match Log and Replay History after corrections or duplicate cleanup.",
        ),
    ),
    AdminWorkflowDefinition(
        key="player_editor",
        label="Player Editor",
        streamlit_page_key="player_editor",
        next_route="/admin/players",
        api_scope="player_roster_detail_create_update_merge_social",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR",
        status_when_disabled="streamlit_fallback",
        next_action="Use guarded player create/update, league-rating edits, social identity linking, and high-friction merge; run Replay History ALL after production merge.",
        safety_notes=(
            "Create/update requires Supabase JWT auth with manage_players permission.",
            "Player merge rewires match history and is audit-flagged.",
            "Replay History ALL is required after production merge operations.",
        ),
    ),
    AdminWorkflowDefinition(
        key="player_updates",
        label="Player Update Emails",
        streamlit_page_key="player_updates_admin",
        next_route="/admin/player-updates",
        api_scope="player_update_digest_range_send",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES",
        status_when_disabled="streamlit_fallback",
        next_action="Generate and send date-range player update email reports to verified subscribers after completed batch sessions.",
        safety_notes=(
            "Requires Supabase JWT auth with manage_subscriptions permission.",
            "Uses existing SMTP_* transactional email configuration and JUPR_EMAIL_MODE safety controls.",
            "Automatic post-batch sending is separately gated by JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS.",
        ),
    ),
    AdminWorkflowDefinition(
        key="support_requests",
        label="Support / Correction / Privacy Requests",
        streamlit_page_key="data_corrections",
        next_route="/admin/support-requests",
        api_scope="public_support_request_review_queue",
        access="staff",
        risk="medium",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_SUPPORT_REQUESTS",
        status_when_disabled="streamlit_fallback",
        next_action="Review public data-correction and profile-privacy intake rows, assign status, and route actual fixes through the appropriate admin workflow.",
        safety_notes=(
            "Public intake never mutates match, rating, player, badge, or tournament data.",
            "Status updates are audit-attributed and club-scoped.",
            "Actual corrections still happen through Match Log, Player Editor, Tournament Admin, or Replay History.",
        ),
    ),
    AdminWorkflowDefinition(
        key="league_manager",
        label="League Manager",
        streamlit_page_key="league_manager",
        next_route="/admin/league-manager",
        api_scope="league_settings_roster_lifecycle_print_and_persisted_awards",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER",
        status_when_disabled="streamlit_fallback",
        next_action="Pilot league settings, roster, and the separately flagged recoverable Awards wizard; keep League Live submission staged behind its own review/publish slices.",
        safety_notes=(
            "League Manager routes require Supabase JWT auth with manage_matches permission.",
            "Settings and roster writes are audit-attributed and club-scoped.",
            "League Awards writes require JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE and verified mint evidence before archive.",
        ),
    ),
    AdminWorkflowDefinition(
        key="tournament_admin",
        label="Tournament Admin / Ops",
        streamlit_page_key="tournaments",
        next_route="/admin/tournaments",
        api_scope="teams_draws_scores_podiums_trophies_official_publish",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS",
        status_when_disabled="streamlit_fallback",
        next_action="Use guarded registration, draw, scoring, podium, award, official publish, singles, and winner-bonus workflows; Tournament Live is the separate in-play runner.",
        safety_notes=("Draw creation, score finalization, podiums, trophy awards, and official match publish are staff-controlled.",),
    ),
    AdminWorkflowDefinition(
        key="tournament_live",
        label="Tournament Live Runner",
        streamlit_page_key="tournament_live",
        next_route="/admin/tournament-live",
        api_scope="draw_live_scoring_progression_official_publish",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS",
        status_when_disabled="streamlit_fallback",
        next_action="Use as the tournament-specific in-play control room after Tournament Ops has created/imported the draw and teams.",
        safety_notes=(
            "Tournament Live reuses guarded Tournament Ops FastAPI endpoints rather than JUPR Live one-off event flows.",
            "Official publish remains explicit and can affect ratings and player update emails.",
            "Tournament Ops remains the setup/import workspace; Tournament Live is focused on live draw running.",
        ),
    ),
    AdminWorkflowDefinition(
        key="challenge_ladder_admin",
        label="Challenge Ladder Admin",
        streamlit_page_key="challenge_ladder_admin",
        next_route="/admin/challenge-ladder",
        api_scope="ladder_roster_challenges_results_rank_movement",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER",
        status_when_disabled="streamlit_fallback",
        next_action="Keep lowest priority until core score/correction, tournament, League Manager, player update, and email safety workflows are proven.",
        safety_notes=("Challenge completion can insert matches and move ladder ranks.",),
    ),
    AdminWorkflowDefinition(
        key="weekly_recap_admin",
        label="Weekly Recap Admin",
        streamlit_page_key="weekly_recap_admin",
        next_route="/admin/weekly-recap",
        api_scope="recap_generate_edit_publish",
        access="staff",
        risk="medium",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP",
        status_when_disabled="streamlit_fallback",
        next_action="Port once admin auth shell and audit attribution are available.",
        safety_notes=("Publishes public-facing recap content but does not directly mutate ratings.",),
    ),
    AdminWorkflowDefinition(
        key="admin_tools",
        label="Admin Tools / Workers / Backfills",
        streamlit_page_key="admin_tools",
        next_route="/admin/tools",
        api_scope="diagnostics_badge_workers_backfills",
        access="super_admin",
        risk="critical",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_TOOLS",
        status_when_disabled="streamlit_only",
        next_action="Keep Streamlit-only until job and super-admin authorization contracts are hardened.",
        safety_notes=("Includes backfill and worker operations that can affect large data ranges.",),
    ),
)


def _workflow_payload(workflow: AdminWorkflowDefinition, *, pilot_enabled: bool) -> dict[str, Any]:
    flag_enabled = _truthy_env(workflow.env_flag)
    payload = {
        **asdict(workflow),
        "enabled": bool(flag_enabled),
        "pilot_enabled": bool(pilot_enabled),
        "effective_status": "enabled_for_pilot" if flag_enabled else workflow.status_when_disabled,
        "can_enable_for_pilot": bool(pilot_enabled or workflow.key == "admin_shell"),
        "requires_review_before_enablement": workflow.risk in {"high", "critical"},
        "safety_notes": list(workflow.safety_notes),
    }
    if workflow.key == "match_log":
        payload["apply_env_flag"] = "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY"
        payload["apply_enabled"] = _truthy_env("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY")
    if workflow.key == "player_updates":
        payload["auto_send_env_flag"] = "JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS"
        payload["auto_send_enabled"] = _truthy_env("JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS")
    if workflow.key == "league_manager":
        payload["awards_write_env_flag"] = "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE"
        payload["awards_write_enabled"] = _truthy_env("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE")
    return payload


def build_admin_operations_status() -> dict[str, Any]:
    """Return public-safe status for the Next admin migration cockpit."""

    env = _env_text("JUPR_ENV", "local").lower()
    pilot_enabled = _truthy_env("JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT")
    workflows = [_workflow_payload(workflow, pilot_enabled=pilot_enabled) for workflow in WORKFLOWS]
    enabled_workflows = [workflow["key"] for workflow in workflows if workflow.get("enabled")]
    return {
        "service": "jupr-api",
        "environment": env,
        "mode": "closed_club_production_write_pilot" if pilot_enabled else "guarded_public_read_spine",
        "write_pilot_enabled": bool(pilot_enabled),
        "streamlit_fallback_url": _env_text("JUPR_STREAMLIT_FALLBACK_URL", STREAMLIT_FALLBACK_DEFAULT),
        "strict_audit_required": _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"),
        "service_role_configured": bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()),
        "jwt_verification_configured": jwt_verification_configured(),
        "jwt_verification_mode": jwt_verification_mode(),
        "enabled_workflows": enabled_workflows,
        "recommended_sequence": [
            "admin_shell",
            "match_log",
            "replay_history",
            "score_entry",
            "match_uploader",
            "player_updates",
            "support_requests",
            "player_editor",
            "league_manager",
            "tournament_admin",
            "tournament_live",
            "weekly_recap_admin",
            "challenge_ladder_admin",
            "admin_tools",
        ],
        "pilot_gates": list(PILOT_GATES),
        "permanent_guardrails": list(PERMANENT_GUARDRAILS),
        "workflows": workflows,
    }
