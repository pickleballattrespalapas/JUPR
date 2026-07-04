from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

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
        key="score_entry",
        label="Score Entry MVP",
        streamlit_page_key="match_uploader",
        next_route="/clubs/tres-palapas/admin/score-entry",
        api_scope="single_match_batch",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY",
        status_when_disabled="guarded_off",
        next_action="Keep disabled until auth, role, audit, and correction-path checks are satisfied for the pilot operator set.",
        safety_notes=("Existing guarded endpoint requires Supabase JWT auth when enabled.", "Rated match writes must stay server-side through Python services."),
    ),
    AdminWorkflowDefinition(
        key="match_uploader",
        label="Full Match Uploader",
        streamlit_page_key="match_uploader",
        next_route="/admin/match-uploader",
        api_scope="batch_rr_new_player_score_entry",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER",
        status_when_disabled="not_started",
        next_action="Port after Match Log correction/replay workflow exists.",
        safety_notes=("Includes batch, round-robin, popup/event context, and new-player creation parity.",),
    ),
    AdminWorkflowDefinition(
        key="player_editor",
        label="Player Editor",
        streamlit_page_key="player_editor",
        next_route="/admin/players",
        api_scope="player_crud_merge_league_rating_edits",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR",
        status_when_disabled="not_started",
        next_action="Port after match corrections are safe so player merges and rating edits have recovery paths.",
        safety_notes=("Player merge rewires match and league-rating references; audit and replay are required.",),
    ),
    AdminWorkflowDefinition(
        key="league_manager",
        label="League Manager",
        streamlit_page_key="league_manager",
        next_route="/admin/league-manager",
        api_scope="live_ladder_league_setup_score_movement",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER",
        status_when_disabled="not_started",
        next_action="Port after Match Log, Match Uploader, and Player Editor foundations are proven.",
        safety_notes=("Runs live events, creates matches, and performs movement workflows.",),
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
        next_action="Port after core score/correction foundation or as a tightly scoped pilot for closed-club operations.",
        safety_notes=("Challenge completion can insert matches and move ladder ranks.",),
    ),
    AdminWorkflowDefinition(
        key="tournament_admin",
        label="Tournament Admin / Ops",
        streamlit_page_key="tournaments",
        next_route="/admin/tournaments",
        api_scope="teams_draws_scores_podiums_trophies",
        access="staff",
        risk="high",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS",
        status_when_disabled="streamlit_fallback",
        next_action="Port after public registration intake is validated and score/correction foundations exist.",
        safety_notes=("Draw creation, score finalization, podiums, and trophy awards are staff-controlled.",),
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
        label="Admin Tools / Replay / Workers",
        streamlit_page_key="admin_tools",
        next_route="/admin/tools",
        api_scope="diagnostics_replay_badge_workers_backfills",
        access="super_admin",
        risk="critical",
        env_flag="JUPR_ENABLE_NEXT_ADMIN_TOOLS",
        status_when_disabled="streamlit_only",
        next_action="Keep Streamlit-only until job, replay, and super-admin authorization contracts are hardened.",
        safety_notes=("Includes replay, backfill, and worker operations that can affect large data ranges.",),
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
        "enabled_workflows": enabled_workflows,
        "recommended_sequence": [
            "admin_shell",
            "match_log",
            "score_entry",
            "match_uploader",
            "player_editor",
            "league_manager",
            "challenge_ladder_admin",
            "tournament_admin",
            "weekly_recap_admin",
            "admin_tools",
        ],
        "pilot_gates": list(PILOT_GATES),
        "permanent_guardrails": list(PERMANENT_GUARDRAILS),
        "workflows": workflows,
    }
