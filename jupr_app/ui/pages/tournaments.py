from __future__ import annotations

from datetime import date, datetime
from typing import Any

import streamlit as st

from jupr_app.domain.event_tags import derive_default_date_tags, normalize_event_tags
from jupr_app.domain.tournament_registration_repo import (
    archive_tournament,
    build_public_urls,
    delete_unused_draft_tournament,
    get_registration_settings,
    list_existing_tournaments,
    tournament_can_be_deleted,
    unarchive_tournament,
    upsert_registration_settings,
)
from jupr_app.ui.layout import page_shell

LEGACY_DEFAULT_TEAM_COUNT = 4
TOURNAMENT_STATUS_OPTIONS = ["DRAFT", "REGISTRATION", "REGISTRATION_OPEN", "REGISTRATION_CLOSED", "ARCHIVED"]
TOURNAMENT_LOCALE_OPTIONS = ["en", "es", "bilingual"]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _coerce_date(value: Any, fallback: date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return fallback


def _parse_optional_date(value: Any) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    text = _safe_text(value)
    return text or None


def _go_to_page(page: str, tournament_id: Any) -> None:
    st.query_params["page"] = page
    st.query_params["tournament_id"] = tournament_id
    st.rerun()


def _insert_tournament_shell(supabase, payload: dict) -> None:
    try:
        supabase.table("tournaments").insert(payload).execute()
        return
    except Exception:
        pass

    fallback_payload = {
        "club_id": payload["club_id"],
        "name": payload["name"],
        "status": payload.get("status") or "DRAFT",
        "team_count": int(payload.get("team_count") or LEGACY_DEFAULT_TEAM_COUNT),
    }
    supabase.table("tournaments").insert(fallback_payload).execute()


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell(
        "🏆 Tournaments",
        "Tournament library and launcher for setup, operations, live scoring, and public pages.",
        mode_label=mode_label,
    )

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    supabase = getattr(ctx, "supabase", None)
    club_id = getattr(ctx, "club_id", None)
    if supabase is None or club_id is None:
        st.error("Missing database context.")
        st.stop()

    st.subheader("Create Tournament Shell")
    c1, c2 = st.columns(2)
    with c1:
        tournament_name = st.text_input("Tournament name *", key="tourney_create_name")
        start_date = st.date_input("Start date", value=_coerce_date(st.session_state.get("tourney_create_start", date.today()), date.today()), key="tourney_create_start")
        end_date = st.date_input("End date", value=_coerce_date(st.session_state.get("tourney_create_end", date.today()), date.today()), min_value=start_date, key="tourney_create_end")
        registration_enabled = st.checkbox("Registration enabled", value=True, key="tourney_create_reg_enabled")
    with c2:
        status = st.selectbox("Status", TOURNAMENT_STATUS_OPTIONS, index=0, key="tourney_create_status")
        public_slug = st.text_input("Public slug (optional)", key="tourney_create_slug")
        locale = st.selectbox("Locale", TOURNAMENT_LOCALE_OPTIONS, index=0, key="tourney_create_locale")
        reg_open_at = st.text_input("Registration opens (optional ISO/local)", key="tourney_create_reg_open")
        reg_close_at = st.text_input("Registration closes (optional ISO/local)", key="tourney_create_reg_close")

    if st.button("Create Tournament", type="primary"):
        if not tournament_name.strip():
            st.error("Tournament name is required.")
        else:
            payload = {
                "club_id": str(club_id),
                "name": tournament_name.strip(),
                "status": status,
                "team_count": None,
                "start_date": _parse_optional_date(start_date),
                "end_date": _parse_optional_date(end_date),
                "registration_enabled": bool(registration_enabled),
                "public_slug": _safe_text(public_slug) or None,
                "locale": locale,
                "registration_open_at": _parse_optional_date(reg_open_at),
                "registration_close_at": _parse_optional_date(reg_close_at),
                "event_tags": normalize_event_tags({
                    "skill_levels": [],
                    "date_tags": derive_default_date_tags(start_date=start_date, end_date=end_date),
                }),
            }
            _insert_tournament_shell(supabase, payload)
            created = (
                supabase.table("tournaments")
                .select("id,name")
                .eq("club_id", str(club_id))
                .eq("name", tournament_name.strip())
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            created_row = (created.data or [None])[0]
            if created_row:
                try:
                    upsert_registration_settings(
                        supabase,
                        {
                            "tournament_id": created_row["id"],
                            "registration_slug": _safe_text(public_slug) or None,
                            "locale": locale,
                            "registration_status": "open" if registration_enabled else "draft",
                            "registration_open_at": _parse_optional_date(reg_open_at),
                            "registration_close_at": _parse_optional_date(reg_close_at),
                        },
                    )
                except Exception:
                    pass
            st.success("Tournament shell created.")
            st.rerun()

    st.divider()
    st.subheader("Tournament List")
    show_archived = st.checkbox("Show archived", value=False, key="tournaments_show_archived")
    tournaments = list_existing_tournaments(supabase, str(club_id), include_archived=show_archived)
    if not tournaments:
        st.info("No tournaments available for this filter.")
        st.stop()

    preselected = _safe_text(st.query_params.get("tournament_id"))
    labels = [f"{row['name']} ({_safe_text(row.get('status') or 'DRAFT')})" for row in tournaments]
    default_index = next((i for i, row in enumerate(tournaments) if str(row.get("id")) == preselected), 0)
    picked = st.selectbox("Select tournament", labels, index=default_index)
    tournament = tournaments[labels.index(picked)]
    tournament_id = tournament["id"]
    st.query_params["tournament_id"] = tournament_id

    st.subheader("Tournament Overview")
    settings = get_registration_settings(supabase, tournament_id, tournament_name=_safe_text(tournament.get("name")))
    cols = st.columns(4)
    cols[0].metric("Status", _safe_text(tournament.get("status") or "DRAFT"))
    cols[1].metric("Start", _safe_text(tournament.get("start_date") or "—"))
    cols[2].metric("End", _safe_text(tournament.get("end_date") or "—"))
    cols[3].metric("Registration status", _safe_text(settings.get("registration_status") or "draft"))

    launch = st.columns(5)
    if launch[0].button("🛠️ Open Tournament Setup"):
        _go_to_page("tournament_manager", tournament_id)
    if launch[1].button("📋 Open Tournament Operations"):
        _go_to_page("tournament_ops", tournament_id)
    if launch[2].button("🔴 Open Tournament Live"):
        _go_to_page("tournament_live", tournament_id)

    links = build_public_urls(
        base_url=_safe_text(st.session_state.get("base_url")),
        tournament_id=str(tournament_id),
        registration_slug=settings.get("registration_slug"),
    )
    launch[3].link_button("📝 Public Registration", links["registration"])
    launch[4].link_button("🤝 Partner Board", links["partner_board"])

    st.markdown("#### Admin Actions")
    action_cols = st.columns(2)
    if _safe_text(tournament.get("status")).upper() == "ARCHIVED":
        if action_cols[0].button("Unarchive Tournament", key=f"unarchive_tournament_{tournament_id}"):
            unarchive_tournament(supabase, tournament_id)
            st.success("Tournament unarchived and restored to DRAFT.")
            st.rerun()
    else:
        if action_cols[0].button("Archive Tournament", key=f"archive_tournament_{tournament_id}"):
            archive_tournament(supabase, tournament_id)
            st.success("Tournament archived.")
            st.rerun()

    can_delete, usage_summary, delete_reason = tournament_can_be_deleted(supabase, tournament)
    if can_delete:
        delete_confirm = action_cols[1].text_input("Type DELETE to confirm draft deletion", key=f"delete_draft_confirm_{tournament_id}")
        if action_cols[1].button("Delete Draft", key=f"delete_draft_tournament_{tournament_id}"):
            if delete_confirm != "DELETE":
                st.error("Type DELETE exactly to confirm draft deletion.")
            else:
                delete_unused_draft_tournament(supabase, tournament)
                st.success("Draft tournament shell deleted.")
                st.query_params.pop("tournament_id", None)
                st.rerun()
    else:
        usage_bits = [f"{k}: {usage_summary.get(k, 0)}" for k in ["registrations", "registration_selections", "event_draws", "teams", "games", "podium"]]
        action_cols[1].warning(delete_reason or "This tournament has existing records. Archive it instead of deleting.")
        action_cols[1].caption("Usage summary — " + ", ".join(usage_bits))
