from __future__ import annotations

from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from jupr_app.domain.notifications.player_profile_update_repo import (
    REQUEST_STATUS_ACTIVE,
    REQUEST_STATUS_PENDING,
    create_public_request,
    get_open_or_active_subscription,
)
from jupr_app.ui.helpers import qp_get
from jupr_app.ui.components.player_picker import build_player_picker_df, render_player_picker
from jupr_app.ui.layout import page_shell


def _normalize_club_slug(value: str) -> str:
    text = str(value or "").strip().lower()
    normalized = text.replace("-", "_").replace(" ", "_")
    return "".join(ch for ch in normalized if ch.isalnum() or ch == "_")


def resolve_scoped_club_id(*, ctx_club_id: str, query_club_id: str) -> str:
    default_club = str(ctx_club_id or "").strip()
    query_club = str(query_club_id or "").strip()
    if not query_club:
        return default_club
    if _normalize_club_slug(query_club) == _normalize_club_slug(default_club):
        return query_club
    return default_club


def requested_player_id_from_query(*, player_id_q: str, pid_q: str) -> int | None:
    preferred = str(player_id_q or "").strip()
    fallback = str(pid_q or "").strip()
    if preferred.isdigit():
        return int(preferred)
    if fallback.isdigit():
        return int(fallback)
    return None


def resolve_prefill_player_id(
    *,
    options_df: pd.DataFrame,
    player_id_q: str,
    pid_q: str,
) -> int | None:
    requested = requested_player_id_from_query(player_id_q=player_id_q, pid_q=pid_q)
    if requested is None:
        return None
    valid_ids = set(options_df.get("id", pd.Series(dtype=int)).astype(int).tolist())
    if requested in valid_ids:
        return requested
    return None


def render(ctx) -> None:
    page_shell(
        "📬 Subscribe to Player Updates",
        "Choose a player profile, enter your email, and request verified update emails for that player.",
        mode_label="Public",
    )

    club_id = resolve_scoped_club_id(
        ctx_club_id=str(getattr(ctx, "club_id", "") or ""),
        query_club_id=qp_get("club_id", ""),
    )
    pid_q = qp_get("pid", "")
    player_id_q = qp_get("player_id", "")

    df_players_all = getattr(ctx, "df_players_all", None)
    df_players_active = getattr(ctx, "df_players_active", None)
    source_df = df_players_all if df_players_all is not None and not df_players_all.empty else df_players_active
    options_df = build_player_picker_df(source_df if source_df is not None else pd.DataFrame(), include_inactive=True)

    if options_df.empty:
        st.info("No players found.")
        return

    prefill_player_id = resolve_prefill_player_id(
        options_df=options_df,
        player_id_q=player_id_q,
        pid_q=pid_q,
    )
    if player_id_q and prefill_player_id is None and not str(player_id_q).strip().isdigit():
        st.caption("The provided player_id is invalid. Please choose a player below.")
    elif (player_id_q or pid_q) and prefill_player_id is None:
        st.caption("The requested player was not found. Please choose a player below.")

    selected_player_id = render_player_picker(
        source_df if source_df is not None else pd.DataFrame(),
        label="Search player",
        key="verified_updates_player_picker",
        default_player_id=prefill_player_id,
        include_inactive=True,
    )
    if selected_player_id is None:
        return

    open_or_active = get_open_or_active_subscription(
        ctx.supabase,
        str(club_id),
        int(selected_player_id),
    )
    status_now = str((open_or_active or {}).get("request_status") or "").strip().lower()
    if status_now == REQUEST_STATUS_ACTIVE:
        st.info("This player profile already has verified updates enabled.")
    elif status_now == REQUEST_STATUS_PENDING:
        st.info("A verified updates request is already pending for this player profile.")

    with st.form(f"verified_updates_public_request_{int(selected_player_id)}"):
        request_email = st.text_input("Email")
        request_note = st.text_area("Note (optional)")
        submit_request = st.form_submit_button("Submit request")

    if submit_request:
        latest = get_open_or_active_subscription(
            ctx.supabase,
            str(club_id),
            int(selected_player_id),
        )
        latest_status = str((latest or {}).get("request_status") or "").strip().lower()
        if latest_status == REQUEST_STATUS_ACTIVE:
            st.info("This player profile already has verified updates enabled.")
        elif latest_status == REQUEST_STATUS_PENDING:
            st.info("A verified updates request is already pending for this player profile.")
        else:
            try:
                create_public_request(
                    ctx.supabase,
                    club_id=str(club_id),
                    player_id=int(selected_player_id),
                    email=request_email,
                    request_note=request_note,
                )
                st.success("Success! Your verified updates request was submitted for admin review.")
                profile_url = (
                    "/?"
                    + urlencode(
                        {
                            "page": "players",
                            "pid": int(selected_player_id),
                            "player_id": int(selected_player_id),
                            "club_id": str(club_id),
                        }
                    )
                )
                st.caption(f"[View this player profile]({profile_url})")
            except Exception as exc:
                msg = str(exc or "").strip().lower()
                if "already has an active verified subscriber" in msg:
                    st.info("This player profile already has verified updates enabled.")
                elif "already pending" in msg or "already exists" in msg:
                    st.info("A verified updates request is already pending for this player profile.")
                else:
                    st.error(f"Could not submit request: {exc}")
