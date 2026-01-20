# jupr_app/ui/pages/challenge_ladder_admin.py
from __future__ import annotations

import time
from datetime import timedelta
from typing import Any, Callable, Dict, Tuple

import pandas as pd
import streamlit as st
from jupr_app.domain.tier_movement import compute_out_of_tier_streak

from jupr_app.domain.challenge_ladder import (
    TIER_ORDER,
    tier_title,
    tier_for_jupr,
    ladder_nm,
    ladder_parse_dt,
    dt_utc_now,
    month_key_utc,
    ladder_bucket_challenge,
    ladder_compute_status_map,
)

# If you moved this into domain already, import it from there instead.
# You showed build_challenge_notice_message inside your old module code.
# Put it in jupr_app/domain/challenge_messages.py or keep it in domain/challenge_ladder.py and import here.
from jupr_app.domain.build_challenge_notice_message import build_challenge_notice_message

  # adjust if needed


# -------------------------
# Supabase retry (page-local)
# -------------------------
def sb_retry(fn: Callable[[], Any], retries: int = 4, base_sleep: float = 0.35):
    last = None
    for i in range(max(1, int(retries))):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base_sleep * (2**i))
    raise last  # type: ignore[misc]


def _df(resp) -> pd.DataFrame:
    try:
        return pd.DataFrame(getattr(resp, "data", None) or [])
    except Exception:
        return pd.DataFrame()


def _secret_val(key: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(key, default) or default)
    except Exception:
        return default


# -------------------------
# Data loaders (your real schema)
# -------------------------
def ladder_fetch_settings(supabase, club_id: str) -> Dict[str, Any]:
    resp = sb_retry(lambda: (
        supabase.table("ladder_settings")
        .select("*")
        .eq("club_id", club_id)
        .limit(1)
        .execute()
    ))
    if getattr(resp, "data", None):
        return resp.data[0]

    sb_retry(lambda: supabase.table("ladder_settings").insert({"club_id": club_id}).execute())

    resp2 = sb_retry(lambda: (
        supabase.table("ladder_settings")
        .select("*")
        .eq("club_id", club_id)
        .limit(1)
        .execute()
    ))
    if getattr(resp2, "data", None):
        return resp2.data[0]

    return {
        "challenge_range": 3,
        "accept_window_hours": 48,
        "play_window_days": 7,
        "cooldown_hours": 72,
        "protected_hours": 72,
        "pass_hold_hours": 72,
    }


def ladder_load_core(supabase, club_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roster = sb_retry(lambda: (
        supabase.table("ladder_roster")
        .select("id,club_id,player_id,tier_id,rank,is_active,joined_at,left_at,notes,updated_at")
        .eq("club_id", club_id)
        .order("tier_id", desc=False)
        .order("rank", desc=False)
        .execute()
    ))
    df_roster = _df(roster)

    flags = sb_retry(lambda: (
        supabase.table("ladder_player_flags")
        .select("club_id,player_id,vacation_until,reinstate_required,reinstate_notes,updated_at")
        .eq("club_id", club_id)
        .execute()
    ))
    df_flags = _df(flags)

    ch = sb_retry(lambda: (
        supabase.table("ladder_challenges")
        .select("*")
        .eq("club_id", club_id)
        .order("created_at", desc=True)
        .limit(5000)
        .execute()
    ))
    df_ch = _df(ch)

    pu = sb_retry(lambda: (
        supabase.table("ladder_pass_usage")
        .select("*")
        .eq("club_id", club_id)
        .order("used_at", desc=True)
        .limit(2000)
        .execute()
    ))
    df_pass = _df(pu)

    return df_roster, df_flags, df_ch, df_pass


def ladder_audit(supabase, club_id: str, action_type: str, entity_type: str, entity_id: str, before: dict | None, after: dict | None):
    actor = "admin" if st.session_state.get("admin_logged_in", False) else "system"
    payload = {
        "club_id": club_id,
        "actor": actor,
        "action_type": str(action_type),
        "entity_type": str(entity_type),
        "entity_id": str(entity_id),
        "before": before,
        "after": after,
    }
    try:
        sb_retry(lambda: supabase.table("ladder_audit_log").insert(payload).execute())
    except Exception:
        pass


# -------------------------
# Page render
# -------------------------
def render(ctx):
    st.header("🛠️ Challenge Ladder Admin")

    if bool(ctx.public_mode):
        st.error("Admin page is not available in public mode.")
        st.stop()

    if not bool(ctx.admin_logged_in):
        st.error("Admin login required.")
        st.stop()

    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    id_to_name = ctx.id_to_name
    df_players_all = ctx.df_players_all

    settings = ladder_fetch_settings(supabase, club_id)
    df_roster, df_flags, df_ch, df_pass = ladder_load_core(supabase, club_id)

    tabs = st.tabs(["📊 Dashboard", "🧾 Intake", "🗂 Challenge Detail", "👥 Roster", "⬆️⬇️ Tier Movement", "🏖 Overrides", "📜 Audit"])


    # -------------------------
    # TAB 1: DASHBOARD
    # -------------------------
    with tabs[0]:
        st.subheader("📊 Ops Dashboard")

        if df_ch is None or df_ch.empty:
            st.info("No challenges yet.")
        else:
            df = df_ch.copy()
            df["bucket"] = df.apply(lambda r: ladder_bucket_challenge(r.to_dict()), axis=1)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pending Acceptance", int((df["bucket"] == "Pending Acceptance").sum()))
            c2.metric("Acceptance Overdue", int((df["bucket"] == "Acceptance Overdue").sum()))
            c3.metric("Play Overdue", int((df["bucket"] == "Play Overdue").sum()))
            c4.metric("Accepted / In Window", int((df["bucket"] == "Accepted / In Window").sum()))

            st.divider()

            needs = df[df["bucket"].isin(["Acceptance Overdue", "Play Overdue"])].copy()
            if needs.empty:
                st.success("No overdue items.")
            else:
                needs["challenger"] = needs["challenger_id"].apply(lambda x: ladder_nm(int(x), id_to_name))
                needs["defender"] = needs["defender_id"].apply(lambda x: ladder_nm(int(x), id_to_name))
                show_cols = [c for c in ["id", "bucket", "status", "challenger", "defender", "accept_by", "play_by", "created_at"] if c in needs.columns]
                st.dataframe(needs[show_cols], use_container_width=True, hide_index=True)

    # -------------------------
    # TAB 2: INTAKE
    # -------------------------
    with tabs[1]:
        st.subheader("🧾 Enter Challenge (from Pro Shop Ledger)")

        tier_pick = st.selectbox("Tier", TIER_ORDER, format_func=tier_title, key="ladder_intake_tier")

        if df_roster is None or df_roster.empty:
            st.error("Roster not initialized yet. Add players in the Roster tab.")
        else:
            roster_active = df_roster[(df_roster["is_active"] == True) & (df_roster["tier_id"].astype(str) == str(tier_pick))].copy()
            roster_active["name"] = roster_active["player_id"].apply(lambda x: ladder_nm(int(x), id_to_name))
            roster_active = roster_active.sort_values("rank")

            name_to_pid = dict(zip(roster_active["name"], roster_active["player_id"]))
            pid_to_rank = dict(zip(roster_active["player_id"].astype(int), roster_active["rank"].astype(int)))

            status_map = ladder_compute_status_map(df_roster, df_flags, df_ch, df_pass, settings, id_to_name)

            with st.form("ladder_intake_form"):
                challenger_name = st.selectbox("Challenger", [""] + roster_active["name"].tolist())
                defender_name = st.selectbox("Defender", [""] + roster_active["name"].tolist())

                challenger_contact = st.text_input(
                    "Challenger contact (optional — email/text/WhatsApp). Leave blank to fill later.",
                    value="",
                )

                ledger_ref = st.text_input("Ledger reference / notes (optional)", value="")
                override = st.checkbox("Admin override (bypass eligibility rules)", value=False)
                submitted = st.form_submit_button("Create Challenge")

            if submitted:
                if not challenger_name or not defender_name:
                    st.error("Select both Challenger and Defender.")
                    st.stop()

                chal_id = int(name_to_pid[challenger_name])
                def_id = int(name_to_pid[defender_name])

                if chal_id == def_id:
                    st.error("Challenger and Defender must be different.")
                    st.stop()

                chal_rank = int(pid_to_rank.get(chal_id, 999999))
                def_rank = int(pid_to_rank.get(def_id, 999999))
                challenge_range = int(settings.get("challenge_range", 3) or 3)

                errors = []

                # Defender must be ranked above challenger (smaller rank number)
                if def_rank >= chal_rank:
                    errors.append("Defender must be ranked ABOVE Challenger.")

                if (chal_rank - def_rank) > challenge_range:
                    errors.append(f"Rank gap too large. Allowed: {challenge_range}. Gap: {chal_rank - def_rank}.")

                chal_status = status_map.get(chal_id, {}).get("status", "Ready to Defend")
                def_status = status_map.get(def_id, {}).get("status", "Ready to Defend")

                if chal_status not in ("Ready to Defend",) and not override:
                    errors.append(f"Challenger is not eligible to initiate (status: {chal_status}).")
                if def_status not in ("Ready to Defend", "Cooldown") and not override:
                    errors.append(f"Defender is not eligible to be challenged (status: {def_status}).")

                if errors and not override:
                    st.error("Cannot create challenge:\n\n- " + "\n- ".join(errors))
                    st.stop()

                payload = {
                    "club_id": club_id,
                    "challenger_id": chal_id,
                    "defender_id": def_id,
                    "challenger_rank_at_create": chal_rank,
                    "defender_rank_at_create": def_rank,
                    "status": "PENDING_ACCEPTANCE",
                    "created_by": "admin",
                    "ledger_ref": ledger_ref.strip() or None,
                    # IMPORTANT: do NOT start clock here
                    "accept_by": None,
                    "tier_id": str(tier_pick),
                }

                res = sb_retry(lambda: supabase.table("ladder_challenges").insert(payload).execute())
                new_id = int(res.data[0]["id"]) if getattr(res, "data", None) else None

                ladder_audit(supabase, club_id, "challenge_create", "ladder_challenges", str(new_id or ""), None, payload)

                admin_name = _secret_val("LADDER_ADMIN_NAME", "Ladder Admin")
                admin_contact = _secret_val("LADDER_ADMIN_CONTACT", "")

                notice = build_challenge_notice_message(
                    challenge_id=new_id,
                    tier_id=str(tier_pick),
                    challenger_name=str(challenger_name),
                    defender_name=str(defender_name),
                    challenger_contact=str(challenger_contact),
                    admin_name=admin_name,
                    admin_contact=admin_contact,
                    ledger_ref=ledger_ref.strip() if ledger_ref else None,
                )

                st.session_state["last_notice_challenge_id"] = new_id
                st.session_state["last_notice_email_full"] = notice["email_full"]
                st.session_state["last_notice_sms"] = notice["sms"]

                st.success(f"Challenge created. ID = {new_id}")
                st.rerun()

        # Post-create notice tools (outside form; safe even if no new challenge)
        if st.session_state.get("last_notice_challenge_id"):
            ch_id = int(st.session_state["last_notice_challenge_id"])

            st.divider()
            st.subheader("📩 Copy/Paste Notice Message")

            st.text_area("Email (copy/paste)", value=str(st.session_state.get("last_notice_email_full", "") or ""), height=260)
            st.text_area("Text/SMS (copy/paste)", value=str(st.session_state.get("last_notice_sms", "") or ""), height=120)

            st.caption(
                "No timestamps are included above. The 48-hour response window is based on the timestamp on your sent message. "
                "After you SEND the notice, click below to start the 48-hour timer inside the app."
            )

            if st.button("✅ Start 48-hour clock (set Accept By)", key=f"start_clock_{ch_id}"):
                now = dt_utc_now()
                accept_h = int(settings.get("accept_window_hours", 48) or 48)
                accept_by = now + timedelta(hours=accept_h)

                sb_retry(lambda: (
                    supabase.table("ladder_challenges")
                    .update({"accept_by": accept_by.isoformat()})
                    .eq("club_id", club_id)
                    .eq("id", ch_id)
                    .execute()
                ))

                st.success("48-hour clock started in the app.")
                time.sleep(0.2)
                st.rerun()

    # -------------------------
    # TAB 3: CHALLENGE DETAIL
    # -------------------------
    with tabs[2]:
        st.subheader("🗂 Challenge Detail")

        if df_ch is None or df_ch.empty:
            st.info("No challenges yet. Create one in Intake.")
        else:
            # rest of Challenge Detail tab...


            df = df_ch.copy()
            if "id" not in df.columns or "challenger_id" not in df.columns or "defender_id" not in df.columns:
                st.error("ladder_challenges table is missing required columns.")
                st.stop()

            df["label"] = df.apply(
                lambda r: f"#{int(r['id'])} • {ladder_nm(int(r['challenger_id']), id_to_name)} vs {ladder_nm(int(r['defender_id']), id_to_name)} • {r.get('status','')}",
                axis=1,
            )

            pick = st.selectbox("Select challenge", df["label"].tolist(), index=0, key="ladder_admin_pick_challenge")
            hit = df[df["label"] == pick]
            if hit.empty:
                st.warning("Selected challenge not found (refresh and try again).")
                st.stop()

            ch_row = hit.iloc[0].to_dict()
            ch_id = int(ch_row["id"])
            chal_id = int(ch_row["challenger_id"])
            def_id = int(ch_row["defender_id"])

            st.write(f"**Challenge #{ch_id}**")
            st.write(f"- Challenger: **{ladder_nm(chal_id, id_to_name)}** (rank at create: {ch_row.get('challenger_rank_at_create')})")
            st.write(f"- Defender: **{ladder_nm(def_id, id_to_name)}** (rank at create: {ch_row.get('defender_rank_at_create')})")
            st.write(f"- Status: **{ch_row.get('status')}**")
            st.write(f"- Accept by: {ch_row.get('accept_by')}")
            st.write(f"- Play by: {ch_row.get('play_by')}")

            st.divider()

            c1, c2, c3, c4 = st.columns(4)

            if c1.button("✅ Mark Accepted", disabled=(str(ch_row.get("status")) != "PENDING_ACCEPTANCE"), key=f"accept_{ch_id}"):
                before = ch_row.copy()
                now = dt_utc_now()
                play_by = now + timedelta(days=int(settings.get("play_window_days", 7) or 7))
                upd = {"accepted_at": now.isoformat(), "play_by": play_by.isoformat(), "status": "ACCEPTED_SCHEDULING"}
                sb_retry(lambda: supabase.table("ladder_challenges").update(upd).eq("club_id", club_id).eq("id", ch_id).execute())
                ladder_audit(supabase, club_id, "challenge_accept", "ladder_challenges", str(ch_id), before, {**before, **upd})
                st.success("Accepted.")
                st.rerun()

            if c2.button("🗑 Cancel (Admin)", type="secondary", key=f"cancel_{ch_id}"):
                before = ch_row.copy()
                upd = {"status": "CANCELED", "resolution_notes": "Admin canceled", "completed_at": dt_utc_now().isoformat()}
                sb_retry(lambda: supabase.table("ladder_challenges").update(upd).eq("club_id", club_id).eq("id", ch_id).execute())
                ladder_audit(supabase, club_id, "challenge_cancel", "ladder_challenges", str(ch_id), before, {**before, **upd})
                st.success("Canceled.")
                st.rerun()

            with c3:
                forfeit_by = st.selectbox("Forfeit by", ["", ladder_nm(chal_id, id_to_name), ladder_nm(def_id, id_to_name)], key=f"ff_by_{ch_id}")
            if c3.button("🏳️ Record Forfeit", disabled=(forfeit_by == ""), key=f"ff_btn_{ch_id}"):
                before = ch_row.copy()
                fb = chal_id if forfeit_by == ladder_nm(chal_id, id_to_name) else def_id
                winner = def_id if fb == chal_id else chal_id
                upd = {
                    "status": "FORFEITED",
                    "forfeit_by": int(fb),
                    "forfeit_reason": "Forfeit (admin entry)",
                    "winner_id": int(winner),
                    "completed_at": dt_utc_now().isoformat(),
                }
                sb_retry(lambda: supabase.table("ladder_challenges").update(upd).eq("club_id", club_id).eq("id", ch_id).execute())

                # Optional rank swap RPC (only if you have it deployed)
                if int(winner) == chal_id:
                    try:
                        sb_retry(lambda: supabase.rpc("ladder_swap_ranks", {"p_club_id": club_id, "p_player_a": chal_id, "p_player_b": def_id}).execute())
                    except Exception:
                        st.warning("Rank swap RPC (ladder_swap_ranks) not available; forfeit recorded without swapping.")

                ladder_audit(supabase, club_id, "challenge_forfeit", "ladder_challenges", str(ch_id), before, {**before, **upd})
                st.success("Forfeit recorded.")
                st.rerun()

            with c4:
                pass_user = st.selectbox("Pass used by", ["", ladder_nm(chal_id, id_to_name), ladder_nm(def_id, id_to_name)], key=f"pass_by_{ch_id}")
            if c4.button("🎟 Record Pass Used", disabled=(pass_user == ""), key=f"pass_btn_{ch_id}"):
                before = ch_row.copy()
                pu_pid = chal_id if pass_user == ladder_nm(chal_id, id_to_name) else def_id
                now = dt_utc_now()
                mk = month_key_utc(now)

                sb_retry(lambda: supabase.table("ladder_pass_usage").insert({
                    "club_id": club_id,
                    "player_id": int(pu_pid),
                    "month_key": mk,
                    "used_at": now.isoformat(),
                    "challenge_id": ch_id,
                }).execute())

                upd = {
                    "status": "CANCELED",
                    "pass_used_by": int(pu_pid),
                    "pass_used_at": now.isoformat(),
                    "resolution_notes": "Pass used",
                    "completed_at": now.isoformat(),
                }
                sb_retry(lambda: supabase.table("ladder_challenges").update(upd).eq("club_id", club_id).eq("id", ch_id).execute())
                ladder_audit(supabase, club_id, "challenge_pass_used", "ladder_challenges", str(ch_id), before, {**before, **upd})
                st.success("Pass recorded (challenge closed).")
                st.rerun()

    # -------------------------
    # TAB: ROSTER (full tools)
    # -------------------------
    with tabs[3]:
        st.subheader("👥 Ladder Roster")

        if df_roster is None or df_roster.empty:
            st.info("No roster yet.")
            st.stop()

        supabase = ctx.supabase
        club_id = str(ctx.club_id)
        id_to_name = ctx.id_to_name
        name_to_id = ctx.name_to_id

        # Tier context
        tier_ctx = st.selectbox(
            "Tier to manage",
            TIER_ORDER,
            format_func=tier_title,
            key="ladder_roster_tier_ctx",
        )

        # Prepare roster DF
        r0 = df_roster.copy()
        if "player_id" in r0.columns:
            r0["player_id"] = r0["player_id"].astype(int)
        if "tier_id" in r0.columns:
            r0["tier_id"] = r0["tier_id"].astype(str)
        if "is_active" in r0.columns:
            # Supabase booleans come back as bool; keep as-is
            pass

        r0["name"] = r0["player_id"].apply(lambda x: ladder_nm(int(x), id_to_name))

        # -------------------------
        # Add ONE player (append to bottom of tier)
        # -------------------------
        st.markdown("#### ➕ Add one player (appends to bottom)")
        st.caption(
            "Adds an existing player to the bottom of the selected tier. "
            "If the player is already on the ladder but inactive, they will be reactivated and appended."
        )

        all_names = sorted(list(name_to_id.keys())) if isinstance(name_to_id, dict) else []

        with st.form("ladder_add_one_form"):
            existing_pick = st.selectbox("Pick an existing player", [""] + all_names, index=0, key="ladder_add_one_pick")
            typed_name = st.text_input("Or type the player name exactly", value="", key="ladder_add_one_typed")

            auto_assign = st.checkbox(
                "Auto-assign tier from OVERALL JUPR (ignores tier selection)",
                value=False,
                key="ladder_add_one_auto_tier",
            )

            add_one = st.form_submit_button("Add to bottom")

        if add_one:
            nm = (typed_name.strip() or existing_pick.strip())
            if not nm:
                st.error("Pick a player OR type a name.")
                st.stop()

            if nm not in name_to_id:
                st.error(f"'{nm}' is not in your Players database yet. Create the player first, then add to roster.")
                st.stop()

            pid = int(name_to_id[nm])

            # Determine tier for player
            tier_for_player = str(tier_ctx)
            if bool(st.session_state.get("ladder_add_one_auto_tier", False)):
                # Use ctx.df_players_all rating to determine tier
                dfp = ctx.df_players_all
                if dfp is None or dfp.empty or "id" not in dfp.columns:
                    st.error("Players table not loaded; cannot auto-assign tier.")
                    st.stop()

                hit = dfp[dfp["id"].astype(int) == pid]
                if hit.empty:
                    st.error("Could not find player row for auto-tier assignment.")
                    st.stop()

                elo = float(hit.iloc[0].get("rating", 1200.0) or 1200.0)
                tier_for_player = tier_for_jupr(elo / 400.0)

            now_iso = dt_utc_now().isoformat()

            # Next rank within that tier
            max_rank_resp = sb_retry(lambda: (
                supabase.table("ladder_roster")
                .select("rank")
                .eq("club_id", club_id)
                .eq("tier_id", tier_for_player)
                .eq("is_active", True)
                .order("rank", desc=True)
                .limit(1)
                .execute()
            ))
            next_rank = (int(max_rank_resp.data[0]["rank"]) + 1) if getattr(max_rank_resp, "data", None) else 1

            # Existing roster row?
            ex = sb_retry(lambda: (
                supabase.table("ladder_roster")
                .select("id,is_active,rank,tier_id,notes,joined_at,left_at")
                .eq("club_id", club_id)
                .eq("player_id", pid)
                .limit(1)
                .execute()
            ))
            ex_df = pd.DataFrame(getattr(ex, "data", None) or [])
            before = ex_df.iloc[0].to_dict() if not ex_df.empty else None

            if before and bool(before.get("is_active", True)):
                st.info(f"'{nm}' is already ACTIVE on the ladder (tier {before.get('tier_id')}, rank {before.get('rank')}).")
                st.stop()

            upd = {
                "is_active": True,
                "tier_id": tier_for_player,
                "rank": int(next_rank),
                "left_at": None,
                "joined_at": now_iso,
            }

            if before:
                sb_retry(lambda: (
                    supabase.table("ladder_roster")
                    .update(upd)
                    .eq("club_id", club_id)
                    .eq("player_id", pid)
                    .execute()
                ))
                ladder_audit(supabase, club_id, "roster_reactivate_append", "ladder_roster", f"{club_id}:{pid}", before, {**before, **upd})
                st.success(f"Reactivated '{nm}' into {tier_title(tier_for_player)} at rank {next_rank}.")
            else:
                ins = {
                    "club_id": club_id,
                    "player_id": pid,
                    "tier_id": tier_for_player,
                    "rank": int(next_rank),
                    "is_active": True,
                    "joined_at": now_iso,
                    "left_at": None,
                }
                sb_retry(lambda: supabase.table("ladder_roster").insert(ins).execute())
                ladder_audit(supabase, club_id, "roster_append", "ladder_roster", f"{club_id}:{pid}", None, ins)
                st.success(f"Added '{nm}' into {tier_title(tier_for_player)} at rank {next_rank}.")

            st.rerun()

        st.divider()

        # -------------------------
        # Move ACTIVE player to a different tier (admin)
        # -------------------------
        st.markdown("#### 🔁 Move active player to a different tier")
        st.caption(
            "Moves an ACTIVE ladder player to a new tier and appends them to the bottom of that tier. "
            "Optionally re-compresses ranks in the tier they leave to avoid gaps."
        )

        active_all = r0[r0["is_active"] == True].copy()
        active_all = active_all.sort_values(["tier_id", "rank"])

        # Build label -> pid map for active roster
        active_all["label"] = active_all.apply(
            lambda rr: f"{rr['name']}  •  {tier_title(str(rr['tier_id']))} (rank {int(rr['rank'])})",
            axis=1,
        )
        label_to_pid = dict(zip(active_all["label"], active_all["player_id"]))

        with st.form("ladder_move_tier_form"):
            pick_label = st.selectbox("Pick an ACTIVE player", [""] + active_all["label"].tolist(), index=0)
            dest_tier = st.selectbox("Destination tier", TIER_ORDER, format_func=tier_title, key="ladder_move_dest_tier")
            recompress_old = st.checkbox("Re-compress ranks in the tier they leave (recommended)", value=True)
            move_notes = st.text_input("Optional notes (audit only)", value="")
            do_move = st.form_submit_button("Move player")

        if do_move:
            if not pick_label:
                st.error("Pick an active player.")
                st.stop()

            pid = int(label_to_pid[pick_label])

            # Pull current row (authoritative from DB)
            ex = sb_retry(lambda: (
                supabase.table("ladder_roster")
                .select("id,club_id,player_id,tier_id,rank,is_active,joined_at,left_at,notes,updated_at")
                .eq("club_id", club_id)
                .eq("player_id", pid)
                .limit(1)
                .execute()
            ))
            ex_df = pd.DataFrame(getattr(ex, "data", None) or [])
            if ex_df.empty:
                st.error("Could not load roster row for this player.")
                st.stop()

            before = ex_df.iloc[0].to_dict()
            if not bool(before.get("is_active", False)):
                st.error("That player is not active on the ladder right now.")
                st.stop()

            cur_tier = str(before.get("tier_id"))
            cur_rank = int(before.get("rank") or 999999)
            dest_tier = str(dest_tier)

            if dest_tier == cur_tier:
                st.info("Destination tier is the same as current tier. No move performed.")
                st.stop()

            now_iso = dt_utc_now().isoformat()

            # Compute next rank in destination tier
            max_rank_resp = sb_retry(lambda: (
                supabase.table("ladder_roster")
                .select("rank")
                .eq("club_id", club_id)
                .eq("tier_id", dest_tier)
                .eq("is_active", True)
                .order("rank", desc=True)
                .limit(1)
                .execute()
            ))
            next_rank = (int(max_rank_resp.data[0]["rank"]) + 1) if getattr(max_rank_resp, "data", None) else 1

            # Update player's roster row (move + append)
            upd = {
                "tier_id": dest_tier,
                "rank": int(next_rank),
                "updated_at": now_iso,
            }

            sb_retry(lambda: (
                supabase.table("ladder_roster")
                .update(upd)
                .eq("club_id", club_id)
                .eq("player_id", pid)
                .execute()
            ))

            ladder_audit(
                supabase,
                club_id,
                "roster_move_tier",
                "ladder_roster",
                f"{club_id}:{pid}",
                before,
                {**before, **upd, "admin_notes": (move_notes.strip() or None)},
            )

            # Optional: recompress ranks in the tier they left (close gaps)
            if recompress_old:
                # Fetch active players in old tier, excluding moved pid, ordered by rank
                old_resp = sb_retry(lambda: (
                    supabase.table("ladder_roster")
                    .select("player_id,rank")
                    .eq("club_id", club_id)
                    .eq("tier_id", cur_tier)
                    .eq("is_active", True)
                    .order("rank", desc=False)
                    .execute()
                ))
                old_df = pd.DataFrame(getattr(old_resp, "data", None) or [])
                if not old_df.empty:
                    old_df["player_id"] = old_df["player_id"].astype(int)
                    old_df = old_df[old_df["player_id"] != pid].copy()
                    old_df = old_df.sort_values("rank")

                    # Re-rank 1..N
                    for i, rr in enumerate(old_df.itertuples(index=False), start=1):
                        p2 = int(rr.player_id)
                        if int(rr.rank) != i:
                            sb_retry(lambda p2=p2, i=i: (
                                supabase.table("ladder_roster")
                                .update({"rank": int(i), "updated_at": now_iso})
                                .eq("club_id", club_id)
                                .eq("player_id", p2)
                                .execute()
                            ))

            st.success(
                f"Moved {ladder_nm(pid, ctx.id_to_name)} from {tier_title(cur_tier)} (rank {cur_rank}) "
                f"to {tier_title(dest_tier)} (rank {next_rank})."
            )
            st.rerun()


        # -------------------------
        # Replace tier roster (paste ranked list)
        # -------------------------
        st.markdown("#### Initialize / Replace Tier Roster (paste ranked list)")
        st.caption("Paste names top-to-bottom. This will REPLACE the selected tier roster only (history preserved as inactive).")

        raw = st.text_area("Ranked roster (top to bottom)", height=160, key="ladder_init_raw")

        if st.button("🚀 Replace Tier Roster", type="primary", key="ladder_replace_tier_btn"):
            names = [x.strip() for x in (raw or "").split("\n") if x.strip()]
            if not names:
                st.error("Paste at least one name.")
                st.stop()

            missing = [nm for nm in names if nm not in name_to_id]
            if missing:
                st.error("These names are not in your Players database yet. Create them first:\n\n- " + "\n- ".join(missing))
                st.stop()

            # Soft-clear ONLY this tier (keeps history)
            now_iso = dt_utc_now().isoformat()
            sb_retry(lambda: (
                supabase.table("ladder_roster")
                .update({"is_active": False, "left_at": now_iso})
                .eq("club_id", club_id)
                .eq("tier_id", str(tier_ctx))
                .execute()
            ))

            rows = []
            for i, nm in enumerate(names, start=1):
                pid = int(name_to_id[nm])
                rows.append({
                    "club_id": club_id,
                    "player_id": pid,
                    "tier_id": str(tier_ctx),
                    "rank": int(i),
                    "is_active": True,
                    "joined_at": now_iso,
                    "left_at": None,
                })

            # Upsert by (club_id, player_id) so existing rows are reactivated/re-ranked
            sb_retry(lambda: supabase.table("ladder_roster").upsert(rows, on_conflict="club_id,player_id").execute())
            ladder_audit(supabase, club_id, "roster_replace_tier", "ladder_roster", f"{club_id}:{tier_ctx}", None, {"tier": str(tier_ctx), "count": len(rows)})

            st.success(f"Tier roster replaced for {tier_title(tier_ctx)}.")
            st.rerun()

        st.divider()

        # -------------------------
        # Display tier roster tables
        # -------------------------
        st.markdown(f"### Active roster — {tier_title(tier_ctx)}")
        active_df = r0[(r0["is_active"] == True) & (r0["tier_id"] == str(tier_ctx))].copy().sort_values("rank")
        show_cols = [c for c in ["rank", "name", "player_id", "notes"] if c in active_df.columns]
        st.dataframe(active_df[show_cols], use_container_width=True, hide_index=True)

        st.markdown(f"### Inactive roster — {tier_title(tier_ctx)}")
        inactive_df = r0[(r0["is_active"] == False) & (r0["tier_id"] == str(tier_ctx))].copy()
        if "rank" in inactive_df.columns:
            inactive_df = inactive_df.sort_values("rank")
        show_cols2 = [c for c in ["rank", "name", "player_id", "left_at", "notes"] if c in inactive_df.columns]
        st.dataframe(inactive_df[show_cols2], use_container_width=True, hide_index=True)


    # -------------------------
    # TAB: TIER MOVEMENT
    # -------------------------
    with tabs[4]:
        st.subheader("⬆️⬇️ Tier Movement (Admin Review Queue)")
        st.caption("Triggers when a player has 10 consecutive rated matches where their post-match tier differs from their assigned tier.")

        if df_roster is None or df_roster.empty:
            st.info("Roster required.")
        elif ctx.df_matches is None or ctx.df_matches.empty:
            st.info("No matches loaded; cannot evaluate tier movement.")
        else:
            # Flags lookup (kept for future use; not required for evaluation)
            flags_df = df_flags.copy() if df_flags is not None else pd.DataFrame()
            if flags_df.empty:
                flags_df = pd.DataFrame(columns=["player_id"])

            # Active roster only
            r = df_roster[df_roster.get("is_active", True) == True].copy()
            if r.empty:
                st.info("No active ladder players.")
            else:
                # Normalize types
                r["player_id"] = r["player_id"].astype(int)
                r["tier_id"] = r["tier_id"].astype(str)

                # joined_at parsing for streak filtering
                if "joined_at" in r.columns:
                    r["joined_at_dt"] = pd.to_datetime(r["joined_at"], utc=True, errors="coerce")
                else:
                    r["joined_at_dt"] = pd.NaT

                # Evaluate streaks
                rows = []
                for _, rr in r.iterrows():
                    pid = int(rr["player_id"])
                    cur_tier = str(rr["tier_id"])
                    joined_dt = rr["joined_at_dt"].to_pydatetime() if pd.notna(rr["joined_at_dt"]) else None

                    streak = compute_out_of_tier_streak(
                        pid=pid,
                        joined_at_utc=joined_dt,
                        current_tier_id=cur_tier,
                        df_matches=ctx.df_matches,
                    )

                    dest = streak.get("dest_tier")
                    count = int(streak.get("count", 0) or 0)
                    latest = streak.get("latest_match_at")

                    if dest and dest != cur_tier and count >= 10:
                        rows.append({
                            "player_id": pid,
                            "name": ladder_nm(pid, ctx.id_to_name),
                            "current_tier": cur_tier,
                            "dest_tier": dest,
                            "count": count,
                            "latest_match_at": latest,
                        })

                if not rows:
                    st.success("No tier-move triggers right now.")
                else:
                    qdf = pd.DataFrame(rows).sort_values(["count", "latest_match_at"], ascending=[False, False])
                    st.dataframe(qdf, use_container_width=True, hide_index=True)

                    st.divider()
                    st.markdown("### Approve a move")

                    pick_pid = st.selectbox(
                        "Player",
                        options=qdf["player_id"].tolist(),
                        format_func=lambda x: f"{ladder_nm(int(x), ctx.id_to_name)} (#{int(x)})",
                        key="tier_move_pick_pid",
                    )

                    pick_row = qdf[qdf["player_id"] == int(pick_pid)].iloc[0].to_dict()
                    dest_tier = str(pick_row["dest_tier"])
                    cur_tier = str(pick_row["current_tier"])

                    st.write(f"Current tier: **{tier_title(cur_tier)}**")
                    st.write(f"Proposed tier: **{tier_title(dest_tier)}**")

                    if st.button("✅ Approve move (append to bottom of destination tier)", key="approve_tier_move_btn"):
                        pid = int(pick_pid)
                        now_iso = dt_utc_now().isoformat()

                        # Compute next rank in destination tier
                        max_rank_resp = sb_retry(lambda: (
                            supabase.table("ladder_roster")
                            .select("rank")
                            .eq("club_id", club_id)
                            .eq("tier_id", dest_tier)
                            .eq("is_active", True)
                            .order("rank", desc=True)
                            .limit(1)
                            .execute()
                        ))
                        next_rank = (int(max_rank_resp.data[0]["rank"]) + 1) if getattr(max_rank_resp, "data", None) else 1

                        # Update roster row
                        before_row = r[r["player_id"] == pid].iloc[0].to_dict()
                        upd = {
                            "tier_id": dest_tier,
                            "rank": int(next_rank),
                            "updated_at": now_iso,
                        }
                        sb_retry(lambda: (
                            supabase.table("ladder_roster")
                            .update(upd)
                            .eq("club_id", club_id)
                            .eq("player_id", pid)
                            .execute()
                        ))
                        ladder_audit(supabase, club_id, "tier_move_approve", "ladder_roster", f"{club_id}:{pid}", before_row, {**before_row, **upd})

                        # Clear tier-move flags (non-blocking)
                        flag_upd = {
                            "tier_move_flag": False,
                            "tier_move_dest_tier": None,
                            "tier_move_count": 0,
                            "tier_move_triggered_at": now_iso,
                            "tier_move_last_eval_at": now_iso,
                        }
                        try:
                            sb_retry(lambda: (
                                supabase.table("ladder_player_flags")
                                .upsert({"club_id": club_id, "player_id": pid, **flag_upd}, on_conflict="club_id,player_id")
                                .execute()
                            ))
                        except Exception:
                            pass

                        st.success(f"Moved {ladder_nm(pid, ctx.id_to_name)} to {tier_title(dest_tier)} at rank {next_rank}.")
                        st.rerun()


    # -------------------------
    # TAB 5: OVERRIDES
    # -------------------------
    with tabs[5]:
        st.subheader("🏖 Vacation / Reinstate Overrides")

        # Use active roster for selection
        if df_roster is None or df_roster.empty:
            st.info("Roster required.")
            st.stop()

        roster_active = df_roster[df_roster.get("is_active", True) == True].copy()
        if roster_active.empty:
            st.info("No active ladder players.")
            st.stop()

        roster_active["player_id"] = roster_active["player_id"].apply(lambda x: int(float(x)) if x is not None else -1)
        roster_active = roster_active[roster_active["player_id"] > 0].copy()

        pid = st.selectbox(
            "Player",
            options=sorted(roster_active["player_id"].unique().tolist()),
            format_func=lambda x: ladder_nm(int(x), id_to_name),
            key="ladder_override_pid",
        )
        pid = int(pid)

        cur = None
        if df_flags is not None and not df_flags.empty and "player_id" in df_flags.columns:
            hit = df_flags[df_flags["player_id"].astype(int) == pid]
            if not hit.empty:
                cur = hit.iloc[0].to_dict()

        vac_default = cur.get("vacation_until") if cur else None
        rein_default = bool(cur.get("reinstate_required", False)) if cur else False
        notes_default = str(cur.get("reinstate_notes", "") or "") if cur else ""

        vac = st.text_input("Vacation until (ISO, UTC) — leave blank to clear", value=str(vac_default or ""))
        rein = st.checkbox("Reinstate Required", value=rein_default)
        notes = st.text_area("Reinstate notes", value=notes_default, height=80)

        if st.button("💾 Save Overrides", key="save_overrides_btn"):
            before = cur
            payload = {
                "club_id": club_id,
                "player_id": pid,
                "vacation_until": (vac.strip() or None),
                "reinstate_required": bool(rein),
                "reinstate_notes": notes.strip() or None,
            }
            sb_retry(lambda: supabase.table("ladder_player_flags").upsert(payload, on_conflict="club_id,player_id").execute())
            ladder_audit(supabase, club_id, "flags_save", "ladder_player_flags", f"{club_id}:{pid}", before, payload)
            st.success("Saved.")
            st.rerun()

    # -------------------------
    # TAB 6: AUDIT
    # -------------------------
    with tabs[6]:
        st.subheader("📜 Ladder Audit Log")
        resp = sb_retry(lambda: (
            supabase.table("ladder_audit_log")
            .select("*")
            .eq("club_id", club_id)
            .order("created_at", desc=True)
            .limit(500)
            .execute()
        ))
        df_a = _df(resp)
        if df_a.empty:
            st.info("No audit entries yet.")
        else:
            cols = [c for c in ["created_at", "actor", "action_type", "entity_type", "entity_id"] if c in df_a.columns]
            st.dataframe(df_a[cols], use_container_width=True, hide_index=True)
