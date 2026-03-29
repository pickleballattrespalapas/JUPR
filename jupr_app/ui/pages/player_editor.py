import streamlit as st
import pandas as pd
import time
from datetime import datetime, timezone

from jupr_app.ui.layout import page_shell
from jupr_app.domain.live_social import auto_link_exact_matches, social_person_rollup_rows
from jupr_app.domain.player_ops import safe_add_player

def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("👥 Player Editor", "Edit player records and league ratings.", mode_label=mode_label)
    PICK_KEY = "player_editor_pick"

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    supabase = ctx.supabase
    club_id = str(ctx.club_id)

    df_players_all = getattr(ctx, "df_players_all", pd.DataFrame())
    df_meta = getattr(ctx, "df_meta", pd.DataFrame())

    # -------------------------
    # Add New Player
    # -------------------------
    with st.expander("➕ Add New Player", expanded=False):
        with st.form("add_player_form"):
            name = st.text_input("Name")
            rating = st.number_input("Starting JUPR", 1.0, 7.0, 3.5, step=0.1)
            if st.form_submit_button("Add Player"):
                name_clean = name.strip()
                if not name_clean:
                    st.error("Name required.")
                    st.stop()
                existing = (
                    supabase.table("players")
                    .select("id,name")
                    .eq("club_id", club_id)
                    .eq("name", name_clean)
                    .limit(1)
                    .execute()
                    .data
                    or []
                )
                if existing:
                    st.info("Player already exists — opening existing record.")
                    st.session_state[PICK_KEY] = name_clean
                    time.sleep(0.1)
                    st.rerun()
                payload = {
                    "club_id": club_id,
                    "name": name_clean,
                    "rating": float(rating) * 400.0,
                    "starting_rating": float(rating) * 400.0,
                    "wins": 0,
                    "losses": 0,
                    "matches_played": 0,
                    "active": True,
                    "inactive_at": None,
                }
                ok, err = safe_add_player(
                    supabase=supabase,
                    club_id=club_id,
                    name=name_clean,
                    rating_jupr=float(rating),
                )
                if not ok:
                    st.error(err or "Unable to add player.")
                    st.stop()
                st.success("Added.")
                st.session_state[PICK_KEY] = name_clean
                time.sleep(0.2)
                st.rerun()

    st.divider()

    # -------------------------
    # Select Player
    # -------------------------
    if df_players_all is None or df_players_all.empty:
        st.info("No players loaded.")
        st.stop()

    all_names = sorted(df_players_all["name"].astype(str).tolist())
    pick = st.selectbox("Select Player", [""] + all_names, index=0, key=PICK_KEY)

    if not pick:
        st.info("Pick a player to edit.")
        st.stop()

    row = df_players_all[df_players_all["name"].astype(str) == str(pick)].iloc[0]
    pid = int(row["id"])

    # -------------------------
    # Edit Player
    # -------------------------
    st.subheader("Manage Player")
    with st.form("edit_player_form"):
        new_name = st.text_input("Name", value=str(row.get("name", "")))
        new_rating = st.number_input("Overall JUPR", 1.0, 7.0, float(row.get("rating", 1200.0) or 1200.0) / 400.0, step=0.01)
        active = st.checkbox("Active", value=bool(row.get("active", True)))
        if st.form_submit_button("Save Player"):
            supabase.table("players").update(
                {"name": new_name.strip(), "rating": float(new_rating) * 400.0, "active": bool(active)}
            ).eq("club_id", club_id).eq("id", pid).execute()
            st.success("Saved. Use Refresh in sidebar if leaderboards still show old values.")
            time.sleep(0.4)
            st.rerun()

    st.divider()

    # -------------------------
    # League Ratings Editor
    # -------------------------
    st.subheader("🏟️ League Ratings")

    league_opts = []
    if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
        league_opts = sorted(df_meta["league_name"].dropna().astype(str).unique().tolist())

    # Fetch league rows live (admin page; fine)
    lr_resp = supabase.table("league_ratings").select(
        "id,league_name,rating,starting_rating,wins,losses,matches_played,is_active,inactive_at"
    ).eq("club_id", club_id).eq("player_id", pid).execute()

    lr_df = pd.DataFrame(lr_resp.data or [])
    if lr_df.empty:
        st.info("No league rows for this player yet.")
    else:
        lr_df["JUPR"] = lr_df["rating"].astype(float) / 400.0
        lr_df["Start JUPR"] = lr_df["starting_rating"].astype(float) / 400.0

        edited = st.data_editor(
            lr_df[["id", "league_name", "is_active", "inactive_at", "JUPR", "Start JUPR", "wins", "losses", "matches_played"]],
            hide_index=True,
            use_container_width=True,
            disabled=["id", "league_name", "inactive_at"],
            column_config={
                "is_active": st.column_config.CheckboxColumn("Active"),
                "JUPR": st.column_config.NumberColumn("League JUPR", min_value=1.0, max_value=7.0, step=0.01),
                "Start JUPR": st.column_config.NumberColumn("Start JUPR", min_value=1.0, max_value=7.0, step=0.01),
                "wins": st.column_config.NumberColumn("W", min_value=0, step=1),
                "losses": st.column_config.NumberColumn("L", min_value=0, step=1),
                "matches_played": st.column_config.NumberColumn("MP", min_value=0, step=1),
            },
        )

        if st.button("💾 Save League Edits"):
            now_iso = datetime.now(timezone.utc).isoformat()
            for _, r in edited.iterrows():
                rid = int(r["id"])
                next_active = bool(r.get("is_active", True))
                payload = {
                    "rating": float(r["JUPR"]) * 400.0,
                    "starting_rating": float(r["Start JUPR"]) * 400.0,
                    "wins": int(r["wins"]),
                    "losses": int(r["losses"]),
                    "matches_played": int(r["matches_played"]),
                    "is_active": next_active,
                    "inactive_at": None if next_active else (lr_df[lr_df["id"] == rid]["inactive_at"].iloc[0] or now_iso),
                }
                supabase.table("league_ratings").update(payload).eq("club_id", club_id).eq("id", rid).execute()

            st.success("Saved league ratings. Refresh cached data if needed.")
            time.sleep(0.4)
            st.rerun()

    st.divider()

    # -------------------------
    # Merge Player Accounts
    # -------------------------
    st.subheader("🧬 Merge Player Accounts")
    st.caption("Rewires Source → Target in matches + league_ratings. After merge: run Admin Tools → Replay History → ALL.")

    # Load all players live
    allp = (
        supabase.table("players")
        .select("id,name,active,inactive_at")
        .eq("club_id", club_id)
        .order("name", desc=False)
        .execute()
        .data
        or []
    )
    dfp = pd.DataFrame(allp)
    if dfp.empty:
        st.info("No players found.")
        return

    def _is_active(row) -> bool:
        if "inactive_at" in row and pd.notna(row.get("inactive_at")):
            return False
        return bool(row.get("active", True))

    dfp["label"] = dfp.apply(
        lambda r: f"{r['name']} (#{int(r['id'])})" + ("" if _is_active(r) else " [inactive]"),
        axis=1,
    )
    label_to_id = dict(zip(dfp["label"], dfp["id"]))

    cA, cB = st.columns(2)
    with cA:
        src_label = st.selectbox("Source (duplicate to remove)", dfp["label"].tolist(), key="merge_src_label")
    with cB:
        dst_label = st.selectbox("Target (keeper)", dfp["label"].tolist(), key="merge_dst_label")

    src_id = int(label_to_id.get(src_label))
    dst_id = int(label_to_id.get(dst_label))

    if src_id == dst_id:
        st.warning("Pick two different players.")
        return

    def count_eq(table: str, col: str, val: int) -> int:
        resp = supabase.table(table).select("id", count="exact").eq("club_id", club_id).eq(col, int(val)).execute()
        return int(getattr(resp, "count", 0) or 0)

    with st.expander("Dry-run impact (counts)", expanded=True):
        st.write("Matches referencing Source:")
        st.json({c: count_eq("matches", c, src_id) for c in ["t1_p1", "t1_p2", "t2_p1", "t2_p2"]})
        st.write("League rows for Source:")
        st.write(count_eq("league_ratings", "player_id", src_id))

    confirm = st.text_input("Type MERGE to confirm", value="", key="merge_confirm_text")
    if st.button("🧬 Execute Merge Now", type="primary", disabled=(confirm.strip().upper() != "MERGE")):
        # Update matches
        for col in ["t1_p1", "t1_p2", "t2_p1", "t2_p2"]:
            supabase.table("matches").update({col: int(dst_id)}).eq("club_id", club_id).eq(col, int(src_id)).execute()

        # Move league_ratings
        supabase.table("league_ratings").update({"player_id": int(dst_id)}).eq("club_id", club_id).eq("player_id", int(src_id)).execute()

        # Deactivate source player
        src_p = supabase.table("players").select("name").eq("club_id", club_id).eq("id", int(src_id)).limit(1).execute().data
        dst_p = supabase.table("players").select("name").eq("club_id", club_id).eq("id", int(dst_id)).limit(1).execute().data
        src_name = str(src_p[0]["name"]) if src_p else f"#{src_id}"
        dst_name = str(dst_p[0]["name"]) if dst_p else f"#{dst_id}"
        now_iso = datetime.now(timezone.utc).isoformat()
        supabase.table("players").update(
            {
                "active": False,
                "inactive_at": now_iso,
                "name": f"{src_name} (MERGED into {dst_name} #{dst_id})",
            }
        ).eq("club_id", club_id).eq("id", int(src_id)).execute()

        st.success("Merge completed. Now run Admin Tools → Replay History → ALL.")
        time.sleep(0.4)
        st.rerun()

    st.divider()

    # -------------------------
    # Social Identity Linking
    # -------------------------
    st.subheader("🧩 Social Identity Linking")
    st.caption("Review club_people rows and explicitly link social-only identities to existing players.")

    rollup_rows = social_person_rollup_rows(supabase, club_id)
    if not rollup_rows:
        st.info("No social club_people rows found yet.")
        return

    id_to_player = {}
    player_options = []
    for _, player in df_players_all.sort_values(by=["name"]).iterrows():
        pid = int(player["id"])
        label = f"{str(player.get('name') or '').strip()} (#{pid})"
        player_options.append(label)
        id_to_player[label] = pid

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Auto-link exact matches", key="club_people_auto_link_exact"):
            result = auto_link_exact_matches(
                supabase,
                club_id=club_id,
                club_people_rows=rollup_rows,
                df_players_all=df_players_all,
            )
            st.success(
                f"Linked {result['linked_count']} rows. "
                f"Skipped {result['skipped_count']} (already linked, unmatched, or ambiguous)."
            )
            time.sleep(0.2)
            st.rerun()
    with c2:
        st.caption("Auto-link only applies exact normalized-name matches with one unique player candidate.")

    table_df = pd.DataFrame(rollup_rows)
    if not table_df.empty:
        table_df = table_df.rename(
            columns={
                "linked_player_id": "linked_player_id",
                "first_seen_on": "first_seen_on",
                "last_seen_on": "last_seen_on",
                "social_event_count": "social_event_count",
                "social_match_count": "social_match_count",
            }
        )
        st.dataframe(
            table_df[
                [
                    "display_name",
                    "normalized_name",
                    "linked_player_id",
                    "first_seen_on",
                    "last_seen_on",
                    "social_event_count",
                    "social_match_count",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    unlinked_rows = [row for row in rollup_rows if row.get("linked_player_id") in (None, "")]
    if not unlinked_rows:
        st.success("All club_people rows are linked.")
        return

    st.markdown("#### Manual link controls (unlinked only)")
    for row in unlinked_rows:
        person_id = str(row.get("id"))
        name = str(row.get("display_name") or "Unknown")
        with st.form(f"manual_link_{person_id}"):
            st.write(f"**{name}**  \nNormalized: `{row.get('normalized_name') or ''}`")
            selection = st.selectbox(
                "Link to player",
                options=[""] + player_options,
                index=0,
                key=f"manual_link_pick_{person_id}",
            )
            submitted = st.form_submit_button("Save Link")
            if submitted:
                if not selection:
                    st.warning("Pick a player before saving.")
                else:
                    player_id = int(id_to_player[selection])
                    supabase.table("club_people").update({"linked_player_id": player_id}).eq("club_id", club_id).eq(
                        "id", person_id
                    ).execute()
                    st.success(f"Linked {name} to {selection}.")
                    time.sleep(0.2)
                    st.rerun()
