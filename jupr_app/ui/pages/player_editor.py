import streamlit as st
import pandas as pd
import time

from jupr_app.ui.layout import page_shell
from jupr_app.domain.player_merge import merge_player_into

def render(ctx):
    st.write("BUILD MARKER: rebuild branch player_editor v1")
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("👥 Player Editor", "Edit player records and league ratings.", mode_label=mode_label)
    PICK_KEY = "player_editor_pick"

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    df_players_all = getattr(ctx, "df_players_all", pd.DataFrame())

    # -------------------------
    # Add New Player
    # -------------------------
    with st.expander("➕ Add New Player", expanded=False):
        with st.form("add_player_form"):
            name = st.text_input("Name")
            rating = st.number_input("Starting JUPR", 1.0, 7.0, 3.5, step=0.1)
            submit = st.form_submit_button("Add Player")
            st.write("Submit value:", submit)

            if submit:
                st.write("Submit is TRUE")
                st.write("DEBUG club_id:", club_id)
                st.write("DEBUG name:", name)
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
                resp = supabase.table("players").insert({
                    "club_id": club_id,
                    "name": "TEST_PLAYER_DIRECT",
                    "normalized_name": "test_player_direct",
                    "active": True,
                }).execute()
                st.write("DIRECT INSERT RESPONSE:", resp.data)
                st.success("Direct insert attempted.")
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
    # Player Rating State (Read-only)
    # -------------------------
    st.subheader("Manage Player")
    st.caption("Ratings are derived from match history. Use Replay or match corrections.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Name": str(row.get("name", "")),
                    "Overall JUPR": float(row.get("rating", 1200.0) or 1200.0) / 400.0,
                    "Wins": int(row.get("wins", 0) or 0),
                    "Losses": int(row.get("losses", 0) or 0),
                    "Matches Played": int(row.get("matches_played", 0) or 0),
                    "Active": bool(row.get("active", True)),
                }
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    # -------------------------
    # League Ratings (Read-only)
    # -------------------------
    st.subheader("🏟️ League Ratings")
    st.caption("Ratings are derived from match history. Use Replay or match corrections.")

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

        st.dataframe(
            lr_df[["id", "league_name", "is_active", "inactive_at", "JUPR", "Start JUPR", "wins", "losses", "matches_played"]],
            hide_index=True,
            use_container_width=True,
            column_config={"is_active": st.column_config.CheckboxColumn("Active")},
        )

    st.divider()

    # -------------------------
    # Merge Player Accounts
    # -------------------------
    st.subheader("🧬 Merge Player Accounts")
    st.caption("Merges Source → Target through the domain pipeline. Ratings are rebuilt from match history.")

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

    confirm = st.text_input("Type MERGE to confirm", value="", key="merge_confirm_text")
    if st.button("🧬 Execute Merge Now", type="primary", disabled=(confirm.strip().upper() != "MERGE")):
        result = merge_player_into(
            supabase=supabase,
            club_id=str(club_id),
            source_player_id=int(src_id),
            destination_player_id=int(dst_id),
            actor="player_editor",
        )
        if result.get("success"):
            st.success("Merge completed. Replay was suggested for downstream recalculation.")
        else:
            st.error(result.get("error") or "Merge failed.")
        time.sleep(0.4)
        st.rerun()
