import streamlit as st
import pandas as pd
import time
from datetime import datetime, timezone

from jupr_app.ui.layout import page_shell
from jupr_app.domain.player_ops import safe_add_player
st.header("FORM TEST")

with st.form("test_form"):
    name = st.text_input("Name")
    submit = st.form_submit_button("Add Player")

st.write("Submit value:", submit)

if submit:
    st.success("FORM WORKS")
def render(ctx):
    import streamlit as st
    st.write("STATIC PAGE")

    with st.form("static_form"):
        submit = st.form_submit_button("Click Me")

    st.write("Submit:", submit)

    if submit:
        st.success("IT WORKS")
    # -------------------------
    # Add New Player
    # -------------------------
    with st.expander("➕ Add New Player", expanded=False):
    
        with st.form("add_player_form", clear_on_submit=True):
            name = st.text_input("Name")
            rating = st.number_input("Starting JUPR", 1.0, 7.0, 3.5, step=0.1)
            submit = st.form_submit_button("Add Player")
    
        if submit:
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
                st.rerun()
    
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
