# jupr_app/ui/pages/challenge_ladder.py
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Tuple

import pandas as pd
import streamlit as st

from jupr_app.domain.challenge_ladder import (
    TIER_ORDER,
    normalize_tier_id,
    tier_title,
    ladder_nm,
    ladder_bucket_challenge,
    ladder_compute_status_map,
)
from jupr_app.ui.layout import page_shell

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


def _df_from_resp(resp) -> pd.DataFrame:
    try:
        data = getattr(resp, "data", None)
        return pd.DataFrame(data or [])
    except Exception:
        return pd.DataFrame()


# -------------------------
# Data loaders (match your schema)
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

    # Ensure exists (same behavior as your old code)
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

    # Hard fallback
    return {
        "challenge_range": 3,
        "accept_window_hours": 48,
        "play_window_days": 7,
        "cooldown_hours": 72,
        "protected_hours": 72,
        "pass_hold_hours": 72,
    }


def ladder_load_core(supabase, club_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Roster
    roster = sb_retry(lambda: (
        supabase.table("ladder_roster")
        .select("id,club_id,player_id,tier_id,rank,is_active,joined_at,left_at,notes,updated_at")
        .eq("club_id", club_id)
        .order("rank", desc=False)
        .execute()
    ))
    df_roster = _df_from_resp(roster)

    # Flags
    flags = sb_retry(lambda: (
        supabase.table("ladder_player_flags")
        .select(
            "club_id,player_id,"
            "vacation_until,reinstate_required,reinstate_notes,"
            "tier_move_flag,tier_move_dest_tier,tier_move_count,tier_move_triggered_at,tier_move_last_eval_at,"
            "updated_at"
        )
        .eq("club_id", club_id)
        .execute()
    ))
    df_flags = _df_from_resp(flags)

    # Challenges
    ch = sb_retry(lambda: (
        supabase.table("ladder_challenges")
        .select("*")
        .eq("club_id", club_id)
        .order("created_at", desc=True)
        .limit(5000)
        .execute()
    ))
    df_ch = _df_from_resp(ch)

    # Pass usage
    pu = sb_retry(lambda: (
        supabase.table("ladder_pass_usage")
        .select("*")
        .eq("club_id", club_id)
        .order("used_at", desc=True)
        .limit(2000)
        .execute()
    ))
    df_pass = _df_from_resp(pu)

    return df_roster, df_flags, df_ch, df_pass


# -------------------------
# Page render
# -------------------------
def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("🪜 Challenge Ladder", "Current ladder standings and active challenges.", mode_label=mode_label)

    settings = ladder_fetch_settings(ctx.supabase, ctx.club_id)
    df_roster, df_flags, df_ch, df_pass = ladder_load_core(ctx.supabase, ctx.club_id)
    if df_roster is not None and not df_roster.empty and "tier_id" in df_roster.columns:
        df_roster["tier_id"] = df_roster["tier_id"].astype(str).apply(normalize_tier_id)
    active_ids = None
    if getattr(ctx, "df_players_active", None) is not None and not ctx.df_players_active.empty:
        if "id" in ctx.df_players_active.columns:
            active_ids = set(ctx.df_players_active["id"].astype(int).tolist())

    tab_ladder, tab_active, tab_quick = st.tabs(["🪜 Ladder", "⚔️ Active Challenges", "📘 Quick Rules"])

    # -------------------------
    # TAB 1: LADDER
    # -------------------------
    with tab_ladder:
        if df_roster is None or df_roster.empty:
            st.info("Ladder roster not initialized yet.")
        else:
            status_map = ladder_compute_status_map(
                df_roster=df_roster,
                df_flags=df_flags,
                df_ch=df_ch,
                df_pass=df_pass,
                settings=settings,
                id_to_name=ctx.id_to_name,
            )

            t_tabs = st.tabs([tier_title(tid) for tid in TIER_ORDER])

            for i, tid in enumerate(TIER_ORDER):
                with t_tabs[i]:
                    # Defensive filtering (pandas can hold bools as object)
                    sub = df_roster.copy()
                    if "is_active" in sub.columns:
                        sub = sub[sub["is_active"] == True]
                    if active_ids is not None and "player_id" in sub.columns:
                        sub = sub[sub["player_id"].astype(int).isin(active_ids)]
                    if "tier_id" in sub.columns:
                        sub = sub[sub["tier_id"].astype(str) == str(tid)]
                    else:
                        st.warning("Roster is missing 'tier_id'. Cannot render tier tabs properly.")
                        continue

                    if sub.empty:
                        st.info("No players in this tier.")
                        continue

                    sub["name"] = sub["player_id"].apply(lambda x: ladder_nm(int(x), ctx.id_to_name))
                    sub["status"] = sub["player_id"].apply(lambda pid: status_map.get(int(pid), {}).get("status", "Ready to Defend"))
                    sub["detail"] = sub["player_id"].apply(lambda pid: status_map.get(int(pid), {}).get("detail", ""))

                    q = st.text_input(f"Search ({tier_title(tid)})", value="", key=f"challenge_ladder_search_{tid}")
                    if q.strip():
                        sub = sub[sub["name"].str.contains(q.strip(), case=False, na=False)].copy()

                    if "rank" in sub.columns:
                        sub = sub.sort_values("rank", ascending=True).copy()

                        def rank_badge(r):
                            r = int(r)
                            if r == 1:
                                return "🥇 1"
                            if r == 2:
                                return "🥈 2"
                            if r == 3:
                                return "🥉 3"
                            return str(r)

                        sub["Rank"] = sub["rank"].astype(int).apply(rank_badge)
                        st.dataframe(sub[["Rank", "name", "status", "detail"]], use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(sub[["name", "status", "detail"]], use_container_width=True, hide_index=True)

    # -------------------------
    # TAB 2: ACTIVE CHALLENGES
    # -------------------------
    with tab_active:
        if df_ch is None or df_ch.empty:
            st.info("No challenges yet.")
        else:
            df = df_ch.copy()

            # Name columns
            if "challenger_id" in df.columns:
                df["challenger_name"] = df["challenger_id"].apply(lambda x: ladder_nm(int(x), ctx.id_to_name) if pd.notna(x) else "")
            else:
                df["challenger_name"] = ""

            if "defender_id" in df.columns:
                df["defender_name"] = df["defender_id"].apply(lambda x: ladder_nm(int(x), ctx.id_to_name) if pd.notna(x) else "")
            else:
                df["defender_name"] = ""

            df["bucket"] = df.apply(lambda r: ladder_bucket_challenge(r.to_dict()), axis=1)

            tab_names = [
                "Pending Acceptance",
                "Accepted / In Window",
                "Acceptance Overdue",
                "Play Overdue",
                "Recently Completed",
            ]
            tabs = st.tabs(tab_names)

            for i, tname in enumerate(tab_names):
                with tabs[i]:
                    view = df[df["bucket"] == tname].copy()
                    if view.empty:
                        st.info("No items.")
                        continue

                    if "created_at" in view.columns:
                        view["created_at"] = pd.to_datetime(view["created_at"], utc=True, errors="coerce")
                        view = view.sort_values("created_at", ascending=False)

                    show_cols = ["id", "status", "challenger_name", "defender_name", "created_at", "accept_by", "play_by", "winner_id"]
                    show_cols = [c for c in show_cols if c in view.columns]
                    show = view[show_cols].copy()

                    def winner_name(x):
                        if x is None or (isinstance(x, float) and pd.isna(x)):
                            return ""
                        try:
                            return ladder_nm(int(x), ctx.id_to_name)
                        except Exception:
                            return ""

                    if "winner_id" in show.columns:
                        show["winner"] = show["winner_id"].apply(winner_name)
                        show = show.drop(columns=["winner_id"])

                    st.dataframe(show, use_container_width=True, hide_index=True)

    # -------------------------
    # TAB 3: QUICK RULES
    # -------------------------
    with tab_quick:
        st.subheader("📘 Challenge Ladder — Quick Rules")
        st.markdown(
            """
**The Challenge Ladder is a challenge-anytime ranking system (in-season).**  
You move up by **challenging and defeating** players ranked above you **within your tier**.

Join the Challenge Ladder by email/text Joe or register at the Tres Palapas Pro Shop

**Full Challenge Ladder Rulebook** is available at the Pro Shop.

---

## How to Play

**Step 1 — Check status**  
Confirm your status on the ladder. Your status controls what you can do.

**Step 2 — Pick an eligible opponent**  
You may challenge someone **ranked above you**, up to **7 ranks higher**, **within your tier**, as long as both players are eligible.

**Step 3 — Make it official at the Pro Shop**  
A challenge is **only official once recorded by staff** in the **Pro Shop Challenge Ledger**.

**Step 4 — Defender responds (48 hours)**  
The defender must **Accept** within 48 hours.  
No response = **forfeit**.

**Step 5 — Play the match (7 days after acceptance)**  
Once accepted, the match must be completed within **7 days**.

**Step 6 — Report scores + verify**  
Submit scores to the Pro Shop so the result can be recorded and verified in the Ledger.

---

## Core Rules

1. **One active challenge at a time (single-threaded)**  
   You may only be involved in **one** challenge at a time—either as challenger or defender.  
   If you are in an active challenge, you are **Locked**.

2. **Who you can challenge**  
   - Must be **above you** in your tier  
   - Up to **7 ranks higher**  
   - Both players must be eligible by status

3. **Status rules (eligibility)**  
   You may not initiate or receive challenges if you are **Locked**, on **Vacation**, **Reinstate Required**, or **Inactive**.  
   - **Cooldown (72h):** you may be challenged, but cannot initiate  
   - **Protected (72h):** you may initiate, but cannot be challenged  
   - **Ready to Defend:** normal mode (can initiate and be challenged)

4. **Challenges must be officially recorded in the Pro Shop**  
   A challenge is only official once recorded by staff in the **Pro Shop Challenge Ledger**.

5. **48-hour acceptance window**  
   The defending player has **48 hours** to accept once the challenge is recorded in the Ledger.

6. **Monthly Pass (1 per calendar month)**  
   The defender may decline **one** challenge per calendar month **without losing rank**, as long as the Pass is used within the 48-hour window and recorded in the Ledger.

7. **No response = forfeit**  
   If the defender does not accept within 48 hours and does not use a Monthly Pass, the defender **forfeits** and ladder movement is applied accordingly.

8. **7-day play window**  
   Once accepted, the match must be completed within **7 days** of acceptance (Ledger timestamp).  
   Failure to play by the deadline may result in an **admin-determined outcome** based on good-faith scheduling.

9. **Match format: Swing Partner Swap**  
   - Each player will bring a partner to the match (Swing Partner)
   - Two doubles matches are played (best 2 of 3 games to 11, win by 2)
   - Ranked players stay **opponents** in both matches  
   - Swing Partners swap between matches  
   - Swing Partners **don't move** on the ladder

10. **How the winner is decided**  
   - Win both matches = win the challenge  
   - Split matches = total games won  
   - Still tied = total point differential  
   - Exact tie favors the **defender**

11. **Ladder movement**  
   - If the challenger wins, the two ranked players **swap ranks**  
   - If the defender wins, **no rank changes**  
   - Swing Partners never move

12. **Post-match timers (72 hours)**  
   - Challenger enters **Cooldown (72h)**  
   - Defender enters **Protected (72h)**

13. **Vacation & reinstatement (admin-controlled)**  
   Vacation status is controlled by the Ladder Admin (typically **48 hours’ notice** when possible).  
   Returning from Vacation requires a **Reinstatement Match** before normal ladder activity resumes.

14. **Disputes & enforcement**  
   The Ladder Admin resolves disputes and enforces rules using the **Pro Shop Challenge Ledger** as the official record.
"""
        )
