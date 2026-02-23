# jupr_app/ui/pages/league_manager.py
from __future__ import annotations

from jupr_app.data.sb_write import sb_insert, sb_update

import json
import logging
from datetime import date, datetime, timedelta, timezone
from jupr_app.domain.player_ops import get_or_create_player
from jupr_app.domain.ratings import jupr_to_elo

import pandas as pd
import streamlit as st
from jupr_court_board import court_board

from jupr_app.domain.constants import DEFAULT_K_FACTOR
from jupr_app.domain.live_ladder import build_movement_preview, compute_round_stats, validate_courts
from jupr_app.domain.league_night_roster import (
    RosterChangeError,
    apply_roster_change,
    roster_change_availability,
    suggest_court_sizes,
)
from jupr_app.domain.leagues import (
    compute_top_performer_awards_for_config,
    get_league_meta_row,
    mint_top_performer_badges,
    normalize_league_status,
)
from jupr_app.domain.roster import (
    compress_courts,
    courts_to_roster_df,
    move_player_to_court,
    move_within_court,
    normalize_slots,
    roster_df_to_courts,
    swap_players,
)
from jupr_app.domain.schedule import get_match_schedule
from jupr_app.ui.layout import page_shell


logger = logging.getLogger(__name__)


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_load(value, default):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _parse_date(value: object | None) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return pd.to_datetime(value, errors="coerce").date()
    except Exception:
        return None


def _parse_blackout_dates(raw: str) -> list[str]:
    entries = [part.strip() for part in (raw or "").replace("\n", ",").split(",") if part.strip()]
    results: list[str] = []
    for entry in entries:
        parsed = _parse_date(entry)
        if parsed:
            results.append(parsed.isoformat())
    return sorted(set(results))


def _build_schedule_preview(schedule_cfg: dict) -> pd.DataFrame:
    start_date = _parse_date(schedule_cfg.get("start_date"))
    if not start_date:
        return pd.DataFrame()
    weekday = schedule_cfg.get("weekday")
    if weekday is None:
        return pd.DataFrame()
    try:
        weekday = int(weekday)
    except Exception:
        return pd.DataFrame()
    weeks = schedule_cfg.get("weeks")
    end_date = _parse_date(schedule_cfg.get("end_date"))
    time_start = schedule_cfg.get("time_start", "")
    time_end = schedule_cfg.get("time_end", "")
    blackout = {d for d in (schedule_cfg.get("blackout_dates") or []) if d}

    first_date = start_date + timedelta(days=(weekday - start_date.weekday()) % 7)
    dates: list[date] = []
    if weeks:
        try:
            total = int(weeks)
        except Exception:
            total = 0
        for idx in range(total):
            dates.append(first_date + timedelta(weeks=idx))
    elif end_date:
        current = first_date
        while current <= end_date:
            dates.append(current)
            current = current + timedelta(weeks=1)

    rows = []
    for idx, day in enumerate(dates, start=1):
        if day.isoformat() in blackout:
            continue
        rows.append(
            {
                "Session": idx,
                "Date": day.isoformat(),
                "Start": time_start,
                "End": time_end,
            }
        )
    return pd.DataFrame(rows)


def _schedule_to_ics(schedule_cfg: dict) -> str:
    preview = _build_schedule_preview(schedule_cfg)
    if preview.empty:
        return ""
    tz = schedule_cfg.get("timezone") or "UTC"
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//JUPR//League Schedule//EN",
    ]
    for _, row in preview.iterrows():
        date_str = row.get("Date")
        time_start = row.get("Start") or "18:00"
        time_end = row.get("End") or "20:00"
        start_stamp = f"{date_str.replace('-', '')}T{time_start.replace(':', '')}00"
        end_stamp = f"{date_str.replace('-', '')}T{time_end.replace(':', '')}00"
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"DTSTART;TZID={tz}:{start_stamp}",
                f"DTEND;TZID={tz}:{end_stamp}",
                "SUMMARY:League Session",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    return "\n".join(lines)


def _league_options(df_meta: pd.DataFrame) -> list[str]:
    if df_meta is not None and not df_meta.empty and "is_active" in df_meta.columns and "league_name" in df_meta.columns:
        active_mask = df_meta["is_active"].fillna(False).astype(bool)
        opts = sorted(df_meta[active_mask]["league_name"].dropna().astype(str).tolist())
        return opts if opts else ["Default"]
    return ["Default"]


def _extract_court_board_defaults(meta_row: dict | None) -> dict:
    defaults = _safe_json_load((meta_row or {}).get("court_board_defaults"), {}) or {}
    return {
        "max_used_courts": int(defaults.get("max_used_courts") or 0) or 0,
        "players_per_court": str(defaults.get("players_per_court") or "4"),
        "rotation_mode": str(defaults.get("rotation_mode") or "fixed"),
        "game_format_points": int(defaults.get("game_format_points") or 11),
        "game_format_time": int(defaults.get("game_format_time") or 15),
        "total_courts": int(defaults.get("total_courts") or 0),
        "court_identifiers": list(defaults.get("court_identifiers") or []),
    }


def _seed_rating_for_player(pid: int, league_name: str, df_players_all: pd.DataFrame, df_leagues: pd.DataFrame) -> float:
    # League rating if exists, else overall
    if df_leagues is not None and not df_leagues.empty:
        hit = df_leagues[
            (df_leagues["player_id"].astype(int) == int(pid))
            & (df_leagues["league_name"].astype(str).str.strip() == str(league_name).strip())
        ]
        if not hit.empty:
            return float(hit.iloc[0].get("rating", 1200.0) or 1200.0)

    hit2 = df_players_all[df_players_all["id"].astype(int) == int(pid)]
    if not hit2.empty:
        return float(hit2.iloc[0].get("rating", 1200.0) or 1200.0)

    return 1200.0


def _summarize_roster(roster_df: pd.DataFrame) -> pd.DataFrame:
    if roster_df is None or roster_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["court", "slot", "name", "player_id", "rating"] if c in roster_df.columns]
    df = roster_df[cols].copy()
    df["court"] = pd.to_numeric(df.get("court"), errors="coerce").fillna(0).astype(int)
    df["slot"] = pd.to_numeric(df.get("slot"), errors="coerce").fillna(0).astype(int)
    return df.sort_values(["court", "slot"]).reset_index(drop=True)


def _render_court_board_grid(roster_df: pd.DataFrame, max_per_row: int = 4) -> None:
    _ = max_per_row
    roster_now = compress_courts(normalize_slots(roster_df.copy()))
    if roster_now.empty:
        raise RuntimeError("Court board render aborted: ladder roster is empty.")
    if "player_id" not in roster_now.columns:
        raise RuntimeError("Court board render aborted: roster is missing player_id.")
    if roster_now["player_id"].astype(str).duplicated().any():
        dupes = roster_now.loc[roster_now["player_id"].astype(str).duplicated(), "player_id"].astype(str).tolist()
        raise RuntimeError(f"Court board render aborted: duplicate player_id values in roster: {dupes}")

    print("DEBUG: ladder_live_roster shape:", roster_now.shape)
    print("DEBUG: ladder_live_roster columns:", list(roster_now.columns))

    courts_payload = roster_df_to_courts(roster_now, ladder_court_sizes=st.session_state.get("ladder_court_sizes"))
    for bench_row in list(st.session_state.get("ladder_bench_players", [])):
        pid = bench_row.get("player_id")
        if pid is None:
            continue
        courts_payload[-1]["players"].append(
            {
                "player_id": str(int(pid)),
                "name": str(bench_row.get("name") or f"#{pid}"),
                "rating": float(bench_row.get("rating", 1200.0)) / 400.0,
            }
        )

    round_num = int(st.session_state.get("ladder_round_num", 1))
    result = court_board(courts_payload, key=f"court_board_confirm_start_r{round_num}")
    if not (result and isinstance(result, dict) and "courts" in result):
        return

    updated_courts = result["courts"]
    new_roster = courts_to_roster_df(updated_courts, roster_now)

    player_lookup = {}
    for _, row in roster_now.iterrows():
        pid = int(row.get("player_id"))
        player_lookup[pid] = {
            "player_id": pid,
            "name": str(row.get("name") or f"#{pid}"),
            "rating": float(row.get("rating", 1200.0)),
        }
    for row in list(st.session_state.get("ladder_bench_players", [])):
        pid = row.get("player_id")
        if pid is None:
            continue
        pid = int(pid)
        player_lookup[pid] = {
            "player_id": pid,
            "name": str(row.get("name") or f"#{pid}"),
            "rating": float(row.get("rating", 1200.0)),
        }

    new_bench: list[dict] = []
    for court in updated_courts:
        if str(court.get("court_id", "")).strip().lower() != "bench":
            continue
        for player in list(court.get("players") or []):
            raw_pid = player.get("player_id")
            if raw_pid is None:
                continue
            pid = int(raw_pid)
            source = player_lookup.get(pid, {})
            new_bench.append(
                {
                    "player_id": pid,
                    "name": str(source.get("name") or player.get("name") or f"#{pid}"),
                    "rating": float(source.get("rating", 1200.0)),
                }
            )

    new_roster = compress_courts(normalize_slots(new_roster.copy()))
    old_bench = list(st.session_state.get("ladder_bench_players", []))
    if new_roster.equals(roster_now) and new_bench == old_bench:
        return

    st.session_state.ladder_live_roster = new_roster
    st.session_state["ladder_bench_players"] = new_bench
    st.session_state.pop("current_schedule", None)
    st.rerun()


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("🏟️ League Manager", "Run live events and manage ladders.", mode_label=mode_label)

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    # ---- state init ----
    st.session_state.setdefault("ladder_state", "SETUP")
    st.session_state.setdefault("ladder_round_num", 1)
    st.session_state.setdefault("ladder_total_rounds", 5)
    st.session_state.setdefault("ladder_roster", [])
    st.session_state.setdefault("ladder_court_sizes", [])

    df_players_all = ctx.df_players_all
    df_leagues = ctx.df_leagues
    df_meta = ctx.df_meta
    print("DEBUG: admin_logged_in:", bool(getattr(ctx, "admin_logged_in", False)))
    print("DEBUG: public_mode:", bool(getattr(ctx, "public_mode", False)))
    print("DEBUG: df_leagues shape:", df_leagues.shape if isinstance(df_leagues, pd.DataFrame) else None)
    print("DEBUG: df_leagues columns:", list(df_leagues.columns) if isinstance(df_leagues, pd.DataFrame) else None)
    id_to_name = ctx.id_to_name
    name_to_id = ctx.name_to_id

    tabs = st.tabs(["🏃‍♂️ Run Live Event (Ladder)", "⚙️ Settings"])

    # ============================================================
    # TAB 1: LIVE LADDER
    # ============================================================
    with tabs[0]:
        st.subheader("Ladder Management")

        # -------------------------
        # 1) SETUP
        # -------------------------
        if st.session_state.get("ladder_state", "SETUP") == "SETUP":
            st.markdown("#### Step 1: Select League & Roster")

            opts = _league_options(df_meta)
            lg_select = st.selectbox("Select League", opts, key="ladder_lg")
            meta_row = get_league_meta_row(df_meta, lg_select) or {}
            if st.session_state.get("ladder_defaults_league") != lg_select:
                defaults = _extract_court_board_defaults(meta_row)
                st.session_state["ladder_defaults_league"] = lg_select
                st.session_state["ladder_max_used_courts"] = defaults.get("max_used_courts", 0)
                st.session_state["ladder_players_per_court"] = defaults.get("players_per_court", "4")
                st.session_state["ladder_rotation_mode"] = defaults.get("rotation_mode", "fixed")
                st.session_state["ladder_game_format_points"] = defaults.get("game_format_points", 11)
                st.session_state["ladder_game_format_time"] = defaults.get("game_format_time", 15)
            with st.expander("Court Board Defaults", expanded=False):
                st.caption("Defaults sourced from league settings. Adjust for this event if needed.")
                st.number_input(
                    "Max used courts",
                    min_value=0,
                    step=1,
                    value=int(st.session_state.get("ladder_max_used_courts", 0)),
                    key="ladder_max_used_courts",
                )
                st.selectbox(
                    "Players per court preference",
                    ["4", "5", "6+"],
                    index=["4", "5", "6+"].index(str(st.session_state.get("ladder_players_per_court", "4"))),
                    key="ladder_players_per_court",
                )
                st.selectbox(
                    "Rotation mode",
                    ["fixed", "queue"],
                    index=["fixed", "queue"].index(str(st.session_state.get("ladder_rotation_mode", "fixed"))),
                    key="ladder_rotation_mode",
                )
                st.number_input(
                    "Game format points cap",
                    min_value=1,
                    step=1,
                    value=int(st.session_state.get("ladder_game_format_points", 11)),
                    key="ladder_game_format_points",
                )
                st.number_input(
                    "Game format time cap (minutes)",
                    min_value=1,
                    step=1,
                    value=int(st.session_state.get("ladder_game_format_time", 15)),
                    key="ladder_game_format_time",
                )
            week_select = st.selectbox("Week", [f"Week {i}" for i in range(1, 13)] + ["Playoffs"], key="ladder_wk")
            num_rounds = st.number_input(
                "Total Rounds to Play",
                1, 20,
                value=int(st.session_state.get("ladder_total_rounds", 5)),
                step=1,
                key="ladder_total_rounds_input",
            )
            raw = st.text_area("Paste Player List (one per line)", height=150, key="ladder_raw_input")

            if st.button("Analyze & Seed"):
                st.session_state.saved_ladder_lg = str(lg_select)
                st.session_state.saved_ladder_wk = str(week_select)
                st.session_state.ladder_total_rounds = int(num_rounds)

                parsed = [x.strip() for x in (raw or "").replace("\n", ",").split(",") if x.strip()]
                duplicate_names = sorted({name for name in parsed if parsed.count(name) > 1})
                if duplicate_names:
                    raise RuntimeError(f"Duplicate player names found in roster input: {duplicate_names}")
                roster_data = []
                new_ps = []

                for n in parsed:
                    if n in name_to_id:
                        pid = int(name_to_id[n])
                        r = _seed_rating_for_player(pid, lg_select, df_players_all, df_leagues)
                        roster_data.append({"name": n, "rating": float(r), "id": pid})
                    else:
                        new_ps.append(n)

                st.session_state.ladder_temp_roster = roster_data
                st.session_state.ladder_temp_new = new_ps
                st.session_state.ladder_state = "REVIEW_ROSTER"
                st.session_state.ladder_round_num = 1
                _rerun()
                return

        # -------------------------
        # 2) REVIEW / NEW PLAYERS
        # -------------------------
        if st.session_state.get("ladder_state") == "REVIEW_ROSTER":
            c_back, _ = st.columns([1, 5])
            if c_back.button("⬅️ Back (edit league/week/rounds/roster)"):
                st.session_state.ladder_state = "SETUP"
                _rerun()
                return

            st.markdown("#### Step 2: Confirm Roster")

            new_names = st.session_state.get("ladder_temp_new", [])
            # Re-check DB for "missing" names (cache/name_to_id may be stale)
            if new_names:
                normalized = [str(x).strip() for x in new_names if str(x).strip()]
                resp = (
                    ctx.supabase.table("players")
                    .select("id,name,rating")
                    .eq("club_id", str(ctx.club_id))
                    .in_("name", normalized)
                    .execute()
                )
                existing = resp.data or []
                if existing:
                    # Move found ones into the roster immediately
                    base_roster = st.session_state.get("ladder_temp_roster", []) or []
                    base_names = {str(x.get("name", "")).strip() for x in base_roster}

                    for row in existing:
                        nm = str(row["name"]).strip()
                        if nm in base_names:
                            continue
                        base_roster.append({
                            "name": nm,
                            "rating": float(row.get("rating", 1200.0) or 1200.0),
                            "id": int(row["id"]),
                        })

                    st.session_state.ladder_temp_roster = base_roster

                    # Remove those from the "new" list
                    existing_names = {str(r["name"]).strip() for r in existing}
                    st.session_state.ladder_temp_new = [n for n in normalized if str(n).strip() not in existing_names]
                    new_names = st.session_state.get("ladder_temp_new", [])

                st.caption("Set a starting JUPR for each new player, then click Save & Continue. (This creates the accounts.)")

                df_new = pd.DataFrame(
                    {"Name": [str(x) for x in new_names], "Starting JUPR": [3.5] * len(new_names)}
                )

                edited_new = st.data_editor(
                    df_new,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Starting JUPR": st.column_config.NumberColumn(
                            "Starting JUPR",
                            min_value=1.0,
                            max_value=7.0,
                            step=0.1,
                        )
                    },
                    key="lm_new_players_editor",
                )

                c1, c2 = st.columns([1, 3])
                if c1.button("Save New Players & Continue", type="primary"):
                    blocking_errs = 0
                    created_or_existing: dict[str, dict] = {}
                    for _, r in edited_new.iterrows():
                        nm = str(r["Name"]).strip()
                        normalized_name = " ".join(nm.lower().split())
                        player_email = str(r.get("Email", "") or "").strip()
                        try:
                            jupr = float(r["Starting JUPR"])
                        except (TypeError, ValueError, KeyError):
                            blocking_errs += 1
                            logger.warning(
                                "Ladder step2 add player validation failed club_id=%s normalized_name=%s name=%s email=%s err=%s",
                                str(ctx.club_id),
                                normalized_name,
                                nm,
                                player_email,
                                "invalid Starting JUPR value",
                            )
                            st.error(f"Could not add {nm}: invalid Starting JUPR value.")
                            continue

                        payload = {
                            "club_id": str(ctx.club_id),
                            "name": nm,
                            "normalized_name": normalized_name,
                            "rating": float(jupr_to_elo(jupr)),
                        }
                        try:
                            ok, player_row, err = get_or_create_player(
                                supabase=ctx.supabase,
                                club_id=str(ctx.club_id),
                                normalized_name=normalized_name,
                                payload=payload,
                            )
                        except Exception as exc:
                            ok, player_row, err = False, None, str(exc)

                        if not ok:
                            blocking_errs += 1
                            logger.warning(
                                "Ladder step2 add player failed club_id=%s normalized_name=%s name=%s email=%s err=%s",
                                str(ctx.club_id),
                                normalized_name,
                                nm,
                                player_email,
                                err,
                            )
                            st.error(f"Could not add {nm}: {err}")
                            continue

                        if err == "already_exists":
                            logger.info(
                                "Ladder step2 player already exists club_id=%s normalized_name=%s name=%s email=%s",
                                str(ctx.club_id),
                                normalized_name,
                                nm,
                                player_email,
                            )
                            st.info(f"{nm} is already in your club roster. Using existing player record.")

                        if isinstance(player_row, dict):
                            created_or_existing[normalized_name] = player_row

                    # Required operations failed; keep user on step 2.
                    if blocking_errs > 0:
                        st.warning("Some players could not be processed. Fix the errors above and try again.")
                        return

                    created_names = [str(x).strip() for x in edited_new["Name"].tolist() if str(x).strip()]

                    base_roster = st.session_state.get("ladder_temp_roster", []) or []
                    base_roster_names = {str(x.get("name", "")).strip() for x in base_roster}

                    for nm in created_names:
                        if nm in base_roster_names:
                            continue
                        row = created_or_existing.get(" ".join(nm.lower().split()))
                        if not row:
                            logger.warning(
                                "Ladder step2 missing returned row club_id=%s normalized_name=%s name=%s email=%s err=%s",
                                str(ctx.club_id),
                                " ".join(nm.lower().split()),
                                nm,
                                "",
                                "player row not returned from create/get",
                            )
                            st.error(f"Could not load {nm} after processing. Please retry.")
                            blocking_errs += 1
                            continue
                        base_roster.append({
                            "name": str(row["name"]).strip(),
                            "rating": float(row.get("rating", 1200.0) or 1200.0),
                            "id": int(row["id"]),
                        })

                    if blocking_errs > 0:
                        st.warning("Some players could not be loaded into the roster yet. Please retry.")
                        return

                    # Clear "new players" list so Step 2 doesn't trap you again
                    st.session_state.ladder_temp_roster = base_roster
                    st.session_state.ladder_temp_new = []

                    # Promote to real roster and advance
                    st.session_state.ladder_roster = sorted(
                        base_roster,
                        key=lambda x: float(x.get("rating", 1200.0)),
                        reverse=True,
                    )
                    st.session_state.ladder_state = "CONFIG_COURTS"

                    # Optional: refresh global cached data so other pages see them immediately
                    return

                # Keep this to prevent falling through while new players exist
                st.stop()



            st.success("All players found.")
            if st.button("Proceed to Court Setup"):
                st.session_state.ladder_roster = sorted(
                    st.session_state.get("ladder_temp_roster", []),
                    key=lambda x: float(x.get("rating", 1200.0)),
                    reverse=True,
                )
                st.session_state.ladder_state = "CONFIG_COURTS"
                st.session_state.pop("current_schedule", None)
                st.session_state.pop("current_schedule_round", None)
                _rerun()
                return


        # -------------------------
        # 3) CONFIG COURTS
        # -------------------------
        if st.session_state.get("ladder_state") == "CONFIG_COURTS":
            c_back, _ = st.columns([1, 5])
            if c_back.button("⬅️ Back (edit roster)"):
                st.session_state.ladder_state = "REVIEW_ROSTER"
                _rerun()
                return

            st.markdown("#### Step 3: Configure Courts")
            total_p = len(st.session_state.get("ladder_roster", []))
            st.info(f"Total Players: {total_p}")

            auto = suggest_court_sizes(total_p)
            use_auto = st.checkbox("Use suggested setup (4s/5s only)", value=True, key="ladder_use_auto_courts")

            if use_auto and auto["ok"]:
                court_sizes = auto["sizes"]
                if auto.get("bench", 0) > 0:
                    st.warning(f"Auto setup requires {auto['bench']} bench player(s). Bench players will not be scheduled.")
                st.caption(auto["note"])
            else:
                num_courts = st.number_input("Number of Courts", 1, 10, key="ladder_num_courts", value=3)
                court_sizes = []
                cols = st.columns(int(num_courts))
                for i in range(int(num_courts)):
                    with cols[i]:
                        s = st.number_input(f"Ct {i+1} Size", 4, 12, 4, key=f"cs_{i}")
                        court_sizes.append(int(s))

            if sum(court_sizes) != total_p:
                st.error(f"Court sizes sum to {sum(court_sizes)}, but you have {total_p} players.")
            else:
                if st.button("Preview Assignments"):
                    current_idx = 0
                    final_assignments = []

                    for c_idx, size in enumerate(court_sizes):
                        roster_seed = st.session_state.get("ladder_roster", [])
                        group = roster_seed[current_idx: current_idx + int(size)]
                        for pl in group:
                            final_assignments.append({
                                "player_id": int(pl["id"]),
                                "name": str(pl["name"]),
                                "rating": float(pl["rating"]),
                                "court": int(c_idx + 1),
                            })
                        current_idx += int(size)

                    final_roster = pd.DataFrame(final_assignments)
                    final_roster = final_roster.sort_values(["court", "rating"], ascending=[True, False]).copy()
                    final_roster["slot"] = final_roster.groupby("court").cumcount() + 1

                    st.session_state.ladder_live_roster = final_roster[["player_id", "name", "rating", "court", "slot"]].copy()
                    st.session_state.ladder_court_sizes = list(map(int, court_sizes))

                    # Printable sheet
                    print_df = final_roster.copy()
                    print_df["JUPR"] = (print_df["rating"].astype(float) / 400.0).round(3)
                    print_df = print_df.sort_values(["court", "slot"], ascending=[True, True])
                    print_df = print_df[["court", "slot", "name", "JUPR"]].rename(
                        columns={"court": "Court", "slot": "Slot", "name": "Player", "JUPR": "JUPR"}
                    )
                    st.session_state.ladder_print_sheet = print_df

                    st.session_state.ladder_state = "CONFIRM_START"
                    _rerun()
                    return

        # -------------------------
        # 3.5) CONFIRM START (Court Board)
        # -------------------------
        if st.session_state.get("ladder_state") == "CONFIRM_START":
            c_back, _ = st.columns([1, 5])
            if c_back.button("⬅️ Back (edit courts)"):
                st.session_state.pop("ladder_live_roster", None)
                st.session_state.ladder_state = "CONFIG_COURTS"
                return

            st.markdown("#### Step 4: Court Board Preview")
            st.caption("Use the Court Board controls to reorder players, move between courts, and manage bench.")

            ps = st.session_state.get("ladder_print_sheet", None)
            if isinstance(ps, pd.DataFrame) and not ps.empty:
                csv_bytes = ps.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Court Sheet (CSV)",
                    data=csv_bytes,
                    file_name=f"court_sheet_{st.session_state.get('saved_ladder_lg','league')}_{st.session_state.get('saved_ladder_wk','week')}.csv",
                    mime="text/csv",
                )
                st.dataframe(ps, use_container_width=True, hide_index=True)

            roster_df = compress_courts(normalize_slots(st.session_state.get("ladder_live_roster", pd.DataFrame()).copy()))
            st.session_state.ladder_live_roster = roster_df

            _render_court_board_grid(roster_df, max_per_row=4)

            round_num = int(st.session_state.get("ladder_round_num", 1))

            # Validation
            target_sizes = st.session_state.get("ladder_court_sizes", None)
            v = validate_courts(roster_df, min_players_per_court=4, target_sizes=target_sizes)

            if v["warnings"]:
                st.info("Court sizes don't need to be perfect, but double-check:\n\n- " + "\n- ".join(v["warnings"]))
            if v["problems"]:
                st.warning("Fix these before starting:\n\n- " + "\n- ".join(v["problems"]))

            can_start = bool(v["can_start"])
            total_r = int(st.session_state.get("ladder_total_rounds", 1))
            start_label = "✅ Start Event (Round 1)" if round_num == 1 else f"✅ Start Round {round_num} / {total_r}"

            if st.button(start_label, disabled=not can_start, key=f"start_round_btn_{round_num}"):
                st.session_state.ladder_state = "PLAY_ROUND"
                st.session_state.pop("current_schedule", None)
                st.session_state.pop("current_schedule_round", None)
                _rerun()
                return

        # -------------------------
        # 4) PLAY ROUND (scoring + save + movement)
        # -------------------------
        if st.session_state.get("ladder_state") == "PLAY_ROUND":
            current_r = int(st.session_state.get("ladder_round_num", 1))
            total_r = int(st.session_state.get("ladder_total_rounds", 1))
            st.markdown(f"### 🎾 Round {current_r} / {total_r}")

            roster_now = compress_courts(normalize_slots(st.session_state.get("ladder_live_roster", pd.DataFrame()).copy()))
            st.session_state.ladder_live_roster = roster_now

            # Quick edits
            with st.expander("✏️ Quick court edits (before scoring)", expanded=False):
                roster_df = roster_now.copy()
                names_now = roster_df["name"].astype(str).tolist()
                court_list = sorted(roster_df["court"].astype(int).unique().tolist())

                cA, cB, cC = st.columns([2, 2, 1])
                a = cA.selectbox("Swap Player A", names_now, key=f"swap_a_r{current_r}")
                b = cB.selectbox("with Player B", names_now, key=f"swap_b_r{current_r}", index=1 if len(names_now) > 1 else 0)
                if cC.button("Swap", key=f"swap_btn_r{current_r}"):
                    st.session_state.ladder_live_roster = compress_courts(swap_players(roster_df, a, b))
                    st.session_state.pop("current_schedule", None)
                    _rerun()
                    return

                st.divider()

                c1, c2, c3 = st.columns([2, 2, 1])
                chosen_court = c1.selectbox("Court to reorder", court_list, key=f"re_ct_r{current_r}")
                court_players = roster_df[roster_df["court"].astype(int) == int(chosen_court)].sort_values("slot")["name"].tolist()
                p = c2.selectbox("Player", court_players, key=f"re_p_r{current_r}")
                new_pos = c3.number_input("New position", min_value=1, max_value=max(1, len(court_players)), value=1, step=1, key=f"re_pos_r{current_r}")
                if st.button("Apply reorder", key=f"re_btn_r{current_r}"):
                    st.session_state.ladder_live_roster = compress_courts(move_within_court(roster_df, p, int(new_pos)))
                    st.session_state.pop("current_schedule", None)
                    _rerun()
                    return

                st.divider()

                st.markdown("#### 🔁 Move player to a different court")
                m1, m2, m3, m4 = st.columns([2, 1, 1, 1])
                mv_player = m1.selectbox("Player to move", names_now, key=f"mv_p_r{current_r}")
                target_court = m2.selectbox("To court", court_list, key=f"mv_ct_r{current_r}")
                target_names = roster_df[roster_df["court"].astype(int) == int(target_court)].sort_values("slot")["name"].tolist()
                target_pos = m3.number_input("Insert pos", min_value=1, max_value=max(1, len(target_names) + 1), value=1, step=1, key=f"mv_pos_r{current_r}")
                if m4.button("Move", key=f"mv_btn_r{current_r}"):
                    st.session_state.ladder_live_roster = move_player_to_court(roster_df, mv_player, int(target_court), int(target_pos))
                    st.session_state.pop("current_schedule", None)
                    _rerun()
                    return

            # Build schedule once per round unless roster changed
            if ("current_schedule" not in st.session_state) or (st.session_state.get("current_schedule_round") != current_r):
                schedule = []
                for c_num in sorted(roster_now["court"].astype(int).unique().tolist()):
                    court_df = roster_now[roster_now["court"].astype(int) == int(c_num)].sort_values("slot")
                    pids = court_df["player_id"].astype(int).tolist()
                    fmt = f"{len(pids)}-Player"
                    matches = get_match_schedule(fmt, pids)
                    schedule.append({"c": int(c_num), "matches": matches})
                st.session_state.current_schedule = schedule
                st.session_state.current_schedule_round = current_r

            all_results = []
            with st.form("round_score_form"):
                for c_data in st.session_state.get("current_schedule", []):
                    st.markdown(f"### Court {c_data['c']}")
                    for m_idx, mm in enumerate(c_data["matches"]):
                        label = mm.get("desc", f"Game {m_idx+1}")
                        t1 = mm["t1"]
                        t2 = mm["t2"]

                        p1 = id_to_name.get(int(t1[0]), f"#{t1[0]}")
                        p2 = id_to_name.get(int(t1[1]), f"#{t1[1]}")
                        p3 = id_to_name.get(int(t2[0]), f"#{t2[0]}")
                        p4 = id_to_name.get(int(t2[1]), f"#{t2[1]}")

                        c1, c2, c3, c4 = st.columns([3, 1, 1, 3])
                        c1.text(f"{label}: {p1} & {p2}")
                        s1 = c2.number_input("S1", 0, 99, 0, 1, key=f"s1_r{current_r}_c{c_data['c']}_{m_idx}")
                        s2 = c3.number_input("S2", 0, 99, 0, 1, key=f"s2_r{current_r}_c{c_data['c']}_{m_idx}")
                        c4.text(f"{p3} & {p4}")

                        all_results.append({
                            "t1_p1": int(t1[0]),
                            "t1_p2": int(t1[1]),
                            "t2_p1": int(t2[0]),
                            "t2_p2": int(t2[1]),
                            "s1": int(s1),
                            "s2": int(s2),
                        })

                submitted = st.form_submit_button("Submit Round & Calculate Movement")

            if submitted:
                roster_pids = roster_now["player_id"].astype(int).tolist()
                round_stats = compute_round_stats(all_results, roster_pids)
                max_court = int(roster_now["court"].astype(int).max()) if not roster_now.empty else 1
                movement_df = build_movement_preview(roster_now.copy(), round_stats, max_court=max_court)

                if movement_df is None or movement_df.empty:
                    st.error("Unable to compute movement preview. Check round scores and try again.")
                    st.stop()

                st.session_state.ladder_movement_preview = movement_df
                st.session_state.ladder_state = "CONFIRM_MOVEMENT"
                _rerun()
                st.stop()

        # -------------------------
        # 5) CONFIRM MOVEMENT
        # -------------------------
        if st.session_state.get("ladder_state") == "CONFIRM_MOVEMENT":
            st.markdown("#### Round Results & Movement")

            movement_df = st.session_state.get("ladder_movement_preview", pd.DataFrame())
            if movement_df is None or movement_df.empty:
                st.error("No movement preview found.")
                st.session_state.ladder_state = "PLAY_ROUND"
                _rerun()
                return

            # Display per court
            for c_num in sorted(movement_df["court"].astype(int).unique()):
                st.markdown(f"### Court {int(c_num)} Results")
                c_players = movement_df[movement_df["court"].astype(int) == int(c_num)]

                for _, p in c_players.iterrows():
                    move = "➖"
                    if int(p["Proposed Court"]) < int(p["court"]):
                        move = f"⬆️ To Ct {int(p['Proposed Court'])}"
                    elif int(p["Proposed Court"]) > int(p["court"]):
                        move = f"⬇️ To Ct {int(p['Proposed Court'])}"

                    st.write(f"**{p['name']}** — W:{int(p['Round Wins'])} | Diff:{int(p['Round Diff'])} | Pts:{int(p['Round Pts'])} — {move}")

                st.divider()

            st.markdown("#### 🛠️ Manual Override")
            st.info("If the arrows look right, click Start Next Round. Otherwise edit Proposed Court below.")

            edit_view = movement_df[["player_id", "name", "rating", "court", "Proposed Court"]].copy()
            editor_df = st.data_editor(
                edit_view,
                column_config={
                    "player_id": st.column_config.NumberColumn("ID", disabled=True),
                    "court": st.column_config.NumberColumn("Old Ct", disabled=True),
                    "Proposed Court": st.column_config.NumberColumn("New Ct", min_value=1, max_value=10, step=1),
                },
                hide_index=True,
                use_container_width=True,
            )

            current_r = int(st.session_state.get("ladder_round_num", 1))
            total_r = int(st.session_state.get("ladder_total_rounds", 1))
            btn_label = "Start Next Round" if current_r < total_r else "🏁 Finish League Night"

            base_next_roster = editor_df.copy()
            base_next_roster["court"] = base_next_roster["Proposed Court"].astype(int)
            base_next_roster = base_next_roster.sort_values(["court", "rating"], ascending=[True, False]).copy()
            base_next_roster["slot"] = base_next_roster.groupby("court").cumcount() + 1
            base_next_roster = base_next_roster[["player_id", "name", "rating", "court", "slot"]].copy()

            if st.session_state.get("ladder_next_roster_override") is not None:
                st.info("Roster changes queued for the next round.")
                st.dataframe(
                    _summarize_roster(st.session_state.get("ladder_next_roster_override")),
                    use_container_width=True,
                    hide_index=True,
                )
                if st.session_state.get("ladder_roster_change_note"):
                    st.caption(str(st.session_state.ladder_roster_change_note))
                bench_ids = st.session_state.get("ladder_roster_bench_ids") or []
                if bench_ids:
                    bench_names = [id_to_name.get(int(pid), f"#{pid}") for pid in bench_ids]
                    st.warning(f"Sit-out this round: {', '.join(bench_names)}")

            st.markdown("#### Roster Changes (Between Rounds)")
            st.caption("Roster changes apply to the next round only. Completed rounds stay unchanged.")

            roster_change_enabled, roster_change_msg = roster_change_availability(
                ladder_state=st.session_state.get("ladder_state"),
                current_round=current_r,
                total_rounds=total_r,
                is_admin=bool(ctx.admin_logged_in),
            )
            if st.button(
                "Substitute / Add Player",
                key="ladder_roster_change_btn",
                disabled=not roster_change_enabled,
            ):
                st.session_state.ladder_show_roster_change_dialog = True
            if not roster_change_enabled and roster_change_msg:
                st.caption(roster_change_msg)

            if st.session_state.get("ladder_show_roster_change_dialog", False):
                @st.dialog("Roster Changes (Next Round)")
                def roster_changes_dialog() -> None:
                    required_cols = {"name", "player_id", "rating"}
                    if movement_df is None or movement_df.empty or not required_cols.issubset(set(movement_df.columns)):
                        st.error(
                            "Roster changes are unavailable because the movement preview is missing required data "
                            "(name, player_id, rating). Please regenerate movement preview and try again."
                        )
                        st.stop()

                    next_round = int(current_r + 1)
                    final_round = int(total_r)
                    tabs_rc = st.tabs(["Substitute", "Add Player"])

                    close_cols = st.columns([1, 1, 6])
                    with close_cols[0]:
                        if st.button("Close", key="ladder_roster_change_close"):
                            st.session_state.ladder_show_roster_change_dialog = False
                            return

                    def _player_options() -> tuple[list[str], dict[str, dict]]:
                        rows = ctx.df_players_all if ctx.df_players_all is not None else pd.DataFrame()
                        options = []
                        mapping = {}
                        if rows is None or rows.empty:
                            return options, mapping
                        for _, row in rows.iterrows():
                            pid = int(row.get("id"))
                            nm = str(row.get("name", "")).strip()
                            if not nm:
                                continue
                            label = f"{nm} (#{pid})"
                            options.append(label)
                            mapping[label] = {
                                "id": pid,
                                "name": nm,
                                "rating": float(row.get("rating", 1200.0) or 1200.0),
                            }
                        return options, mapping

                    options, option_map = _player_options()

                    with tabs_rc[0]:
                        st.markdown("Replace player (active roster):")
                        active_names = movement_df["name"].astype(str).tolist()
                        active_map = {
                            str(r["name"]): {
                                "id": int(r["player_id"]),
                                "name": str(r["name"]),
                                "rating": float(r.get("rating", 1200.0)),
                            }
                            for _, r in movement_df.iterrows()
                        }
                        replaced_name = st.selectbox("Replace player", active_names, key="ladder_sub_out")
                        replace_info = active_map.get(replaced_name, {})

                        st.markdown("Substitute with:")
                        use_guest_sub = st.checkbox("Create guest substitute", value=False, key="ladder_sub_guest")
                        if use_guest_sub:
                            guest_name = st.text_input("Guest name", key="ladder_sub_guest_name")
                            guest_rating = st.number_input(
                                "Guest starting JUPR", min_value=1.0, max_value=7.0, step=0.1, value=3.5, key="ladder_sub_guest_rating"
                            )
                            sub_player = {"name": guest_name.strip(), "rating": float(guest_rating)}
                        else:
                            sub_label = st.selectbox("Select substitute", options, key="ladder_sub_in")
                            sub_player = option_map.get(sub_label, {})

                        st.text(f"Effective round: {next_round}")
                        st.warning(
                            f"This will regenerate matchups for rounds {next_round}–{final_round}. Completed rounds will not change."
                        )
                        confirm_sub = st.checkbox("I understand and want to apply this substitute.", key="ladder_sub_confirm")

                        if st.button("Apply Substitute", type="primary", key="ladder_sub_apply"):
                            if not confirm_sub:
                                st.error("Please confirm the roster change before applying.")
                                st.stop()
                            try:
                                if use_guest_sub:
                                    if not sub_player.get("name"):
                                        raise RosterChangeError("Guest name is required.")
                                    normalized_name = " ".join(str(sub_player["name"]).strip().lower().split())
                                    payload = {
                                        "club_id": str(ctx.club_id),
                                        "name": str(sub_player["name"]).strip(),
                                        "normalized_name": normalized_name,
                                        "rating": float(jupr_to_elo(sub_player.get("rating", 3.5))),
                                    }
                                    ok, player_row, err = get_or_create_player(
                                        supabase=ctx.supabase,
                                        club_id=str(ctx.club_id),
                                        normalized_name=normalized_name,
                                        payload=payload,
                                    )
                                    if not ok:
                                        raise RosterChangeError(f"Could not add guest: {err}")
                                    if not isinstance(player_row, dict):
                                        raise RosterChangeError("Guest player was created but could not be loaded.")
                                    sub_player = {
                                        "id": int(player_row["id"]),
                                        "name": str(player_row["name"]),
                                        "rating": float(player_row.get("rating", 1200.0) or 1200.0),
                                    }
                                    id_to_name[int(player_row["id"])] = str(player_row["name"])
                                    name_to_id[str(player_row["name"])] = int(player_row["id"])

                                result = apply_roster_change(
                                    roster_df=base_next_roster,
                                    change_type="substitute",
                                    replaced_player_id=int(replace_info.get("id")),
                                    new_player=sub_player,
                                    court_sizes=st.session_state.get("ladder_court_sizes"),
                                    roster_locked=False,
                                )
                                st.session_state.ladder_next_roster_override = result.roster_df
                                st.session_state.ladder_next_court_sizes = result.court_sizes
                                st.session_state.ladder_roster_change_note = result.note
                                st.session_state.ladder_roster_bench_ids = result.bench_ids
                                st.session_state.ladder_show_roster_change_dialog = False
                                st.success("Substitution queued for next round.")
                                return
                            except RosterChangeError as exc:
                                st.error(str(exc))

                    with tabs_rc[1]:
                        st.markdown("Add player (late arrival):")
                        use_guest_add = st.checkbox("Create guest player", value=False, key="ladder_add_guest")
                        if use_guest_add:
                            guest_name = st.text_input("Guest name", key="ladder_add_guest_name")
                            guest_rating = st.number_input(
                                "Guest starting JUPR",
                                min_value=1.0,
                                max_value=7.0,
                                step=0.1,
                                value=3.5,
                                key="ladder_add_guest_rating",
                            )
                            add_player = {"name": guest_name.strip(), "rating": float(guest_rating)}
                        else:
                            add_label = st.selectbox("Select player", options, key="ladder_add_player")
                            add_player = option_map.get(add_label, {})

                        st.text(f"Effective round: {next_round}")
                        st.warning(
                            f"This will regenerate matchups for rounds {next_round}–{final_round}. Completed rounds will not change."
                        )
                        confirm_add = st.checkbox("I understand and want to add this player.", key="ladder_add_confirm")

                        if st.button("Apply Add Player", type="primary", key="ladder_add_apply"):
                            if not confirm_add:
                                st.error("Please confirm the roster change before applying.")
                                st.stop()
                            try:
                                if use_guest_add:
                                    if not add_player.get("name"):
                                        raise RosterChangeError("Guest name is required.")
                                    normalized_name = " ".join(str(add_player["name"]).strip().lower().split())
                                    payload = {
                                        "club_id": str(ctx.club_id),
                                        "name": str(add_player["name"]).strip(),
                                        "normalized_name": normalized_name,
                                        "rating": float(jupr_to_elo(add_player.get("rating", 3.5))),
                                    }
                                    ok, player_row, err = get_or_create_player(
                                        supabase=ctx.supabase,
                                        club_id=str(ctx.club_id),
                                        normalized_name=normalized_name,
                                        payload=payload,
                                    )
                                    if not ok:
                                        raise RosterChangeError(f"Could not add guest: {err}")
                                    if not isinstance(player_row, dict):
                                        raise RosterChangeError("Guest player was created but could not be loaded.")
                                    add_player = {
                                        "id": int(player_row["id"]),
                                        "name": str(player_row["name"]),
                                        "rating": float(player_row.get("rating", 1200.0) or 1200.0),
                                    }
                                    id_to_name[int(player_row["id"])] = str(player_row["name"])
                                    name_to_id[str(player_row["name"])] = int(player_row["id"])

                                result = apply_roster_change(
                                    roster_df=base_next_roster,
                                    change_type="add",
                                    new_player=add_player,
                                    court_sizes=st.session_state.get("ladder_court_sizes"),
                                    roster_locked=False,
                                )
                                st.session_state.ladder_next_roster_override = result.roster_df
                                st.session_state.ladder_next_court_sizes = result.court_sizes
                                st.session_state.ladder_roster_change_note = result.note
                                st.session_state.ladder_roster_bench_ids = result.bench_ids
                                st.session_state.ladder_show_roster_change_dialog = False
                                st.success("Player added for next round.")
                                return
                            except RosterChangeError as exc:
                                st.error(str(exc))

                roster_changes_dialog()

            if st.button(btn_label):
                if current_r >= total_r:
                    st.success("League Night Complete! All matches saved.")
                    for k in [
                        "ladder_round_num",
                        "ladder_live_roster",
                        "ladder_court_sizes",
                        "ladder_movement_preview",
                        "current_schedule",
                        "current_schedule_round",
                        "ladder_temp_roster",
                        "ladder_temp_new",
                        "ladder_roster",
                        "ladder_print_sheet",
                    ]:
                        st.session_state.pop(k, None)
                    st.session_state.ladder_state = "SETUP"
                    _rerun()
                    return

                override = st.session_state.get("ladder_next_roster_override")
                if override is None:
                    new_roster = base_next_roster
                else:
                    if hasattr(override, "empty"):
                        new_roster = override if not override.empty else base_next_roster
                    else:
                        try:
                            new_roster = override if len(override) > 0 else base_next_roster
                        except Exception:
                            new_roster = override
                new_court_sizes = st.session_state.get("ladder_next_court_sizes") or [
                    int(x) for x in new_roster["court"].value_counts().sort_index().tolist()
                ]

                st.session_state.ladder_live_roster = new_roster.copy()
                st.session_state.ladder_court_sizes = new_court_sizes
                st.session_state.pop("ladder_next_roster_override", None)
                st.session_state.pop("ladder_next_court_sizes", None)
                st.session_state.pop("ladder_roster_change_note", None)
                st.session_state.pop("ladder_roster_bench_ids", None)

                st.session_state.ladder_round_num = current_r + 1
                st.session_state.ladder_state = "CONFIRM_START"
                st.session_state.pop("current_schedule", None)
                _rerun()
                return

    # ============================================================
    # TAB 2: SETTINGS
    # ============================================================
    with tabs[1]:
        st.subheader("Premium League Editor")

        with st.expander("➕ Create New League", expanded=False):
            default_k = int(DEFAULT_K_FACTOR) if DEFAULT_K_FACTOR is not None else 32
            league_name = st.text_input("League name (required)", key="create_league_name")
            description = st.text_area("Description (optional)", key="create_league_description")
            min_games = st.number_input("Minimum games", min_value=0, step=1, value=6, key="create_league_min_games")
            k_factor = st.number_input(
                "K-factor",
                min_value=1,
                step=1,
                value=default_k,
                key="create_league_k_factor",
            )

            if st.button("Create Draft", type="primary"):
                trimmed_name = str(league_name or "").strip()
                if not trimmed_name:
                    st.error("League name is required.")
                    st.stop()

                normalized_input = trimmed_name.lower()
                if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
                    existing = (
                        df_meta["league_name"]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .tolist()
                    )
                    if normalized_input in existing:
                        st.error("A league with that name already exists for this club.")
                        st.stop()

                payload = {
                    "club_id": str(ctx.club_id),
                    "league_name": trimmed_name,
                    "description": str(description or "").strip(),
                    "min_games": int(min_games or 0),
                    "k_factor": int(k_factor or default_k),
                    "is_active": False,
                    "status": "draft",
                }

                try:
                    resp = sb_insert(ctx.supabase, "leagues_metadata", payload)
                except Exception as exc:
                    st.error(f"Could not create league: {exc}")
                    st.stop()

                if getattr(resp, "error", None):
                    st.error(f"Could not create league: {resp.error}")
                    st.stop()

                return

        if df_meta is None or df_meta.empty:
            st.info("No league metadata loaded.")
            return

        league_names = (
            df_meta["league_name"].dropna().astype(str).str.strip().unique().tolist()
            if "league_name" in df_meta.columns
            else []
        )
        if not league_names:
            st.info("No leagues available.")
            return

        league_names = sorted({name for name in league_names if name})
        selected_league = st.selectbox("League", league_names, key="league_editor_select")
        meta_row = get_league_meta_row(df_meta, selected_league) or {}
        status = normalize_league_status(meta_row)
        status_label = status.title()
        st.markdown(f"**Status:** {status_label}")
        if status == "active":
            st.warning("Active leagues are locked to safe edits only (description and awards visibility).")
        if status in {"ended", "archived"}:
            st.info("Ended leagues are read-only, except for archiving and award review.")

        schedule_cfg = _safe_json_load(meta_row.get("schedule_config"), {})
        court_cfg = _safe_json_load(meta_row.get("court_board_defaults"), {})
        rules_cfg = _safe_json_load(meta_row.get("rules_config"), {})
        awards_cfg = _safe_json_load(meta_row.get("awards_config"), {})
        overview_cfg = rules_cfg.get("overview", {}) if isinstance(rules_cfg, dict) else {}
        competition_cfg = rules_cfg.get("competition", {}) if isinstance(rules_cfg, dict) else {}

        def _update_league(payload: dict) -> None:
            if not meta_row.get("id"):
                st.error("League metadata ID is missing.")
                return
            sb_update(
                ctx.supabase,
                "leagues_metadata",
                payload,
                filters={"id": int(meta_row["id"]), "club_id": str(ctx.club_id)},
            )
            return

        def _build_payload(
            *,
            status_override: str | None = None,
            is_active_override: bool | None = None,
            started_at_override: str | None = None,
        ) -> dict:
            min_games_val = int(st.session_state.get("le_min_games", 0) or 0)
            k_factor_val = int(st.session_state.get("le_k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR)
            divisions_raw = str(st.session_state.get("le_divisions", "") or "")
            divisions = [d.strip() for d in divisions_raw.split(",") if d.strip()]
            rules_payload = {
                "overview": {
                    "league_type": str(st.session_state.get("le_type", "") or "").strip(),
                    "divisions": divisions,
                    "summary": str(st.session_state.get("le_summary", "") or "").strip(),
                },
                "competition": {
                    "scoring_rules": str(st.session_state.get("le_scoring", "") or "").strip(),
                    "match_format": str(st.session_state.get("le_match_format", "") or "").strip(),
                    "tie_break_rules": str(st.session_state.get("le_tie_break", "") or "").strip(),
                    "dispute_window": str(st.session_state.get("le_dispute_window", "") or "").strip(),
                    "dispute_policy": str(st.session_state.get("le_dispute_policy", "") or "").strip(),
                },
            }
            end_date_value = st.session_state.get("le_end_date") if st.session_state.get("le_use_end_date") else ""
            schedule_payload = {
                "start_date": str(st.session_state.get("le_start_date") or ""),
                "weeks": int(st.session_state.get("le_weeks") or 0) or None,
                "end_date": str(end_date_value or ""),
                "weekday": int(st.session_state.get("le_weekday") or 0),
                "time_start": str(st.session_state.get("le_time_start") or ""),
                "time_end": str(st.session_state.get("le_time_end") or ""),
                "timezone": str(st.session_state.get("le_timezone") or ""),
                "blackout_dates": _parse_blackout_dates(st.session_state.get("le_blackouts", "") or ""),
                "session_capacity": int(st.session_state.get("le_capacity") or 0) or None,
            }
            court_payload = {
                "total_courts": int(st.session_state.get("le_total_courts") or 0),
                "court_identifiers": [
                    c.strip()
                    for c in str(st.session_state.get("le_court_ids", "") or "").split(",")
                    if c.strip()
                ],
                "max_used_courts": int(st.session_state.get("le_max_used_courts") or 0),
                "players_per_court": str(st.session_state.get("le_players_per_court") or ""),
                "rotation_mode": str(st.session_state.get("le_rotation_mode") or ""),
                "game_format_points": int(st.session_state.get("le_game_format_points") or 0),
                "game_format_time": int(st.session_state.get("le_game_format_time") or 0),
            }
            categories = {}
            award_defaults = int(st.session_state.get("le_award_depth", 1) or 1)
            for key in ["highest_rating", "most_improved", "best_win_pct", "most_wins"]:
                categories[key] = {
                    "enabled": bool(st.session_state.get(f"award_{key}_enabled", True)),
                    "min_games": int(st.session_state.get(f"award_{key}_min_games", min_games_val) or min_games_val),
                    "depth": int(st.session_state.get(f"award_{key}_depth", award_defaults) or award_defaults),
                }
            awards_payload = {
                "default_min_games": min_games_val,
                "default_depth": award_defaults,
                "categories": categories,
            }
            payload = {
                "description": str(st.session_state.get("le_desc", "") or ""),
                "min_games": min_games_val,
                "k_factor": k_factor_val,
                "schedule_config": schedule_payload,
                "court_board_defaults": court_payload,
                "rules_config": rules_payload,
                "awards_config": awards_payload,
            }
            if status_override is not None:
                payload["status"] = status_override
            if is_active_override is not None:
                payload["is_active"] = is_active_override
            if started_at_override is not None:
                payload["started_at"] = started_at_override
            return payload

        action_cols = st.columns(5)
        if action_cols[0].button("Save Draft", disabled=status not in {"draft", "active"}):
            payload = _build_payload(status_override="draft", is_active_override=False)
            _update_league(payload)
        if action_cols[1].button("Publish/Start League", disabled=status in {"ended", "archived"}):
            started_at = meta_row.get("started_at") or _utc_iso_now()
            payload = _build_payload(status_override="active", is_active_override=True, started_at_override=started_at)
            _update_league(payload)
        if action_cols[2].button("End League", disabled=status in {"ended", "archived"}):
            st.session_state["end_league_wizard_open"] = True
        if action_cols[3].button("Archive", disabled=status != "ended"):
            payload = _build_payload(status_override="archived", is_active_override=False)
            _update_league(payload)
        duplicate_name = action_cols[4].text_input("Duplicate as", value=f"{selected_league} Copy", key="dup_name")
        if action_cols[4].button("Duplicate", type="secondary"):
            payload = {
                "club_id": str(ctx.club_id),
                "league_name": str(duplicate_name or "").strip() or f"{selected_league} Copy",
                "description": str(meta_row.get("description") or ""),
                "min_games": int(meta_row.get("min_games") or 0),
                "k_factor": int(meta_row.get("k_factor") or DEFAULT_K_FACTOR),
                "is_active": False,
                "status": "draft",
                "schedule_config": schedule_cfg,
                "court_board_defaults": court_cfg,
                "rules_config": rules_cfg,
                "awards_config": awards_cfg,
            }
            sb_insert(ctx.supabase, "leagues_metadata", payload)
            return

        tabs_editor = st.tabs(
            [
                "Overview",
                "Schedule",
                "Courts & Court Board Defaults",
                "Competition Format & Rules",
                "Ratings & Eligibility",
                "Awards & Trophies",
            ]
        )

        with tabs_editor[0]:
            st.text_input("League name", value=selected_league, disabled=True)
            st.text_area(
                "Description",
                value=str(meta_row.get("description") or ""),
                key="le_desc",
                disabled=status in {"ended", "archived"},
            )
            st.text_input(
                "League type",
                value=str(overview_cfg.get("league_type") or ""),
                key="le_type",
                disabled=status != "draft",
            )
            st.text_input(
                "Divisions (comma separated)",
                value=", ".join(overview_cfg.get("divisions") or []),
                key="le_divisions",
                disabled=status != "draft",
            )
            st.text_area(
                "Summary",
                value=str(overview_cfg.get("summary") or ""),
                key="le_summary",
                disabled=status != "draft",
            )

        with tabs_editor[1]:
            schedule_start = _parse_date(schedule_cfg.get("start_date")) or date.today()
            schedule_end = _parse_date(schedule_cfg.get("end_date"))
            st.date_input(
                "Start date",
                value=schedule_start,
                key="le_start_date",
                disabled=status != "draft",
            )
            st.number_input(
                "Weeks (optional)",
                min_value=0,
                step=1,
                value=int(schedule_cfg.get("weeks") or 0),
                key="le_weeks",
                disabled=status != "draft",
            )
            st.checkbox(
                "Use end date",
                value=bool(schedule_end),
                key="le_use_end_date",
                disabled=status != "draft",
            )
            st.date_input(
                "End date (optional)",
                value=schedule_end or date.today(),
                key="le_end_date",
                disabled=status != "draft",
            )
            weekday_val = int(schedule_cfg.get("weekday") or 0)
            weekday_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            st.selectbox(
                "Weekday",
                list(range(7)),
                format_func=lambda idx: weekday_name[idx],
                index=weekday_val,
                key="le_weekday",
                disabled=status != "draft",
            )
            st.text_input(
                "Time window start (HH:MM)",
                value=str(schedule_cfg.get("time_start") or "18:00"),
                key="le_time_start",
                disabled=status != "draft",
            )
            st.text_input(
                "Time window end (HH:MM)",
                value=str(schedule_cfg.get("time_end") or "20:00"),
                key="le_time_end",
                disabled=status != "draft",
            )
            st.text_input(
                "Timezone",
                value=str(schedule_cfg.get("timezone") or "UTC"),
                key="le_timezone",
                disabled=status != "draft",
            )
            blackout_text = ", ".join(schedule_cfg.get("blackout_dates") or [])
            st.text_area(
                "Blackout dates (comma or newline separated)",
                value=blackout_text,
                key="le_blackouts",
                disabled=status != "draft",
            )
            st.number_input(
                "Session capacity",
                min_value=0,
                step=1,
                value=int(schedule_cfg.get("session_capacity") or 0),
                key="le_capacity",
                disabled=status != "draft",
            )

            preview_end_date = st.session_state.get("le_end_date") if st.session_state.get("le_use_end_date") else ""
            preview_cfg = {
                "start_date": st.session_state.get("le_start_date"),
                "weeks": st.session_state.get("le_weeks"),
                "end_date": preview_end_date,
                "weekday": st.session_state.get("le_weekday"),
                "time_start": st.session_state.get("le_time_start"),
                "time_end": st.session_state.get("le_time_end"),
                "timezone": st.session_state.get("le_timezone"),
                "blackout_dates": _parse_blackout_dates(st.session_state.get("le_blackouts", "")),
            }
            preview_df = _build_schedule_preview(preview_cfg)
            if not preview_df.empty:
                st.dataframe(preview_df, hide_index=True, use_container_width=True)
                ics_content = _schedule_to_ics(preview_cfg)
                if ics_content:
                    st.download_button("Download ICS", data=ics_content, file_name="league_schedule.ics")
            else:
                st.caption("Fill out schedule details to preview sessions.")

        with tabs_editor[2]:
            st.number_input(
                "Total courts",
                min_value=0,
                step=1,
                value=int(court_cfg.get("total_courts") or 0),
                key="le_total_courts",
                disabled=status != "draft",
            )
            st.text_input(
                "Court identifiers (comma separated)",
                value=", ".join(court_cfg.get("court_identifiers") or []),
                key="le_court_ids",
                disabled=status != "draft",
            )
            st.number_input(
                "Max used courts",
                min_value=0,
                step=1,
                value=int(court_cfg.get("max_used_courts") or 0),
                key="le_max_used_courts",
                disabled=status != "draft",
            )
            st.selectbox(
                "Players per court preference",
                ["4", "5", "6+"],
                index=["4", "5", "6+"].index(str(court_cfg.get("players_per_court") or "4")),
                key="le_players_per_court",
                disabled=status != "draft",
            )
            st.selectbox(
                "Rotation mode",
                ["fixed", "queue"],
                index=["fixed", "queue"].index(str(court_cfg.get("rotation_mode") or "fixed")),
                key="le_rotation_mode",
                disabled=status != "draft",
            )
            st.number_input(
                "Game format points cap",
                min_value=1,
                step=1,
                value=int(court_cfg.get("game_format_points") or 11),
                key="le_game_format_points",
                disabled=status != "draft",
            )
            st.number_input(
                "Game format time cap (minutes)",
                min_value=1,
                step=1,
                value=int(court_cfg.get("game_format_time") or 15),
                key="le_game_format_time",
                disabled=status != "draft",
            )

        with tabs_editor[3]:
            st.text_area(
                "Scoring rules",
                value=str(competition_cfg.get("scoring_rules") or ""),
                key="le_scoring",
                disabled=status != "draft",
            )
            st.text_area(
                "Match format",
                value=str(competition_cfg.get("match_format") or ""),
                key="le_match_format",
                disabled=status != "draft",
            )
            st.text_area(
                "Tie-break rules",
                value=str(competition_cfg.get("tie_break_rules") or ""),
                key="le_tie_break",
                disabled=status != "draft",
            )
            st.text_input(
                "Dispute window",
                value=str(competition_cfg.get("dispute_window") or ""),
                key="le_dispute_window",
                disabled=status != "draft",
            )
            st.text_input(
                "Who can submit disputes",
                value=str(competition_cfg.get("dispute_policy") or ""),
                key="le_dispute_policy",
                disabled=status != "draft",
            )

        with tabs_editor[4]:
            st.number_input(
                "Minimum games",
                min_value=0,
                step=1,
                value=int(meta_row.get("min_games") or 0),
                key="le_min_games",
                disabled=status != "draft",
            )
            st.number_input(
                "K-factor",
                min_value=1,
                step=1,
                value=int(meta_row.get("k_factor") or DEFAULT_K_FACTOR),
                key="le_k_factor",
                disabled=status != "draft",
            )

        with tabs_editor[5]:
            st.number_input(
                "Award depth (top 1 vs top 3)",
                min_value=1,
                max_value=3,
                step=2,
                value=int(awards_cfg.get("default_depth") or 1),
                key="le_award_depth",
                disabled=status != "draft",
            )
            for key, label in [
                ("highest_rating", "Highest Rating"),
                ("most_improved", "Most Improved"),
                ("best_win_pct", "Best Win %"),
                ("most_wins", "Most Wins"),
            ]:
                cat_cfg = (awards_cfg.get("categories") or {}).get(key, {})
                st.checkbox(
                    f"{label} enabled",
                    value=bool(cat_cfg.get("enabled", True)),
                    key=f"award_{key}_enabled",
                    disabled=status != "draft",
                )
                st.number_input(
                    f"{label} min games",
                    min_value=0,
                    step=1,
                    value=int(cat_cfg.get("min_games") or meta_row.get("min_games") or 0),
                    key=f"award_{key}_min_games",
                    disabled=status != "draft",
                )
                st.number_input(
                    f"{label} award depth",
                    min_value=1,
                    max_value=3,
                    step=2,
                    value=int(cat_cfg.get("depth") or awards_cfg.get("default_depth") or 1),
                    key=f"award_{key}_depth",
                    disabled=status != "draft",
                )

        wizard_open = st.session_state.get("end_league_wizard_open", False)
        with st.expander("🏁 End League Wizard", expanded=wizard_open):
            step = int(st.session_state.get("end_league_step", 1))
            st.markdown(f"**Step {step} of 5**")

            if step == 1:
                st.info("Freezing will mark the league as ended and lock settings.")
                if st.button("Freeze & Continue", type="primary"):
                    ended_at = _utc_iso_now()
                    payload = _build_payload(status_override="ended", is_active_override=False)
                    payload["ended_at"] = ended_at
                    payload["ended_by"] = "admin"
                    _update_league(payload)
                    st.session_state["end_league_frozen_at"] = ended_at
                    st.session_state["end_league_step"] = 2
                    st.session_state["end_league_wizard_open"] = True
                    _rerun()
                    return

            elif step == 2:
                awards = compute_top_performer_awards_for_config(
                    df_leagues,
                    df_meta,
                    id_to_name,
                    selected_league,
                    awards_config=_build_payload().get("awards_config"),
                )
                st.session_state["end_league_awards"] = awards
                if awards:
                    preview_rows = [
                        {
                            "Category": award.get("category_label") or award.get("category_key"),
                            "Rank": award.get("rank"),
                            "Player": award.get("player_name") or award.get("player_id"),
                            "Metric": award.get("metric_display"),
                        }
                        for award in awards
                    ]
                    st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)
                    if st.button("Continue to Overrides"):
                        st.session_state["end_league_step"] = 3
                        st.session_state["end_league_wizard_open"] = True
                        _rerun()
                        return
                else:
                    st.warning("No winners found. Check eligibility rules or standings.")

            elif step == 3:
                awards = st.session_state.get("end_league_awards", [])
                league_players = pd.DataFrame()
                if df_leagues is not None and not df_leagues.empty and "league_name" in df_leagues.columns:
                    league_players = df_leagues[df_leagues["league_name"].astype(str) == str(selected_league)].copy()
                player_ids = sorted(league_players.get("player_id", pd.Series(dtype="int")).dropna().astype(int).unique().tolist())
                if not awards:
                    st.info("No awards to override.")
                if not player_ids:
                    st.warning("No players found for this league.")
                for award in awards:
                    category_key = award.get("category_key")
                    rank = int(award.get("rank") or 1)
                    override_key = f"{category_key}:{rank}"
                    st.markdown(f"**{award.get('category_label')} #{rank}**")
                    if player_ids:
                        default_player = (
                            int(award.get("player_id")) if award.get("player_id") is not None else player_ids[0]
                        )
                        st.selectbox(
                            "Winner",
                            player_ids,
                            index=player_ids.index(default_player) if default_player in player_ids else 0,
                            format_func=lambda pid: f"{id_to_name.get(int(pid), pid)} (#{pid})",
                            key=f"override_player_{override_key}",
                        )
                    st.text_area(
                        "Override note (required if changed)",
                        key=f"override_note_{override_key}",
                    )
                if st.button("Continue to Confirm"):
                    st.session_state["end_league_step"] = 4
                    st.session_state["end_league_wizard_open"] = True
                    _rerun()
                    return

            elif step == 4:
                awards = st.session_state.get("end_league_awards", [])
                if not awards:
                    st.info("No awards to confirm.")
                override_notes = {}
                final_awards = []
                for award in awards:
                    category_key = award.get("category_key")
                    rank = int(award.get("rank") or 1)
                    override_key = f"{category_key}:{rank}"
                    selected_pid = st.session_state.get(f"override_player_{override_key}", award.get("player_id"))
                    note = str(st.session_state.get(f"override_note_{override_key}", "") or "").strip()
                    if int(selected_pid) != int(award.get("player_id")) and not note:
                        st.error(f"Override note required for {award.get('category_label')} #{rank}.")
                        st.stop()
                    if note:
                        override_notes[override_key] = note
                    updated_award = dict(award)
                    updated_award["player_id"] = int(selected_pid)
                    updated_award["player_name"] = id_to_name.get(int(selected_pid), str(selected_pid))
                    final_awards.append(updated_award)
                st.dataframe(pd.DataFrame(final_awards), hide_index=True, use_container_width=True)
                if st.button("Confirm & Mint Badges", type="primary"):
                    ended_at = meta_row.get("ended_at") or st.session_state.get("end_league_frozen_at") or _utc_iso_now()
                    created = mint_top_performer_badges(
                        ctx.supabase,
                        club_id=str(ctx.club_id),
                        league_id=str(selected_league),
                        awards=final_awards,
                        ended_at=ended_at,
                        override_notes=override_notes,
                    )
                    st.success(f"Minted {len(created)} top performer badges.")
                    st.session_state["end_league_step"] = 5
                    st.session_state["end_league_wizard_open"] = True
                    _rerun()
                    return

            elif step == 5:
                st.success("League closed. You can archive it now.")
                if st.button("Archive league", type="secondary"):
                    payload = _build_payload(status_override="archived", is_active_override=False)
                    _update_league(payload)

        with st.expander("Advanced (Legacy Grid)", expanded=False):
            cols = [
                c
                for c in ["id", "league_name", "is_active", "min_games", "description", "k_factor"]
                if c in df_meta.columns
            ]
            editor = st.data_editor(
                df_meta[cols],
                disabled=["id", "league_name"],
                hide_index=True,
                use_container_width=True,
            )

            if st.button("Save Config"):
                for _, r in editor.iterrows():
                    sb_update(
                        ctx.supabase,
                        "leagues_metadata",
                        {
                            "is_active": bool(r.get("is_active", True)),
                            "min_games": int(r.get("min_games", 0) or 0),
                            "k_factor": int(r.get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR),
                            "description": str(r.get("description", "") or ""),
                        },
                        filters={"id": int(r["id"]), "club_id": str(ctx.club_id)},
                    )

                st.success("Saved.")
                return
