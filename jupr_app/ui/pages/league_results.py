from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st

from jupr_app.ui.layout import page_shell
from jupr_app.ui.pages.leaderboards import _build_player_link
from jupr_app.ui.url import qp_get

try:
    import altair as alt
except Exception:
    alt = None

try:
    from jupr_app.domain.ratings import calculate_hybrid_elo
    from jupr_app.domain.constants import DEFAULT_K_FACTOR, MIN_WIN_DELTA_ELO, CAP_LOSER_GAIN_ELO
    _RATING_REPLAY_AVAILABLE = True
except Exception:
    calculate_hybrid_elo = None
    DEFAULT_K_FACTOR = 32
    MIN_WIN_DELTA_ELO = 1.0
    CAP_LOSER_GAIN_ELO = 16.0
    _RATING_REPLAY_AVAILABLE = False


def _safe_int(value, default=None):
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(value)
    except Exception:
        return default


def _safe_float(value, default=None):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _parse_week_num(week_tag: str) -> int | None:
    if week_tag is None:
        return None
    match = re.search(r"(\d+)", str(week_tag))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _coerce_week_value(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, (float,)):
        try:
            return int(value)
        except Exception:
            return None
    return _parse_week_num(value)


def _build_league_weeks(
    df_meta: pd.DataFrame | None, league_name: str, league_matches: pd.DataFrame
) -> pd.DataFrame:
    week_nums: list[int] = []
    if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
        meta = df_meta.copy()
        meta["league_name"] = meta["league_name"].fillna("").astype(str).str.strip()
        hit = meta[meta["league_name"] == str(league_name).strip()]
        if not hit.empty:
            row = hit.iloc[0]
            start_week = _coerce_week_value(row.get("start_week", row.get("week_start")))
            end_week = _coerce_week_value(row.get("end_week", row.get("week_end")))
            if start_week is not None and end_week is not None and end_week >= start_week and start_week > 0:
                week_nums = list(range(int(start_week), int(end_week) + 1))
            else:
                total_weeks = _coerce_week_value(row.get("num_weeks", row.get("total_weeks", row.get("weeks"))))
                if total_weeks is not None and int(total_weeks) > 0:
                    week_nums = list(range(1, int(total_weeks) + 1))

    if not week_nums and league_matches is not None and not league_matches.empty:
        match_weeks = league_matches["week_num"].dropna().astype(int).unique().tolist()
        if match_weeks:
            week_nums = list(range(min(match_weeks), max(match_weeks) + 1))

    weeks_df = pd.DataFrame({"week_num": week_nums})
    if weeks_df.empty:
        return weeks_df
    weeks_df["week_label"] = weeks_df["week_num"].apply(lambda x: f"Week {int(x)}")
    return weeks_df


def _week_label_order(weeks_df: pd.DataFrame, data_df: pd.DataFrame | None = None) -> list[str]:
    if weeks_df is not None and not weeks_df.empty and "week_label" in weeks_df.columns:
        return weeks_df["week_label"].tolist()
    if data_df is None or data_df.empty or "week_num" not in data_df.columns:
        return []
    week_nums = (
        data_df.dropna(subset=["week_num"])
        .assign(week_num=lambda d: d["week_num"].astype(int))
        .sort_values("week_num")["week_num"]
        .unique()
        .tolist()
    )
    return [f"Week {int(x)}" for x in week_nums]


def _attach_week_label(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    data = df.copy()
    if "week_label" not in data.columns:
        data["week_label"] = data["week_num"].apply(lambda x: f"Week {int(x)}" if pd.notna(x) else None)
    return data


def _league_options(df_meta: pd.DataFrame | None, df_matches: pd.DataFrame | None, df_leagues: pd.DataFrame | None) -> list[str]:
    leagues = []
    if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
        meta = df_meta.copy()
        meta["league_name"] = meta["league_name"].fillna("").astype(str).str.strip()
        if "is_active" in meta.columns:
            meta = meta[meta["is_active"] == True]
        leagues = [x for x in meta["league_name"].dropna().tolist() if x and x.upper() != "OVERALL"]
    if not leagues and df_leagues is not None and not df_leagues.empty and "league_name" in df_leagues.columns:
        leagues = df_leagues["league_name"].fillna("").astype(str).str.strip().unique().tolist()
        leagues = [x for x in leagues if x and x.upper() != "OVERALL"]
    if not leagues and df_matches is not None and not df_matches.empty and "league" in df_matches.columns:
        leagues = df_matches["league"].fillna("").astype(str).str.strip().unique().tolist()
        leagues = [x for x in leagues if x and x.upper() != "OVERALL"]
    return sorted(set(leagues))


def _filter_league_matches(df_matches: pd.DataFrame, league_name: str) -> pd.DataFrame:
    df = df_matches.copy()
    df["league"] = df.get("league", "").fillna("").astype(str).str.strip()
    df["match_type"] = df.get("match_type", "").fillna("").astype(str).str.strip()
    df = df[df["league"] == str(league_name).strip()].copy()
    if "match_type" in df.columns:
        df = df[df["match_type"] != "PopUp"].copy()
    df["score_t1"] = pd.to_numeric(df.get("score_t1", 0), errors="coerce").fillna(0).astype(int)
    df["score_t2"] = pd.to_numeric(df.get("score_t2", 0), errors="coerce").fillna(0).astype(int)
    df = df[(df["score_t1"] + df["score_t2"]) > 0].copy()
    df["week_tag"] = df.get("week_tag", "").fillna("").astype(str)
    df["week_num"] = df["week_tag"].map(_parse_week_num)
    df["date_dt"] = pd.to_datetime(df.get("date", None), errors="coerce", utc=True)
    return df


def _expand_player_matches(df_matches: pd.DataFrame, id_to_name: dict[int, str]) -> pd.DataFrame:
    rows: list[dict] = []
    for _, match in df_matches.iterrows():
        try:
            p1 = _safe_int(match.get("t1_p1"))
            p2 = _safe_int(match.get("t1_p2"))
            p3 = _safe_int(match.get("t2_p1"))
            p4 = _safe_int(match.get("t2_p2"))
            s1 = int(match.get("score_t1", 0) or 0)
            s2 = int(match.get("score_t2", 0) or 0)
        except Exception:
            continue
        if any(pid is None for pid in (p1, p2, p3, p4)):
            continue
        if (s1 + s2) <= 0:
            continue
        t1_win = s1 > s2
        t2_win = s2 > s1
        for pid, team in [(p1, 1), (p2, 1), (p3, 2), (p4, 2)]:
            win = 1 if (t1_win and team == 1) or (t2_win and team == 2) else 0
            loss = 1 if (t1_win and team == 2) or (t2_win and team == 1) else 0
            rows.append(
                {
                    "match_id": _safe_int(match.get("id")),
                    "date_dt": match.get("date_dt"),
                    "week_tag": match.get("week_tag", ""),
                    "week_num": match.get("week_num"),
                    "player_id": int(pid),
                    "player_name": id_to_name.get(int(pid), f"#{pid}"),
                    "games": 1,
                    "wins": int(win),
                    "losses": int(loss),
                }
            )
    return pd.DataFrame(rows)


def _summarize_player_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(
            games=("games", "sum"),
            wins=("wins", "sum"),
            losses=("losses", "sum"),
        )
    )
    grouped["win_pct"] = grouped.apply(
        lambda r: (float(r["wins"]) / float(r["games"]) * 100.0) if r["games"] > 0 else pd.NA,
        axis=1,
    )
    return grouped


def _build_league_standings(ctx, league_name: str) -> pd.DataFrame:
    df_leagues = getattr(ctx, "df_leagues", None)
    id_to_name = getattr(ctx, "id_to_name", {})
    active_ids = None
    df_players_active = getattr(ctx, "df_players_active", None)
    if df_players_active is not None and not df_players_active.empty and "id" in df_players_active.columns:
        active_ids = set(df_players_active["id"].astype(int).tolist())
    if df_leagues is None or df_leagues.empty or "league_name" not in df_leagues.columns:
        return pd.DataFrame()

    league_df = df_leagues.copy()
    league_df["league_name"] = league_df["league_name"].fillna("").astype(str).str.strip()
    league_df = league_df[league_df["league_name"] == str(league_name).strip()].copy()
    if league_df.empty:
        return pd.DataFrame()

    league_df["player_id"] = league_df["player_id"].astype(int)
    if active_ids is not None:
        league_df = league_df[league_df["player_id"].isin(active_ids)].copy()
        if league_df.empty:
            return pd.DataFrame()
    league_df["name"] = league_df["player_id"].map(id_to_name)
    league_df["rating"] = pd.to_numeric(league_df.get("rating", 0), errors="coerce").fillna(0.0)
    league_df["starting_rating"] = pd.to_numeric(
        league_df.get("starting_rating", league_df["rating"]), errors="coerce"
    ).fillna(league_df["rating"])
    for col in ["wins", "losses", "matches_played"]:
        league_df[col] = pd.to_numeric(league_df.get(col, 0), errors="coerce").fillna(0).astype(int)

    league_df["JUPR"] = league_df["rating"].astype(float) / 400.0
    league_df["rating_delta"] = (league_df["rating"] - league_df["starting_rating"]).astype(float) / 400.0
    league_df["win_pct"] = league_df.apply(
        lambda r: (float(r["wins"]) / float(r["matches_played"]) * 100.0)
        if int(r["matches_played"]) > 0
        else pd.NA,
        axis=1,
    )
    league_df = league_df.sort_values(["rating", "matches_played"], ascending=[False, False]).copy()
    league_df["rank"] = range(1, len(league_df) + 1)
    return league_df


def _render_html_table(headers: list[str], rows: list[list[str]]) -> None:
    header_html = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    row_html = ""
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        row_html += f"<tr>{cells}</tr>"

    table_html = f"""
    <div class="lb-table-wrap">
        <table class="lb-table">
            <thead>
                <tr>{header_html}</tr>
            </thead>
            <tbody>
                {row_html}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def _replay_league_ratings(
    df_matches: pd.DataFrame,
    league_name: str,
    df_meta: pd.DataFrame | None,
    df_players_all: pd.DataFrame | None,
) -> pd.DataFrame:
    if not _RATING_REPLAY_AVAILABLE or df_matches.empty:
        return pd.DataFrame()

    df = df_matches.copy()
    if df.empty:
        return pd.DataFrame()

    k_val = int(DEFAULT_K_FACTOR)
    try:
        if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
            hit = df_meta[df_meta["league_name"].astype(str).str.strip() == str(league_name).strip()]
            if not hit.empty:
                k_val = int(hit.iloc[0].get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR)
    except Exception:
        k_val = int(DEFAULT_K_FACTOR)

    overall_seed = {}
    try:
        if df_players_all is not None and not df_players_all.empty:
            overall_seed = dict(zip(df_players_all["id"].astype(int), df_players_all["rating"].astype(float)))
    except Exception:
        overall_seed = {}

    used_snap = all(c in df.columns for c in ["t1_p1_r", "t1_p2_r", "t2_p1_r", "t2_p2_r"])
    island: dict[int, float] = {}
    rows: list[dict] = []

    def seed_from_row(row, pid: int) -> float:
        pid = int(pid)
        if used_snap:
            try:
                if pid == _safe_int(row.get("t1_p1")):
                    snap = row.get("t1_p1_r")
                elif pid == _safe_int(row.get("t1_p2")):
                    snap = row.get("t1_p2_r")
                elif pid == _safe_int(row.get("t2_p1")):
                    snap = row.get("t2_p1_r")
                elif pid == _safe_int(row.get("t2_p2")):
                    snap = row.get("t2_p2_r")
                else:
                    snap = None
                if snap is not None and str(snap).strip() != "":
                    return float(snap)
            except Exception:
                pass
        return float(overall_seed.get(pid, 1200.0))

    def get_rating(row, pid: int) -> float:
        pid = int(pid)
        if pid not in island:
            island[pid] = seed_from_row(row, pid)
        return float(island[pid])

    df = df.sort_values(["date_dt", "id"], ascending=[True, True])

    for _, match in df.iterrows():
        try:
            mid = int(match.get("id"))
            p1 = int(match.get("t1_p1"))
            p2 = int(match.get("t1_p2"))
            p3 = int(match.get("t2_p1"))
            p4 = int(match.get("t2_p2"))
            s1 = int(match.get("score_t1", 0) or 0)
            s2 = int(match.get("score_t2", 0) or 0)
        except Exception:
            continue

        if (s1 + s2) <= 0:
            continue

        r1, r2, r3, r4 = get_rating(match, p1), get_rating(match, p2), get_rating(match, p3), get_rating(match, p4)

        d1, d2 = calculate_hybrid_elo(
            (r1 + r2) / 2.0,
            (r3 + r4) / 2.0,
            s1,
            s2,
            k_factor=int(k_val),
            min_win_delta=float(MIN_WIN_DELTA_ELO),
            cap_loser_gain=float(CAP_LOSER_GAIN_ELO),
        )

        island[p1] = r1 + float(d1)
        island[p2] = r2 + float(d1)
        island[p3] = r3 + float(d2)
        island[p4] = r4 + float(d2)

        for pid, start, end in [
            (p1, r1, island[p1]),
            (p2, r2, island[p2]),
            (p3, r3, island[p3]),
            (p4, r4, island[p4]),
        ]:
            rows.append(
                {
                    "match_id": mid,
                    "player_id": int(pid),
                    "week_tag": match.get("week_tag", ""),
                    "week_num": match.get("week_num"),
                    "date_dt": match.get("date_dt"),
                    "start_rating": float(start),
                    "end_rating": float(end),
                }
            )

    return pd.DataFrame(rows)


def _weekly_rating_summary(replay_df: pd.DataFrame) -> pd.DataFrame:
    if replay_df is None or replay_df.empty:
        return pd.DataFrame()
    data = replay_df.dropna(subset=["week_num"]).copy()
    if data.empty:
        return pd.DataFrame()
    data["week_num"] = data["week_num"].astype(int)
    data = data.sort_values(["date_dt", "match_id"], ascending=[True, True])
    first = data.groupby(["player_id", "week_num"], as_index=False).first()
    last = data.groupby(["player_id", "week_num"], as_index=False).last()
    summary = first[["player_id", "week_num", "start_rating"]].merge(
        last[["player_id", "week_num", "end_rating"]], on=["player_id", "week_num"], how="inner"
    )
    summary["rating_delta"] = (summary["end_rating"] - summary["start_rating"]).astype(float) / 400.0

    ranks = []
    for week_num, week_df in summary.groupby("week_num"):
        week_df = week_df.copy()
        week_df["rank"] = week_df["end_rating"].rank(method="dense", ascending=False).astype(int)
        week_df["week_num"] = int(week_num)
        ranks.append(week_df)
    ranked = pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame()
    if ranked.empty:
        return summary

    ranked = ranked.sort_values(["player_id", "week_num"]).copy()
    ranked["prev_rank"] = ranked.groupby("player_id")["rank"].shift(1)
    ranked["rank_delta"] = ranked["prev_rank"] - ranked["rank"]
    return ranked


def render(ctx):
    PUBLIC_MODE = bool(getattr(ctx, "public_mode", False))
    mode_label = "Public" if PUBLIC_MODE else "Admin"
    page_shell("📊 League Results", "League-specific results and trends.", mode_label=mode_label)

    st.markdown(
        """
        <style>
        .lb-link {
            color: inherit;
            text-decoration: none;
        }
        .lb-link:hover {
            text-decoration: underline;
            opacity: 0.95;
        }
        .lb-table-wrap {
            overflow-x: auto;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: var(--panel);
        }
        .lb-table {
            width: 100%;
            border-collapse: collapse;
            min-width: 720px;
        }
        .lb-table th,
        .lb-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
        }
        .lb-table th {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            background: var(--table-stripe);
            position: sticky;
            top: 0;
        }
        .lb-table tbody tr:hover {
            background: var(--accent-soft);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df_matches = getattr(ctx, "df_matches", None)
    df_meta = getattr(ctx, "df_meta", None)
    df_players_all = getattr(ctx, "df_players_all", None)
    df_players_active = getattr(ctx, "df_players_active", None)

    if df_matches is None or df_matches.empty:
        st.info("No matches loaded.")
        return

    leagues = _league_options(df_meta, df_matches, getattr(ctx, "df_leagues", None))
    if not leagues:
        st.info("No leagues found.")
        return

    requested_league = (qp_get("league", "") or "").strip()
    default_league = requested_league if requested_league in leagues else leagues[0]
    league_name = st.selectbox("Select league", leagues, index=leagues.index(default_league))

    league_matches = _filter_league_matches(df_matches, league_name)
    if league_matches.empty:
        st.info("No league matches found yet.")
        return

    player_frame = _expand_player_matches(league_matches, getattr(ctx, "id_to_name", {}))
    if player_frame.empty:
        st.info("No player results found in this league.")
        return

    if df_players_active is not None and not df_players_active.empty and "id" in df_players_active.columns:
        active_ids = set(df_players_active["id"].astype(int).tolist())
        player_frame = player_frame[player_frame["player_id"].astype(int).isin(active_ids)].copy()
        if player_frame.empty:
            st.info("No active players found for this league.")
            return

    overall_stats = _summarize_player_stats(player_frame, ["player_id", "player_name"])
    weekly_stats = _summarize_player_stats(player_frame, ["player_id", "player_name", "week_num"]) if not player_frame.empty else pd.DataFrame()

    replay_df = _replay_league_ratings(league_matches, league_name, df_meta, df_players_all)
    weekly_rating = _weekly_rating_summary(replay_df)
    weeks_df = _build_league_weeks(df_meta, league_name, league_matches)
    league_week_nums = weeks_df["week_num"].tolist() if not weeks_df.empty else []

    tabs = st.tabs(["Overall", "Weekly", "Player"])

    with tabs[0]:
        st.subheader("Current Standings")
        standings = _build_league_standings(ctx, league_name)
        if standings.empty:
            st.info("No league rating data found yet.")
        else:
            table_rows = []
            for _, row in standings.iterrows():
                player_link = _build_player_link(row["player_id"], row["name"], PUBLIC_MODE, ctx)
                win_pct = f"{float(row['win_pct']):.1f}%" if pd.notna(row["win_pct"]) else "—"
                rating_delta = row.get("rating_delta")
                rating_delta_display = f"{float(rating_delta):+.3f}" if pd.notna(rating_delta) else "—"
                table_rows.append(
                    [
                        str(int(row["rank"])),
                        player_link,
                        f"{float(row['JUPR']):.3f}",
                        str(int(row["matches_played"])),
                        str(int(row["wins"])),
                        str(int(row["losses"])),
                        win_pct,
                        rating_delta_display,
                    ]
                )

            _render_html_table(
                ["Rank", "Player", "Rating (JUPR)", "Games", "Wins", "Losses", "Win %", "Rating Δ"],
                table_rows,
            )

        st.markdown("### Cumulative Performance (League Only)")
        if overall_stats.empty:
            st.info("No cumulative stats available yet.")
        else:
            overall_stats = overall_stats.sort_values(["wins", "games"], ascending=[False, False])
            table_rows = []
            for _, row in overall_stats.iterrows():
                player_link = _build_player_link(row["player_id"], row["player_name"], PUBLIC_MODE, ctx)
                win_pct = f"{float(row['win_pct']):.1f}%" if pd.notna(row["win_pct"]) else "—"
                table_rows.append(
                    [
                        player_link,
                        str(int(row["games"])),
                        str(int(row["wins"])),
                        str(int(row["losses"])),
                        win_pct,
                    ]
                )
            _render_html_table(["Player", "Games", "Wins", "Losses", "Win %"], table_rows)

    with tabs[1]:
        st.subheader("Weekly Results")
        week_nums = league_week_nums or sorted(player_frame["week_num"].dropna().astype(int).unique().tolist())
        if not week_nums:
            st.info("No week numbers found in league matches yet.")
            return

        default_week = max(week_nums)
        selected_week = st.selectbox(
            "Week", week_nums, index=week_nums.index(default_week), format_func=lambda x: f"Week {x}"
        )

        weekly_view = weekly_stats[weekly_stats["week_num"] == int(selected_week)].copy() if not weekly_stats.empty else pd.DataFrame()
        weekly_view = weekly_view.sort_values(["wins", "games"], ascending=[False, False])

        if not weekly_rating.empty:
            weekly_delta = weekly_rating[weekly_rating["week_num"] == int(selected_week)].copy()
            weekly_view = weekly_view.merge(weekly_delta, on=["player_id", "week_num"], how="left")

        if weekly_view.empty:
            st.info("No weekly data found.")
        else:
            table_rows = []
            for _, row in weekly_view.iterrows():
                player_link = _build_player_link(row["player_id"], row["player_name"], PUBLIC_MODE, ctx)
                win_pct = f"{float(row['win_pct']):.1f}%" if pd.notna(row["win_pct"]) else "—"
                rating_delta = row.get("rating_delta")
                rating_delta_display = f"{float(rating_delta):+.3f}" if pd.notna(rating_delta) else "—"
                rank_delta = row.get("rank_delta")
                rank_delta_display = f"{int(rank_delta):+d}" if pd.notna(rank_delta) else "—"
                table_rows.append(
                    [
                        player_link,
                        str(int(row["games"])),
                        str(int(row["wins"])),
                        str(int(row["losses"])),
                        win_pct,
                        rating_delta_display,
                        rank_delta_display,
                    ]
                )

            _render_html_table(
                ["Player", "Games", "Wins", "Losses", "Win %", "Weekly Rating Δ", "Rank Δ"],
                table_rows,
            )

        st.markdown("### Weekly Highlights")
        highlight_cols = st.columns(3)

        with highlight_cols[0]:
            st.markdown("**Biggest climbers**")
            if "rating_delta" in weekly_view.columns and weekly_view["rating_delta"].notna().any():
                climbers = weekly_view.sort_values("rating_delta", ascending=False).head(3)
                for _, row in climbers.iterrows():
                    st.write(f"{row['player_name']} ({float(row['rating_delta']):+.3f})")
            else:
                climbers = weekly_view.sort_values(["wins", "games"], ascending=[False, False]).head(3)
                for _, row in climbers.iterrows():
                    st.write(f"{row['player_name']} ({int(row['wins'])} wins)")

        with highlight_cols[1]:
            st.markdown("**Best win %**")
            min_games = st.number_input("Min games", min_value=1, max_value=20, value=4, step=1)
            qualified = weekly_view[weekly_view["games"] >= int(min_games)].copy()
            qualified = qualified[pd.notna(qualified["win_pct"])].copy()
            qualified = qualified.sort_values(["win_pct", "games"], ascending=[False, False]).head(3)
            if qualified.empty:
                st.caption("Not enough games to qualify.")
            else:
                for _, row in qualified.iterrows():
                    st.write(f"{row['player_name']} ({float(row['win_pct']):.1f}%)")

        with highlight_cols[2]:
            st.markdown("**Most active**")
            active = weekly_view.sort_values(["games", "wins"], ascending=[False, False]).head(3)
            for _, row in active.iterrows():
                st.write(f"{row['player_name']} ({int(row['games'])} games)")

        if weekly_rating.empty:
            st.caption("Weekly rating deltas are unavailable (no league replay data).")

    with tabs[2]:
        st.subheader("Player Summary")
        player_ids = sorted(set(overall_stats["player_id"].tolist())) if not overall_stats.empty else []
        if standings is not None and not standings.empty:
            player_ids = sorted(set(player_ids + standings["player_id"].tolist()))

        if not player_ids:
            st.info("No players found for this league.")
            return

        id_to_name = getattr(ctx, "id_to_name", {})
        player_id = st.selectbox(
            "Player", player_ids, format_func=lambda pid: id_to_name.get(int(pid), f"#{pid}")
        )
        player_id = int(player_id)

        name = id_to_name.get(player_id, f"#{player_id}")
        st.markdown(f"**{name}**")

        rank_value = "—"
        rating_value = "—"
        if standings is not None and not standings.empty:
            hit = standings[standings["player_id"] == player_id]
            if not hit.empty:
                rank_value = int(hit.iloc[0]["rank"])
                rating_value = f"{float(hit.iloc[0]['JUPR']):.3f}"

        player_stats = overall_stats[overall_stats["player_id"] == player_id]
        games_value = int(player_stats.iloc[0]["games"]) if not player_stats.empty else 0
        win_pct_value = (
            f"{float(player_stats.iloc[0]['win_pct']):.1f}%" if not player_stats.empty and pd.notna(player_stats.iloc[0]["win_pct"]) else "—"
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Rank", rank_value)
        metric_cols[1].metric("League JUPR", rating_value)
        metric_cols[2].metric("Games", games_value)
        metric_cols[3].metric("Win %", win_pct_value)

        st.markdown("### Rank over time")
        player_weekly_stats = weekly_stats[weekly_stats["player_id"] == player_id].copy() if not weekly_stats.empty else pd.DataFrame()
        if not weeks_df.empty:
            player_weekly_stats = weeks_df.merge(player_weekly_stats, on="week_num", how="left")
            player_weekly_stats["games"] = player_weekly_stats["games"].fillna(0).astype(int)
            player_weekly_stats["wins"] = player_weekly_stats["wins"].fillna(0).astype(int)
            player_weekly_stats["losses"] = player_weekly_stats["losses"].fillna(0).astype(int)
            player_weekly_stats["win_pct"] = player_weekly_stats["win_pct"].where(player_weekly_stats["games"] > 0, pd.NA)
        player_weekly_stats = _attach_week_label(player_weekly_stats).sort_values("week_num")

        rank_series = pd.DataFrame()
        rank_label = "Weekly performance rank"
        if not weekly_rating.empty:
            rating_player = weekly_rating[weekly_rating["player_id"] == player_id].copy()
            if not rating_player.empty:
                rank_series = rating_player[["week_num", "rank"]].copy()
                rank_label = "Rating rank (end of week)"

        if rank_series.empty and not weekly_stats.empty:
            perf = weekly_stats.copy()
            perf = perf[pd.notna(perf["week_num"])].copy()
            perf["week_num"] = perf["week_num"].astype(int)
            perf = perf.sort_values(["week_num", "win_pct", "wins", "games"], ascending=[True, False, False, False])
            perf["perf_rank"] = perf.groupby("week_num").cumcount() + 1
            rank_series = perf[perf["player_id"] == player_id][["week_num", "perf_rank"]].rename(
                columns={"perf_rank": "rank"}
            )
            rank_label = "Weekly performance rank (Win % then Wins)"

        if rank_series.empty:
            st.caption("Not enough weekly data to build a rank series.")
        else:
            if not weeks_df.empty:
                rank_series = weeks_df.merge(rank_series, on="week_num", how="left")
            rank_series = _attach_week_label(rank_series).sort_values("week_num")
            week_labels = _week_label_order(weeks_df, rank_series)
            if alt is not None:
                chart = (
                    alt.Chart(rank_series)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "week_label:N",
                            title="Week",
                            sort=week_labels,
                            axis=alt.Axis(labelAngle=0, labelFontSize=14),
                        ),
                        y=alt.Y("rank:Q", title=rank_label, scale=alt.Scale(reverse=True)),
                        tooltip=[alt.Tooltip("week_num:O", title="Week"), alt.Tooltip("rank:Q", title="Rank")],
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.line_chart(rank_series.set_index("week_num")["rank"])

        st.markdown("### Weekly trend")
        if player_weekly_stats.empty:
            st.caption("No weekly results available for this player.")
        else:
            trend_cols = st.columns(2)
            games_series = player_weekly_stats[["week_num", "week_label", "games"]].copy()
            win_series = player_weekly_stats[["week_num", "week_label", "win_pct"]].copy()
            week_labels = _week_label_order(weeks_df, player_weekly_stats)

            with trend_cols[0]:
                st.markdown("**Games by week**")
                if alt is not None:
                    game_chart = (
                        alt.Chart(games_series)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "week_label:N",
                                title="Week",
                                sort=week_labels,
                                axis=alt.Axis(labelAngle=0, labelFontSize=14),
                            ),
                            y=alt.Y("games:Q", title="Games"),
                            tooltip=[alt.Tooltip("week_num:O", title="Week"), alt.Tooltip("games:Q", title="Games")],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(game_chart, use_container_width=True)
                else:
                    st.bar_chart(games_series.set_index("week_num")["games"])

            with trend_cols[1]:
                st.markdown("**Win % by week**")
                if win_series.dropna(subset=["win_pct"]).empty:
                    st.caption("No win % data.")
                elif alt is not None:
                    win_chart = (
                        alt.Chart(win_series)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X(
                                "week_label:N",
                                title="Week",
                                sort=week_labels,
                                axis=alt.Axis(labelAngle=0, labelFontSize=14),
                            ),
                            y=alt.Y("win_pct:Q", title="Win %"),
                            tooltip=[alt.Tooltip("week_num:O", title="Week"), alt.Tooltip("win_pct:Q", title="Win %", format=".1f")],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(win_chart, use_container_width=True)
                else:
                    st.line_chart(win_series.set_index("week_num")["win_pct"])

            rating_weekly = weekly_rating[weekly_rating["player_id"] == player_id] if not weekly_rating.empty else pd.DataFrame()
            if not rating_weekly.empty:
                st.markdown("**Weekly rating Δ**")
                if not weeks_df.empty:
                    rating_weekly = weeks_df.merge(rating_weekly, on="week_num", how="left")
                rating_weekly = _attach_week_label(rating_weekly).sort_values("week_num")
                week_labels = _week_label_order(weeks_df, rating_weekly)
                if alt is not None:
                    delta_chart = (
                        alt.Chart(rating_weekly)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "week_label:N",
                                title="Week",
                                sort=week_labels,
                                axis=alt.Axis(labelAngle=0, labelFontSize=14),
                            ),
                            y=alt.Y("rating_delta:Q", title="Rating Δ"),
                            tooltip=[alt.Tooltip("week_num:O", title="Week"), alt.Tooltip("rating_delta:Q", title="Rating Δ", format="+.3f")],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(delta_chart, use_container_width=True)
                else:
                    st.bar_chart(rating_weekly.set_index("week_num")["rating_delta"])
            else:
                st.caption("Weekly rating delta unavailable.")

        st.markdown("### Recent matches")
        recent = league_matches.copy()
        recent = recent[
            (recent.get("t1_p1") == player_id)
            | (recent.get("t1_p2") == player_id)
            | (recent.get("t2_p1") == player_id)
            | (recent.get("t2_p2") == player_id)
        ].copy()
        recent = recent.sort_values(["date_dt", "id"], ascending=[False, False]).head(15)

        if recent.empty:
            st.caption("No recent league matches found.")
        else:
            rows = []
            for _, match in recent.iterrows():
                p1 = _safe_int(match.get("t1_p1"))
                p2 = _safe_int(match.get("t1_p2"))
                p3 = _safe_int(match.get("t2_p1"))
                p4 = _safe_int(match.get("t2_p2"))
                s1 = int(match.get("score_t1", 0) or 0)
                s2 = int(match.get("score_t2", 0) or 0)
                date_str = match.get("date_dt")
                date_display = date_str.strftime("%Y-%m-%d") if pd.notna(date_str) else "—"

                if player_id in {p1, p2}:
                    partner = p2 if player_id == p1 else p1
                    opps = [p3, p4]
                    score = f"{s1}-{s2}"
                    result = "W" if s1 > s2 else "L" if s2 > s1 else "D"
                else:
                    partner = p4 if player_id == p3 else p3
                    opps = [p1, p2]
                    score = f"{s2}-{s1}"
                    result = "W" if s2 > s1 else "L" if s1 > s2 else "D"

                rows.append(
                    [
                        date_display,
                        match.get("week_tag", ""),
                        id_to_name.get(partner, f"#{partner}"),
                        ", ".join(id_to_name.get(opp, f"#{opp}") for opp in opps),
                        result,
                        score,
                    ]
                )

            _render_html_table(
                ["Date", "Week", "Partner", "Opponents", "Result", "Score"],
                rows,
            )
