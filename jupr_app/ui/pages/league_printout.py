from __future__ import annotations

import json
import re
from urllib.parse import quote

import pandas as pd
import streamlit as st

from jupr_app.domain.leagues import compute_top_performer_awards_for_config, get_league_meta_row
from jupr_app.ui.layout import page_shell
from jupr_app.ui.url import qp_get

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


def _league_options(df_meta: pd.DataFrame | None, df_matches: pd.DataFrame | None, df_leagues: pd.DataFrame | None) -> list[str]:
    leagues: list[str] = []
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
                    "week_tag": match.get("week_tag", ""),
                    "week_num": match.get("week_num"),
                    "date_dt": match.get("date_dt"),
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
        .agg(games=("games", "sum"), wins=("wins", "sum"), losses=("losses", "sum"))
        .sort_values(["wins", "games"], ascending=[False, False])
    )
    return grouped


def _replay_league_ratings(
    df_matches: pd.DataFrame,
    league_name: str,
    df_meta: pd.DataFrame | None,
    df_players_all: pd.DataFrame | None,
) -> pd.DataFrame:
    if not _RATING_REPLAY_AVAILABLE or df_matches.empty:
        return pd.DataFrame()

    df = df_matches.copy()
    k_val = int(DEFAULT_K_FACTOR)
    try:
        if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
            hit = df_meta[df_meta["league_name"].astype(str).str.strip() == str(league_name).strip()]
            if not hit.empty:
                k_val = int(hit.iloc[0].get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR)
    except Exception:
        k_val = int(DEFAULT_K_FACTOR)

    overall_seed: dict[int, float] = {}
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

        for pid, start, end in [(p1, r1, island[p1]), (p2, r2, island[p2]), (p3, r3, island[p3]), (p4, r4, island[p4])]:
            rows.append(
                {
                    "match_id": mid,
                    "player_id": int(pid),
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
    return summary


def _safe_awards_config(meta_row: dict) -> dict:
    awards_cfg = meta_row.get("awards_config") if isinstance(meta_row, dict) else {}
    if isinstance(awards_cfg, str):
        try:
            parsed = json.loads(awards_cfg)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    if isinstance(awards_cfg, dict):
        return awards_cfg
    return {}


def _build_printout_html(league_name: str, week_num: int, weekly_rating: pd.DataFrame, weekly_wins: pd.DataFrame, awards: list[dict]) -> str:
    def leaderboard_rows(df: pd.DataFrame, cols: list[str]) -> str:
        lines = []
        for _, row in df.iterrows():
            cells = "".join(f"<td>{row[c]}</td>" for c in cols)
            lines.append(f"<tr>{cells}</tr>")
        return "\n".join(lines)

    awards_rows = ""
    if awards:
        for award in awards:
            awards_rows += (
                f"<tr><td>{award.get('category_label','')}</td>"
                f"<td>{award.get('player_name','')}</td>"
                f"<td>{award.get('metric_display','')}</td></tr>"
            )
    else:
        awards_rows = "<tr><td colspan='3'>No Top Performer config set for this league.</td></tr>"

    return f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>League Night Printout</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
h1 {{ margin: 0 0 6px 0; }}
.subtitle {{ color: #444; margin-bottom: 18px; }}
h2 {{ margin-top: 24px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }}
th {{ background: #f6f6f6; }}
@media print {{
  .no-print {{ display: none !important; }}
  body {{ margin: 10mm; }}
  @page {{ size: A4; margin: 10mm; }}
}}
</style>
</head>
<body>
<div class=\"no-print\">Tip: use your browser Print dialog to print or Save as PDF.</div>
<h1>League Night Printout</h1>
<div class=\"subtitle\"><strong>{league_name}</strong> • Week {week_num}</div>
<h2>Weekly Leaders</h2>
<h3>Highest Rating Gained</h3>
<table>
<thead><tr><th>Player</th><th>Δ JUPR</th></tr></thead>
<tbody>{leaderboard_rows(weekly_rating, ['player_name', 'rating_delta_display'])}</tbody>
</table>
<h3>Most Wins</h3>
<table>
<thead><tr><th>Player</th><th>Wins</th><th>Games</th></tr></thead>
<tbody>{leaderboard_rows(weekly_wins, ['player_name', 'wins', 'games'])}</tbody>
</table>
<h2>Season Leaders (Top Performers)</h2>
<table>
<thead><tr><th>Category</th><th>Winner</th><th>Metric</th></tr></thead>
<tbody>{awards_rows}</tbody>
</table>
</body>
</html>"""


def _build_printout_txt(league_name: str, week_num: int, weekly_rating: pd.DataFrame, weekly_wins: pd.DataFrame, awards: list[dict]) -> str:
    lines = [
        "LEAGUE NIGHT PRINTOUT",
        f"League: {league_name}",
        f"Week: {week_num}",
        "",
        "WEEKLY LEADERS",
        "- Highest Rating Gained",
    ]
    if weekly_rating.empty:
        lines.append("  (no weekly rating data)")
    else:
        for idx, row in enumerate(weekly_rating.itertuples(index=False), start=1):
            lines.append(f"  {idx}. {row.player_name} — {row.rating_delta_display} JUPR")

    lines.append("- Most Wins")
    if weekly_wins.empty:
        lines.append("  (no match data)")
    else:
        for idx, row in enumerate(weekly_wins.itertuples(index=False), start=1):
            lines.append(f"  {idx}. {row.player_name} — {int(row.wins)} wins ({int(row.games)} games)")

    lines.extend(["", "SEASON LEADERS (TOP PERFORMERS)"])
    if not awards:
        lines.append("No Top Performer config set for this league.")
    else:
        by_cat: dict[str, list[dict]] = {}
        for award in awards:
            by_cat.setdefault(str(award.get("category_label", "")), []).append(award)
        for category, items in by_cat.items():
            lines.append(f"- {category}")
            for item in items:
                lines.append(f"  • {item.get('player_name', '')} — {item.get('metric_display', '')}")
    return "\n".join(lines)


def render(ctx):
    PUBLIC_MODE = bool(getattr(ctx, "public_mode", False))
    mode_label = "Public" if PUBLIC_MODE else "Admin"
    page_shell("🖨️ League Night Printout", "Shareable weekly + season summary.", mode_label=mode_label)

    df_matches = getattr(ctx, "df_matches", None)
    df_meta = getattr(ctx, "df_meta", None)
    df_leagues = getattr(ctx, "df_leagues", None)
    df_players_all = getattr(ctx, "df_players_all", None)
    id_to_name = getattr(ctx, "id_to_name", {}) or {}

    league_options = _league_options(df_meta, df_matches, df_leagues)
    if not league_options:
        st.info("No active leagues available.")
        return

    qp_league = (qp_get("league", "") or "").strip()
    league_default_idx = league_options.index(qp_league) if qp_league in league_options else 0
    selected_league = st.selectbox("Select league", league_options, index=league_default_idx)

    if df_matches is None or df_matches.empty:
        st.info("No matches available yet.")
        return

    league_matches = _filter_league_matches(df_matches, selected_league)
    available_weeks = sorted(league_matches["week_num"].dropna().astype(int).unique().tolist())
    if not available_weeks:
        st.info("No scored weeks available for this league yet.")
        return

    qp_week = _parse_week_num(qp_get("week", ""))
    default_week = qp_week if qp_week in available_weeks else max(available_weeks)
    selected_week = st.selectbox("Select week", available_weeks, index=available_weeks.index(default_week), format_func=lambda w: f"Week {w}")

    weekly_matches = league_matches[league_matches["week_num"] == int(selected_week)].copy()

    players_expanded = _expand_player_matches(weekly_matches, id_to_name)
    weekly_wins = _summarize_player_stats(players_expanded, ["player_id", "player_name"]).head(5)

    replay_df = _replay_league_ratings(league_matches, selected_league, df_meta, df_players_all)
    weekly_rating = _weekly_rating_summary(replay_df)
    weekly_rating = weekly_rating[weekly_rating["week_num"] == int(selected_week)].copy() if not weekly_rating.empty else pd.DataFrame()
    if not weekly_rating.empty:
        weekly_rating["player_name"] = weekly_rating["player_id"].map(id_to_name).fillna(weekly_rating["player_id"].map(lambda x: f"#{x}"))
        weekly_rating = weekly_rating.sort_values("rating_delta", ascending=False).head(5)
        weekly_rating["rating_delta_display"] = weekly_rating["rating_delta"].map(lambda v: f"{float(v):+.3f}")

    meta_row = get_league_meta_row(df_meta, selected_league) or {}
    awards_cfg = _safe_awards_config(meta_row)
    awards = compute_top_performer_awards_for_config(
        df_leagues,
        df_meta,
        id_to_name,
        selected_league,
        awards_config=awards_cfg,
    )

    st.subheader("Weekly Leaders")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Highest Rating Gained**")
        if weekly_rating.empty:
            st.caption("No rating replay data available.")
        else:
            st.dataframe(
                weekly_rating[["player_name", "rating_delta_display"]].rename(
                    columns={"player_name": "Player", "rating_delta_display": "Δ JUPR"}
                ),
                use_container_width=True,
                hide_index=True,
            )
    with c2:
        st.markdown("**Most Wins**")
        if weekly_wins.empty:
            st.caption("No scored matches found for this week.")
        else:
            st.dataframe(
                weekly_wins[["player_name", "wins", "games"]].rename(
                    columns={"player_name": "Player", "wins": "Wins", "games": "Games"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Season Leaders (Top Performers)")
    if not awards_cfg:
        st.info("No Top Performer config set for this league.")
    if not awards:
        st.caption("No Top Performer winners available yet.")
    else:
        awards_df = pd.DataFrame(awards)
        awards_df["winner"] = awards_df["player_name"].fillna("")
        grouped = (
            awards_df.groupby(["category_label", "metric_display"], as_index=False)
            .agg(winner=("winner", lambda vals: ", ".join([v for v in vals if v])))
            .rename(columns={"category_label": "Category", "winner": "Winner(s)", "metric_display": "Metric"})
        )
        st.dataframe(grouped, use_container_width=True, hide_index=True)

    html_text = _build_printout_html(selected_league, int(selected_week), weekly_rating, weekly_wins, awards)
    txt_text = _build_printout_txt(selected_league, int(selected_week), weekly_rating, weekly_wins, awards)

    st.markdown("### Export")
    ex1, ex2, ex3 = st.columns([1, 1, 2])
    ex1.download_button(
        "Download HTML",
        data=html_text,
        file_name=f"league_printout_{quote(selected_league)}_week_{int(selected_week)}.html",
        mime="text/html",
        use_container_width=True,
    )
    ex2.download_button(
        "Download TXT",
        data=txt_text,
        file_name=f"league_printout_{quote(selected_league)}_week_{int(selected_week)}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    base_url = st.session_state.get("base_url", "")
    public_url = f"{base_url}/?page=league_printout&public=1&league={quote(selected_league)}&week={int(selected_week)}"
    ex3.caption("Public share link")
    ex3.code(public_url)
