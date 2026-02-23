from __future__ import annotations

import csv
import difflib
import io
import re
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from jupr_app.config import FEATURE_SESSION_LADDER
from jupr_app.ui.layout import page_shell
from services import session_ladder_api

_ROUTE_SESSION = re.compile(r"^sessions/([^/]+)$")
_ROUTE_COURT = re.compile(r"^sessions/([^/]+)/rounds/(\d+)/courts/([^/]+)$")


def build_session_route(session_id: str, *, round_number: int | None = None, court_id: str | int | None = None) -> str:
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    if round_number is None or court_id is None:
        return f"sessions/{sid}"
    return f"sessions/{sid}/rounds/{int(round_number)}/courts/{court_id}"


def parse_session_route(route: str) -> dict[str, Any] | None:
    normalized = str(route or "").strip().strip("/")
    if not normalized:
        return None
    m_court = _ROUTE_COURT.fullmatch(normalized)
    if m_court:
        return {
            "session_id": m_court.group(1),
            "round_number": int(m_court.group(2)),
            "court_id": m_court.group(3),
            "is_court": True,
        }
    m_session = _ROUTE_SESSION.fullmatch(normalized)
    if m_session:
        return {"session_id": m_session.group(1), "round_number": None, "court_id": None, "is_court": False}
    return None


def parse_intake_names(text: str) -> list[str]:
    raw = str(text or "")
    tokens = re.split(r"[\n,;]+", raw)
    cleaned = [" ".join(tok.strip().split()) for tok in tokens if tok and tok.strip()]
    dedup: list[str] = []
    seen: set[str] = set()
    for item in cleaned:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


def build_fuzzy_preview(names: list[str], existing_names: list[str]) -> list[dict[str, Any]]:
    existing_lower = {n.lower(): n for n in existing_names}
    out: list[dict[str, Any]] = []
    for name in names:
        if name.lower() in existing_lower:
            out.append({"input": name, "match": existing_lower[name.lower()], "confidence": 1.0, "mode": "exact"})
            continue
        best = difflib.get_close_matches(name, existing_names, n=1, cutoff=0.75)
        if best:
            ratio = difflib.SequenceMatcher(a=name.lower(), b=best[0].lower()).ratio()
            out.append({"input": name, "match": best[0], "confidence": round(float(ratio), 3), "mode": "fuzzy"})
        else:
            out.append({"input": name, "match": None, "confidence": 0.0, "mode": "new"})
    return out





def _render_step_help_links(*, state: str) -> None:
    state_norm = str(state or "").upper()
    step = "Roster"
    if state_norm == "SEEDED_LOCKED":
        step = "Seeding"
    elif state_norm.startswith("ROUND_1"):
        step = "Round 1"
    elif state_norm.startswith("ROUND_2"):
        step = "Round 2"
    elif state_norm.startswith("ROUND_3"):
        step = "Round 3"
    elif state_norm in {"COMPLETED", "PUBLISHED"}:
        step = "Publish"

    with st.expander(f"Help for this step: {step}", expanded=False):
        st.markdown(
            "- Manager guide: [`docs/session-ladder/manager-guide.md`](./docs/session-ladder/manager-guide.md)\n"
            "- Player guide: [`docs/session-ladder/player-guide.md`](./docs/session-ladder/player-guide.md)\n"
            "- Tie-break chain: **Wins → PD → PF → H2H → Playoff**\n"
            "- Movement rule: **1 court only**"
        )

def _step_index(state: str) -> int:
    state_norm = str(state or "").upper()
    if state_norm in {"DRAFT", "ROSTER_OPEN"}:
        return 0
    if state_norm == "SEEDED_LOCKED":
        return 1
    if state_norm.startswith("ROUND_1"):
        return 2
    if state_norm.startswith("ROUND_2"):
        return 3
    if state_norm.startswith("ROUND_3"):
        return 4
    return 5


def _store_resume_pointer(user_key: str, route: str) -> None:
    st.session_state["session_console_last_route"] = str(route)
    st.session_state["session_console_last_user"] = str(user_key)
    st.query_params["sl_last_route"] = str(route)

    safe_user = user_key.replace("'", "")
    safe_route = str(route).replace("'", "")
    components.html(
        f"""
        <script>
          try {{
            localStorage.setItem('jupr.session_console.last_user', '{safe_user}');
            localStorage.setItem('jupr.session_console.last_route', '{safe_route}');
          }} catch (e) {{}}
        </script>
        """,
        height=0,
    )


def _auth_payload(ctx) -> dict[str, Any]:
    jwt_payload = st.session_state.get("jwt_payload") or {}
    return {
        "club_id": str(ctx.club_id),
        "role": str(jwt_payload.get("role") or "manager"),
        "user_id": str(jwt_payload.get("sub") or jwt_payload.get("email") or "streamlit"),
        "admin_logged_in": bool(getattr(ctx, "admin_logged_in", False)),
    }


def _load_roster(supabase: Any, session_id: str) -> list[dict[str, Any]]:
    rows = (
        supabase.table("session_ladder_roster_entries")
        .select("*")
        .eq("session_id", str(session_id))
        .execute()
        .data
        or []
    )
    return sorted(rows, key=lambda r: (str(r.get("status") or ""), -float(r.get("rating_snapshot") or 0), int(r.get("player_id") or 0)))


def _normalize_csv_rows(text: str, *, name_col: str, rating_col: str | None, status_col: str | None) -> str:
    reader = csv.DictReader(io.StringIO(text))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["name", "rating", "status"])
    writer.writeheader()
    for row in reader:
        name = str(row.get(name_col) or "").strip()
        if not name:
            continue
        rating = str(row.get(rating_col) or "1200").strip() if rating_col else "1200"
        status = str(row.get(status_col) or "EXPECTED").strip() if status_col else "EXPECTED"
        writer.writerow({"name": name, "rating": rating, "status": status})
    return output.getvalue()


def _render_roster_intake(ctx, *, session_id: str, roster_rows: list[dict[str, Any]], player_records: list[dict[str, Any]]) -> None:
    auth = _auth_payload(ctx)
    roster_player_ids = {int(r.get("player_id") or 0) for r in roster_rows}

    st.subheader("Roster intake")
    mode = st.radio(
        "Intake mode",
        options=[
            "Search/select existing",
            "Manual new player",
            "Copy/paste names",
            "CSV upload",
        ],
        horizontal=True,
        key=f"sl_intake_mode_{session_id}",
    )

    if mode == "Search/select existing":
        candidates = [p for p in player_records if int(p.get("id") or 0) not in roster_player_ids]
        label_to_player = {f"{p.get('name')} (#{p.get('id')})": p for p in candidates}
        picked_labels = st.multiselect(
            "Select players",
            options=list(label_to_player.keys()),
            key=f"sl_existing_pick_{session_id}",
        )
        default_status = st.selectbox("Status", ["EXPECTED", "CHECKED_IN", "NO_SHOW", "WALK_IN"], index=1)
        if st.button("Add selected", key=f"sl_existing_add_{session_id}", type="primary"):
            for label in picked_labels:
                player = label_to_player[label]
                session_ladder_api.post_add_roster_entries(
                    supabase=ctx.supabase,
                    auth=auth,
                    payload={
                        "session_id": session_id,
                        "mode": "manual_existing",
                        "player_id": int(player["id"]),
                        "status": default_status,
                        "rating_snapshot": float(player.get("rating") or 1200),
                    },
                )
            st.success(f"Added {len(picked_labels)} players.")
            st.rerun()

    elif mode == "Manual new player":
        col1, col2, col3 = st.columns([3, 1, 1])
        name = col1.text_input("Name", key=f"sl_manual_name_{session_id}")
        rating = col2.number_input("Rating", min_value=0.0, max_value=2500.0, value=0.0, step=25.0, key=f"sl_manual_rating_{session_id}")
        status = col3.selectbox("Status", ["WALK_IN", "EXPECTED", "CHECKED_IN"], key=f"sl_manual_status_{session_id}")
        if st.button("Create + add", key=f"sl_manual_add_{session_id}", type="primary"):
            session_ladder_api.post_add_roster_entries(
                supabase=ctx.supabase,
                auth=auth,
                payload={
                    "session_id": session_id,
                    "mode": "create_new_player",
                    "name": name,
                    "rating": float(rating),
                    "status": status,
                },
            )
            st.success("Player added.")
            st.rerun()

    elif mode == "Copy/paste names":
        text = st.text_area("Paste names (newline/comma separated)", key=f"sl_bulk_text_{session_id}", height=120)
        names = parse_intake_names(text)
        preview = build_fuzzy_preview(names, [str(p.get("name") or "") for p in player_records])
        st.caption("Preview")
        st.dataframe(preview, use_container_width=True, hide_index=True)
        if st.button("Confirm add", key=f"sl_bulk_add_{session_id}", type="primary"):
            session_ladder_api.post_add_roster_entries(
                supabase=ctx.supabase,
                auth=auth,
                payload={
                    "session_id": session_id,
                    "mode": "bulk_text",
                    "bulk_text": "\n".join(names),
                },
            )
            st.success(f"Processed {len(names)} names.")
            st.rerun()

    else:
        file = st.file_uploader("CSV", type=["csv"], key=f"sl_csv_uploader_{session_id}")
        if file is not None:
            text = file.read().decode("utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(text))
            cols = reader.fieldnames or []
            if not cols:
                st.warning("CSV appears empty.")
                return
            c1, c2, c3 = st.columns(3)
            name_col = c1.selectbox("Name column", cols, index=0, key=f"sl_csv_name_col_{session_id}")
            rating_col = c2.selectbox("Rating column", ["<none>"] + cols, index=0, key=f"sl_csv_rating_col_{session_id}")
            status_col = c3.selectbox("Status column", ["<none>"] + cols, index=0, key=f"sl_csv_status_col_{session_id}")
            normalized = _normalize_csv_rows(
                text,
                name_col=name_col,
                rating_col=None if rating_col == "<none>" else rating_col,
                status_col=None if status_col == "<none>" else status_col,
            )
            preview_rows = list(csv.DictReader(io.StringIO(normalized)))
            st.dataframe(preview_rows[:50], use_container_width=True, hide_index=True)
            if st.button("Confirm CSV add", key=f"sl_csv_add_{session_id}", type="primary"):
                session_ladder_api.post_add_roster_entries(
                    supabase=ctx.supabase,
                    auth=auth,
                    payload={
                        "session_id": session_id,
                        "mode": "csv_upload",
                        "csv_text": normalized,
                    },
                )
                st.success(f"Processed {len(preview_rows)} CSV rows.")
                st.rerun()


def _render_roster_table(ctx, *, session_id: str, roster_rows: list[dict[str, Any]], player_records: list[dict[str, Any]]) -> None:
    name_by_id = {int(p.get("id") or 0): str(p.get("name") or "") for p in player_records}
    rows = []
    for row in roster_rows:
        pid = int(row.get("player_id") or 0)
        rows.append(
            {
                "player_id": pid,
                "name": name_by_id.get(pid, f"Player #{pid}"),
                "status": str(row.get("status") or "EXPECTED"),
                "rating_snapshot": float(row.get("rating_snapshot") or 0),
            }
        )
    st.subheader("Roster")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    needs = [r for r in rows if float(r.get("rating_snapshot") or 0) <= 0]
    st.subheader("Needs rating queue")
    if not needs:
        st.caption("No players require rating input.")
        return

    for item in needs:
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        col1.write(item["name"])
        preset = col2.selectbox(
            "Preset",
            options=[800, 1000, 1200, 1400, 1600],
            index=2,
            key=f"sl_rate_preset_{session_id}_{item['player_id']}",
        )
        slider = col3.slider(
            "Rating",
            min_value=600,
            max_value=2200,
            value=int(preset),
            step=25,
            key=f"sl_rate_slider_{session_id}_{item['player_id']}",
        )
        if col4.button("Save", key=f"sl_rate_save_{session_id}_{item['player_id']}"):
            session_ladder_api.post_add_roster_entries(
                supabase=ctx.supabase,
                auth=_auth_payload(ctx),
                payload={
                    "session_id": session_id,
                    "mode": "manual_existing",
                    "player_id": int(item["player_id"]),
                    "status": "CHECKED_IN",
                    "rating_snapshot": float(slider),
                },
            )
            st.success(f"Saved rating for {item['name']}.")
            st.rerun()




def _can_edit_games(ctx) -> bool:
    jwt_payload = st.session_state.get("jwt_payload") or {}
    role = str(jwt_payload.get("role") or "").strip().lower()
    return bool(getattr(ctx, "admin_logged_in", False)) or role in {"admin", "manager"}


def _derive_game_inputs(game: dict[str, Any]) -> tuple[str, int]:
    sa = game.get("score_a")
    sb = game.get("score_b")
    if sa is None or sb is None:
        return ("teamA", 0)
    sa_i, sb_i = int(sa), int(sb)
    if sa_i >= sb_i:
        return ("teamA", min(sb_i, 10))
    return ("teamB", min(sa_i, 10))


def _save_game_from_inputs(ctx, *, session_id: str, court_pod_id: str, game: dict[str, Any], winner: str, losing_points: int) -> None:
    team_a = list(game.get("team_a_player_ids") or [])
    team_b = list(game.get("team_b_player_ids") or [])
    lose_pts = max(0, min(int(losing_points), 10))
    if winner == "teamA":
        score_a, score_b = 11, lose_pts
    else:
        score_a, score_b = lose_pts, 11
    session_ladder_api.post_submit_game_result(
        supabase=ctx.supabase,
        auth=_auth_payload(ctx),
        payload={
            "session_id": session_id,
            "court_pod_id": str(court_pod_id),
            "game_number": int(game.get("game_number") or 0),
            "teamA_player_ids": team_a,
            "teamB_player_ids": team_b,
            "scoreA": int(score_a),
            "scoreB": int(score_b),
        },
    )


def _render_standings_with_tiebreaks(standings: list[dict[str, Any]]) -> None:
    if not standings:
        st.caption("No completed games yet.")
        return
    rows = []
    for row in standings:
        rows.append(
            {
                "Rank": int(row.get("rank") or 0),
                "Player": int(row.get("player_id") or 0),
                "W": int(row.get("wins") or 0),
                "L": int(row.get("losses") or 0),
                "PF": int(row.get("pf") or 0),
                "PA": int(row.get("pa") or 0),
                "PD": int(row.get("pd") or 0),
                "Tie-break": str(row.get("tie_break") or "Wins->PD->PF"),
                "Playoff": bool(row.get("playoff_required") or False),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_court_sheet(ctx, *, session_id: str, court_item: dict[str, Any]) -> None:
    pod = court_item.get("pod") or {}
    players = sorted(court_item.get("players") or [], key=lambda x: int(x.get("player_order") or 0))
    games = sorted(court_item.get("games") or [], key=lambda x: int(x.get("game_number") or 0))
    standings = court_item.get("standings") or []

    st.subheader(f"Court {int(pod.get('court_number') or 0)} • Round {int(pod.get('round_number') or 0)}")
    st.caption("Score entry rule: choose winning team and losing points (0-10); winner auto-saves as 11.")

    pchips = [f"P{int(p.get('player_order') or 0)}: {int(p.get('player_id') or 0)}" for p in players]
    st.write("Players:", " • ".join(pchips) if pchips else "(none)")

    editable = _can_edit_games(ctx)
    if not editable:
        st.warning("Read-only: only manager/admin can edit games.")

    for game in games:
        team_a = list(game.get("team_a_player_ids") or [])
        team_b = list(game.get("team_b_player_ids") or [])
        winner_default, lose_default = _derive_game_inputs(game)

        st.markdown(f"**Game {int(game.get('game_number') or 0)}**")
        c1, c2, c3, c4 = st.columns([4, 2, 2, 2])
        c1.write(f"Team A {team_a} vs Team B {team_b}")
        winner = c2.selectbox(
            "Winner",
            options=["teamA", "teamB"],
            index=0 if winner_default == "teamA" else 1,
            key=f"sl_game_winner_{session_id}_{pod.get('id')}_{game.get('game_number')}",
            disabled=not editable,
            label_visibility="collapsed",
        )
        lose_pts = c3.number_input(
            "Lose pts",
            min_value=0,
            max_value=10,
            step=1,
            value=int(lose_default),
            key=f"sl_game_lose_{session_id}_{pod.get('id')}_{game.get('game_number')}",
            disabled=not editable,
            label_visibility="collapsed",
        )
        if c4.button("Autosave", key=f"sl_game_save_{session_id}_{pod.get('id')}_{game.get('game_number')}", disabled=not editable):
            _save_game_from_inputs(
                ctx,
                session_id=session_id,
                court_pod_id=str(pod.get("id") or ""),
                game=game,
                winner=str(winner),
                losing_points=int(lose_pts),
            )
            st.success(f"Saved game {int(game.get('game_number') or 0)}")
            st.rerun()

    st.subheader("Standings")
    _render_standings_with_tiebreaks(standings)

    playoff_candidates = [r for r in standings if bool(r.get("playoff_required"))]
    if playoff_candidates:
        st.error("Playoff Required: tie remains after Wins → PD → PF → H2H")
        options = [str(r.get("player_id")) for r in playoff_candidates]
        p1, p2, p3 = st.columns([3, 2, 1])
        winner = p1.selectbox("Playoff winner", options=options, key=f"sl_playoff_winner_{session_id}_{pod.get('id')}")
        losing = p2.number_input("Optional losing score", min_value=0, max_value=10, step=1, value=0, key=f"sl_playoff_loser_{session_id}_{pod.get('id')}")
        if p3.button("Record", key=f"sl_playoff_record_{session_id}_{pod.get('id')}", disabled=not editable):
            st.session_state[f"sl_playoff_recorded_{session_id}_{pod.get('id')}"] = {"winner": winner, "losing_score": int(losing)}
            st.success("Playoff winner recorded for manager workflow.")

        recorded = st.session_state.get(f"sl_playoff_recorded_{session_id}_{pod.get('id')}")
        if recorded:
            st.info(f"Recorded playoff winner: {recorded['winner']} (losing score {recorded['losing_score']})")






def _render_publish_payload(session: dict[str, Any]) -> None:
    recap = session.get("recap_json") or session.get("recap") or {}
    leaderboard = session.get("leaderboard_json") or session.get("leaderboard") or []
    if recap:
        st.subheader("Session recap")
        st.json(recap)
    if leaderboard:
        st.subheader("Updated ratings leaderboard")
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

def _is_round_complete(court_items: list[dict[str, Any]], players_per_court: int) -> bool:
    required = 3 if int(players_per_court) == 4 else 5
    for item in court_items:
        games = item.get("games") or []
        complete = [g for g in games if g.get("score_a") is not None and g.get("score_b") is not None]
        if len(complete) < required:
            return False
    return True


def _round_has_unresolved_playoff(court_items: list[dict[str, Any]]) -> bool:
    for item in court_items:
        standings = item.get("standings") or []
        if any(bool(r.get("playoff_required")) for r in standings):
            return True
    return False


def build_print_pack_html(session: dict[str, Any], grouped: dict[int, list[dict[str, Any]]]) -> str:
    session_id = str(session.get("id") or "")
    parts = [
        "<html><head><style>",
        "body{font-family:Arial,sans-serif;color:#111;margin:16px;} h1,h2,h3{margin:0 0 8px;} .page{page-break-after:always;} table{width:100%;border-collapse:collapse;margin:8px 0;} th,td{border:1px solid #ccc;padding:6px;font-size:12px;} .muted{color:#555;} @media print{ .page{page-break-after:always;} }",
        "</style></head><body>",
        f"<h1>Session Print Pack</h1><p class='muted'>Session {session_id}</p>",
    ]

    parts.append("<h2>Court Assignments Summary</h2><table><tr><th>Round</th><th>Court</th><th>Players</th></tr>")
    for rnd in sorted(grouped):
        for item in sorted(grouped[rnd], key=lambda x: int((x.get("pod") or {}).get("court_number") or 0)):
            pod = item.get("pod") or {}
            players = item.get("players") or []
            ptxt = ", ".join(str(int(p.get("player_id") or 0)) for p in players)
            parts.append(f"<tr><td>{rnd}</td><td>{int(pod.get('court_number') or 0)}</td><td>{ptxt}</td></tr>")
    parts.append("</table>")

    for rnd in sorted(grouped):
        for item in sorted(grouped[rnd], key=lambda x: int((x.get("pod") or {}).get("court_number") or 0)):
            pod = item.get("pod") or {}
            players = item.get("players") or []
            games = item.get("games") or []
            parts.append("<div class='page'>")
            parts.append(f"<h2>Round {rnd} • Court {int(pod.get('court_number') or 0)}</h2>")
            parts.append("<h3>Players</h3><table><tr><th>Order</th><th>Player</th></tr>")
            for p in players:
                parts.append(f"<tr><td>{int(p.get('player_order') or 0)}</td><td>{int(p.get('player_id') or 0)}</td></tr>")
            parts.append("</table>")
            parts.append("<h3>Games</h3><table><tr><th>#</th><th>Team A</th><th>Team B</th><th>Score</th></tr>")
            for g in games:
                ta = ",".join(str(x) for x in (g.get("team_a_player_ids") or []))
                tb = ",".join(str(x) for x in (g.get("team_b_player_ids") or []))
                sa = g.get("score_a")
                sb = g.get("score_b")
                score = "-" if sa is None or sb is None else f"{int(sa)}-{int(sb)}"
                parts.append(f"<tr><td>{int(g.get('game_number') or 0)}</td><td>{ta}</td><td>{tb}</td><td>{score}</td></tr>")
            parts.append("</table></div>")

    parts.append("</body></html>")
    return "".join(parts)

def render(ctx):
    mode_label = "Public" if bool(getattr(ctx, "public_mode", False)) else "Admin"
    page_shell("🗂️ Session Console", "Session Ladder shell: routing, stepper, data load, resume pointer", mode_label=mode_label)

    if not bool(FEATURE_SESSION_LADDER):
        st.info("Session Ladder is disabled by feature flag.")
        return

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    raw_route = str(st.query_params.get("route", "") or "").strip().strip("/")
    route_info = parse_session_route(raw_route)
    user_key = str((st.session_state.get("jwt_payload") or {}).get("sub") or "anonymous")

    if route_info is None:
        fallback = str(st.session_state.get("session_console_last_route") or st.query_params.get("sl_last_route") or "").strip()
        if fallback and parse_session_route(fallback):
            st.query_params["route"] = fallback
            st.rerun()
        st.info("Open a session route, e.g. /sessions/:id or /sessions/:id/rounds/:round/courts/:courtId")
        return

    session_id = str(route_info["session_id"])
    try:
        details = session_ladder_api.get_session_details(
            supabase=ctx.supabase,
            auth=_auth_payload(ctx),
            session_id=session_id,
        )
    except Exception as exc:
        st.error(f"Unable to load session details: {exc}")
        return

    session = details.get("session") or {}
    courts = details.get("courts") or []
    state = str(session.get("state") or "DRAFT")
    roster_rows = _load_roster(ctx.supabase, session_id)
    player_records = []
    if getattr(ctx, "df_players_all", None) is not None and not ctx.df_players_all.empty:
        player_records = ctx.df_players_all.to_dict(orient="records")

    steps = ["Roster", "Seeding", "Round 1", "Round 2", "Round 3", "Publish"]
    if not any(str(p.get("pod", {}).get("round_number")) == "3" for p in courts):
        steps = ["Roster", "Seeding", "Round 1", "Round 2", "Publish"]
    active_step = min(_step_index(state), len(steps) - 1)

    st.write("**Progress**")
    st.progress((active_step + 1) / max(1, len(steps)))
    st.caption(" → ".join([f"[{s}]" if i == active_step else s for i, s in enumerate(steps)]))

    st.write(f"Session: `{session_id}` | State: `{state}`")

    _render_step_help_links(state=state)

    _render_roster_intake(ctx, session_id=session_id, roster_rows=roster_rows, player_records=player_records)
    _render_roster_table(ctx, session_id=session_id, roster_rows=roster_rows, player_records=player_records)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in courts:
        round_number = int((item.get("pod") or {}).get("round_number") or 0)
        grouped.setdefault(round_number, []).append(item)

    if route_info["is_court"]:
        selected_round = int(route_info["round_number"])
        selected_court = str(route_info["court_id"])
        selected = None
        for item in courts:
            pod = item.get("pod") or {}
            if int(pod.get("round_number") or 0) == selected_round and str(pod.get("court_number") or "") == selected_court:
                selected = item
                break
        if selected is not None:
            _render_court_sheet(ctx, session_id=session_id, court_item=selected)
        else:
            st.warning("Court route not found in this session.")

    # Round closeout + next-round generation controls
    if grouped:
        st.subheader("Round closeout")
        round_options = sorted(grouped.keys())
        selected_round_to_close = st.selectbox("Round", options=round_options, key=f"sl_close_round_pick_{session_id}")
        selected_items = grouped.get(int(selected_round_to_close), [])
        complete_ok = _is_round_complete(selected_items, int(session.get("players_per_court") or 4))
        playoff_ok = not _round_has_unresolved_playoff(selected_items)

        override = st.checkbox("Manager override", value=False, key=f"sl_close_override_{session_id}")
        reason = st.text_input("Override reason", key=f"sl_close_override_reason_{session_id}") if override else ""

        if not complete_ok:
            st.warning("Round cannot be closed yet: not all courts have complete games.")
        if not playoff_ok:
            st.warning("Round cannot be closed yet: unresolved playoff required.")

        can_close = (complete_ok and playoff_ok) or (override and str(reason).strip())
        movers_per_court = st.selectbox("Movers per court", options=[1, 2], index=0, key=f"sl_movers_per_court_{session_id}")
        if st.button("Close Round", type="primary", disabled=not can_close, key=f"sl_close_round_btn_{session_id}"):
            try:
                out = session_ladder_api.post_close_round(
                    supabase=ctx.supabase,
                    auth=_auth_payload(ctx),
                    session_id=session_id,
                    round_number=int(selected_round_to_close),
                    movers_per_court=int(movers_per_court),
                    allow_override=bool(override),
                    override_reason=str(reason).strip() if override else None,
                )
                st.success(f"Round {selected_round_to_close} closed. Next round generated: {bool(out.get('generated_next_round'))}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.subheader("Completion & publish")
    c1, c2 = st.columns(2)
    if c1.button("Complete Session", key=f"sl_complete_{session_id}", disabled=str(state).upper() in {"COMPLETED", "PUBLISHED"}):
        try:
            out = session_ladder_api.post_complete_session(
                supabase=ctx.supabase,
                auth=_auth_payload(ctx),
                session_id=session_id,
            )
            st.success(f"Session completed. Ratings updated: {out.get('session', {}).get('rating_update', {})}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if c2.button("Publish Session", key=f"sl_publish_{session_id}", disabled=str(state).upper() == "PUBLISHED"):
        try:
            out = session_ladder_api.post_publish_session(
                supabase=ctx.supabase,
                auth=_auth_payload(ctx),
                session_id=session_id,
            )
            st.success("Session published.")
            if out.get("session"):
                _render_publish_payload(out.get("session") or {})
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if str(state).upper() == "PUBLISHED":
        _render_publish_payload(session)

    st.subheader("Print Pack")
    html = build_print_pack_html(session, grouped)
    st.download_button(
        "Download Print Pack (HTML)",
        data=html.encode("utf-8"),
        file_name=f"session_{session_id}_print_pack.html",
        mime="text/html",
        key=f"sl_print_pack_{session_id}",
    )

    st.subheader("Rounds")
    for round_number in sorted(grouped):
        with st.expander(f"Round {round_number}", expanded=bool(route_info.get("round_number") == round_number)):
            for item in sorted(grouped[round_number], key=lambda x: int((x.get("pod") or {}).get("court_number") or 0)):
                pod = item.get("pod") or {}
                court_number = int(pod.get("court_number") or 0)
                route = build_session_route(session_id, round_number=round_number, court_id=court_number)
                cols = st.columns([3, 2])
                cols[0].write(f"Court {court_number} • {len(item.get('players') or [])} players • {len(item.get('games') or [])} games")
                if cols[1].button("Open court sheet", key=f"open_court_{session_id}_{round_number}_{court_number}"):
                    st.query_params["route"] = route
                    st.session_state["_nav_pending"] = "🗂️ Session Console"
                    st.rerun()

    _store_resume_pointer(user_key, raw_route)
