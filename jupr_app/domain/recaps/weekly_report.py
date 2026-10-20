"""Weekly report snapshots. Private source records never leave this module.

Results are read from canonical saved games and published documents. Generator
state supplies format-specific presentation only; it never adds games twice.
"""
from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd

from jupr_app.data.paged_reads import read_all_rows


def records(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    return list(value or [])


def clean(value):
    return str(value or "").strip()


def played_day(value, tz_name="America/Mazatlan"):
    raw = clean(value)
    if len(raw) == 10:
        return raw
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    return stamp.tz_convert(tz_name).date().isoformat() if not pd.isna(stamp) else ""


def completed_games(event):
    for round_row in event.get("rounds", []):
        if round_row.get("status") != "saved":
            continue
        games = round_row.get("matches") or [m for c in round_row.get("courts", []) for m in c.get("matches", [])]
        for game in games:
            a, b = game.get("scoreA"), game.get("scoreB")
            if a is not None and b is not None and a != b:
                yield round_row, game


def round_robin_summary(rows, *, final=True):
    leaders = [r for r in rows if r.get("rank") == 1]
    if not leaders:
        return "No completed results"
    names = ", ".join(f"{r['name']} ({r.get('wins', 0)}–{r.get('losses', 0)})" for r in leaders)
    if any(r.get("winnerNeedsAdmin") for r in leaders):
        return f"Tied for first: {names} · awaiting the club's winner decision"
    return ("Winner: " if final and len(leaders) == 1 else "Standings leader: ") + names


def generator_results_href(row, event):
    """Only link to an enabled, readable public results surface."""
    from jupr_app.services.public_play_generator_service import _not_expired

    state = row.get("state") or {}
    if (event.get("generatorKind") == "round_robin"
            and (state.get("generator_sharing") or {}).get("enabled")
            and row.get("status") in {"active", "completed"}
            and (state.get("mode") == "admin_play_generator" or _not_expired(row))
            and clean(row.get("session_key"))):
        return f"round-robin-generator/shared/{quote(str(row['session_key']), safe='')}/standings"
    return ""


def social_result(event_row, social_data):
    """Use the event's recorded RR tiebreak decision, never the weekly spotlight."""
    from jupr_app.domain.live_beta_engine import round_robin_standings, league_aggregate_standings

    event_id = str(event_row["id"])
    event = event_row.get("raw_event_json") or {}
    if not isinstance(event, dict) or not event.get("rounds"):
        # Older saved events may predate raw snapshots. Rebuild only saved seats.
        people = [p for p in social_data.get("participants", []) if str(p.get("event_id")) == event_id]
        ids = {str(p["id"]) for p in people}
        games = []
        for game in social_data.get("matches", []):
            if str(game.get("event_id")) != event_id:
                continue
            seats = [str(game.get(k) or "") for k in ("t1_p1_participant_id", "t1_p2_participant_id", "t2_p1_participant_id", "t2_p2_participant_id")]
            a, b = game.get("score_t1"), game.get("score_t2")
            if len(set(seats)) != 4 or not set(seats) <= ids or a is None or b is None or a == b:
                continue
            games.append({"id": str(game.get("id")), "teamA": seats[:2], "teamB": seats[2:], "scoreA": a, "scoreB": b})
        event = {"type": "round_robin", "participants": [{"id": str(p["id"]), "name": p.get("display_name_snapshot") or "Player"} for p in people], "rounds": [{"matches": games}]}
    is_rr = event_row.get("event_type") == "round_robin"
    # Raw ladder snapshots keep the court/mini-round structure used by the engine.
    rows = (round_robin_standings if is_rr or event.get("type") == "round_robin" else league_aggregate_standings)(event)
    rows = [r for r in rows if int(r.get("matches") or 0) > 0]
    if not rows:
        return None
    details = [f"{r.get('rank', i)}. {r['name']} — {r['wins']}–{r['losses']} · point difference {r['differential']:+d}" for i, r in enumerate(rows[:5], 1)]
    return {"id": event_id, "title": event_row.get("name") or "Social round robin", "kind": "Social round robin" if is_rr else "Social ladder", "date": event_row.get("event_date") or "", "summary": round_robin_summary(rows) if is_rr else f"Standings leader: {rows[0]['name']} ({rows[0]['wins']}–{rows[0]['losses']})", "details": details, "href": ""}


def league_weekly_reports(name, matches):
    from jupr_app.services.public_league_results_service import _parse_week_num
    from jupr_app.domain.recaps.weekly_recap import _resolve_event_key

    weeks = sorted({week for m in matches if _resolve_event_key(m) == ("LEAGUE", name)
                    and (week := _parse_week_num(m.get("week_tag"))) is not None})
    return [{"label": f"Weekly report · Week {week}", "href": f"leagues/{quote(name, safe='')}/weekly-history?week={week}"} for week in weeks]


def generator_result(row, start_date, end_date, matches=None):
    """Only approved/published generators become public result summaries."""
    from jupr_app.domain.adaptive_play_engine import generator_event_standings
    from jupr_app.domain.moneyball_expectations import is_moneyball, settlement

    state = row.get("state") or {}
    event = (state.get("page_state") or {}).get("event") or {}
    submission = state.get("generator_submission") or {}
    official = state.get("official_publish") or {}
    reviewed_date = ((submission.get("review") or {}).get("request") or {}).get("match_date")
    played_on = clean(reviewed_date or submission.get("match_date") or official.get("match_date"))[:10]
    if not played_on:
        contexts = set((official.get("match_context_by_live_id") or {}).values())
        dates = [played_day(m.get("date")) for m in (matches or []) if clean(m.get("context_id")) in contexts]
        played_on = min(dates) if dates else ""
    if not played_on or not start_date <= played_on <= end_date:
        return None, None
    title = clean(row.get("title") or event.get("name") or "Round robin")
    coverage = {"title": title, "date": played_on, "status": clean(submission.get("status") or "published")}
    if submission.get("status") != "approved" and not official.get("published_match_ids"):
        return None, coverage
    games = list(completed_games(event))
    if not games:
        return None, coverage
    participants = {str(p["id"]): p for p in event.get("participants", [])}
    played = {str(pid) for _, game in games for side in ("A", "B") for pid in game.get(f"side{side}", game.get(f"team{side}", []))}
    rows = generator_event_standings(event)
    rows = [r for r in rows if int(r.get("matches") or 0) > 0]
    details = [f"{len(played)} players · {len(games)} games · {'Unrated' if event.get('ratingMode') == 'unrated' else 'Rated'}"]
    kind = "Ladder" if event.get("generatorKind") == "ladder" else "Round robin"
    if is_moneyball(event):
        kind = "Moneyball"
        outcome = settlement(event, rows)
        names = [w["name"] for w in outcome.get("winners", [])]
        summary = f"Winner{'s' if len(names) != 1 else ''}: {', '.join(names)}" if names else "No eligible winner" if outcome.get("final") else "Results remain provisional"
        for r in rows[:5]:
            eligible = r.get("moneyballEligible", True)
            details.append(f"{r['name']}: {r['wins']} wins vs {r['expectedWins']:.1f} expected; point difference {r['differential']:+d} vs {r['expectedDifferential']:+.1f} expected" + (" · ineligible after substitution" if not eligible else ""))
    else:
        summary = round_robin_summary(rows, final=event.get("status") == "completed" and event.get("type") == "round_robin" and not event.get("playoff"))
        for round_row, game in games:
            if round_row.get("stage") != "playoff":
                continue
            side = "A" if game["scoreA"] > game["scoreB"] else "B"
            names = [clean(participants.get(str(pid), {}).get("name")) for pid in game.get(f"side{side}", [])]
            details.append(f"{game.get('label') or round_row.get('label') or 'Playoff'}: {' + '.join(names)} won {max(game['scoreA'], game['scoreB'])}–{min(game['scoreA'], game['scoreB'])}")
    return {"id": clean(row.get("session_key")), "title": title, "kind": kind, "date": played_on, "summary": summary, "details": details, "href": generator_results_href(row, event)}, coverage


def _season_races(ctx, start_date, end_date):
    from jupr_app.services.admin_league_awards_service import (
        _award_inputs, _award_minimum, _award_race_entries, _has_explicit_award_categories,
    )
    players = records(getattr(ctx, "df_players_all", None))
    all_matches = records(getattr(ctx, "df_matches", None))
    leagues = records(getattr(ctx, "df_leagues", None))
    metadata = records(getattr(ctx, "df_meta", None))
    results = []
    for meta in metadata:
        name = clean(meta.get("league_name"))
        # A recap should not revive archived seasons with current rating data.
        if not name or meta.get("is_archived") is True or clean(meta.get("status")).lower() in {"archived", "closed", "completed"}:
            continue
        if not any(clean(m.get("league")) == name and start_date <= played_day(m.get("date")) <= end_date for m in all_matches):
            continue
        if not _has_explicit_award_categories(meta):
            continue
        config = meta.get("awards_config") or {}
        if not isinstance(config, dict):
            continue
        snapshots = []
        for cutoff in (end_date, (date.fromisoformat(start_date) - timedelta(days=1)).isoformat()):
            subset = [m for m in all_matches if played_day(m.get("date")) <= cutoff]
            enriched, *_ = _award_inputs(ctx.supabase, club_id=str(ctx.club_id), league_name=name, metadata=meta,
                league_rows=[r for r in leagues if clean(r.get("league")) == name], match_rows=subset, player_rows=players, team_rows=[], fixture_rows=[])
            snapshots.append(enriched.get("_analytics") or {})
        current, previous = snapshots
        catalog = {s["key"]: s for s in current.get("catalog", [])}
        for key, setting in (config.get("categories") or {}).items():
            if not isinstance(setting, dict) or setting.get("enabled") is False or key not in catalog:
                continue
            spec = catalog[key]
            if spec.get("recipient_type") == "team":
                continue
            minimum = _award_minimum(setting, meta, config)
            eligible = _award_race_entries(spec, current.get("player_analytics", []), minimum=minimum)
            prior = _award_race_entries(spec, previous.get("player_analytics", []), minimum=minimum)
            prior_rank = {r.get("player_id"): r.get("rank") for r in prior}
            lines = []
            for entry in eligible[:3]:
                before = prior_rank.get(entry.get("player_id"))
                change = f" · up {before - entry['rank']}" if before and before > entry["rank"] else f" · down {entry['rank'] - before}" if before and before < entry["rank"] else ""
                lines.append(f"{entry['rank']}. {entry['recipient_name']} — {entry['metric_display']}{change}")
            metric = spec.get("minimum_metric") or "games"
            pending = sorted([p for p in current.get("player_analytics", []) if 0 < float(p.get(metric) or 0) < minimum], key=lambda p: (-float(p.get(metric) or 0), clean(p.get("player_name"))))
            for p in pending[:2]:
                lines.append(f"{p.get('player_name')}: {int(p.get(metric) or 0)} of {minimum} {metric.replace('_', ' ')} to qualify")
            if lines:
                results.append({"title": f"{name} · {spec['label']}", "lines": lines, "href": f"leagues/{quote(name, safe='')}/standings"})
    return results


def _achievements(ctx, matches):
    from jupr_app.domain.recaps.player_weekly_digest import _award_trigger
    from jupr_app.domain.gamification.badge_registry import badge_schema_by_id

    match_ids = {clean(m.get("id")) for m in matches if m.get("id") is not None}
    names = getattr(ctx, "id_to_name", {}) or {}
    definitions = {clean(r.get("badge_id")): clean(r.get("name")) for r in records(getattr(ctx, "df_badges", None))}
    definitions.update({str(k): v.title for k, v in badge_schema_by_id().items() if str(k) not in definitions})
    lines, seen = [], set()
    badge_rows = read_all_rows(lambda: ctx.supabase.table("player_badges")
        .select("id,player_id,badge_id,match_id,value_json,revoked_at").eq("club_id", str(ctx.club_id))) if getattr(ctx, "supabase", None) is not None else records(getattr(ctx, "df_player_badges", None))
    for badge in badge_rows:
        if badge.get("revoked_at") and not pd.isna(badge.get("revoked_at")):
            continue
        trigger = _award_trigger(badge)
        if trigger not in match_ids:
            continue
        identity = (clean(badge.get("player_id")), clean(badge.get("badge_id")))
        if identity in seen:
            continue
        seen.add(identity)
        pid = int(badge["player_id"])
        lines.append(f"{names.get(pid, f'Player #{pid}')} — {definitions.get(identity[1], identity[1])}")
    return lines


def build_weekly_report(ctx, recap, df_week, social_data, start_date, end_date):
    """Optional sections fail visibly in admin coverage, never fabricate results."""
    from jupr_app.domain.recaps.recap_evidence import generator_courts, league_courts, social_courts
    start, end = start_date.isoformat(), end_date.isoformat()
    db, club_id = getattr(ctx, "supabase", None), str(ctx.club_id)
    report = {"version": 1, "club_name": "Your club", "club_slug": "", "start_date": start, "end_date": end, "timezone": (recap.get("meta") or {}).get("tz_name", "America/Mazatlan"), "headline": "This week on the courts", "opening": "", "event_notes": "", "upcoming": [], "event_results": [], "season_races": [], "achievements": [], "new_faces": [], "coverage": {"included": [], "pending": [], "warnings": []}}
    scoped_leagues = getattr(ctx, "recap_leagues", []) or []
    report["scope_label"] = ", ".join(scoped_leagues) if scoped_leagues else "Whole club"
    coverage = report["coverage"]
    matches = records(df_week)
    if db is not None:
        try:
            clubs = db.table("clubs").select("name,slug").eq("id", club_id).execute().data or []
            if clubs:
                report.update(club_name=clubs[0].get("name") or "Your club", club_slug=clubs[0].get("slug") or "")
        except Exception:
            coverage["warnings"].append("Club branding could not be loaded.")
        try:
            sessions = read_all_rows(lambda: db.table("live_sessions").select("session_key,title,state,status,expires_at")
                .eq("club_id", club_id).in_("state->>mode", ["admin_play_generator", "public_play_generator"]), order="session_key")
            for row in ([] if scoped_leagues else sessions):
                try:
                    result, source = generator_result(row, start, end, matches)
                    if result:
                        report["event_results"].append(result)
                        report["event_results"].extend(generator_courts(row, result))
                        coverage["included"].append(source)
                    elif source and source["status"] in {"pending", "processing"}:
                        coverage["pending"].append(source)
                except Exception:
                    coverage["warnings"].append(f"The result for {clean(row.get('title')) or 'a generator session'} could not be verified.")
        except Exception:
            coverage["warnings"].append("Generator session coverage could not be verified. Check results before publishing.")
        try:
            courts, warnings = league_courts(ctx, matches, start, end)
            report["event_results"].extend(courts)
            coverage["warnings"].extend(warnings)
        except Exception:
            coverage["warnings"].append("Ladder court results could not be verified.")
        try:
            report["season_races"] = _season_races(ctx, start, end)
        except Exception:
            coverage["warnings"].append("Season award progress could not be verified.")
        try:
            report["achievements"] = _achievements(ctx, matches)
        except Exception:
            coverage["warnings"].append("New achievements could not be verified.")
        try:
            if not scoped_leagues:
                _add_tournament_draws(db, club_id, matches, report, getattr(ctx, "id_to_name", {}) or {})
        except Exception:
            coverage["warnings"].append("Tournament division podiums could not be verified.")
        try:
            if not scoped_leagues:
                _add_interclub(db, club_id, start, end, report)
        except Exception:
            coverage["warnings"].append("Published interclub results could not be verified.")
        try:
            report["new_faces"] = _new_faces(db, club_id, matches, social_data, start, end, getattr(ctx, "id_to_name", {}) or {})
            recap["numbers"]["new_faces"] = len(report["new_faces"])
            for card in recap.get("numbers_cards", []):
                if card.get("key") == "new_faces":
                    card["value"] = len(report["new_faces"])
        except Exception:
            coverage["warnings"].append("First-time player history could not be fully verified.")
    # Retain named leagues and legacy social events, even without generators.
    for item in recap.get("around_club", {}).get("leagues", []):
        report["event_results"].append({"title": item["league_name"], "kind": "League", "date": "", "summary": "", "details": [h["display"] for h in item.get("highlights", [])], "href": f"leagues/{quote(item['league_name'], safe='')}/standings", "weekly_reports": league_weekly_reports(item["league_name"], matches)})
    for item in recap.get("around_club", {}).get("round_robins", []):
        report["event_results"].append({"title": item["event_name"], "kind": "Round robin", "date": "", "summary": "", "details": [h["display"] for h in item.get("highlights", [])], "href": ""})
    for event in social_data.get("events", []):
        try:
            result = social_result(event, social_data)
        except Exception:
            coverage["warnings"].append(f"The social result for {clean(event.get('name')) or 'an event'} could not be verified.")
            continue
        if not result:
            continue
        report["event_results"].append(result)
        try:
            report["event_results"].extend(social_courts(event, result))
        except Exception:
            coverage["warnings"].append(f"Court results for {clean(event.get('name')) or 'a social event'} could not be verified.")
        coverage["included"].append({"title": event.get("name") or "Social play", "date": event.get("event_date"), "status": "saved"})
    report["event_results"].sort(key=lambda e: (e.get("date") or "", e["title"]))
    coverage["games"] = int(recap.get("numbers", {}).get("matches", 0))
    coverage["players"] = int(recap.get("numbers", {}).get("players", 0))
    return report


def _add_tournament_draws(db, club_id, matches, report, names):
    from jupr_app.services.public_tournament_results_service import build_public_tournament_results
    from jupr_app.services.public_tournament_team_service import build_public_team_tournament_results
    from jupr_app.domain.recaps.weekly_recap import get_date_range_bounds
    start, end = get_date_range_bounds(date.fromisoformat(report["start_date"]), date.fromisoformat(report["end_date"]), report.get("timezone", "America/Mazatlan"))
    receipts = read_all_rows(lambda: db.table("tournament_lifecycle_receipts")
        .select("id,tournament_id,created_at").eq("club_id", club_id).eq("action", "complete")
        .gte("created_at", start.isoformat()).lte("created_at", end.isoformat()))
    ids = sorted({str(row["tournament_id"]) for row in receipts})
    if not ids:
        return
    tournaments = read_all_rows(lambda: db.table("tournaments").select("id,name")
        .eq("club_id", club_id).in_("id", ids).in_("status", ["COMPLETE", "COMPLETED"]))
    for tournament in tournaments:
        result = build_public_tournament_results(db, club_id=club_id, tournament_id=tournament["id"])
        draws = list(result.get("draws") or [])
        team_draws = read_all_rows(lambda: db.table("tournament_event_draws").select("id")
            .eq("tournament_id", tournament["id"]).eq("draw_kind", "TEAM_PARENT").eq("status", "published"))
        for team_draw in team_draws:
            team_result = build_public_team_tournament_results(db, club_id=club_id, tournament_id=tournament["id"], draw_id=team_draw["id"])
            draw = {**team_result.get("draw", {}), "podium": team_result.get("podium", []), "team_draw_id": team_draw["id"]}
            draws.append(draw)
        for draw in draws:
            medals = draw.get("podium") or []
            if not medals:
                continue
            division = " · ".join(dict.fromkeys(filter(None, [draw.get("event_family_label"), draw.get("division_name")]))) or draw.get("name") or "Division"
            href = f"tournament-results?tournament_id={quote(str(tournament['id']))}"
            if draw.get("team_draw_id"):
                href = f"tournament-team-results/{quote(str(tournament['id']))}/{quote(str(draw['team_draw_id']))}"
            report["event_results"].append({"title": f"{tournament['name']} · {division}", "kind": "Tournament", "date": "", "summary": "Division podium", "details": [f"{p['placement']}. {p.get('team_name') or p.get('display_name') or 'Team'}" for p in sorted(medals, key=lambda p: p.get("placement") or 99)], "href": href})


def _add_interclub(db, club_id, start, end, report):
    memberships = read_all_rows(lambda: db.table("pcs_interclub_participations").select("season_id").eq("club_id", club_id).eq("status", "accepted"), order="season_id")
    for membership in memberships:
        publications = db.table("pcs_interclub_publications").select("published").eq("season_id", membership["season_id"]).execute().data or []
        for publication in publications:
            doc = publication.get("published") or {}
            if not any(c.get("id") == club_id for c in doc.get("clubs", [])):
                continue
            clubs = {c["id"]: c.get("name") or c["id"] for c in doc.get("clubs", [])}
            dates = {m["id"]: played_day(m.get("starts_at"), doc.get("timezone") or "America/Mazatlan") for m in doc.get("meets", [])}
            for result in doc.get("competition_results", []):
                played_on = dates.get(result.get("meet_id"), "")
                if not start <= played_on <= end or club_id not in {result.get("club_a"), result.get("club_b")}:
                    continue
                outcome = result.get("outcome") or {}
                details = [f"Points: {outcome.get('points_a', 0)}–{outcome.get('points_b', 0)}", f"Games won: {outcome.get('games_a', 0)}–{outcome.get('games_b', 0)}"]
                report["event_results"].append({"title": f"{doc.get('name', 'Interclub')} · {result.get('division', '')}", "kind": "Interclub", "date": played_on, "summary": f"{clubs.get(result['club_a'], 'Club A')} vs {clubs.get(result['club_b'], 'Club B')}", "details": details, "href": f"/interclub/{quote(str(membership['season_id']), safe='')}"})


def _new_faces(db, club_id, matches, social_data, start, end, names):
    """First recorded play across both official and social histories."""
    identities = {}
    for m in matches:
        for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            value = m.get(key)
            if value is not None and not pd.isna(value):
                identities[("player", str(int(value)))] = names.get(int(value), f"Player #{int(value)}")
    from jupr_app.domain.recaps.weekly_recap import _social_identity_from_row, _compute_social_stats
    social_stats, _, _ = _compute_social_stats(social_data, names)
    for identity, stats in social_stats.items():
        identities[(identity[0], str(identity[1]))] = stats["display_name"]
    if not identities:
        return []
    # Full history is deliberately paged; a truncated page cannot prove "new".
    historical = read_all_rows(lambda: db.table("matches").select("id,date,t1_p1,t1_p2,t2_p1,t2_p2,score_t1,score_t2").eq("club_id", club_id))
    prior = set()
    for m in historical:
        if played_day(m.get("date")) >= start or m.get("score_t1") == m.get("score_t2"):
            continue
        for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            if m.get(key) is not None:
                prior.add(("player", str(m[key])))
    old_events = read_all_rows(lambda: db.table("live_events").select("id,event_date").eq("club_id", club_id).eq("result_mode", "social_unrated").eq("status", "saved").lt("event_date", start))
    for offset in range(0, len(old_events), 200):
        ids = [e["id"] for e in old_events[offset:offset + 200]]
        people = read_all_rows(lambda: db.table("live_event_participants").select("id,club_person_id,linked_player_id").in_("event_id", ids))
        games = read_all_rows(lambda: db.table("live_event_matches").select("id,t1_p1_participant_id,t1_p2_participant_id,t2_p1_participant_id,t2_p2_participant_id,score_t1,score_t2").in_("event_id", ids))
        appeared = {str(g.get(k)) for g in games for k in ("t1_p1_participant_id", "t1_p2_participant_id", "t2_p1_participant_id", "t2_p2_participant_id") if g.get("score_t1") != g.get("score_t2")}
        for p in people:
            identity = _social_identity_from_row(p)
            if identity and str(p["id"]) in appeared:
                prior.add((identity[0], str(identity[1])))
    return sorted([name for key, name in identities.items() if key not in prior], key=str.casefold)
