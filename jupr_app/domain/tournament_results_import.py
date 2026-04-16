from __future__ import annotations

import csv
from collections import Counter
from io import StringIO
from typing import Any


STAGE_OPTIONS = ["ROUND_ROBIN", "PLAYOFF", "FINAL", "BRONZE", "OTHER"]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_name(value: Any) -> str:
    return " ".join(_safe_text(value).lower().split())


def _normalize_dupr_id(value: Any) -> str | None:
    raw = _safe_text(value).upper().replace(" ", "")
    return raw or None


def _to_int(value: Any) -> int | None:
    text = _safe_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _team_key(p1_key: str, p2_key: str | None) -> str:
    return "|".join([p1_key, p2_key or "__SINGLES__"])


def parse_dupr_results_csv(uploaded_bytes: bytes) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    players_by_key: dict[str, dict[str, Any]] = {}
    team_index: dict[str, dict[str, Any]] = {}
    matches: list[dict[str, Any]] = []

    try:
        text = uploaded_bytes.decode("utf-8-sig")
    except Exception:
        text = uploaded_bytes.decode("latin-1", errors="ignore")

    reader = csv.DictReader(StringIO(text))
    headers = [str(h or "").strip() for h in (reader.fieldnames or [])]
    required_cols = ["playerA1", "playerB1"]
    missing = [col for col in required_cols if col not in headers]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return {"players": [], "teams": [], "matches": [], "event_meta": {}, "warnings": warnings, "errors": errors}

    def ensure_player(name: str, dupr_id: str | None, source_row: int) -> str:
        display_name = _safe_text(name)
        normalized_name = _normalize_name(display_name)
        norm_dupr = _normalize_dupr_id(dupr_id)
        import_key = f"dupr:{norm_dupr}" if norm_dupr else f"name:{normalized_name}"
        if import_key not in players_by_key:
            players_by_key[import_key] = {
                "import_key": import_key,
                "display_name": display_name,
                "normalized_name": normalized_name,
                "dupr_id": norm_dupr,
                "source_rows": [source_row],
            }
        else:
            players_by_key[import_key]["source_rows"].append(source_row)
            if display_name and not players_by_key[import_key].get("display_name"):
                players_by_key[import_key]["display_name"] = display_name
        return import_key

    for idx, row in enumerate(reader, start=2):
        a1 = _safe_text(row.get("playerA1"))
        b1 = _safe_text(row.get("playerB1"))
        a2 = _safe_text(row.get("playerA2"))
        b2 = _safe_text(row.get("playerB2"))
        if not a1 or not b1:
            warnings.append(f"Row {idx}: missing required side-A or side-B primary player; row skipped.")
            continue

        a1_key = ensure_player(a1, row.get("playerA1DuprId"), idx)
        b1_key = ensure_player(b1, row.get("playerB1DuprId"), idx)
        a2_key = ensure_player(a2, row.get("playerA2DuprId"), idx) if a2 else None
        b2_key = ensure_player(b2, row.get("playerB2DuprId"), idx) if b2 else None

        team_a_key = _team_key(a1_key, a2_key)
        team_b_key = _team_key(b1_key, b2_key)

        if team_a_key == team_b_key:
            warnings.append(f"Row {idx}: same team appears on both sides.")

        if team_a_key not in team_index:
            team_index[team_a_key] = {"team_key": team_a_key, "player_keys": [a1_key, a2_key], "source_rows": [idx]}
        else:
            team_index[team_a_key]["source_rows"].append(idx)
        if team_b_key not in team_index:
            team_index[team_b_key] = {"team_key": team_b_key, "player_keys": [b1_key, b2_key], "source_rows": [idx]}
        else:
            team_index[team_b_key]["source_rows"].append(idx)

        games: list[dict[str, int]] = []
        wins_a, wins_b = 0, 0
        for game_num in range(1, 6):
            sa = _to_int(row.get(f"teamAGame{game_num}"))
            sb = _to_int(row.get(f"teamBGame{game_num}"))
            if sa is None and sb is None:
                continue
            if sa is None or sb is None:
                warnings.append(f"Row {idx}: partial score for game {game_num}; ignored.")
                continue
            games.append({"game_number": game_num, "score_a": sa, "score_b": sb})
            if sa > sb:
                wins_a += 1
            elif sb > sa:
                wins_b += 1

        winner_side = "A" if wins_a > wins_b else "B" if wins_b > wins_a else None
        if games and winner_side is None:
            warnings.append(f"Row {idx}: winner could not be inferred from game scores.")
        if not games:
            warnings.append(f"Row {idx}: no valid game scores provided.")

        matches.append(
            {
                "source_row": idx,
                "team_a_key": team_a_key,
                "team_b_key": team_b_key,
                "games": games,
                "winner_side": winner_side,
                "stage": "PLAYOFF",
                "include": True,
                "score_summary": ", ".join([f"{g['score_a']}-{g['score_b']}" for g in games]) if games else "—",
                "match_type": _safe_text(row.get("matchType")),
                "event": _safe_text(row.get("event")),
                "date": _safe_text(row.get("date")),
            }
        )

    team_rows = list(team_index.values())
    duplicate_team_count = sum(1 for row in team_rows if len(row.get("source_rows") or []) > 1)
    if duplicate_team_count:
        warnings.append(f"Detected {duplicate_team_count} duplicated imported teams across rows.")

    match_dupe_counter = Counter((m["team_a_key"], m["team_b_key"], m["score_summary"]) for m in matches)
    duplicated_matches = sum(1 for count in match_dupe_counter.values() if count > 1)
    if duplicated_matches:
        warnings.append(f"Detected {duplicated_matches} duplicated imported matches.")

    return {
        "players": sorted(players_by_key.values(), key=lambda row: (row.get("display_name") or "", row.get("import_key") or "")),
        "teams": team_rows,
        "matches": matches,
        "event_meta": {
            "events": sorted({m.get("event") for m in matches if _safe_text(m.get("event"))}),
            "match_types": sorted({m.get("match_type") for m in matches if _safe_text(m.get("match_type"))}),
            "dates": sorted({m.get("date") for m in matches if _safe_text(m.get("date"))}),
        },
        "warnings": warnings,
        "errors": errors,
    }


def suggest_player_matches(imported_players: list[dict[str, Any]], existing_players: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_dupr: dict[str, list[dict[str, Any]]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in existing_players:
        dupr = _normalize_dupr_id(row.get("dupr_id"))
        if dupr:
            by_dupr.setdefault(dupr, []).append(row)
        by_name.setdefault(_normalize_name(row.get("name")), []).append(row)

    suggestions: dict[str, dict[str, Any]] = {}
    for player in imported_players:
        import_key = player.get("import_key")
        dupr = _normalize_dupr_id(player.get("dupr_id"))
        norm_name = _normalize_name(player.get("display_name"))
        suggestion = {"suggested_player_id": None, "reason": "none", "ambiguous_ids": []}

        if dupr and len(by_dupr.get(dupr, [])) == 1:
            suggestion = {"suggested_player_id": by_dupr[dupr][0].get("id"), "reason": "dupr_exact", "ambiguous_ids": []}
        elif dupr and len(by_dupr.get(dupr, [])) > 1:
            suggestion = {
                "suggested_player_id": None,
                "reason": "dupr_ambiguous",
                "ambiguous_ids": [row.get("id") for row in by_dupr[dupr] if row.get("id") is not None],
            }
        elif norm_name and len(by_name.get(norm_name, [])) == 1:
            suggestion = {"suggested_player_id": by_name[norm_name][0].get("id"), "reason": "name_exact", "ambiguous_ids": []}
        elif norm_name and len(by_name.get(norm_name, [])) > 1:
            suggestion = {
                "suggested_player_id": None,
                "reason": "name_ambiguous",
                "ambiguous_ids": [row.get("id") for row in by_name[norm_name] if row.get("id") is not None],
            }

        suggestions[str(import_key)] = suggestion
    return suggestions


def build_draw_import_payload(
    *,
    bundle: dict[str, Any],
    mapping_decisions: dict[str, dict[str, Any]],
    match_reviews: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = list(bundle.get("warnings") or [])

    mapped_player_refs: dict[str, str] = {}
    create_import_keys: list[str] = []

    for imported in bundle.get("players") or []:
        import_key = str(imported.get("import_key"))
        decision = mapping_decisions.get(import_key) or {}
        action = decision.get("action")
        player_id = decision.get("player_id")
        if action == "use_existing" and player_id:
            mapped_player_refs[import_key] = f"existing:{player_id}"
        elif action == "create_new":
            mapped_player_refs[import_key] = f"create:{import_key}"
            create_import_keys.append(import_key)
        else:
            errors.append(f"Unresolved player mapping for {imported.get('display_name') or import_key}.")

    if errors:
        return {"errors": errors, "warnings": warnings}

    existing_targets = [ref for ref in mapped_player_refs.values() if ref.startswith("existing:")]
    duplicate_mapped_existing = [ref for ref, count in Counter(existing_targets).items() if count > 1]
    if duplicate_mapped_existing:
        warnings.append("Multiple imported players are mapped to the same existing JUPR player.")

    team_payloads: list[dict[str, Any]] = []
    team_ref_by_key: dict[str, str] = {}
    seen_teams: set[str] = set()

    for imported_team in bundle.get("teams") or []:
        team_key = imported_team.get("team_key")
        pkeys = imported_team.get("player_keys") or []
        p1 = mapped_player_refs.get(pkeys[0]) if len(pkeys) > 0 else None
        p2 = mapped_player_refs.get(pkeys[1]) if len(pkeys) > 1 and pkeys[1] else None
        if not p1:
            errors.append(f"Team {team_key}: missing mapped player A.")
            continue
        if p2 and p1 == p2:
            errors.append(f"Team {team_key}: same mapped player appears twice.")
            continue
        canonical = "|".join([p1, p2 or "__SINGLES__"])
        if canonical in seen_teams:
            warnings.append(f"Duplicate team detected for mapped roster {canonical}.")
            continue
        seen_teams.add(canonical)
        team_ref = f"team:{len(team_payloads) + 1}"
        team_ref_by_key[team_key] = team_ref
        team_payloads.append({"team_ref": team_ref, "p1_ref": p1, "p2_ref": p2, "source_team_key": team_key})

    match_payloads: list[dict[str, Any]] = []
    for match in bundle.get("matches") or []:
        source_row = int(match.get("source_row") or 0)
        reviewed = match_reviews.get(str(source_row)) or {}
        include = bool(reviewed.get("include", True))
        stage = str(reviewed.get("stage") or match.get("stage") or "PLAYOFF").upper()
        if stage not in STAGE_OPTIONS:
            stage = "PLAYOFF"
        if not include:
            continue
        team_a_ref = team_ref_by_key.get(match.get("team_a_key"))
        team_b_ref = team_ref_by_key.get(match.get("team_b_key"))
        if not team_a_ref or not team_b_ref:
            errors.append(f"Row {source_row}: team mapping failed.")
            continue
        if team_a_ref == team_b_ref:
            errors.append(f"Row {source_row}: mapped same team on both sides.")
            continue
        games = match.get("games") or []
        score_a = sum(int(g.get("score_a") or 0) for g in games) if games else None
        score_b = sum(int(g.get("score_b") or 0) for g in games) if games else None
        match_payloads.append(
            {
                "source_row": source_row,
                "team_a_ref": team_a_ref,
                "team_b_ref": team_b_ref,
                "games": games,
                "score_a": score_a,
                "score_b": score_b,
                "winner_side": match.get("winner_side"),
                "stage": stage,
            }
        )

    podium_wins: Counter[str] = Counter()
    for row in match_payloads:
        if row.get("winner_side") == "A":
            podium_wins[row["team_a_ref"]] += 1
        elif row.get("winner_side") == "B":
            podium_wins[row["team_b_ref"]] += 1
    podium_candidates = [team_ref for team_ref, _wins in podium_wins.most_common(3)]

    return {
        "errors": errors,
        "warnings": warnings,
        "create_import_keys": create_import_keys,
        "mapped_player_refs": mapped_player_refs,
        "teams": team_payloads,
        "matches": match_payloads,
        "podium_candidates": podium_candidates,
    }
