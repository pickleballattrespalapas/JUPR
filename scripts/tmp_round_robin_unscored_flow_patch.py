from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one marker, found {count}: {old[:140]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_count(path: str, old: str, new: str, expected: int) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise SystemExit(f"{path}: expected {expected} markers, found {count}: {old[:140]!r}")
    target.write_text(text.replace(old, new), encoding="utf-8")


# ---------------------------------------------------------------------------
# Domain: scored/unscored Round-Robin sessions and played rounds
# ---------------------------------------------------------------------------
engine = "jupr_app/domain/adaptive_play_engine.py"
replace_once(
    engine,
    'STANDINGS_SORTS = {"wins", "points", "differential"}\n\n\ndef normalize_standings_sort(value: Any) -> str:',
    'STANDINGS_SORTS = {"wins", "points", "differential"}\nSCORING_MODES = {"scored", "unscored"}\n\n\ndef normalize_scoring_mode(value: Any) -> str:\n    mode = str(value or "scored").strip().lower().replace("-", "_")\n    aliases = {\n        "score": "scored",\n        "scores": "scored",\n        "no_scores": "unscored",\n        "round_played": "unscored",\n        "played": "unscored",\n    }\n    mode = aliases.get(mode, mode)\n    return mode if mode in SCORING_MODES else "scored"\n\n\ndef normalize_standings_sort(value: Any) -> str:',
)
replace_once(
    engine,
    '        if status not in {"saved", "active", "preview"}:',
    '        if status not in {"saved", "played", "active", "preview"}:',
)
replace_once(
    engine,
    '    court_count: int = 0,\n    standings_sort: str = "wins",\n) -> dict[str, Any]:',
    '    court_count: int = 0,\n    standings_sort: str = "wins",\n    scoring_mode: str = "scored",\n) -> dict[str, Any]:',
)
replace_once(
    engine,
    '    if fmt not in {"singles", "doubles"}:\n        raise ValueError("play_format must be singles or doubles")\n    names = [_clean_name(x) for x in participant_names if _clean_name(x)]',
    '    if fmt not in {"singles", "doubles"}:\n        raise ValueError("play_format must be singles or doubles")\n    scoring = normalize_scoring_mode(scoring_mode)\n    if kind == "ladder" and scoring != "scored":\n        raise ValueError("Ladder Generator requires scored rounds because later rounds depend on results.")\n    names = [_clean_name(x) for x in participant_names if _clean_name(x)]',
)
replace_once(
    engine,
    '        "playFormat": fmt,\n        "standingsSort": normalize_standings_sort(standings_sort),',
    '        "playFormat": fmt,\n        "scoringMode": scoring,\n        "standingsSort": normalize_standings_sort(standings_sort),',
)
replace_once(
    engine,
    '                "standings_sort": event["standingsSort"],\n                "schedule": event["rounds"],',
    '                "standings_sort": event["standingsSort"],\n                "scoring_mode": event["scoringMode"],\n                "schedule": event["rounds"],',
)
replace_once(
    engine,
    '    next_event = copy.deepcopy(event)\n    row = _get_round(next_event, round_number)\n    if str(row.get("status")) not in {"active", "preview"}:\n        raise ValueError("Only an active round can be scored.")',
    '    next_event = copy.deepcopy(event)\n    if normalize_scoring_mode(next_event.get("scoringMode")) != "scored":\n        raise ValueError("This unscored Round-Robin uses Round Played instead of score entry.")\n    row = _get_round(next_event, round_number)\n    if str(row.get("status")) not in {"active", "preview"}:\n        raise ValueError("Only an active round can be scored.")',
)
played_function = '''\n\ndef mark_generator_round_played(event: dict[str, Any], *, round_number: int) -> dict[str, Any]:\n    next_event = copy.deepcopy(event)\n    if str(next_event.get("generatorKind") or "") != "round_robin":\n        raise ValueError("Round Played is available only for Round-Robin Generator sessions.")\n    if normalize_scoring_mode(next_event.get("scoringMode")) != "unscored":\n        raise ValueError("Scored Round-Robins must save scores or skip the round.")\n    row = _get_round(next_event, round_number)\n    if str(row.get("status")) not in {"active", "preview"}:\n        raise ValueError("Only an active round can be marked played.")\n    if _round_has_any_scores(row):\n        raise ValueError("Clear entered scores before marking this round played.")\n    row["status"] = "played"\n    row["playedAt"] = _now_iso()\n    row["savedAt"] = None\n    row["skippedAt"] = None\n    for match in row.get("matches") or []:\n        match["status"] = "played"\n    for court in row.get("courts") or []:\n        for match in court.get("matches") or []:\n            match["status"] = "played"\n    return next_event\n'''
replace_once(
    engine,
    '\ndef skip_generator_round(event: dict[str, Any], *, round_number: int, reason: str = "") -> dict[str, Any]:',
    played_function + '\n\ndef skip_generator_round(event: dict[str, Any], *, round_number: int, reason: str = "") -> dict[str, Any]:',
)
replace_once(
    engine,
    'def generator_event_standings(event: dict[str, Any]) -> list[dict[str, Any]]:\n    """Return complete saved-round standings using the event\'s selected primary sort.',
    'def generator_event_standings(event: dict[str, Any]) -> list[dict[str, Any]]:\n    """Return complete saved-round standings using the event\'s selected primary sort.',
)
replace_once(
    engine,
    '    participants = _participant_map(event)\n    stats: dict[str, dict[str, Any]] = {}',
    '    if normalize_scoring_mode(event.get("scoringMode")) == "unscored":\n        return []\n    participants = _participant_map(event)\n    stats: dict[str, dict[str, Any]] = {}',
)
replace_once(
    engine,
    '    if str(row.get("status")) not in {"saved", "skipped"}:\n        raise ValueError("Save or skip the current round before continuing.")',
    '    if str(row.get("status")) not in {"saved", "played", "skipped"}:\n        raise ValueError("Save scores, mark the round played, or skip it before continuing.")',
)
replace_once(
    engine,
    '    if str(row.get("status")) in {"saved", "skipped"} or _round_has_any_scores(row):',
    '    if str(row.get("status")) in {"saved", "played", "skipped"} or _round_has_any_scores(row):',
)

# ---------------------------------------------------------------------------
# Services: carry scoring mode and persist Round Played
# ---------------------------------------------------------------------------
admin_service = "jupr_app/services/admin_play_generator_service.py"
replace_once(
    admin_service,
    '    generator_event_standings,\n    mutate_generator_roster,',
    '    generator_event_standings,\n    mark_generator_round_played,\n    mutate_generator_roster,',
)
replace_once(
    admin_service,
    '        "standings_sort": str(event.get("standingsSort") or "wins") if event else "wins",\n        "standings": generator_event_standings(event) if event else [],',
    '        "scoring_mode": str(event.get("scoringMode") or "scored") if event else "scored",\n        "standings_sort": str(event.get("standingsSort") or "wins") if event else "wins",\n        "standings": generator_event_standings(event) if event else [],',
)
replace_once(
    admin_service,
    '    court_count: int,\n    standings_sort: str = "wins",\n) -> dict[str, Any]:',
    '    court_count: int,\n    standings_sort: str = "wins",\n    scoring_mode: str = "scored",\n) -> dict[str, Any]:',
)
replace_once(
    admin_service,
    '        court_count=court_count,\n        standings_sort=standings_sort,\n    )\n    return {',
    '        court_count=court_count,\n        standings_sort=standings_sort,\n        scoring_mode=scoring_mode,\n    )\n    return {',
)
replace_once(
    admin_service,
    '    source: str,\n    standings_sort: str = "wins",\n) -> dict[str, Any]:',
    '    source: str,\n    standings_sort: str = "wins",\n    scoring_mode: str = "scored",\n) -> dict[str, Any]:',
)
replace_once(
    admin_service,
    '        court_count=court_count,\n        standings_sort=standings_sort,\n    )["preview"]',
    '        court_count=court_count,\n        standings_sort=standings_sort,\n        scoring_mode=scoring_mode,\n    )["preview"]',
)
admin_played_service = '''\n\ndef mark_play_generator_round_played(\n    supabase: Any,\n    *,\n    club_id: str,\n    session_key: str,\n    round_number: int,\n    expected_version: str,\n    actor_email: str,\n    actor_role: str,\n    source: str,\n) -> dict[str, Any]:\n    before = _live_row(supabase, club_id=str(club_id), session_key=str(session_key))\n    event = mark_generator_round_played(\n        _event_from_state(_state(before)),\n        round_number=int(round_number),\n    )\n    updated = _persist_event(\n        supabase,\n        before=before,\n        event=event,\n        expected_version=expected_version,\n    )\n    session = _session_payload(updated)\n    _audit(\n        supabase,\n        club_id=club_id,\n        actor_email=actor_email,\n        actor_role=actor_role,\n        action_type="mark_play_generator_round_played",\n        entity_id=session_key,\n        before_json={"session": _session_payload(before)},\n        after_json={"session": session, "round_number": int(round_number)},\n        source=source,\n    )\n    return {"ok": True, "mode": "play_generator_round_played", "session": session}\n'''
replace_once(
    admin_service,
    '\ndef skip_play_generator_round(\n',
    admin_played_service + '\n\ndef skip_play_generator_round(\n',
)
replace_once(
    admin_service,
    '    if current_row and str(current_row.get("status")) not in {"saved", "skipped"}:\n        raise ValueError("Save or skip the current round before completing the session.")',
    '    if current_row and str(current_row.get("status")) not in {"saved", "played", "skipped"}:\n        raise ValueError("Save scores, mark the round played, or skip it before completing the session.")',
)

public_service = "jupr_app/services/public_play_generator_service.py"
replace_once(
    public_service,
    '    generator_event_standings,\n    mutate_generator_roster,',
    '    generator_event_standings,\n    mark_generator_round_played,\n    mutate_generator_roster,',
)
replace_once(
    public_service,
    '        "standings_sort": str(event.get("standingsSort") or "wins") if event else "wins",\n        "standings": generator_event_standings(event) if event else [],',
    '        "scoring_mode": str(event.get("scoringMode") or "scored") if event else "scored",\n        "standings_sort": str(event.get("standingsSort") or "wins") if event else "wins",\n        "standings": generator_event_standings(event) if event else [],',
)
replace_once(
    public_service,
    '    court_count: int,\n    standings_sort: str = "wins",\n) -> dict[str, Any]:',
    '    court_count: int,\n    standings_sort: str = "wins",\n    scoring_mode: str = "scored",\n) -> dict[str, Any]:',
)
replace_once(
    public_service,
    '            court_count=max(0, min(int(court_count or 0), 20)),\n            standings_sort=standings_sort,\n        )',
    '            court_count=max(0, min(int(court_count or 0), 20)),\n            standings_sort=standings_sort,\n            scoring_mode=scoring_mode,\n        )',
)
replace_once(
    public_service,
    '    token_secret: str | None = None,\n    standings_sort: str = "wins",\n) -> dict[str, Any]:',
    '    token_secret: str | None = None,\n    standings_sort: str = "wins",\n    scoring_mode: str = "scored",\n) -> dict[str, Any]:',
)
replace_once(
    public_service,
    '        court_count=court_count,\n        standings_sort=standings_sort,\n    )\n    preview = preview_result["preview"]',
    '        court_count=court_count,\n        standings_sort=standings_sort,\n        scoring_mode=scoring_mode,\n    )\n    preview = preview_result["preview"]',
)
replace_once(
    public_service,
    '        "standings_sort": str(preview.get("standingsSort") or "wins"),\n        "live_mode": "quick",',
    '        "standings_sort": str(preview.get("standingsSort") or "wins"),\n        "scoring_mode": str(preview.get("scoringMode") or "scored"),\n        "live_mode": "quick",',
)
public_played_service = '''\n\ndef mark_public_play_generator_round_played(\n    supabase: Any,\n    *,\n    club_id: str,\n    session_key: str,\n    round_number: int,\n    edit_token: str,\n    expected_version: int,\n    idempotency_key: str,\n    requester_hash: str,\n) -> dict[str, Any]:\n    return _run_mutation(\n        supabase,\n        club_id=club_id,\n        session_key=session_key,\n        edit_token=edit_token,\n        expected_version=expected_version,\n        idempotency_key=idempotency_key,\n        requester_hash=requester_hash,\n        action="played",\n        request_payload={"round_number": int(round_number)},\n        mutate=lambda event: (\n            mark_generator_round_played(event, round_number=int(round_number)),\n            {},\n        ),\n    )\n'''
replace_once(
    public_service,
    '\ndef skip_public_play_generator_round(\n',
    public_played_service + '\n\ndef skip_public_play_generator_round(\n',
)

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
admin_routes = "services/api/admin_play_generator_routes.py"
replace_once(
    admin_routes,
    '    list_play_generator_sessions,\n    mutate_play_generator_roster,',
    '    list_play_generator_sessions,\n    mark_play_generator_round_played,\n    mutate_play_generator_roster,',
)
replace_once(
    admin_routes,
    '    standings_sort: str = Field(default="wins", pattern=r"^(wins|points|differential)$")\n',
    '    standings_sort: str = Field(default="wins", pattern=r"^(wins|points|differential)$")\n    scoring_mode: str = Field(default="scored", pattern=r"^(scored|unscored)$")\n',
)
replace_once(
    admin_routes,
    'class GeneratorSkipRequest(GeneratorDurableRequest):\n    reason: str = Field(default="", max_length=300)\n    source: str = "next_play_generator_skip"\n',
    'class GeneratorPlayedRequest(GeneratorDurableRequest):\n    source: str = "next_play_generator_played"\n\n\nclass GeneratorSkipRequest(GeneratorDurableRequest):\n    reason: str = Field(default="", max_length=300)\n    source: str = "next_play_generator_skip"\n',
)
replace_count(
    admin_routes,
    '                standings_sort=payload.standings_sort,\n',
    '                standings_sort=payload.standings_sort,\n                scoring_mode=payload.scoring_mode,\n',
    2,
)
admin_played_route = '''\n    @app.post(\n        "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/played"\n    )\n    def post_generator_round_played(\n        club_id: str,\n        session_key: str,\n        round_number: int,\n        payload: GeneratorPlayedRequest,\n        authorization: str | None = auth_header(),\n    ) -> dict[str, Any]:\n        _require_write_gate()\n        supabase = get_supabase_client()\n        actor_email, actor_role = _resolve_role_or_403(\n            supabase=supabase,\n            club_id=str(club_id),\n            authorization=authorization,\n            source=payload.source,\n        )\n        try:\n            current = get_play_generator_session(\n                supabase,\n                club_id=str(club_id),\n                session_key=str(session_key),\n            )["session"]\n            return run_durable_admin_operation(\n                supabase,\n                club_id=str(club_id),\n                surface="play_generator",\n                operation_type="mark_round_played",\n                entity_id=str(session_key),\n                idempotency_key=payload.idempotency_key,\n                expected_version=payload.expected_version,\n                current_version=str(current.get("version") or ""),\n                request_payload=_model_payload(payload),\n                recovery=operation_recovery_handoff(\n                    surface="play_generator",\n                    entity_id=str(session_key),\n                ),\n                actor_email=actor_email,\n                actor_role=actor_role,\n                source=payload.source,\n                mutate=lambda: mark_play_generator_round_played(\n                    supabase,\n                    club_id=str(club_id),\n                    session_key=str(session_key),\n                    round_number=int(round_number),\n                    expected_version=payload.expected_version,\n                    actor_email=actor_email,\n                    actor_role=actor_role,\n                    source=payload.source,\n                ),\n                current_version_resolver=lambda: str(\n                    get_play_generator_session(\n                        supabase,\n                        club_id=str(club_id),\n                        session_key=str(session_key),\n                    )["session"].get("version")\n                    or ""\n                ),\n            )\n        except Exception as exc:\n            _handle(exc)\n'''
replace_once(
    admin_routes,
    '    @app.post(\n        "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/skip"\n    )',
    admin_played_route + '\n\n    @app.post(\n        "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/skip"\n    )',
)

public_routes = "services/api/public_play_generator_routes.py"
replace_once(
    public_routes,
    '    list_public_play_generator_sessions,\n    mutate_public_play_generator_roster,',
    '    list_public_play_generator_sessions,\n    mark_public_play_generator_round_played,\n    mutate_public_play_generator_roster,',
)
replace_once(
    public_routes,
    '    standings_sort: str = Field(default="wins", pattern=r"^(wins|points|differential)$")\n',
    '    standings_sort: str = Field(default="wins", pattern=r"^(wins|points|differential)$")\n    scoring_mode: str = Field(default="scored", pattern=r"^(scored|unscored)$")\n',
)
replace_count(
    public_routes,
    '                standings_sort=payload.standings_sort,\n',
    '                standings_sort=payload.standings_sort,\n                scoring_mode=payload.scoring_mode,\n',
    2,
)
public_played_route = '''\n    @app.post("/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/played")\n    def post_public_generator_round_played(\n        club_slug: str,\n        session_key: str,\n        round_number: int,\n        payload: PublicGeneratorMutationRequest,\n        request: Request,\n    ) -> dict[str, Any]:\n        require_public_writes()\n        require_service_role()\n        club, club_id, supabase = context(club_slug)\n        try:\n            result = mark_public_play_generator_round_played(\n                supabase,\n                club_id=club_id,\n                session_key=session_key,\n                round_number=round_number,\n                edit_token=payload.edit_token,\n                expected_version=payload.expected_version,\n                idempotency_key=payload.idempotency_key,\n                requester_hash=requester_hash(request),\n            )\n        except Exception as exc:\n            raise_public_error(exc)\n            raise\n        return {"club": public_club_payload(club, club_slug), **result}\n'''
replace_once(
    public_routes,
    '    @app.post("/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/skip")',
    public_played_route + '\n\n    @app.post("/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/skip")',
)

# ---------------------------------------------------------------------------
# Browser draft persistence
# ---------------------------------------------------------------------------
draft = "apps/web/lib/playGeneratorDraft.ts"
replace_once(
    draft,
    '  standingsSort?: "wins" | "points" | "differential";\n  targetCount: number;',
    '  standingsSort?: "wins" | "points" | "differential";\n  scoringMode?: "scored" | "unscored";\n  targetCount: number;',
)

# ---------------------------------------------------------------------------
# Setup UI: scored or unscored Round-Robin
# ---------------------------------------------------------------------------
def patch_workspace(path: str) -> None:
    replace_once(
        path,
        'type StandingsSort = "wins" | "points" | "differential";\n',
        'type StandingsSort = "wins" | "points" | "differential";\ntype ScoringMode = "scored" | "unscored";\n',
    )
    replace_once(
        path,
        '  standingsSort?: StandingsSort;\n  totalRounds: number;',
        '  standingsSort?: StandingsSort;\n  scoringMode?: ScoringMode;\n  totalRounds: number;',
    )
    replace_once(
        path,
        '  play_format: PlayFormat;\n  current_round_number?: number | null;',
        '  play_format: PlayFormat;\n  scoring_mode?: ScoringMode;\n  current_round_number?: number | null;',
    )
    replace_once(
        path,
        '  const [standingsSort, setStandingsSort] = useState<StandingsSort>("wins");\n  const [targetCount, setTargetCount] = useState(8);',
        '  const [standingsSort, setStandingsSort] = useState<StandingsSort>("wins");\n  const [scoringMode, setScoringMode] = useState<ScoringMode>("scored");\n  const [targetCount, setTargetCount] = useState(8);',
    )
    replace_once(
        path,
        '      setStandingsSort(stored.standingsSort || "wins");\n      setTargetCount(stored.targetCount);',
        '      setStandingsSort(stored.standingsSort || "wins");\n      setScoringMode(generatorKind === "round_robin" ? stored.scoringMode || "scored" : "scored");\n      setTargetCount(stored.targetCount);',
    )
    replace_once(
        path,
        '      standingsSort,\n      targetCount,',
        '      standingsSort,\n      scoringMode,\n      targetCount,',
    )
    replace_once(
        path,
        '    standingsSort,\n    targetCount,',
        '    standingsSort,\n    scoringMode,\n    targetCount,',
    )
    replace_once(
        path,
        '      standings_sort: standingsSort,\n      title: title.trim(),',
        '      standings_sort: standingsSort,\n      scoring_mode: generatorKind === "round_robin" ? scoringMode : "scored",\n      title: title.trim(),',
    )
    old_block = '''          {generatorKind === "round_robin" ? (\n            <label>\n              Standings ranked by\n              <br />\n              <select\n                value={standingsSort}\n                onChange={(event) => {\n                  setStandingsSort(event.target.value as StandingsSort);\n                  invalidatePreview();\n                }}\n                style={inputStyle}\n              >\n                <option value="wins">Total wins</option>\n                <option value="points">Total points</option>\n                <option value="differential">Point differential</option>\n              </select>\n              <small style={{ display: "block", marginTop: "0.35rem", color: "#64748b" }}>\n                This primary ranking applies to the full session standings.\n              </small>\n            </label>\n          ) : null}'''
    new_block = '''          {generatorKind === "round_robin" ? (\n            <>\n              <label>\n                Round scoring\n                <br />\n                <select\n                  value={scoringMode}\n                  onChange={(event) => {\n                    setScoringMode(event.target.value as ScoringMode);\n                    invalidatePreview();\n                  }}\n                  style={inputStyle}\n                >\n                  <option value="scored">Scored — enter scores and show standings</option>\n                  <option value="unscored">Unscored — mark each round played</option>\n                </select>\n                <small style={{ display: "block", marginTop: "0.35rem", color: "#64748b" }}>\n                  Unscored sessions have no score fields or standings between rounds.\n                </small>\n              </label>\n              {scoringMode === "scored" ? (\n                <label>\n                  Standings ranked by\n                  <br />\n                  <select\n                    value={standingsSort}\n                    onChange={(event) => {\n                      setStandingsSort(event.target.value as StandingsSort);\n                      invalidatePreview();\n                    }}\n                    style={inputStyle}\n                  >\n                    <option value="wins">Total wins</option>\n                    <option value="points">Total points</option>\n                    <option value="differential">Point differential</option>\n                  </select>\n                  <small style={{ display: "block", marginTop: "0.35rem", color: "#64748b" }}>\n                    This primary ranking applies to the full session standings.\n                  </small>\n                </label>\n              ) : null}\n            </>\n          ) : null}'''
    replace_once(path, old_block, new_block)
    replace_once(
        path,
        '            {generatorKind === "ladder"\n              ? "Only Round 1 is shown. Round 2 and later are generated from saved results."\n              : "Review every planned round, matchup, and bye. Change the roster order above and regenerate when needed."}',
        '            {generatorKind === "ladder"\n              ? "Only Round 1 is shown. Round 2 and later are generated from saved results."\n              : scoringMode === "unscored"\n                ? "Review every planned round, matchup, and bye. During play, use Round Played to move directly to the next round."\n                : "Review every planned round, matchup, and bye. Change the roster order above and regenerate when needed."}',
    )


patch_workspace("apps/web/app/admin/play-generators/GeneratorWorkspace.tsx")
patch_workspace("apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx")

# ---------------------------------------------------------------------------
# Round pages: scored flow through standings, unscored Round Played flow
# ---------------------------------------------------------------------------
def patch_round_runner(path: str, *, public: bool) -> None:
    replace_once(
        path,
        'type GeneratorKind = "round_robin" | "ladder";\n',
        'type GeneratorKind = "round_robin" | "ladder";\ntype ScoringMode = "scored" | "unscored";\n',
    )
    replace_once(
        path,
        '  status: "preview" | "active" | "saved" | "skipped" | string;',
        '  status: "preview" | "active" | "saved" | "played" | "skipped" | string;',
    )
    replace_once(
        path,
        '  playFormat: "singles" | "doubles";\n  status: string;',
        '  playFormat: "singles" | "doubles";\n  scoringMode?: ScoringMode;\n  status: string;',
    )
    replace_once(
        path,
        '  play_format: "singles" | "doubles";\n  current_round_number?: number | null;',
        '  play_format: "singles" | "doubles";\n  scoring_mode?: ScoringMode;\n  current_round_number?: number | null;',
    )
    replace_once(
        path,
        '  const canEditRound =\n    Boolean(session) &&',
        '  const scoringMode: ScoringMode = session?.scoring_mode || event?.scoringMode || "scored";\n  const scoredSession = scoringMode === "scored";\n  const canEditRound =\n    Boolean(session) &&',
    )
    replace_once(
        path,
        '  const anyDraftScore = Object.values(scores).some((value) => value !== "");',
        '  const anyDraftScore = scoredSession && Object.values(scores).some((value) => value !== "");',
    )
    # Insert Round Played before skipRound.
    if public:
        played_fn = '''\n  async function markRoundPlayed(): Promise<void> {\n    if (!session || !round || generatorKind !== "round_robin" || scoredSession) return;\n    setBusy(true);\n    setMessage(null);\n    try {\n      const playedPayload = await requestJson<MutationResponse>(\n        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(\n          sessionKey\n        )}/rounds/${roundNumber}/played`,\n        {\n          method: "POST",\n          body: JSON.stringify({\n            edit_token: editToken,\n            expected_version: Number(session.version),\n            idempotency_key: operationKey("played")\n          })\n        }\n      );\n      if (!playedPayload.session) throw new Error("Round marked played without a refreshed session.");\n      applySession(playedPayload.session);\n      const advancedPayload = await requestJson<MutationResponse>(\n        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(\n          sessionKey\n        )}/advance`,\n        {\n          method: "POST",\n          body: JSON.stringify({\n            edit_token: editToken,\n            expected_version: Number(playedPayload.session.version),\n            idempotency_key: operationKey("advance-after-played")\n          })\n        }\n      );\n      if (!advancedPayload.session) throw new Error("Round advanced without a refreshed session.");\n      applySession(advancedPayload.session);\n      if (advancedPayload.session.status === "completed") {\n        setMessage("Session completed.");\n        router.refresh();\n        return;\n      }\n      const nextRound = advancedPayload.session.current_round_number || roundNumber + 1;\n      router.push(roundPath(generatorKind, clubId, sessionKey, nextRound));\n      router.refresh();\n    } catch (error) {\n      setMessage(error instanceof Error ? error.message : "Unable to mark the round played.");\n    } finally {\n      setBusy(false);\n    }\n  }\n'''
    else:
        played_fn = '''\n  async function markRoundPlayed(): Promise<void> {\n    if (!session || !round || generatorKind !== "round_robin" || scoredSession) return;\n    setBusy(true);\n    setMessage(null);\n    try {\n      const playedPayload = await requestJson<MutationResponse>(\n        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(\n          sessionKey\n        )}/rounds/${roundNumber}/played`,\n        {\n          method: "POST",\n          body: JSON.stringify({\n            expected_version: session.version,\n            idempotency_key: operationKey("played")\n          })\n        }\n      );\n      if (!playedPayload.session) throw new Error("Round marked played without a refreshed session.");\n      applySession(playedPayload.session);\n      const advancedPayload = await requestJson<MutationResponse>(\n        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(\n          sessionKey\n        )}/advance`,\n        {\n          method: "POST",\n          body: JSON.stringify({\n            expected_version: playedPayload.session.version,\n            idempotency_key: operationKey("advance-after-played")\n          })\n        }\n      );\n      if (!advancedPayload.session) throw new Error("Round advanced without a refreshed session.");\n      applySession(advancedPayload.session);\n      if (advancedPayload.session.status === "completed") {\n        setMessage("Session completed.");\n        router.refresh();\n        return;\n      }\n      const nextRound = advancedPayload.session.current_round_number || roundNumber + 1;\n      router.push(roundPath(generatorKind, sessionKey, nextRound));\n      router.refresh();\n    } catch (error) {\n      setMessage(error instanceof Error ? error.message : "Unable to mark the round played.");\n    } finally {\n      setBusy(false);\n    }\n  }\n'''
    replace_once(path, '\n  async function skipRound(): Promise<void> {', played_fn + '\n  async function skipRound(): Promise<void> {')
    # Auto-advance skipped unscored Round-Robins.
    if public:
        skip_tail = '''      if (!payload.session) throw new Error("Round skipped without a refreshed session.");\n      applySession(payload.session);\n      setMessage(`Round ${roundNumber} skipped.`);'''
        skip_new = '''      if (!payload.session) throw new Error("Round skipped without a refreshed session.");\n      applySession(payload.session);\n      if (generatorKind === "round_robin" && !scoredSession) {\n        const advancedPayload = await requestJson<MutationResponse>(\n          `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(\n            sessionKey\n          )}/advance`,\n          {\n            method: "POST",\n            body: JSON.stringify({\n              edit_token: editToken,\n              expected_version: Number(payload.session.version),\n              idempotency_key: operationKey("advance-after-skip")\n            })\n          }\n        );\n        if (!advancedPayload.session) throw new Error("Skipped round advanced without a refreshed session.");\n        applySession(advancedPayload.session);\n        if (advancedPayload.session.status === "completed") {\n          setMessage("Session completed.");\n          router.refresh();\n        } else {\n          const nextRound = advancedPayload.session.current_round_number || roundNumber + 1;\n          router.push(roundPath(generatorKind, clubId, sessionKey, nextRound));\n          router.refresh();\n        }\n        return;\n      }\n      setMessage(`Round ${roundNumber} skipped.`);'''
    else:
        skip_tail = '''      if (!payload.session) throw new Error("Round skipped without a refreshed session.");\n      applySession(payload.session);\n      setMessage(`Round ${roundNumber} skipped.`);'''
        skip_new = '''      if (!payload.session) throw new Error("Round skipped without a refreshed session.");\n      applySession(payload.session);\n      if (generatorKind === "round_robin" && !scoredSession) {\n        const advancedPayload = await requestJson<MutationResponse>(\n          `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(\n            sessionKey\n          )}/advance`,\n          {\n            method: "POST",\n            body: JSON.stringify({\n              expected_version: payload.session.version,\n              idempotency_key: operationKey("advance-after-skip")\n            })\n          }\n        );\n        if (!advancedPayload.session) throw new Error("Skipped round advanced without a refreshed session.");\n        applySession(advancedPayload.session);\n        if (advancedPayload.session.status === "completed") {\n          setMessage("Session completed.");\n          router.refresh();\n        } else {\n          const nextRound = advancedPayload.session.current_round_number || roundNumber + 1;\n          router.push(roundPath(generatorKind, sessionKey, nextRound));\n          router.refresh();\n        }\n        return;\n      }\n      setMessage(`Round ${roundNumber} skipped.`);'''
    replace_once(path, skip_tail, skip_new)
    # Header status includes scoring mode.
    replace_once(
        path,
        '{event.totalRounds} · {round.status}',
        '{event.totalRounds} · {scoredSession ? "Scored" : "Unscored"} · {round.status}',
    )
    # Hide standings for unscored sessions.
    replace_once(
        path,
        '{generatorKind === "round_robin" ? (\n              <Link href={standingsPath(',
        '{generatorKind === "round_robin" && scoredSession ? (\n              <Link href={standingsPath(',
    )
    # Score inputs only for scored sessions.
    replace_once(path, '            const editable = canEditRound;', '            const editable = canEditRound && scoredSession;')
    replace_once(
        path,
        '                      {match.scoreA == null || match.scoreB == null\n                        ? "—"\n                        : `${match.scoreA}–${match.scoreB}`}',
        '                      {!scoredSession\n                        ? "vs."\n                        : match.scoreA == null || match.scoreB == null\n                          ? "—"\n                          : `${match.scoreA}–${match.scoreB}`}',
    )
    # Replace current-round action controls.
    old_actions = '''        {canEditRound ? (\n          <div style={{ marginTop: "1rem", display: "grid", gap: "0.75rem" }}>\n            <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>\n              <button type="button" onClick={() => void saveRound()} disabled={busy} style={primaryButton}>\n                {busy ? "Saving…" : "Save round scores"}\n              </button>\n              <input\n                value={skipReason}\n                onChange={(event_) => setSkipReason(event_.target.value)}\n                placeholder="Optional skip reason"\n                style={{ ...inputStyle, maxWidth: 260 }}\n              />\n              <button type="button" onClick={() => void skipRound()} disabled={busy} style={secondaryButton}>\n                Skip round\n              </button>\n            </div>\n          </div>\n        ) : null}'''
    new_actions = '''        {canEditRound ? (\n          <div style={{ marginTop: "1rem", display: "grid", gap: "0.75rem" }}>\n            <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>\n              {scoredSession ? (\n                <button type="button" onClick={() => void saveRound()} disabled={busy} style={primaryButton}>\n                  {busy ? "Saving…" : "Save round scores"}\n                </button>\n              ) : (\n                <button type="button" onClick={() => void markRoundPlayed()} disabled={busy} style={primaryButton}>\n                  {busy ? "Saving…" : "Round Played"}\n                </button>\n              )}\n              <input\n                value={skipReason}\n                onChange={(event_) => setSkipReason(event_.target.value)}\n                placeholder="Optional skip reason"\n                style={{ ...inputStyle, maxWidth: 260 }}\n              />\n              <button type="button" onClick={() => void skipRound()} disabled={busy} style={secondaryButton}>\n                Skip round\n              </button>\n            </div>\n          </div>\n        ) : null}'''
    replace_once(path, old_actions, new_actions)
    # Results and full standings only for scored sessions.
    replace_once(path, '{round.status === "saved" ? (', '{round.status === "saved" && scoredSession ? (')
    replace_once(
        path,
        '{generatorKind === "round_robin" ? (\n                <Link href={standingsPath(',
        '{generatorKind === "round_robin" && scoredSession ? (\n                <Link href={standingsPath(',
    )
    # Add played status message.
    replace_once(
        path,
        '        {round.status === "skipped" ? (',
        '        {round.status === "played" ? (\n          <p style={{ marginTop: "1rem", padding: "0.7rem", background: "#dcfce7", borderRadius: "8px" }}>\n            Round {roundNumber} was marked played.\n          </p>\n        ) : null}\n\n        {round.status === "skipped" ? (',
    )
    # Replace progression button with scored standings flow.
    old_progress = '''        {isCurrent && ["saved", "skipped"].includes(round.status) && session.status === "active" ? (\n          <button\n            type="button"\n            onClick={() => void advanceRound()}\n            disabled={busy}\n            style={{ ...primaryButton, marginTop: "1rem" }}\n          >\n            {roundNumber >= event.totalRounds\n              ? "Finish session"\n              : generatorKind === "ladder"\n                ? `Generate Round ${roundNumber + 1}`\n                : `Go to Round ${roundNumber + 1}`}\n          </button>\n        ) : null}'''
    if public:
        standings_href = 'standingsPath(clubId, sessionKey)'
    else:
        standings_href = 'standingsPath(sessionKey)'
    new_progress = f'''        {{isCurrent && ["saved", "played", "skipped"].includes(round.status) && session.status === "active" ? (\n          generatorKind === "round_robin" && scoredSession ? (\n            <Link href={{{standings_href}}} style={{{{ ...primaryButton, display: "inline-flex", marginTop: "1rem", textDecoration: "none" }}}}>\n              View standings and continue\n            </Link>\n          ) : (\n            <button\n              type="button"\n              onClick={{() => void advanceRound()}}\n              disabled={{busy}}\n              style={{{{ ...primaryButton, marginTop: "1rem" }}}}\n            >\n              {{roundNumber >= event.totalRounds\n                ? "Finish session"\n                : generatorKind === "ladder"\n                  ? `Generate Round ${{roundNumber + 1}}`\n                  : `Continue to Round ${{roundNumber + 1}}`}}\n            </button>\n          )\n        ) : null}}'''
    replace_once(path, old_progress, new_progress)


patch_round_runner("apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx", public=False)
patch_round_runner("apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx", public=True)

# ---------------------------------------------------------------------------
# Standings pages own scored Round-Robin progression to the next round
# ---------------------------------------------------------------------------
admin_standings = r'''"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PlayGeneratorStandingsTable, {
  PlayGeneratorStanding,
  standingsSortLabel
} from "@/components/PlayGeneratorStandingsTable";
import { useAdminSession } from "@/lib/useAdminSession";

type StandingsSort = "wins" | "points" | "differential";
type ScoringMode = "scored" | "unscored";

type Session = {
  session_key: string;
  title: string;
  status: string;
  version: string;
  generator_kind: string;
  play_format: string;
  scoring_mode?: ScoringMode;
  current_round_number?: number | null;
  total_rounds?: number | null;
  standings_sort?: StandingsSort;
  standings?: PlayGeneratorStanding[];
  event: {
    scoringMode?: ScoringMode;
    standingsSort?: StandingsSort;
    currentRoundNumber?: number;
    totalRounds?: number;
    rounds?: Array<{ number: number; status: string }>;
  };
};

type Props = { apiBase: string | null; clubId: string; sessionKey: string };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const linkButton = { display: "inline-flex", alignItems: "center", minHeight: "38px", padding: "0.45rem 0.75rem", border: "1px solid #cbd5e1", borderRadius: "999px", color: "#0f172a", fontWeight: 800, textDecoration: "none" };
const primaryButton = { border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#0f172a", color: "white", fontWeight: 800, cursor: "pointer" };

function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
function operationKey(action: string): string { return `${action}-${Date.now()}-${Math.random().toString(16).slice(2)}`; }

export default function AdminGeneratorStandings({ apiBase, clubId, sessionKey }: Props) {
  const router = useRouter();
  const { accessToken } = useAdminSession();
  const [session, setSession] = useState<Session | null>(null);
  const [message, setMessage] = useState("Loading standings…");
  const [busy, setBusy] = useState(false);

  async function loadSession(): Promise<void> {
    if (!apiBase || !accessToken) return;
    try {
      const response = await fetch(
        apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}`),
        { headers: { Authorization: `Bearer ${accessToken}` }, cache: "no-store" }
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
      if (payload?.session?.generator_kind !== "round_robin") throw new Error("Standings are available for Round-Robin Generator sessions.");
      setSession(payload.session as Session);
      setMessage("");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load standings.");
    }
  }

  useEffect(() => { void loadSession(); }, [accessToken, apiBase, clubId, sessionKey]);

  async function continueSession(): Promise<void> {
    if (!apiBase || !accessToken || !session) return;
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch(
        apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/advance`),
        {
          method: "POST",
          headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
          body: JSON.stringify({ expected_version: session.version, idempotency_key: operationKey("standings-advance") })
        }
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
      const next = payload?.session as Session | undefined;
      if (!next) throw new Error("Session advanced without a refreshed session.");
      setSession(next);
      if (next.status === "completed") {
        setMessage("Session completed.");
        return;
      }
      const nextRound = Number(next.current_round_number || 1);
      router.push(`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${nextRound}`);
      router.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to continue the session.");
    } finally {
      setBusy(false);
    }
  }

  if (!session) return <article style={cardStyle}><h1>Round-Robin standings</h1><p>{message}</p></article>;

  const scoringMode = session.scoring_mode || session.event.scoringMode || "scored";
  const currentRound = Number(session.current_round_number || session.event.currentRoundNumber || 1);
  const totalRounds = Number(session.total_rounds || session.event.totalRounds || 1);
  const currentStatus = session.event.rounds?.find((row) => row.number === currentRound)?.status || "";
  const sortMode = session.standings_sort || session.event.standingsSort || "wins";
  const visibleRounds = (session.event.rounds || []).filter((row) => row.number <= currentRound);
  const canContinue = scoringMode === "scored" && session.status === "active" && ["saved", "skipped"].includes(currentStatus);

  if (scoringMode === "unscored") {
    return <article style={cardStyle}><h1>{session.title}</h1><p>This unscored Round-Robin does not use standings.</p><Link href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`}>Return to current round</Link></article>;
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <p style={{ margin: "0 0 0.4rem" }}><Link href="/admin/round-robin-generator">← Round-Robin Generator</Link></p>
        <h1 style={{ margin: "0 0 0.35rem" }}>{session.title} standings</h1>
        <p style={{ margin: 0, color: "#475569" }}>{session.play_format === "singles" ? "Singles" : "Doubles"} · {standingsSortLabel(sortMode)} · {session.status}</p>
      </article>
      <nav aria-label="Round-Robin session navigation" style={{ ...cardStyle, display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <Link href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`} style={linkButton}>Current round</Link>
        {visibleRounds.map((row) => <Link key={row.number} href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${row.number}`} style={linkButton}>Round {row.number}</Link>)}
      </nav>
      <PlayGeneratorStandingsTable rows={session.standings || []} sortMode={sortMode} />
      {canContinue ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>{currentRound >= totalRounds ? "Finish the session" : `Continue to Round ${currentRound + 1}`}</h2>
          <p style={{ color: "#475569" }}>The completed round results are included above. Continue when the organizer is ready for the next round.</p>
          <button type="button" onClick={() => void continueSession()} disabled={busy} style={primaryButton}>{busy ? "Continuing…" : currentRound >= totalRounds ? "Finish session" : `Continue to Round ${currentRound + 1}`}</button>
        </article>
      ) : null}
      {message ? <p role="status">{message}</p> : null}
    </div>
  );
}
'''
(ROOT / "apps/web/app/admin/play-generators/GeneratorStandings.tsx").write_text(admin_standings, encoding="utf-8")

public_standings = admin_standings.replace(
    'import { useAdminSession } from "@/lib/useAdminSession";\n',
    '',
).replace(
    'export default function AdminGeneratorStandings({ apiBase, clubId, sessionKey }: Props) {\n  const router = useRouter();\n  const { accessToken } = useAdminSession();',
    'export default function PublicGeneratorStandings({ apiBase, clubId, sessionKey }: Props) {\n  const router = useRouter();\n  const [editToken, setEditToken] = useState("");',
).replace(
    '    if (!apiBase || !accessToken) return;',
    '    if (!apiBase) return;',
).replace(
    '`/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}`',
    '`/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}`',
).replace(
    '{ headers: { Authorization: `Bearer ${accessToken}` }, cache: "no-store" }',
    '{ cache: "no-store" }',
).replace(
    '  useEffect(() => { void loadSession(); }, [accessToken, apiBase, clubId, sessionKey]);',
    '''  useEffect(() => {\n    const storageKey = `public-generator-edit:${clubId}:${sessionKey}`;\n    const hash = new URLSearchParams(window.location.hash.replace(/^#/, ""));\n    const discovered = hash.get("edit") || sessionStorage.getItem(storageKey) || "";\n    if (discovered) { sessionStorage.setItem(storageKey, discovered); setEditToken(discovered); }\n    if (hash.has("edit")) window.history.replaceState({}, "", `${window.location.pathname}${window.location.search}`);\n    void loadSession();\n  }, [apiBase, clubId, sessionKey]);''',
).replace(
    '    if (!apiBase || !accessToken || !session) return;',
    '    if (!apiBase || !editToken || !session) return;',
).replace(
    '`/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/advance`',
    '`/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/advance`',
).replace(
    '          headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },\n          body: JSON.stringify({ expected_version: session.version, idempotency_key: operationKey("standings-advance") })',
    '          headers: { "Content-Type": "application/json" },\n          body: JSON.stringify({ edit_token: editToken, expected_version: Number(session.version), idempotency_key: operationKey("standings-advance") })',
).replace(
    '      router.push(`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${nextRound}`);',
    '      router.push(`/clubs/${clubId}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${nextRound}`);',
).replace(
    '<Link href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`}>Return to current round</Link>',
    '<Link href={`/clubs/${clubId}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`}>Return to current round</Link>',
).replace(
    '<Link href="/admin/round-robin-generator">← Round-Robin Generator</Link>',
    '<Link href={`/clubs/${clubId}/round-robin-generator`}>← Round-Robin Generator</Link>',
).replace(
    'href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`}',
    'href={`/clubs/${clubId}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`}',
).replace(
    'href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${row.number}`}',
    'href={`/clubs/${clubId}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${row.number}`}',
).replace(
    '  const canContinue = scoringMode === "scored" && session.status === "active" && ["saved", "skipped"].includes(currentStatus);',
    '  const canContinue = Boolean(editToken) && scoringMode === "scored" && session.status === "active" && ["saved", "skipped"].includes(currentStatus);',
)
(ROOT / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings.tsx").write_text(public_standings, encoding="utf-8")

# ---------------------------------------------------------------------------
# Focused regression tests
# ---------------------------------------------------------------------------
(ROOT / "tests/test_round_robin_unscored_flow.py").write_text(r'''from pathlib import Path

import pytest

from jupr_app.domain.adaptive_play_engine import (
    advance_generator_event,
    create_generator_preview,
    generator_event_standings,
    history_before_round,
    mark_generator_round_played,
    save_generator_round,
    start_generator_event,
)

ROOT = Path(__file__).resolve().parents[1]


def test_unscored_round_robin_marks_played_and_advances() -> None:
    preview = create_generator_preview(
        generator_kind="round_robin",
        play_format="singles",
        title="Unscored",
        participant_names=["A", "B", "C"],
        total_rounds=3,
        court_count=1,
        scoring_mode="unscored",
    )
    event = start_generator_event(preview)
    assert event["scoringMode"] == "unscored"
    assert generator_event_standings(event) == []
    with pytest.raises(ValueError, match="Round Played"):
        save_generator_round(event, round_number=1, scores=[])
    played = mark_generator_round_played(event, round_number=1)
    assert played["rounds"][0]["status"] == "played"
    history = history_before_round(played, 2)
    assert sum(history["games"].values()) == 2
    advanced = advance_generator_event(played)
    assert advanced["currentRoundNumber"] == 2
    assert advanced["rounds"][1]["status"] == "active"


def test_ladder_rejects_unscored_mode() -> None:
    with pytest.raises(ValueError, match="requires scored rounds"):
        create_generator_preview(
            generator_kind="ladder",
            play_format="doubles",
            title="Bad ladder",
            participant_names=["A", "B", "C", "D"],
            total_rounds=3,
            court_count=1,
            scoring_mode="unscored",
        )


def test_unscored_routes_setup_and_round_played_controls_exist() -> None:
    admin_routes = (ROOT / "services/api/admin_play_generator_routes.py").read_text()
    public_routes = (ROOT / "services/api/public_play_generator_routes.py").read_text()
    admin_setup = (ROOT / "apps/web/app/admin/play-generators/GeneratorWorkspace.tsx").read_text()
    public_setup = (ROOT / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx").read_text()
    admin_runner = (ROOT / "apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx").read_text()
    public_runner = (ROOT / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx").read_text()
    assert "/played" in admin_routes
    assert "/played" in public_routes
    assert "Unscored — mark each round played" in admin_setup
    assert "Unscored — mark each round played" in public_setup
    assert "Round Played" in admin_runner
    assert "Round Played" in public_runner
    assert "View standings and continue" in admin_runner
    assert "View standings and continue" in public_runner


def test_standings_pages_own_scored_progression() -> None:
    admin = (ROOT / "apps/web/app/admin/play-generators/GeneratorStandings.tsx").read_text()
    public = (ROOT / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings.tsx").read_text()
    for text in (admin, public):
        assert "Continue to Round" in text
        assert "/advance" in text
        assert "This unscored Round-Robin does not use standings." in text
''', encoding="utf-8")

print("Round-Robin unscored and standings progression patch applied.")
