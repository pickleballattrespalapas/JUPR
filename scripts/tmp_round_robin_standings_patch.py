from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one marker, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_count(path: str, old: str, new: str, expected: int) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise SystemExit(f"{path}: expected {expected} markers, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new), encoding="utf-8")


# ---------------------------------------------------------------------------
# Python scheduling and session contracts
# ---------------------------------------------------------------------------
engine = "jupr_app/domain/adaptive_play_engine.py"
replace_once(
    engine,
    "from uuid import uuid4\n\n",
    "from uuid import uuid4\n\n\nSTANDINGS_SORTS = {\"wins\", \"points\", \"differential\"}\n\n\ndef normalize_standings_sort(value: Any) -> str:\n    mode = str(value or \"wins\").strip().lower().replace(\"-\", \"_\")\n    aliases = {\n        \"total_wins\": \"wins\",\n        \"total_points\": \"points\",\n        \"point_differential\": \"differential\",\n        \"diff\": \"differential\",\n    }\n    mode = aliases.get(mode, mode)\n    return mode if mode in STANDINGS_SORTS else \"wins\"\n\n",
)
replace_once(
    engine,
    "    total_rounds: int = 3,\n    court_count: int = 0,\n) -> dict[str, Any]:",
    "    total_rounds: int = 3,\n    court_count: int = 0,\n    standings_sort: str = \"wins\",\n) -> dict[str, Any]:",
)
replace_once(
    engine,
    "        \"playFormat\": fmt,\n        \"status\": \"preview\",",
    "        \"playFormat\": fmt,\n        \"standingsSort\": normalize_standings_sort(standings_sort),\n        \"status\": \"preview\",",
)
replace_once(
    engine,
    "                \"courts\": event[\"courtCount\"],\n                \"schedule\": event[\"rounds\"],",
    "                \"courts\": event[\"courtCount\"],\n                \"standings_sort\": event[\"standingsSort\"],\n                \"schedule\": event[\"rounds\"],",
)
standings_function = '''\n\ndef generator_event_standings(event: dict[str, Any]) -> list[dict[str, Any]]:\n    \"\"\"Return complete saved-round standings using the event's selected primary sort.\n\n    Every participant remains visible, including players who joined late, withdrew,\n    substituted, or have not yet completed a game. Skipped and unsaved rounds do\n    not affect the table.\n    \"\"\"\n    participants = _participant_map(event)\n    stats: dict[str, dict[str, Any]] = {}\n    for participant in event.get(\"participants\") or []:\n        pid = str(participant.get(\"id\") or \"\")\n        if not pid:\n            continue\n        stats[pid] = {\n            \"participantId\": pid,\n            \"name\": str(participant.get(\"name\") or pid),\n            \"matches\": 0,\n            \"wins\": 0,\n            \"losses\": 0,\n            \"pointsFor\": 0,\n            \"pointsAgainst\": 0,\n            \"differential\": 0,\n            \"rosterOrder\": int(participant.get(\"roster_order\") or 9999),\n        }\n\n    for round_row in event.get(\"rounds\") or []:\n        if str(round_row.get(\"status\") or \"\") != \"saved\":\n            continue\n        for match in _round_matches(round_row):\n            if match.get(\"scoreA\") is None or match.get(\"scoreB\") is None:\n                continue\n            score_a = int(match.get(\"scoreA\") or 0)\n            score_b = int(match.get(\"scoreB\") or 0)\n            side_a = [str(value) for value in match.get(\"sideA\") or match.get(\"teamA\") or []]\n            side_b = [str(value) for value in match.get(\"sideB\") or match.get(\"teamB\") or []]\n            for pid, points_for, points_against in [\n                *((pid, score_a, score_b) for pid in side_a),\n                *((pid, score_b, score_a) for pid in side_b),\n            ]:\n                row = stats.get(pid)\n                if row is None:\n                    participant = participants.get(pid, {})\n                    row = {\n                        \"participantId\": pid,\n                        \"name\": str(participant.get(\"name\") or pid),\n                        \"matches\": 0,\n                        \"wins\": 0,\n                        \"losses\": 0,\n                        \"pointsFor\": 0,\n                        \"pointsAgainst\": 0,\n                        \"differential\": 0,\n                        \"rosterOrder\": int(participant.get(\"roster_order\") or 9999),\n                    }\n                    stats[pid] = row\n                row[\"matches\"] += 1\n                row[\"pointsFor\"] += points_for\n                row[\"pointsAgainst\"] += points_against\n                row[\"differential\"] += points_for - points_against\n                row[\"wins\" if points_for > points_against else \"losses\"] += 1\n\n    mode = normalize_standings_sort(event.get(\"standingsSort\"))\n\n    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:\n        common_tail = (\n            int(row.get(\"losses\") or 0),\n            int(row.get(\"rosterOrder\") or 9999),\n            str(row.get(\"name\") or \"\").casefold(),\n            str(row.get(\"participantId\") or \"\"),\n        )\n        if mode == \"points\":\n            return (\n                -int(row.get(\"pointsFor\") or 0),\n                -int(row.get(\"wins\") or 0),\n                -int(row.get(\"differential\") or 0),\n                *common_tail,\n            )\n        if mode == \"differential\":\n            return (\n                -int(row.get(\"differential\") or 0),\n                -int(row.get(\"wins\") or 0),\n                -int(row.get(\"pointsFor\") or 0),\n                *common_tail,\n            )\n        return (\n            -int(row.get(\"wins\") or 0),\n            -int(row.get(\"differential\") or 0),\n            -int(row.get(\"pointsFor\") or 0),\n            *common_tail,\n        )\n\n    rows = sorted(stats.values(), key=sort_key)\n    for rank, row in enumerate(rows, start=1):\n        row[\"rank\"] = rank\n        row[\"standingsSort\"] = mode\n    return rows\n'''
replace_once(
    engine,
    "\ndef _ladder_next_order(event: dict[str, Any], round_row: dict[str, Any], next_round: int) -> list[str]:",
    standings_function + "\n\ndef _ladder_next_order(event: dict[str, Any], round_row: dict[str, Any], next_round: int) -> list[str]:",
)

admin_service = "jupr_app/services/admin_play_generator_service.py"
replace_once(
    admin_service,
    "    create_generator_preview,\n    mutate_generator_roster,",
    "    create_generator_preview,\n    generator_event_standings,\n    mutate_generator_roster,",
)
replace_once(
    admin_service,
    "        \"schedule_rows\": schedule_export_rows(event) if event else [],\n        \"official_publish\": _as_dict(state.get(\"official_publish\")),",
    "        \"schedule_rows\": schedule_export_rows(event) if event else [],\n        \"standings_sort\": str(event.get(\"standingsSort\") or \"wins\") if event else \"wins\",\n        \"standings\": generator_event_standings(event) if event else [],\n        \"official_publish\": _as_dict(state.get(\"official_publish\")),",
)
replace_once(
    admin_service,
    "def preview_play_generator(\n    supabase: Any,\n    *,\n    club_id: str,\n    generator_kind: str,\n    play_format: str,\n    title: str,\n    participant_names: list[str],\n    player_ids: list[int] | None,\n    total_rounds: int,\n    court_count: int,\n) -> dict[str, Any]:",
    "def preview_play_generator(\n    supabase: Any,\n    *,\n    club_id: str,\n    generator_kind: str,\n    play_format: str,\n    title: str,\n    participant_names: list[str],\n    player_ids: list[int] | None,\n    total_rounds: int,\n    court_count: int,\n    standings_sort: str = \"wins\",\n) -> dict[str, Any]:",
)
replace_once(
    admin_service,
    "def create_play_generator_session(\n    supabase: Any,\n    *,\n    club_id: str,\n    generator_kind: str,\n    play_format: str,\n    title: str,\n    participant_names: list[str],\n    player_ids: list[int] | None,\n    total_rounds: int,\n    court_count: int,\n    preview_fingerprint: str | None,\n    actor_email: str,\n    actor_role: str,\n    source: str,\n) -> dict[str, Any]:",
    "def create_play_generator_session(\n    supabase: Any,\n    *,\n    club_id: str,\n    generator_kind: str,\n    play_format: str,\n    title: str,\n    participant_names: list[str],\n    player_ids: list[int] | None,\n    total_rounds: int,\n    court_count: int,\n    preview_fingerprint: str | None,\n    actor_email: str,\n    actor_role: str,\n    source: str,\n    standings_sort: str = \"wins\",\n) -> dict[str, Any]:",
)
replace_count(
    admin_service,
    "        court_count=court_count,\n    )",
    "        court_count=court_count,\n        standings_sort=standings_sort,\n    )",
    2,
)

public_service = "jupr_app/services/public_play_generator_service.py"
replace_once(
    public_service,
    "    create_generator_preview,\n    mutate_generator_roster,",
    "    create_generator_preview,\n    generator_event_standings,\n    mutate_generator_roster,",
)
replace_once(
    public_service,
    "        \"schedule_rows\": schedule_export_rows(event) if event else [],\n        \"unrated\": True,",
    "        \"schedule_rows\": schedule_export_rows(event) if event else [],\n        \"standings_sort\": str(event.get(\"standingsSort\") or \"wins\") if event else \"wins\",\n        \"standings\": generator_event_standings(event) if event else [],\n        \"unrated\": True,",
)
replace_once(
    public_service,
    "def preview_public_play_generator(\n    supabase: Any,\n    *,\n    club_id: str,\n    generator_kind: str,\n    play_format: str,\n    title: str,\n    participant_names: list[Any],\n    participant_player_ids: dict[str, int] | None,\n    total_rounds: int,\n    court_count: int,\n) -> dict[str, Any]:",
    "def preview_public_play_generator(\n    supabase: Any,\n    *,\n    club_id: str,\n    generator_kind: str,\n    play_format: str,\n    title: str,\n    participant_names: list[Any],\n    participant_player_ids: dict[str, int] | None,\n    total_rounds: int,\n    court_count: int,\n    standings_sort: str = \"wins\",\n) -> dict[str, Any]:",
)
replace_once(
    public_service,
    "def create_public_play_generator_session(\n    supabase: Any,\n    *,\n    club_id: str,\n    generator_kind: str,\n    play_format: str,\n    title: str,\n    participant_names: list[Any],\n    participant_player_ids: dict[str, int] | None,\n    total_rounds: int,\n    court_count: int,\n    preview_fingerprint: str | None,\n    idempotency_key: str,\n    requester_hash: str,\n    token_secret: str | None = None,\n) -> dict[str, Any]:",
    "def create_public_play_generator_session(\n    supabase: Any,\n    *,\n    club_id: str,\n    generator_kind: str,\n    play_format: str,\n    title: str,\n    participant_names: list[Any],\n    participant_player_ids: dict[str, int] | None,\n    total_rounds: int,\n    court_count: int,\n    preview_fingerprint: str | None,\n    idempotency_key: str,\n    requester_hash: str,\n    token_secret: str | None = None,\n    standings_sort: str = \"wins\",\n) -> dict[str, Any]:",
)
replace_count(
    public_service,
    "            court_count=max(0, min(int(court_count or 0), 20)),\n        )",
    "            court_count=max(0, min(int(court_count or 0), 20)),\n            standings_sort=standings_sort,\n        )",
    1,
)
replace_once(
    public_service,
    "        court_count=court_count,\n    )\n    preview = preview_result[\"preview\"]",
    "        court_count=court_count,\n        standings_sort=standings_sort,\n    )\n    preview = preview_result[\"preview\"]",
)
replace_once(
    public_service,
    "        \"court_sizes\": [int(preview.get(\"courtCount\") or 0)],\n        \"live_mode\": \"quick\",",
    "        \"court_sizes\": [int(preview.get(\"courtCount\") or 0)],\n        \"standings_sort\": str(preview.get(\"standingsSort\") or \"wins\"),\n        \"live_mode\": \"quick\",",
)

admin_routes = "services/api/admin_play_generator_routes.py"
replace_once(
    admin_routes,
    "    court_count: int = Field(default=0, ge=0, le=20)\n",
    "    court_count: int = Field(default=0, ge=0, le=20)\n    standings_sort: str = Field(default=\"wins\", pattern=r\"^(wins|points|differential)$\")\n",
)
replace_once(
    admin_routes,
    "                total_rounds=payload.total_rounds,\n                court_count=payload.court_count,\n            )",
    "                total_rounds=payload.total_rounds,\n                court_count=payload.court_count,\n                standings_sort=payload.standings_sort,\n            )",
)
replace_once(
    admin_routes,
    "                    court_count=payload.court_count,\n                    preview_fingerprint=payload.preview_fingerprint,",
    "                    court_count=payload.court_count,\n                    preview_fingerprint=payload.preview_fingerprint,\n                    standings_sort=payload.standings_sort,",
)

public_routes = "services/api/public_play_generator_routes.py"
replace_once(
    public_routes,
    "    court_count: int = Field(default=0, ge=0, le=20)\n",
    "    court_count: int = Field(default=0, ge=0, le=20)\n    standings_sort: str = Field(default=\"wins\", pattern=r\"^(wins|points|differential)$\")\n",
)
replace_once(
    public_routes,
    "                court_count=payload.court_count,\n            )",
    "                court_count=payload.court_count,\n                standings_sort=payload.standings_sort,\n            )",
)
replace_once(
    public_routes,
    "                court_count=payload.court_count,\n                preview_fingerprint=payload.preview_fingerprint,",
    "                court_count=payload.court_count,\n                preview_fingerprint=payload.preview_fingerprint,\n                standings_sort=payload.standings_sort,",
)

# ---------------------------------------------------------------------------
# Browser draft and setup controls
# ---------------------------------------------------------------------------
draft = "apps/web/lib/playGeneratorDraft.ts"
replace_once(
    draft,
    "  playFormat: \"singles\" | \"doubles\";\n  targetCount: number;",
    "  playFormat: \"singles\" | \"doubles\";\n  standingsSort?: \"wins\" | \"points\" | \"differential\";\n  targetCount: number;",
)

workspace_paths = [
    "apps/web/app/admin/play-generators/GeneratorWorkspace.tsx",
    "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx",
]
for workspace in workspace_paths:
    replace_once(
        workspace,
        'type PlayFormat = "singles" | "doubles";\n',
        'type PlayFormat = "singles" | "doubles";\ntype StandingsSort = "wins" | "points" | "differential";\n',
    )
    replace_once(
        workspace,
        "  playFormat: PlayFormat;\n  totalRounds: number;",
        "  playFormat: PlayFormat;\n  standingsSort?: StandingsSort;\n  totalRounds: number;",
    )
    replace_once(
        workspace,
        '  const [playFormat, setPlayFormat] = useState<PlayFormat>("doubles");\n  const [targetCount, setTargetCount] = useState(8);',
        '  const [playFormat, setPlayFormat] = useState<PlayFormat>("doubles");\n  const [standingsSort, setStandingsSort] = useState<StandingsSort>("wins");\n  const [targetCount, setTargetCount] = useState(8);',
    )
    replace_once(
        workspace,
        "      setPlayFormat(stored.playFormat);\n      setTargetCount(stored.targetCount);",
        "      setPlayFormat(stored.playFormat);\n      setStandingsSort(stored.standingsSort || \"wins\");\n      setTargetCount(stored.targetCount);",
    )
    replace_once(
        workspace,
        "      title,\n      playFormat,\n      targetCount,",
        "      title,\n      playFormat,\n      standingsSort,\n      targetCount,",
    )
    replace_once(
        workspace,
        "    title,\n    playFormat,\n    targetCount,",
        "    title,\n    playFormat,\n    standingsSort,\n    targetCount,",
    )
    replace_once(
        workspace,
        "      generator_kind: generatorKind,\n      play_format: playFormat,\n      title: title.trim(),",
        "      generator_kind: generatorKind,\n      play_format: playFormat,\n      standings_sort: standingsSort,\n      title: title.trim(),",
    )
    setup_select = '''          {generatorKind === "round_robin" ? (\n            <label>\n              Standings ranked by\n              <br />\n              <select\n                value={standingsSort}\n                onChange={(event) => {\n                  setStandingsSort(event.target.value as StandingsSort);\n                  invalidatePreview();\n                }}\n                style={inputStyle}\n              >\n                <option value="wins">Total wins</option>\n                <option value="points">Total points</option>\n                <option value="differential">Point differential</option>\n              </select>\n              <small style={{ display: "block", marginTop: "0.35rem", color: "#64748b" }}>\n                This primary ranking applies to the full session standings.\n              </small>\n            </label>\n          ) : null}\n'''
    replace_once(
        workspace,
        "          </label>\n        </div>\n\n        <GeneratorRosterSetup",
        "          </label>\n" + setup_select + "        </div>\n\n        <GeneratorRosterSetup",
    )

# ---------------------------------------------------------------------------
# Round-page links and dedicated standings pages
# ---------------------------------------------------------------------------
admin_runner = "apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx"
public_runner = "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx"
replace_once(
    admin_runner,
    "function roundPath(kind: GeneratorKind, sessionKey: string, round: number): string {\n  return `/admin/${generatorSlug(kind)}/sessions/${encodeURIComponent(sessionKey)}/rounds/${round}`;\n}\n",
    "function roundPath(kind: GeneratorKind, sessionKey: string, round: number): string {\n  return `/admin/${generatorSlug(kind)}/sessions/${encodeURIComponent(sessionKey)}/rounds/${round}`;\n}\n\nfunction standingsPath(sessionKey: string): string {\n  return `/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/standings`;\n}\n",
)
replace_once(
    public_runner,
    "function roundPath(kind: GeneratorKind, clubSlug: string, sessionKey: string, round: number): string {\n  return `/clubs/${clubSlug}/${generatorSlug(kind)}/sessions/${encodeURIComponent(sessionKey)}/rounds/${round}`;\n}\n",
    "function roundPath(kind: GeneratorKind, clubSlug: string, sessionKey: string, round: number): string {\n  return `/clubs/${clubSlug}/${generatorSlug(kind)}/sessions/${encodeURIComponent(sessionKey)}/rounds/${round}`;\n}\n\nfunction standingsPath(clubSlug: string, sessionKey: string): string {\n  return `/clubs/${clubSlug}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/standings`;\n}\n",
)
replace_once(
    admin_runner,
    "          <div style={{ display: \"flex\", gap: \"0.5rem\", flexWrap: \"wrap\" }}>\n            {previousRound ? (",
    "          <div style={{ display: \"flex\", gap: \"0.5rem\", flexWrap: \"wrap\" }}>\n            {generatorKind === \"round_robin\" ? (\n              <Link href={standingsPath(sessionKey)} style={secondaryButton}>\n                Standings\n              </Link>\n            ) : null}\n            {previousRound ? (",
)
replace_once(
    public_runner,
    "          <div style={{ display: \"flex\", gap: \"0.5rem\", flexWrap: \"wrap\" }}>\n            {previousRound ? (",
    "          <div style={{ display: \"flex\", gap: \"0.5rem\", flexWrap: \"wrap\" }}>\n            {generatorKind === \"round_robin\" ? (\n              <Link href={standingsPath(clubId, sessionKey)} style={secondaryButton}>\n                Standings\n              </Link>\n            ) : null}\n            {previousRound ? (",
)
replace_once(
    admin_runner,
    "            <h3>Round {roundNumber} results</h3>",
    "            <div style={{ display: \"flex\", justifyContent: \"space-between\", gap: \"0.75rem\", flexWrap: \"wrap\", alignItems: \"center\" }}>\n              <h3 style={{ margin: 0 }}>Round {roundNumber} results</h3>\n              {generatorKind === \"round_robin\" ? (\n                <Link href={standingsPath(sessionKey)} style={secondaryButton}>View full standings</Link>\n              ) : null}\n            </div>",
)
replace_once(
    public_runner,
    "            <h3>Round {roundNumber} results</h3>",
    "            <div style={{ display: \"flex\", justifyContent: \"space-between\", gap: \"0.75rem\", flexWrap: \"wrap\", alignItems: \"center\" }}>\n              <h3 style={{ margin: 0 }}>Round {roundNumber} results</h3>\n              {generatorKind === \"round_robin\" ? (\n                <Link href={standingsPath(clubId, sessionKey)} style={secondaryButton}>View full standings</Link>\n              ) : null}\n            </div>",
)

# Shared full standings table.
(ROOT / "apps/web/components/PlayGeneratorStandingsTable.tsx").write_text('''type StandingsSort = "wins" | "points" | "differential";\n\nexport type PlayGeneratorStanding = {\n  rank: number;\n  participantId: string;\n  name: string;\n  matches: number;\n  wins: number;\n  losses: number;\n  pointsFor: number;\n  pointsAgainst: number;\n  differential: number;\n};\n\ntype Props = {\n  rows: PlayGeneratorStanding[];\n  sortMode: StandingsSort;\n};\n\nexport function standingsSortLabel(mode: StandingsSort): string {\n  if (mode === "points") return "Total points";\n  if (mode === "differential") return "Point differential";\n  return "Total wins";\n}\n\nfunction tieBreakText(mode: StandingsSort): string {\n  if (mode === "points") return "Ties: wins, point differential, then starting roster order.";\n  if (mode === "differential") return "Ties: wins, total points, then starting roster order.";\n  return "Ties: point differential, total points, then starting roster order.";\n}\n\nconst primaryCell = {\n  background: "#ecfdf5",\n  fontWeight: 850\n};\n\nexport default function PlayGeneratorStandingsTable({ rows, sortMode }: Props) {\n  return (\n    <article\n      style={{\n        border: "1px solid #e2e8f0",\n        borderRadius: "14px",\n        padding: "1rem",\n        background: "white"\n      }}\n    >\n      <div style={{ marginBottom: "0.85rem" }}>\n        <h2 style={{ margin: "0 0 0.3rem" }}>Full standings</h2>\n        <p style={{ margin: 0, color: "#475569" }}>\n          Ranked by <strong>{standingsSortLabel(sortMode)}</strong>. {tieBreakText(sortMode)}\n          {" "}Skipped and unplayed rounds do not affect the table.\n        </p>\n      </div>\n\n      <div style={{ overflowX: "auto" }}>\n        <table style={{ width: "100%", minWidth: 680, borderCollapse: "collapse" }}>\n          <thead>\n            <tr style={{ borderBottom: "2px solid #cbd5e1" }}>\n              <th align="center" style={{ padding: "0.55rem" }}>Rank</th>\n              <th align="left" style={{ padding: "0.55rem" }}>Player</th>\n              <th align="center" style={{ padding: "0.55rem" }}>GP</th>\n              <th align="center" style={{ padding: "0.55rem", ...(sortMode === "wins" ? primaryCell : {}) }}>W</th>\n              <th align="center" style={{ padding: "0.55rem" }}>L</th>\n              <th align="center" style={{ padding: "0.55rem", ...(sortMode === "points" ? primaryCell : {}) }}>Points</th>\n              <th align="center" style={{ padding: "0.55rem" }}>Against</th>\n              <th align="center" style={{ padding: "0.55rem", ...(sortMode === "differential" ? primaryCell : {}) }}>Diff</th>\n            </tr>\n          </thead>\n          <tbody>\n            {rows.map((row) => (\n              <tr key={row.participantId} style={{ borderBottom: "1px solid #e2e8f0" }}>\n                <td align="center" style={{ padding: "0.65rem", fontWeight: 850 }}>{row.rank}</td>\n                <td style={{ padding: "0.65rem", fontWeight: 750 }}>{row.name}</td>\n                <td align="center" style={{ padding: "0.65rem" }}>{row.matches}</td>\n                <td align="center" style={{ padding: "0.65rem", ...(sortMode === "wins" ? primaryCell : {}) }}>{row.wins}</td>\n                <td align="center" style={{ padding: "0.65rem" }}>{row.losses}</td>\n                <td align="center" style={{ padding: "0.65rem", ...(sortMode === "points" ? primaryCell : {}) }}>{row.pointsFor}</td>\n                <td align="center" style={{ padding: "0.65rem" }}>{row.pointsAgainst}</td>\n                <td align="center" style={{ padding: "0.65rem", ...(sortMode === "differential" ? primaryCell : {}) }}>\n                  {row.differential > 0 ? "+" : ""}{row.differential}\n                </td>\n              </tr>\n            ))}\n          </tbody>\n        </table>\n      </div>\n    </article>\n  );\n}\n''', encoding="utf-8")

admin_standings = '''"use client";\n\nimport Link from "next/link";\nimport { useEffect, useState } from "react";\nimport PlayGeneratorStandingsTable, {\n  PlayGeneratorStanding,\n  standingsSortLabel\n} from "@/components/PlayGeneratorStandingsTable";\nimport { useAdminSession } from "@/lib/useAdminSession";\n\ntype StandingsSort = "wins" | "points" | "differential";\n\ntype Session = {\n  session_key: string;\n  title: string;\n  status: string;\n  generator_kind: string;\n  play_format: string;\n  current_round_number?: number | null;\n  total_rounds?: number | null;\n  standings_sort?: StandingsSort;\n  standings?: PlayGeneratorStanding[];\n  event: {\n    standingsSort?: StandingsSort;\n    currentRoundNumber?: number;\n    rounds?: Array<{ number: number; status: string }>;\n  };\n};\n\ntype Props = { apiBase: string | null; clubId: string; sessionKey: string };\n\nconst cardStyle = {\n  border: "1px solid #e2e8f0",\n  borderRadius: "14px",\n  padding: "1rem",\n  background: "white"\n};\n\nconst linkButton = {\n  display: "inline-flex",\n  alignItems: "center",\n  minHeight: "38px",\n  padding: "0.45rem 0.75rem",\n  border: "1px solid #cbd5e1",\n  borderRadius: "999px",\n  color: "#0f172a",\n  fontWeight: 800,\n  textDecoration: "none"\n};\n\nfunction apiUrl(apiBase: string, path: string): string {\n  return `${apiBase.replace(/\\/$/, "")}${path}`;\n}\n\nexport default function AdminGeneratorStandings({ apiBase, clubId, sessionKey }: Props) {\n  const { accessToken } = useAdminSession();\n  const [session, setSession] = useState<Session | null>(null);\n  const [message, setMessage] = useState("Loading standings…");\n\n  useEffect(() => {\n    if (!apiBase || !accessToken) return;\n    let active = true;\n    void fetch(\n      apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}`),\n      { headers: { Authorization: `Bearer ${accessToken}` }, cache: "no-store" }\n    )\n      .then(async (response) => {\n        const payload = await response.json().catch(() => null);\n        if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));\n        if (payload?.session?.generator_kind !== "round_robin") {\n          throw new Error("Standings are available for Round-Robin Generator sessions.");\n        }\n        if (active) setSession(payload.session as Session);\n      })\n      .catch((error) => {\n        if (active) setMessage(error instanceof Error ? error.message : "Unable to load standings.");\n      });\n    return () => { active = false; };\n  }, [accessToken, apiBase, clubId, sessionKey]);\n\n  if (!session) {\n    return <article style={cardStyle}><h1>Round-Robin standings</h1><p>{message}</p></article>;\n  }\n\n  const currentRound = Number(session.current_round_number || session.event.currentRoundNumber || 1);\n  const sortMode = session.standings_sort || session.event.standingsSort || "wins";\n  const visibleRounds = (session.event.rounds || []).filter((row) => row.number <= currentRound);\n\n  return (\n    <div style={{ display: "grid", gap: "1rem" }}>\n      <article style={{ ...cardStyle, background: "#f8fafc" }}>\n        <p style={{ margin: "0 0 0.4rem" }}>\n          <Link href="/admin/round-robin-generator">← Round-Robin Generator</Link>\n        </p>\n        <h1 style={{ margin: "0 0 0.35rem" }}>{session.title} standings</h1>\n        <p style={{ margin: 0, color: "#475569" }}>\n          {session.play_format === "singles" ? "Singles" : "Doubles"} · {standingsSortLabel(sortMode)} · {session.status}\n        </p>\n      </article>\n\n      <nav aria-label="Round-Robin session navigation" style={{ ...cardStyle, display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>\n        <Link\n          href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`}\n          style={linkButton}\n        >\n          Current round\n        </Link>\n        {visibleRounds.map((row) => (\n          <Link\n            key={row.number}\n            href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${row.number}`}\n            style={linkButton}\n          >\n            Round {row.number}\n          </Link>\n        ))}\n      </nav>\n\n      <PlayGeneratorStandingsTable rows={session.standings || []} sortMode={sortMode} />\n    </div>\n  );\n}\n'''
(ROOT / "apps/web/app/admin/play-generators/GeneratorStandings.tsx").write_text(admin_standings, encoding="utf-8")

public_standings = admin_standings.replace(
    'import { useAdminSession } from "@/lib/useAdminSession";\n',
    '',
).replace(
    'export default function AdminGeneratorStandings({ apiBase, clubId, sessionKey }: Props) {\n  const { accessToken } = useAdminSession();',
    'export default function PublicGeneratorStandings({ apiBase, clubId, sessionKey }: Props) {',
).replace(
    '    if (!apiBase || !accessToken) return;',
    '    if (!apiBase) return;',
).replace(
    '      { headers: { Authorization: `Bearer ${accessToken}` }, cache: "no-store" }',
    '      { cache: "no-store" }',
).replace(
    '  }, [accessToken, apiBase, clubId, sessionKey]);',
    '  }, [apiBase, clubId, sessionKey]);',
).replace(
    '`/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}`',
    '`/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}`',
).replace(
    '<Link href="/admin/round-robin-generator">← Round-Robin Generator</Link>',
    '<Link href={`/clubs/${clubId}/round-robin-generator`}>← Round-Robin Generator</Link>',
).replace(
    'href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`}',
    'href={`/clubs/${clubId}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`}',
).replace(
    'href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${row.number}`}',
    'href={`/clubs/${clubId}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${row.number}`}',
)
(ROOT / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings.tsx").write_text(public_standings, encoding="utf-8")

admin_page = '''import GeneratorStandings from "@/app/admin/play-generators/GeneratorStandings";\n\ntype Props = { params: { sessionKey: string } };\n\nfunction apiBase(): string | null {\n  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;\n}\n\nexport default function RoundRobinStandingsPage({ params }: Props) {\n  return <GeneratorStandings apiBase={apiBase()} clubId="tres_palapas" sessionKey={params.sessionKey} />;\n}\n'''
admin_page_path = ROOT / "apps/web/app/admin/round-robin-generator/sessions/[sessionKey]/standings/page.tsx"
admin_page_path.parent.mkdir(parents=True, exist_ok=True)
admin_page_path.write_text(admin_page, encoding="utf-8")

public_page = '''import PublicGeneratorStandings from "@/app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings";\n\ntype Props = { params: { clubSlug: string; sessionKey: string } };\n\nfunction apiBase(): string | null {\n  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;\n}\n\nexport default function PublicRoundRobinStandingsPage({ params }: Props) {\n  return (\n    <PublicGeneratorStandings\n      apiBase={apiBase()}\n      clubId={params.clubSlug}\n      sessionKey={params.sessionKey}\n    />\n  );\n}\n'''
public_page_path = ROOT / "apps/web/app/clubs/[clubSlug]/round-robin-generator/sessions/[sessionKey]/standings/page.tsx"
public_page_path.parent.mkdir(parents=True, exist_ok=True)
public_page_path.write_text(public_page, encoding="utf-8")

# ---------------------------------------------------------------------------
# Focused contracts
# ---------------------------------------------------------------------------
(ROOT / "tests/test_play_generator_full_standings.py").write_text('''from pathlib import Path\n\nfrom jupr_app.domain.adaptive_play_engine import generator_event_standings\n\nROOT = Path(__file__).resolve().parents[1]\n\n\ndef _event(sort_mode: str):\n    return {\n        "standingsSort": sort_mode,\n        "participants": [\n            {"id": "p1", "name": "Alex", "roster_order": 1},\n            {"id": "p2", "name": "Blake", "roster_order": 2},\n            {"id": "p3", "name": "Casey", "roster_order": 3},\n            {"id": "p4", "name": "Drew", "roster_order": 4},\n        ],\n        "rounds": [\n            {\n                "number": 1,\n                "status": "saved",\n                "matches": [\n                    {"id": "m1", "sideA": ["p1"], "sideB": ["p2"], "scoreA": 1, "scoreB": 0},\n                    {"id": "m2", "sideA": ["p1"], "sideB": ["p3"], "scoreA": 1, "scoreB": 0},\n                    {"id": "m3", "sideA": ["p2"], "sideB": ["p3"], "scoreA": 20, "scoreB": 19},\n                ],\n            },\n            {\n                "number": 2,\n                "status": "skipped",\n                "matches": [\n                    {"id": "ignored", "sideA": ["p4"], "sideB": ["p1"], "scoreA": 99, "scoreB": 0}\n                ],\n            },\n        ],\n    }\n\n\ndef test_full_standings_follow_selected_primary_sort() -> None:\n    assert [row["name"] for row in generator_event_standings(_event("wins"))] == [\n        "Alex", "Blake", "Casey", "Drew"\n    ]\n    assert [row["name"] for row in generator_event_standings(_event("points"))] == [\n        "Blake", "Casey", "Alex", "Drew"\n    ]\n    assert [row["name"] for row in generator_event_standings(_event("differential"))] == [\n        "Blake", "Alex", "Drew", "Casey"\n    ]\n\n\ndef test_full_standings_include_zero_game_players_and_ignore_skips() -> None:\n    rows = generator_event_standings(_event("wins"))\n    drew = next(row for row in rows if row["name"] == "Drew")\n    assert drew["matches"] == 0\n    assert drew["pointsFor"] == 0\n    assert all(row["pointsFor"] < 99 for row in rows)\n\n\ndef test_round_robin_standings_routes_and_links_exist() -> None:\n    admin_runner = (ROOT / "apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx").read_text()\n    public_runner = (ROOT / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx").read_text()\n    admin_setup = (ROOT / "apps/web/app/admin/play-generators/GeneratorWorkspace.tsx").read_text()\n    public_setup = (ROOT / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx").read_text()\n    assert "View full standings" in admin_runner\n    assert "View full standings" in public_runner\n    assert "Standings ranked by" in admin_setup\n    assert "Standings ranked by" in public_setup\n    assert (ROOT / "apps/web/app/admin/round-robin-generator/sessions/[sessionKey]/standings/page.tsx").exists()\n    assert (ROOT / "apps/web/app/clubs/[clubSlug]/round-robin-generator/sessions/[sessionKey]/standings/page.tsx").exists()\n\n\ndef test_api_accepts_and_persists_standings_sort() -> None:\n    admin_routes = (ROOT / "services/api/admin_play_generator_routes.py").read_text()\n    public_routes = (ROOT / "services/api/public_play_generator_routes.py").read_text()\n    engine = (ROOT / "jupr_app/domain/adaptive_play_engine.py").read_text()\n    assert "standings_sort" in admin_routes\n    assert "standings_sort" in public_routes\n    assert '"standingsSort"' in engine\n''', encoding="utf-8")

print("Round-Robin full standings patch applied.")
