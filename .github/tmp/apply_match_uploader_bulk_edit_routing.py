from pathlib import Path
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def replace_count(text: str, old: str, new: str, expected: int, label: str) -> str:
    count = text.count(old)
    if count != expected:
        raise SystemExit(f"{label}: expected {expected} matches, found {count}")
    return text.replace(old, new)


# ---------------------------------------------------------------------------
# Match Uploader result handoff
# ---------------------------------------------------------------------------
form_path = Path("apps/web/app/admin/match-uploader/MatchUploaderForm.tsx")
form = form_path.read_text(encoding="utf-8")

form = replace_once(
    form,
    '''function SubmissionResultDialog({
    result,
    roundRobinRecords,
    onClose,
  }: {
    result: AdminMatchUploaderWriteResult;
    roundRobinRecords?: PlayerRoundRobinRecords | null;
    onClose: () => void;
  }) {''',
    '''function SubmissionResultDialog({
    result,
    roundRobinRecords,
    submissionKind,
    onClose,
  }: {
    result: AdminMatchUploaderWriteResult;
    roundRobinRecords?: PlayerRoundRobinRecords | null;
    submissionKind: "manual" | "round_robin" | "singles" | null;
    onClose: () => void;
  }) {''',
    "submission result props",
)

form = replace_once(
    form,
    '''    const matchIds = (result.operation?.match_ids || []).map((value) => String(value)).filter(Boolean);
    const correctionMatchId = matchIds[0] || (result.feedback?.latest_match_id == null ? "" : String(result.feedback.latest_match_id));
    const correctionHref = correctionMatchId
      ? `/admin/match-log/edit?match_id=${encodeURIComponent(correctionMatchId)}`
      : (result.recovery?.match_log_route || "/admin/match-log");
    const showRoundRobinRecords = Boolean(roundRobinRecords && Object.keys(roundRobinRecords).length);''',
    '''    const matchIds = (result.operation?.match_ids || []).map((value) => String(value)).filter(Boolean);
    const correctionMatchId = matchIds[0] || (result.feedback?.latest_match_id == null ? "" : String(result.feedback.latest_match_id));
    const [chooseMatchesToEdit, setChooseMatchesToEdit] = useState(false);
    const [selectedCorrectionIds, setSelectedCorrectionIds] = useState<string[]>(() => [...matchIds]);
    const bulkCorrectionHref = (ids: string[]) => {
      const params = new URLSearchParams();
      params.set("match_ids", ids.join(","));
      params.set("selected_ids", ids.join(","));
      params.set("limit", String(Math.max(250, ids.length)));
      return `/admin/match-log/bulk?${params.toString()}`;
    };
    const isRoundRobinBulk = submissionKind === "round_robin" && matchIds.length > 1;
    const isManualMulti = submissionKind === "manual" && matchIds.length > 1;
    const correctionHref = isRoundRobinBulk
      ? bulkCorrectionHref(matchIds)
      : correctionMatchId
        ? `/admin/match-log/edit?match_id=${encodeURIComponent(correctionMatchId)}`
        : (result.recovery?.match_log_route || "/admin/match-log");
    const showRoundRobinRecords = Boolean(roundRobinRecords && Object.keys(roundRobinRecords).length);''',
    "submission result correction routing",
)

form = replace_once(
    form,
    '''        {result.warnings?.length ? <ul style={{ color: "#92400e" }}>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
        <p style={{ display: "flex", justifyContent: "flex-end", gap: "0.5rem", flexWrap: "wrap", marginBottom: 0 }}>
        {correctionMatchId ? <Link href={correctionHref} style={{ ...ghostButtonStyle, textDecoration: "none", display: "inline-flex", alignItems: "center" }}>Edit results</Link> : null}
        <button type="button" onClick={onClose} style={buttonStyle}>OK</button>
      </p>''',
    '''        {result.warnings?.length ? <ul style={{ color: "#92400e" }}>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
        {isManualMulti && chooseMatchesToEdit ? (
          <div style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.85rem", marginTop: "1rem", background: "#f8fafc" }}>
            <h3 style={{ marginTop: 0 }}>Choose matches to edit</h3>
            <p style={{ color: "#475569" }}>Select the uploaded matches that need correction. They will open together in Bulk edit.</p>
            <div style={{ display: "grid", gap: "0.4rem" }}>
              {matchIds.map((id) => (
                <label key={id} style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                  <input
                    type="checkbox"
                    checked={selectedCorrectionIds.includes(id)}
                    onChange={() => setSelectedCorrectionIds((current) => current.includes(id) ? current.filter((value) => value !== id) : [...current, id])}
                  />
                  Match #{id}
                </label>
              ))}
            </div>
            <p style={{ display: "flex", justifyContent: "flex-end", gap: "0.5rem", flexWrap: "wrap", marginBottom: 0 }}>
              <button type="button" onClick={() => setChooseMatchesToEdit(false)} style={ghostButtonStyle}>Back</button>
              {selectedCorrectionIds.length ? (
                <Link href={bulkCorrectionHref(selectedCorrectionIds)} style={{ ...ghostButtonStyle, textDecoration: "none", display: "inline-flex", alignItems: "center" }}>
                  Open selected in bulk editor
                </Link>
              ) : (
                <button type="button" disabled style={ghostButtonStyle}>Open selected in bulk editor</button>
              )}
            </p>
          </div>
        ) : null}
        <p style={{ display: "flex", justifyContent: "flex-end", gap: "0.5rem", flexWrap: "wrap", marginBottom: 0 }}>
        {correctionMatchId && !isManualMulti ? <Link href={correctionHref} style={{ ...ghostButtonStyle, textDecoration: "none", display: "inline-flex", alignItems: "center" }}>Edit results</Link> : null}
        {correctionMatchId && isManualMulti && !chooseMatchesToEdit ? <button type="button" onClick={() => setChooseMatchesToEdit(true)} style={ghostButtonStyle}>Edit results</button> : null}
        <button type="button" onClick={onClose} style={buttonStyle}>OK</button>
      </p>''',
    "submission result selection prompt",
)

form = replace_once(
    form,
    '''      {result ? <SubmissionResultDialog result={result} roundRobinRecords={rrResultRecords} onClose={acknowledgeSubmission} /> : null}''',
    '''      {result ? <SubmissionResultDialog result={result} roundRobinRecords={rrResultRecords} submissionKind={submissionKind} onClose={acknowledgeSubmission} /> : null}''',
    "submission result call",
)

form_path.write_text(form, encoding="utf-8")

# ---------------------------------------------------------------------------
# Match Log URL/filter handoff
# ---------------------------------------------------------------------------
workspace_path = Path("apps/web/app/admin/match-log/MatchLogWorkspace.tsx")
workspace = workspace_path.read_text(encoding="utf-8")

workspace = replace_once(
    workspace,
    '''  match_id?: string;
  league?: string;''',
    '''  match_id?: string;
  match_ids?: string;
  selected_ids?: string;
  league?: string;''',
    "workspace search params",
)

workspace = replace_once(
    workspace,
    '''  { mode: "bulk", path: "/admin/match-log/bulk", label: "Bulk edit", title: "Bulk edit matches", description: "Apply the same correction to a selected group of visible matches." },''',
    '''  { mode: "bulk", path: "/admin/match-log/bulk", label: "Bulk edit", title: "Bulk edit matches", description: "Review selected matches, edit scores individually, and optionally apply shared corrections." },''',
    "bulk workspace copy",
)

workspace = replace_once(
    workspace,
    '''  const matchIdParam = searchParams?.match_id || null;
  const leagueParam = searchParams?.league || null;''',
    '''  const matchIdsParam = searchParams?.match_ids || searchParams?.match_id || null;
  const initialSelectedIds = (searchParams?.selected_ids || "")
    .split(/[\\s,;]+/)
    .map((value) => value.replace(/^#/, "").trim())
    .filter(Boolean)
    .slice(0, 100);
  const initialSelectedIdsKey = initialSelectedIds.join(",");
  const leagueParam = searchParams?.league || null;''',
    "workspace multiple IDs",
)

workspace = replace_once(
    workspace,
    '''    matchIdParam || "",
    leagueParam || "",''',
    '''    matchIdsParam || "",
    leagueParam || "",''',
    "workspace request scope",
)

workspace = replace_once(
    workspace,
    '''        matchId: matchIdParam,
        league: leagueParam,''',
    '''        matchIds: matchIdsParam,
        league: leagueParam,''',
    "workspace API request",
)

workspace = replace_count(
    workspace,
    '''    matchIdParam,
''',
    '''    matchIdsParam,
''',
    1,
    "workspace effect dependency",
)

workspace = replace_once(
    workspace,
    '''            <label>Match ID<br /><input key={`match-${matchIdParam || "all"}`} name="match_id" defaultValue={matchIdParam || ""} style={{ width: "100%" }} /></label>''',
    '''            <label>Match IDs<br /><input key={`matches-${matchIdsParam || "all"}`} name="match_ids" defaultValue={matchIdsParam || ""} placeholder="e.g. 27, 28, 29" style={{ width: "100%" }} /><small>Comma or space-separated.</small></label>''',
    "workspace match IDs filter",
)

workspace = replace_once(
    workspace,
    '''            <MatchLogApplyPanel
              mode={mode === "edit" ? "guided" : mode}''',
    '''            <MatchLogApplyPanel
              key={`${mode}:${initialSelectedIdsKey}:${data.matches.map((match) => match.id).join(",")}`}
              mode={mode === "edit" ? "guided" : mode}''',
    "workspace panel key",
)

workspace = replace_once(
    workspace,
    '''              recentOperations={data.recent_edit_operations || []}
              exclusionOperation={exclusionOperation}''',
    '''              recentOperations={data.recent_edit_operations || []}
              initialSelectedIds={initialSelectedIds}
              exclusionOperation={exclusionOperation}''',
    "workspace initial selected IDs",
)

workspace_path.write_text(workspace, encoding="utf-8")

# ---------------------------------------------------------------------------
# Match Log client query types
# ---------------------------------------------------------------------------
api_path = Path("apps/web/lib/adminMatchLogApi.ts")
api = api_path.read_text(encoding="utf-8")

api = replace_once(
    api,
    '''    match_id?: number | null;
    league?: string | null;''',
    '''    match_id?: number | null;
    match_ids?: number[];
    league?: string | null;''',
    "match log response filter type",
)

api = replace_once(
    api,
    '''  matchId?: string | number | null;
  league?: string | null;''',
    '''  matchId?: string | number | null;
  matchIds?: string | Array<string | number> | null;
  league?: string | null;''',
    "match log request type",
)

api = replace_once(
    api,
    '''  if (params?.filter) query.set("filter", String(params.filter));
  if (params?.matchId) query.set("match_id", String(params.matchId));
  if (params?.league) query.set("league", String(params.league));''',
    '''  if (params?.filter) query.set("filter", String(params.filter));
  if (params?.matchIds) {
    const matchIds = Array.isArray(params.matchIds)
      ? params.matchIds.map((value) => String(value).trim()).filter(Boolean).join(",")
      : String(params.matchIds).trim();
    if (matchIds) query.set("match_ids", matchIds);
  } else if (params?.matchId) {
    query.set("match_id", String(params.matchId));
  }
  if (params?.league) query.set("league", String(params.league));''',
    "match log multiple ID request",
)

api_path.write_text(api, encoding="utf-8")

# ---------------------------------------------------------------------------
# Bulk editor initial selection and per-match scores
# ---------------------------------------------------------------------------
panel_path = Path("apps/web/app/admin/match-log/MatchLogApplyPanel.tsx")
panel = panel_path.read_text(encoding="utf-8")

panel = replace_once(
    panel,
    '''  recentOperations?: AdminMatchEditOperation[];
  exclusionOperation: AdminMatchExclusionOperation | null;''',
    '''  recentOperations?: AdminMatchEditOperation[];
  initialSelectedIds?: string[];
  exclusionOperation: AdminMatchExclusionOperation | null;''',
    "bulk initial selection prop",
)

panel = replace_once(
    panel,
    '''type MatchEditState = {
  league: string;''',
    '''type BulkScoreEdit = {
  scoreT1: string;
  scoreT2: string;
};

type MatchEditState = {
  league: string;''',
    "bulk score edit type",
)

panel = replace_once(
    panel,
    '''function editStateFromMatch(match: AdminMatchLogMatch | null): MatchEditState {
  if (!match) return emptyEditState();''',
    '''function visibleMatchIds(matches: AdminMatchLogMatch[], requestedIds: string[]): string[] {
  const visible = new Set(matches.filter((match) => match.id != null).map((match) => String(match.id)));
  return Array.from(new Set(requestedIds.map((value) => String(value).trim()).filter((value) => value && visible.has(value)))).slice(0, 100);
}

function bulkScoreState(matches: AdminMatchLogMatch[], matchIds: string[]): Record<string, BulkScoreEdit> {
  const selected = new Set(matchIds);
  return Object.fromEntries(
    matches
      .filter((match) => match.id != null && selected.has(String(match.id)))
      .map((match) => [String(match.id), {
        scoreT1: String(match.score?.team1 ?? ""),
        scoreT2: String(match.score?.team2 ?? "")
      }])
  );
}

function editStateFromMatch(match: AdminMatchLogMatch | null): MatchEditState {
  if (!match) return emptyEditState();''',
    "bulk score helpers",
)

panel = replace_once(
    panel,
    '''  recentOperations = [],
  exclusionOperation,''',
    '''  recentOperations = [],
  initialSelectedIds = [],
  exclusionOperation,''',
    "bulk initial selection destructuring",
)

panel = replace_once(
    panel,
    '''  const [recoveryOperationId, setRecoveryOperationId] = useState<string | null>(() => unresolvedEditRecoveryId(recentOperations));
  const [bulkIds, setBulkIds] = useState<string[]>([]);
  const [bulkLeague, setBulkLeague] = useState("");''',
    '''  const [recoveryOperationId, setRecoveryOperationId] = useState<string | null>(() => unresolvedEditRecoveryId(recentOperations));
  const [bulkIds, setBulkIds] = useState<string[]>(() => visibleMatchIds(matches, initialSelectedIds));
  const [bulkScoreEdits, setBulkScoreEdits] = useState<Record<string, BulkScoreEdit>>(() => bulkScoreState(matches, visibleMatchIds(matches, initialSelectedIds)));
  const [bulkLeague, setBulkLeague] = useState("");''',
    "bulk initial state",
)

panel = replace_once(
    panel,
    '''  const selectedMatch = matches.find((match) => match.id != null && String(match.id) === selectedMatchId) || null;
  const visiblePlayerOptions = collectVisiblePlayers(matches);''',
    '''  const selectedMatch = matches.find((match) => match.id != null && String(match.id) === selectedMatchId) || null;
  const selectedBulkMatches = matches.filter((match) => match.id != null && bulkIds.includes(String(match.id)));
  const visiblePlayerOptions = collectVisiblePlayers(matches);''',
    "selected bulk matches",
)

panel = replace_once(
    panel,
    '''  function toggleBulkMatch(matchId: number) {
    const key = String(matchId);
    setBulkIds((current) => current.includes(key) ? current.filter((value) => value !== key) : current.length < 100 ? [...current, key] : current);
  }

  function stageBulkEdits() {''',
    '''  function setBulkSelection(nextIds: string[]) {
    const normalized = visibleMatchIds(matches, nextIds);
    setBulkIds(normalized);
    setBulkScoreEdits((current) => {
      const defaults = bulkScoreState(matches, normalized);
      const next: Record<string, BulkScoreEdit> = {};
      for (const id of normalized) {
        const value = current[id] || defaults[id];
        if (value) next[id] = value;
      }
      return next;
    });
  }

  function toggleBulkMatch(matchId: number) {
    const key = String(matchId);
    setBulkSelection(bulkIds.includes(key) ? bulkIds.filter((value) => value !== key) : [...bulkIds, key]);
  }

  function updateBulkScore(matchId: string, field: keyof BulkScoreEdit, value: string) {
    setBulkScoreEdits((current) => ({
      ...current,
      [matchId]: {
        ...(current[matchId] || { scoreT1: "", scoreT2: "" }),
        [field]: value
      }
    }));
  }

  function stageBulkEdits() {''',
    "bulk selection helpers",
)

panel = replace_once(
    panel,
    '''        if (bulkReplaceSlot) patch[bulkReplaceSlot] = integerInput(bulkReplacementPlayer, "Replacement player");
        return patch;''',
    '''        if (bulkReplaceSlot) patch[bulkReplaceSlot] = integerInput(bulkReplacementPlayer, "Replacement player");
        const scoreEdit = bulkScoreEdits[String(match.id)] || {
          scoreT1: String(match.score?.team1 ?? ""),
          scoreT2: String(match.score?.team2 ?? "")
        };
        const scoreT1 = integerInput(scoreEdit.scoreT1, `Match #${match.id} Team 1 score`);
        const scoreT2 = integerInput(scoreEdit.scoreT2, `Match #${match.id} Team 2 score`);
        if (scoreT1 < 0 || scoreT2 < 0) throw new Error(`Match #${match.id} scores must be non-negative.`);
        if (scoreT1 + scoreT2 <= 0) throw new Error(`Match #${match.id} needs a non-zero score.`);
        if (scoreT1 === scoreT2) throw new Error(`Match #${match.id} cannot have a tied score.`);
        if (scoreT1 !== Number(match.score?.team1 ?? 0)) patch.score_t1 = scoreT1;
        if (scoreT2 !== Number(match.score?.team2 ?? 0)) patch.score_t2 = scoreT2;
        return patch;''',
    "individual bulk score patches",
)

panel = replace_once(
    panel,
    '''        <p style={{ color: "#475569" }}>Select up to 100 rows, then set shared fields, clear notes/week tags, shift dates, or replace one player slot. Nothing is written until the staged operation is confirmed below.</p>
        <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <button type="button" onClick={() => setBulkIds(matches.filter((match) => match.id != null).slice(0, 100).map((match) => String(match.id)))} style={secondaryButtonStyle}>Select first 100 visible</button>
          <button type="button" onClick={() => setBulkIds([])} disabled={!bulkIds.length} style={secondaryButtonStyle}>Clear selection</button>''',
    '''        <p style={{ color: "#475569" }}>Select up to 100 rows. Each selected score can be corrected independently, and shared metadata changes can be added to the same staged operation. Nothing is written until confirmation.</p>
        <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <button type="button" onClick={() => setBulkSelection(matches.filter((match) => match.id != null).slice(0, 100).map((match) => String(match.id)))} style={secondaryButtonStyle}>Select first 100 visible</button>
          <button type="button" onClick={() => setBulkSelection([])} disabled={!bulkIds.length} style={secondaryButtonStyle}>Clear selection</button>''',
    "bulk selection controls",
)

panel = replace_once(
    panel,
    '''        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>''',
    '''        </div>
        {selectedBulkMatches.length ? (
          <div style={{ marginTop: "0.85rem" }}>
            <h4 style={{ marginBottom: "0.5rem" }}>Individual score corrections</h4>
            <p style={{ color: "#475569", marginTop: 0 }}>Review every selected match. Change only the scores that need correction.</p>
            <div style={{ display: "grid", gap: "0.6rem" }}>
              {selectedBulkMatches.map((match) => {
                const scoreEdit = bulkScoreEdits[String(match.id)] || {
                  scoreT1: String(match.score?.team1 ?? ""),
                  scoreT2: String(match.score?.team2 ?? "")
                };
                return (
                  <div key={`bulk-score-${match.id}`} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.6rem", alignItems: "end", border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.65rem", background: "white" }}>
                    <div>
                      <strong>Match #{match.id}</strong>
                      <div style={{ color: "#475569", marginTop: "0.25rem" }}>{playerNames(match.team1)} vs {playerNames(match.team2)}</div>
                    </div>
                    <label><strong>Team 1 score</strong><br /><input type="number" min="0" step="1" value={scoreEdit.scoreT1} onChange={(event) => updateBulkScore(String(match.id), "scoreT1", event.target.value)} style={inputStyle} /></label>
                    <label><strong>Team 2 score</strong><br /><input type="number" min="0" step="1" value={scoreEdit.scoreT2} onChange={(event) => updateBulkScore(String(match.id), "scoreT2", event.target.value)} style={inputStyle} /></label>
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>''',
    "individual bulk score UI",
)

panel_path.write_text(panel, encoding="utf-8")

# ---------------------------------------------------------------------------
# API route and domain multi-ID filters
# ---------------------------------------------------------------------------
routes_path = Path("services/api/admin_match_log_routes.py")
routes = routes_path.read_text(encoding="utf-8")

routes = replace_once(
    routes,
    '''        match_id: int | None = Query(default=None),
        league: str | None = Query(default=None),''',
    '''        match_id: int | None = Query(default=None),
        match_ids: str | None = Query(default=None, max_length=4000),
        league: str | None = Query(default=None),''',
    "match log route multiple IDs",
)

routes = replace_once(
    routes,
    '''                match_id=match_id,
                league=league,''',
    '''                match_id=match_id,
                match_ids=match_ids,
                league=league,''',
    "match log route service call",
)

routes_path.write_text(routes, encoding="utf-8")

service_path = Path("jupr_app/services/admin_match_log_service.py")
service = service_path.read_text(encoding="utf-8")

service = replace_once(
    service,
    '''import os
from collections import Counter''',
    '''import os
import re
from collections import Counter''',
    "match log regex import",
)

service = replace_once(
    service,
    '''MAX_CONTEXT_IDS = 200
MAX_CONTEXT_ID_LENGTH = 200''',
    '''MAX_CONTEXT_IDS = 200
MAX_CONTEXT_ID_LENGTH = 200
MAX_MATCH_IDS = 100''',
    "match ID limit",
)

service = replace_once(
    service,
    '''def _normalize_context_ids(
    *,''',
    '''def _normalize_match_ids(
    *,
    match_id: int | None = None,
    match_ids: str | list[int | str] | tuple[int | str, ...] | None = None,
) -> list[int]:
    raw_values: list[Any] = []
    if match_id not in (None, ""):
        raw_values.append(match_id)
    if isinstance(match_ids, str):
        raw_values.extend(re.split(r"[\\s,;]+", match_ids))
    elif match_ids:
        for value in match_ids:
            raw_values.extend(re.split(r"[\\s,;]+", str(value or "")))

    normalized: list[int] = []
    seen: set[int] = set()
    for value in raw_values:
        token = str(value or "").strip()
        if token.startswith("#"):
            token = token[1:].strip()
        if not token:
            continue
        if not re.fullmatch(r"\\d+", token):
            raise ValueError("Match IDs must be positive whole numbers separated by commas or spaces.")
        parsed = int(token)
        if parsed < 1:
            raise ValueError("Match IDs must be positive whole numbers separated by commas or spaces.")
        if parsed in seen:
            continue
        seen.add(parsed)
        normalized.append(parsed)
    if len(normalized) > MAX_MATCH_IDS:
        raise ValueError(f"No more than {MAX_MATCH_IDS} match IDs may be loaded at once.")
    return normalized


def _normalize_context_ids(
    *,''',
    "match ID normalizer",
)

service = replace_once(
    service,
    '''    match_id: int | None = None,
    context_type: str | None = None,''',
    '''    match_ids: list[int] | None = None,
    context_type: str | None = None,''',
    "fetch multiple ID signature",
)

service = replace_once(
    service,
    '''    requested_context_type = str(context_type or "").strip().casefold() or None
    requested_context_ids = _normalize_context_ids(context_ids=context_ids)''',
    '''    requested_match_ids = _normalize_match_ids(match_ids=match_ids)
    requested_context_type = str(context_type or "").strip().casefold() or None
    requested_context_ids = _normalize_context_ids(context_ids=context_ids)''',
    "fetch requested IDs",
)

service = replace_once(
    service,
    '''        if match_id is not None:
            query = query.eq("id", int(match_id))''',
    '''        if len(requested_match_ids) == 1:
            query = query.eq("id", requested_match_ids[0])
        elif requested_match_ids:
            query = query.in_("id", requested_match_ids)''',
    "fetch ID pushdown",
)

service = replace_once(
    service,
    '''    match_id: int | None,
    league: str | None,''',
    '''    match_ids: list[int],
    league: str | None,''',
    "filter multiple ID signature",
)

service = replace_once(
    service,
    '''    if match_id is not None:
        result = [row for row in result if _safe_int(row.get("id")) == int(match_id)]''',
    '''    if match_ids:
        expected_match_ids = set(match_ids)
        result = [row for row in result if _safe_int(row.get("id")) in expected_match_ids]''',
    "filter multiple IDs",
)

service = replace_once(
    service,
    '''    match_id: int | None = None,
    league: str | None = None,''',
    '''    match_id: int | None = None,
    match_ids: str | list[int | str] | tuple[int | str, ...] | None = None,
    league: str | None = None,''',
    "build multiple IDs signature",
)

service = replace_once(
    service,
    '''    safe_limit = max(1, min(int(limit or 500), MAX_RETURN_ROWS))
    requested_context_ids = _normalize_context_ids(''',
    '''    safe_limit = max(1, min(int(limit or 500), MAX_RETURN_ROWS))
    requested_match_ids = _normalize_match_ids(match_id=match_id, match_ids=match_ids)
    requested_context_ids = _normalize_context_ids(''',
    "build requested IDs",
)

service = replace_once(
    service,
    '''        "filter": filter_type or "All",
        "match_id": match_id,
        "league": league or None,''',
    '''        "filter": filter_type or "All",
        "match_id": requested_match_ids[0] if len(requested_match_ids) == 1 else None,
        "match_ids": requested_match_ids,
        "league": league or None,''',
    "response multiple ID filters",
)

service = replace_once(
    service,
    '''        match_id=match_id,
        context_type=context_type,''',
    '''        match_ids=requested_match_ids,
        context_type=context_type,''',
    "fetch call multiple IDs",
)

service = replace_once(
    service,
    '''        match_id=match_id,
        league=league,''',
    '''        match_ids=requested_match_ids,
        league=league,''',
    "filter call multiple IDs",
)

service_path.write_text(service, encoding="utf-8")

# ---------------------------------------------------------------------------
# Focused source and behavior tests
# ---------------------------------------------------------------------------
manual_test_path = Path("tests/test_manual_acceptance_ux_regressions.py")
manual_test = manual_test_path.read_text(encoding="utf-8")
manual_test = replace_once(
    manual_test,
    '''    assert '[newMatchRow(todayIsoDate(), status.week_tag_options[0] || "Week 1")]' in source''',
    '''    assert '[newMatchRow(todayIsoDate(), initialWeekTag, "", initialLeague)]' in source''',
    "current manual row initialization contract",
)
manual_test_path.write_text(manual_test, encoding="utf-8")

layout_path = Path("apps/web/tests/match-uploader-layout.cjs")
layout = layout_path.read_text(encoding="utf-8")

layout = replace_once(
    layout,
    '''const form = fs.readFileSync(path.join(routeRoot, "MatchUploaderForm.tsx"), "utf8");''',
    '''const form = fs.readFileSync(path.join(routeRoot, "MatchUploaderForm.tsx"), "utf8");
const matchLogRoot = path.join(webRoot, "app", "admin", "match-log");
const matchLogWorkspace = fs.readFileSync(path.join(matchLogRoot, "MatchLogWorkspace.tsx"), "utf8");
const matchLogPanel = fs.readFileSync(path.join(matchLogRoot, "MatchLogApplyPanel.tsx"), "utf8");
const matchLogApi = fs.readFileSync(path.join(webRoot, "lib", "adminMatchLogApi.ts"), "utf8");''',
    "layout source fixtures",
)

layout = replace_once(
    layout,
    '''assert.match(form, /Edit results/, "submission result modal must label the match correction path as Edit results");

console.log("Match Uploader context and responsive layout checks passed.");''',
    '''assert.match(form, /Edit results/, "submission result modal must label the match correction path as Edit results");
assert.match(form, /submissionKind === "round_robin"/, "round-robin results must use a bulk edit handoff");
assert.match(form, /Choose matches to edit/, "multi-match manual uploads must prompt for the matches to edit");
assert.match(form, /\\/admin\\/match-log\\/bulk/, "multi-match results must route to Bulk edit");
assert.match(form, /params\\.set\\("match_ids"/, "bulk handoff must filter all created match IDs");
assert.match(form, /params\\.set\\("selected_ids"/, "bulk handoff must preselect created match IDs");
assert.match(matchLogWorkspace, /match_ids\\?: string/, "Match Log search params must support multiple IDs");
assert.match(matchLogWorkspace, /name="match_ids"/, "Match Log filter must accept multiple IDs");
assert.match(matchLogWorkspace, /initialSelectedIds=\\{initialSelectedIds\\}/, "Bulk edit must receive the uploaded match selection");
assert.match(matchLogPanel, /initialSelectedIds\\?: string\\[\\]/, "Bulk editor must accept initial selected IDs");
assert.match(matchLogPanel, /Individual score corrections/, "Bulk editor must expose per-match score controls");
assert.match(matchLogPanel, /bulkScoreEdits/, "Bulk editor must stage independent score changes");
assert.match(matchLogApi, /matchIds\\?: string \\| Array<string \\| number>/, "Match Log client must send multiple IDs");

console.log("Match Uploader context and responsive layout checks passed.");''',
    "bulk handoff source tests",
)

layout_path.write_text(layout, encoding="utf-8")

service_test_path = Path("tests/test_admin_match_log_service.py")
service_test = service_test_path.read_text(encoding="utf-8")
service_test += '''


def test_admin_match_log_filters_multiple_match_ids_before_limit(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")
    tables = fake_tables()
    template = dict(tables["matches"][0])
    for index in range(6):
        tables["matches"].append(
            {
                **template,
                "id": 100 + index,
                "date": f"2026-04-{index + 1:02d}T10:00:00Z",
            }
        )

    payload = build_admin_match_log(
        FakeSupabase(tables, strict_select_tables=SCHEMA_STRICT_TABLES),
        club_id="club",
        match_ids="1, #3, 1",
        limit=2,
    )

    assert payload["filters"]["match_id"] is None
    assert payload["filters"]["match_ids"] == [1, 3]
    assert payload["summary"]["scanned_matches"] == 2
    assert [match["id"] for match in payload["matches"]] == [3, 1]


def test_admin_match_log_rejects_invalid_multiple_match_ids(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")

    try:
        build_admin_match_log(
            fake_supabase(),
            club_id="club",
            match_ids="1, not-a-match",
        )
    except ValueError as exc:
        assert "positive whole numbers" in str(exc)
    else:
        raise AssertionError("Expected invalid match IDs to be rejected")
'''
service_test_path.write_text(service_test, encoding="utf-8")

api_test_path = Path("tests/test_api_contract_admin_match_log.py")
api_test = api_test_path.read_text(encoding="utf-8")
api_test += '''


def test_admin_match_log_multiple_match_ids_contract(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")
    supabase = fake_supabase()
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).get(
        "/admin/clubs/club/match-log?match_ids=1,3,1&limit=20",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filters"]["match_id"] is None
    assert payload["filters"]["match_ids"] == [1, 3]
    assert [match["id"] for match in payload["matches"]] == [3, 1]
'''
api_test_path.write_text(api_test, encoding="utf-8")
