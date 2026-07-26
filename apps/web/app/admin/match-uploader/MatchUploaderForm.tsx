"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { PublicPlayer } from "@/lib/api";
import type {
  AdminMatchUploaderCreatePlayersResult,
  AdminMatchUploaderRoundRobinPreview,
  AdminMatchUploaderStatusResponse,
  AdminMatchUploaderWriteResult
} from "@/lib/adminMatchUploaderApi";
import {
  clearDirectMatchIdempotencyKey,
  directMatchIdempotencyKey
} from "@/lib/directMatchIdempotency";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  players: PublicPlayer[];
  status: AdminMatchUploaderStatusResponse;
};

type MatchRow = {
  rowId: string;
  date: string;
  weekTag: string;
  ratingScope: "" | "overall_only" | "unrated";
  t1p1: string;
  t1p2: string;
  t2p1: string;
  t2p2: string;
  s1: string;
  s2: string;
};

type SinglesRow = {
  date: string;
  league: string;
  weekTag: string;
  playerA: string;
  playerB: string;
  scoreA: string;
  scoreB: string;
  ratingScope: "" | "unrated";
};

type RrCourtInput = { rowId: string; formatType: string; namesText: string };
type RrScoreRow = {
  rowId: string;
  court: number;
  label: string;
  t1: Array<{ id: number; name: string }>;
  t2: Array<{ id: number; name: string }>;
  t1p1: number;
  t1p2: number;
  t2p1: number;
  t2p2: number;
  s1: string;
  s2: string;
};
type RrCourtSchedule = { court: number; formatType: string; expectedGames?: number | null; matches: RrScoreRow[] };
type RrPayload = { source: string; custom_schedule: string; schedule_mode: string; courts: Array<{ court: number; format_type: string; player_names: string[] }> };
type NewPlayerDraft = { name: string; startingJupr: string };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function todayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function randomId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function newMatchRow(): MatchRow {
  return { rowId: randomId("row"), date: todayIsoDate(), weekTag: "Week 1", ratingScope: "", t1p1: "", t1p2: "", t2p1: "", t2p2: "", s1: "0", s2: "0" };
}

function newSinglesRow(): SinglesRow {
  return { date: todayIsoDate(), league: "Singles", weekTag: "Singles", playerA: "", playerB: "", scoreA: "0", scoreB: "0", ratingScope: "" };
}

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function splitNames(value: string): string[] {
  const seen = new Set<string>();
  const names: string[] = [];
  for (const raw of value.replaceAll(String.fromCharCode(10), ",").split(",")) {
    const name = raw.replace(/\s+/g, " ").trim();
    if (name && !seen.has(name)) {
      seen.add(name);
      names.push(name);
    }
  }
  return names;
}

function ratingLabel(value?: number | null): string {
  return value == null ? "—" : Math.round(Number(value)).toString();
}

function deltaLabel(value?: number | null): string {
  if (value == null) return "—";
  const rounded = Math.round(Number(value));
  return `${rounded >= 0 ? "+" : ""}${rounded}`;
}

function isFilled(row: MatchRow): boolean {
  return Boolean(row.t1p1 || row.t1p2 || row.t2p1 || row.t2p2 || Number(row.s1 || 0) + Number(row.s2 || 0) > 0);
}

function validateRow(row: MatchRow, index: number): string | null {
  if (!isFilled(row)) return null;
  const ids = [row.t1p1, row.t1p2, row.t2p1, row.t2p2].filter(Boolean);
  if (ids.length !== 4) return `Row ${index + 1}: select four players.`;
  if (new Set(ids).size !== 4) return `Row ${index + 1}: select four different players.`;
  const s1 = Number(row.s1 || 0);
  const s2 = Number(row.s2 || 0);
  if (!Number.isFinite(s1) || !Number.isFinite(s2) || s1 < 0 || s2 < 0) return `Row ${index + 1}: scores must be non-negative numbers.`;
  if (s1 + s2 <= 0) return `Row ${index + 1}: enter a non-zero score.`;
  return null;
}

function validateSingles(row: SinglesRow): string | null {
  if (!row.playerA || !row.playerB) return "Select two singles players.";
  if (row.playerA === row.playerB) return "Singles players must be different.";
  const s1 = Number(row.scoreA || 0);
  const s2 = Number(row.scoreB || 0);
  if (!Number.isFinite(s1) || !Number.isFinite(s2) || s1 < 0 || s2 < 0) return "Singles scores must be non-negative numbers.";
  if (s1 + s2 <= 0) return "Enter a non-zero singles score.";
  if (s1 === s2) return "Singles scores cannot be tied.";
  return null;
}

function mergePlayers(current: PublicPlayer[], incoming: NonNullable<AdminMatchUploaderCreatePlayersResult["players"]>): PublicPlayer[] {
  const byId = new Map<string, PublicPlayer>();
  for (const player of current) byId.set(String(player.id), player);
  for (const player of incoming) byId.set(String(player.id), player as PublicPlayer);
  return Array.from(byId.values()).sort((left, right) => String(left.name).localeCompare(String(right.name)));
}

function previewToSchedule(preview: AdminMatchUploaderRoundRobinPreview): RrCourtSchedule[] {
  return (preview.courts || []).map((court) => ({
    court: court.court,
    formatType: court.format_type,
    expectedGames: court.expected_games,
    matches: (court.matches || []).map((match) => ({
      rowId: match.row_id || randomId("rr"),
      court: match.court,
      label: match.label,
      t1: match.t1,
      t2: match.t2,
      t1p1: match.t1_p1,
      t1p2: match.t1_p2,
      t2p1: match.t2_p1,
      t2p2: match.t2_p2,
      s1: "0",
      s2: "0"
    }))
  }));
}

export default function MatchUploaderForm({ apiBase, clubId, players, status }: Props) {
  const firstFormat = status.round_robin_format_options?.[0] || "4-Player";
  const singlesEnabled = Boolean(status.singles_write_enabled && status.singles_submit_endpoint);
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [knownPlayers, setKnownPlayers] = useState<PublicPlayer[]>(players);
  const [entryMethod, setEntryMethod] = useState<"singles" | "manual" | "round_robin">(
    () => singlesEnabled ? "singles" : "manual"
  );
  const [context, setContext] = useState<"league" | "popup">("league");
  const [defaultDate, setDefaultDate] = useState(todayIsoDate());
  const [defaultLeague, setDefaultLeague] = useState(status.league_options[0] || "Open");
  const [defaultWeekTag, setDefaultWeekTag] = useState(status.week_tag_options[0] || "Week 1");
  const [popupEventName, setPopupEventName] = useState("Saturday Social");
  const [singlesRow, setSinglesRow] = useState<SinglesRow>(() => newSinglesRow());
  const [rows, setRows] = useState<MatchRow[]>(() => Array.from({ length: 5 }, () => newMatchRow()));
  const [rrCourts, setRrCourts] = useState<RrCourtInput[]>(() => [{ rowId: randomId("court"), formatType: firstFormat, namesText: "" }]);
  const [rrCustomSchedule, setRrCustomSchedule] = useState("");
  const [rrSchedule, setRrSchedule] = useState<RrCourtSchedule[]>([]);
  const [rrPendingPayload, setRrPendingPayload] = useState<RrPayload | null>(null);
  const [newPlayerDrafts, setNewPlayerDrafts] = useState<NewPlayerDraft[]>([]);
  const [saving, setSaving] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [creatingPlayers, setCreatingPlayers] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AdminMatchUploaderWriteResult | null>(null);

  const playerOptions = useMemo(() => knownPlayers.map((player) => <option key={String(player.id)} value={String(player.id)}>{player.name}</option>), [knownPlayers]);
  const validRows = rows.filter(isFilled);
  const scoredRrRows = rrSchedule.flatMap((court) => court.matches).filter((match) => Number(match.s1 || 0) + Number(match.s2 || 0) > 0);
  const league = context === "popup" ? "POPUP" : defaultLeague;
  const matchType = context === "popup" ? "PopUp" : "Live Match";

  function requireReady(): boolean {
    if (!apiBase) {
      setMessage("API base URL is not configured.");
      return false;
    }
    if (!accessToken) {
      setMessage("Sign in at /admin/login before using Match Uploader.");
      return false;
    }
    if (!status.enabled) {
      setMessage("Next Match Uploader is disabled on the API.");
      return false;
    }
    return true;
  }

  function patchRow(rowId: string, patch: Partial<MatchRow>) {
    setRows((current) => current.map((row) => row.rowId === rowId ? { ...row, ...patch } : row));
  }

  function patchSingles(patch: Partial<SinglesRow>) {
    setSinglesRow((current) => ({ ...current, ...patch }));
  }

  function patchRrCourt(rowId: string, patch: Partial<RrCourtInput>) {
    setRrCourts((current) => current.map((court) => court.rowId === rowId ? { ...court, ...patch } : court));
  }

  function patchRrScore(rowId: string, patch: Partial<Pick<RrScoreRow, "s1" | "s2">>) {
    setRrSchedule((current) => current.map((court) => ({
      ...court,
      matches: court.matches.map((match) => match.rowId === rowId ? { ...match, ...patch } : match)
    })));
  }

  function buildRoundRobinPayload(): RrPayload | null {
    const courts = rrCourts.map((court, index) => ({ court: index + 1, format_type: court.formatType, player_names: splitNames(court.namesText) }));
    const empty = courts.find((court) => court.player_names.length === 0);
    if (empty) {
      setMessage(`Court ${empty.court}: enter player names before generating a schedule.`);
      return null;
    }
    return { source: "next_match_uploader_round_robin_preview", custom_schedule: rrCustomSchedule, schedule_mode: "full", courts };
  }

  async function postJson<T>(path: string, body: unknown): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Match Uploader.");
    const response = await fetch(apiUrl(apiBase, path), {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}` },
      body: JSON.stringify(body)
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function submitSinglesMatch() {
    setMessage(null);
    setResult(null);
    if (!requireReady()) return;
    if (!singlesEnabled) {
      setMessage("Direct singles submission is unavailable until its transactional write and replay path is complete.");
      return;
    }
    const error = validateSingles(singlesRow);
    if (error) {
      setMessage(error);
      return;
    }
    setSaving(true);
    const request = {
      source: "next_match_uploader_singles",
      date: singlesRow.date,
      league: singlesRow.league || "Singles",
      week_tag: singlesRow.weekTag || "Singles",
      t1_p1: Number(singlesRow.playerA),
      t2_p1: Number(singlesRow.playerB),
      score_t1: Number(singlesRow.scoreA),
      score_t2: Number(singlesRow.scoreB),
      rating_scope: singlesRow.ratingScope || undefined
    };
    const idempotencyKey = directMatchIdempotencyKey(
      `match-uploader:${clubId}:singles`,
      request
    );
    try {
      const payload = await postJson<AdminMatchUploaderWriteResult>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/singles`, {
        ...request,
        idempotency_key: idempotencyKey
      });
      setResult(payload);
      clearDirectMatchIdempotencyKey(
        `match-uploader:${clubId}:singles`,
        idempotencyKey
      );
      setMessage(`Submitted singles match; inserted ${payload.result?.inserted ?? 0} rated singles match.`);
      setSinglesRow((current) => ({ ...newSinglesRow(), date: current.date, league: current.league, weekTag: current.weekTag }));
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : "Unable to submit singles match."} Retry this unchanged form; duplicate protection is active.`);
    } finally {
      setSaving(false);
    }
  }

  async function previewRoundRobin(payload: RrPayload) {
    setGenerating(true);
    setResult(null);
    try {
      const preview = await postJson<AdminMatchUploaderRoundRobinPreview>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/round-robin/preview`, payload);
      if (preview.missing_players?.length) {
        setRrSchedule([]);
        setRrPendingPayload(payload);
        setNewPlayerDrafts(preview.missing_players.map((name) => ({ name, startingJupr: "3.5" })));
        setMessage(`Found ${preview.missing_players.length} new player(s). Create profiles to continue.`);
        return;
      }
      setRrPendingPayload(null);
      setNewPlayerDrafts([]);
      setRrSchedule(previewToSchedule(preview));
      setMessage(`Generated ${preview.match_count ?? 0} round-robin game(s). Enter non-zero scores, then submit scored games.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to generate round-robin schedule.");
    } finally {
      setGenerating(false);
    }
  }

  async function generateRoundRobin() {
    setMessage(null);
    setResult(null);
    if (!requireReady()) return;
    const payload = buildRoundRobinPayload();
    if (payload) await previewRoundRobin(payload);
  }

  async function createPlayersAndContinue() {
    setMessage(null);
    setResult(null);
    if (!requireReady() || !rrPendingPayload) return;
    const playersToCreate = newPlayerDrafts.map((draft) => ({ name: draft.name.trim(), starting_jupr: Number(draft.startingJupr) }));
    const invalid = playersToCreate.find((player) => !player.name || !Number.isFinite(player.starting_jupr) || player.starting_jupr < 1 || player.starting_jupr > 7);
    if (invalid) {
      setMessage("Each new player needs a name and a Starting JUPR between 1.0 and 7.0.");
      return;
    }
    setCreatingPlayers(true);
    try {
      const payload = await postJson<AdminMatchUploaderCreatePlayersResult>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/players`, { source: "next_match_uploader_new_players", players: playersToCreate });
      if (payload.players?.length) setKnownPlayers((current) => mergePlayers(current, payload.players || []));
      setMessage(`Created or confirmed ${payload.accepted_count ?? playersToCreate.length} player profile(s). Regenerating schedule…`);
      await previewRoundRobin(rrPendingPayload);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to create players.");
    } finally {
      setCreatingPlayers(false);
    }
  }

  async function submitMatches(matches: Array<Record<string, unknown>>, source: string) {
    setSaving(true);
    const request = { source, matches };
    const operationScope = `match-uploader:${clubId}:batch`;
    const idempotencyKey = directMatchIdempotencyKey(
      operationScope,
      request
    );
    try {
      const payload = await postJson<AdminMatchUploaderWriteResult>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/batch`, {
        ...request,
        idempotency_key: idempotencyKey
      });
      setResult(payload);
      clearDirectMatchIdempotencyKey(operationScope, idempotencyKey);
      const handoff = payload.auto_player_updates;
      const handoffSummary = handoff?.mode === "auto_sent"
        ? ` Player-update email: ${handoff.sent ?? 0} sent, ${handoff.skipped ?? 0} skipped, ${handoff.errors ?? 0} error(s).`
        : handoff?.mode
          ? ` Player-update email: ${handoff.mode}${handoff.reason ? ` — ${handoff.reason}` : ""}.`
          : "";
      setMessage(`Submitted ${payload.submitted_count ?? matches.length} row(s); inserted ${payload.result?.inserted ?? 0} rated match(es).${handoffSummary}`);
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : "Unable to submit matches."} Retry this unchanged batch; duplicate protection is active.`);
    } finally {
      setSaving(false);
    }
  }

  async function submitManualBatch() {
    setMessage(null);
    setResult(null);
    if (!requireReady()) return;
    const errors = rows.map(validateRow).filter(Boolean) as string[];
    if (errors.length) {
      setMessage(errors[0]);
      return;
    }
    const matches = validRows.map((row) => ({
      date: row.date,
      league,
      week_tag: row.weekTag,
      match_type: matchType,
      rating_scope: row.ratingScope || undefined,
      is_popup: context === "popup",
      context_type: context === "popup" ? "event" : null,
      context_name: context === "popup" ? popupEventName : undefined,
      t1_p1: Number(row.t1p1),
      t1_p2: Number(row.t1p2),
      t2_p1: Number(row.t2p1),
      t2_p2: Number(row.t2p2),
      score_t1: Number(row.s1),
      score_t2: Number(row.s2)
    }));
    if (!matches.length) {
      setMessage("Enter at least one complete match row.");
      return;
    }
    await submitMatches(matches, "next_match_uploader_manual_batch");
  }

  async function submitRoundRobinScores() {
    setMessage(null);
    setResult(null);
    if (!requireReady()) return;
    const matches = scoredRrRows.map((row) => ({
      date: defaultDate,
      league,
      week_tag: defaultWeekTag,
      match_type: matchType,
      is_popup: context === "popup",
      context_type: context === "popup" ? "event" : null,
      context_name: context === "popup" ? popupEventName : undefined,
      t1_p1: row.t1p1,
      t1_p2: row.t1p2,
      t2_p1: row.t2p1,
      t2_p2: row.t2p2,
      score_t1: Number(row.s1),
      score_t2: Number(row.s2)
    }));
    if (!matches.length) {
      setMessage("Enter at least one non-zero round-robin score before submitting.");
      return;
    }
    await submitMatches(matches, "next_match_uploader_round_robin");
  }

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Next Match Uploader is disabled</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the Match Uploader pilot flag on FastAPI."}</p>
      </article>
    );
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Match entry setup</h2>
        <p style={{ color: "#475569" }}>
          Doubles manual/batch and round-robin entry use the existing doubles/overall rating path.
          {singlesEnabled ? " Singles input writes to the separate singles rating." : " Direct singles entry remains unavailable until its write and replay path is transaction-safe."}
        </p>
        {status.warnings?.length ? (
          <ul style={{ color: "#92400e" }}>
            {status.warnings.map((warning) => <li key={warning}>{warning}</li>)}
          </ul>
        ) : null}
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "0.75rem" }}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
            {accessToken ? "Ready to send authorized Match Uploader requests." : sessionLoading ? "Checking admin session…" : "Sign in before generating schedules, creating players, or submitting matches."}
          </p>
          {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
          {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
          <label><strong>Entry method</strong><br /><select value={entryMethod} onChange={(event) => setEntryMethod(event.target.value as "singles" | "manual" | "round_robin")} style={inputStyle}>{singlesEnabled ? <option value="singles">Singles match</option> : null}<option value="manual">Doubles manual / batch</option><option value="round_robin">Doubles round robin</option></select></label>
          {entryMethod !== "singles" ? <label><strong>Context</strong><br /><select value={context} onChange={(event) => setContext(event.target.value as "league" | "popup")} style={inputStyle}><option value="league">Official League</option><option value="popup">Pop-Up / Social</option></select></label> : null}
          {entryMethod !== "singles" ? <label><strong>Default date</strong><br /><input value={defaultDate} onChange={(event) => setDefaultDate(event.target.value)} type="date" style={inputStyle} /></label> : null}
          {entryMethod !== "singles" ? <label><strong>Default league</strong><br /><select value={defaultLeague} onChange={(event) => setDefaultLeague(event.target.value)} disabled={context === "popup"} style={inputStyle}>{status.league_options.map((item) => <option key={item}>{item}</option>)}</select></label> : null}
          {entryMethod !== "singles" ? <label><strong>Default week/session</strong><br /><select value={defaultWeekTag} onChange={(event) => setDefaultWeekTag(event.target.value)} style={inputStyle}>{status.week_tag_options.map((item) => <option key={item}>{item}</option>)}</select></label> : null}
          {entryMethod !== "singles" && context === "popup" ? <label><strong>Pop-Up event name</strong><br /><input value={popupEventName} onChange={(event) => setPopupEventName(event.target.value)} style={inputStyle} /></label> : null}
        </div>
      </article>

      {entryMethod === "singles" && singlesEnabled ? (
        <article style={{ ...cardStyle, background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Singles match input</h2>
          <p style={{ color: "#475569" }}>Use this for one-on-one games only. This updates each player’s singles rating and writes an official match row with <code>match_format=singles</code>.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem" }}>
            <label><strong>Date</strong><br /><input type="date" value={singlesRow.date} onChange={(event) => patchSingles({ date: event.target.value })} style={inputStyle} /></label>
            <label><strong>Singles league/tag</strong><br /><input value={singlesRow.league} onChange={(event) => patchSingles({ league: event.target.value })} style={inputStyle} /></label>
            <label><strong>Session</strong><br /><input value={singlesRow.weekTag} onChange={(event) => patchSingles({ weekTag: event.target.value })} style={inputStyle} /></label>
            <label><strong>Rating scope</strong><br /><select value={singlesRow.ratingScope} onChange={(event) => patchSingles({ ratingScope: event.target.value as SinglesRow["ratingScope"] })} style={inputStyle}><option value="">Rated singles</option><option value="unrated">Unrated / record only</option></select></label>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(180px, 1fr) 90px 90px minmax(180px, 1fr)", gap: "0.75rem", alignItems: "end", marginTop: "0.75rem" }}>
            <label><strong>Player A</strong><br /><select value={singlesRow.playerA} onChange={(event) => patchSingles({ playerA: event.target.value })} style={inputStyle}><option value="">Choose player…</option>{playerOptions}</select></label>
            <label><strong>Score A</strong><br /><input type="number" min={0} max={99} value={singlesRow.scoreA} onChange={(event) => patchSingles({ scoreA: event.target.value })} style={inputStyle} /></label>
            <label><strong>Score B</strong><br /><input type="number" min={0} max={99} value={singlesRow.scoreB} onChange={(event) => patchSingles({ scoreB: event.target.value })} style={inputStyle} /></label>
            <label><strong>Player B</strong><br /><select value={singlesRow.playerB} onChange={(event) => patchSingles({ playerB: event.target.value })} style={inputStyle}><option value="">Choose player…</option>{playerOptions}</select></label>
          </div>
          <p><button type="button" onClick={submitSinglesMatch} disabled={saving || !accessToken || !singlesEnabled} style={buttonStyle}>{saving ? "Submitting…" : "Submit singles match"}</button></p>
        </article>
      ) : null}

      {entryMethod === "manual" ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Doubles manual / batch score entry</h2>
          <p style={{ color: "#475569" }}>Empty rows are ignored. Submitted rows must have four distinct players and a non-zero score.</p>
          <p><button type="button" onClick={() => setRows((current) => [...current, ...Array.from({ length: 5 }, () => newMatchRow())])} disabled={rows.length >= status.max_batch_rows} style={ghostButtonStyle}>Add 5 rows</button></p>
          <div style={{ display: "grid", gap: "0.75rem" }}>
            {rows.map((row, index) => (
              <div key={row.rowId} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: isFilled(row) ? "#f8fafc" : "white" }}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem", alignItems: "center", marginBottom: "0.5rem" }}><strong>Match {index + 1}</strong><button type="button" onClick={() => setRows((current) => current.filter((item) => item.rowId !== row.rowId))} disabled={rows.length <= 1}>Remove</button></div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.5rem" }}>
                  <input type="date" value={row.date} onChange={(event) => patchRow(row.rowId, { date: event.target.value })} style={inputStyle} />
                  <input value={league} disabled placeholder="League" style={inputStyle} />
                  <input value={row.weekTag} onChange={(event) => patchRow(row.rowId, { weekTag: event.target.value })} placeholder="Week" style={inputStyle} />
                  <input value={matchType} disabled placeholder="Match type" style={inputStyle} />
                  <select value={row.ratingScope} onChange={(event) => patchRow(row.rowId, { ratingScope: event.target.value as MatchRow["ratingScope"] })} style={inputStyle}><option value="">Overall + league</option><option value="overall_only">Overall only</option><option value="unrated">Unrated / record only</option></select>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.5rem", marginTop: "0.5rem" }}>
                  <select value={row.t1p1} onChange={(event) => patchRow(row.rowId, { t1p1: event.target.value })} style={inputStyle}><option value="">T1 P1</option>{playerOptions}</select>
                  <select value={row.t1p2} onChange={(event) => patchRow(row.rowId, { t1p2: event.target.value })} style={inputStyle}><option value="">T1 P2</option>{playerOptions}</select>
                  <input value={row.s1} onChange={(event) => patchRow(row.rowId, { s1: event.target.value })} type="number" min={0} max={99} style={inputStyle} />
                  <input value={row.s2} onChange={(event) => patchRow(row.rowId, { s2: event.target.value })} type="number" min={0} max={99} style={inputStyle} />
                  <select value={row.t2p1} onChange={(event) => patchRow(row.rowId, { t2p1: event.target.value })} style={inputStyle}><option value="">T2 P1</option>{playerOptions}</select>
                  <select value={row.t2p2} onChange={(event) => patchRow(row.rowId, { t2p2: event.target.value })} style={inputStyle}><option value="">T2 P2</option>{playerOptions}</select>
                </div>
              </div>
            ))}
          </div>
          <p><strong>Ready rows:</strong> {validRows.length} / {rows.length}</p>
          <button type="button" onClick={submitManualBatch} disabled={saving || !accessToken || !validRows.length} style={buttonStyle}>{saving ? "Submitting…" : "Submit batch"}</button>
        </article>
      ) : null}

      {entryMethod === "round_robin" ? (
        <>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Doubles round-robin generator</h2>
            <p style={{ color: "#475569" }}>Enter one court per group. The API checks names, creates missing players when requested, then returns the Python-generated schedule for score entry.</p>
            <div style={{ display: "grid", gap: "0.75rem" }}>
              {rrCourts.map((court, index) => (
                <div key={court.rowId} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: "#f8fafc" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem", alignItems: "center", marginBottom: "0.5rem" }}><strong>Court {index + 1}</strong><button type="button" onClick={() => setRrCourts((current) => current.filter((item) => item.rowId !== court.rowId))} disabled={rrCourts.length <= 1}>Remove</button></div>
                  <div style={{ display: "grid", gridTemplateColumns: "minmax(150px, 220px) 1fr", gap: "0.75rem" }}>
                    <label><strong>Format</strong><br /><select value={court.formatType} onChange={(event) => patchRrCourt(court.rowId, { formatType: event.target.value })} style={inputStyle}>{(status.round_robin_format_options || [firstFormat]).map((format) => <option key={format}>{format}</option>)}</select><span style={{ display: "block", color: "#64748b", fontSize: "0.85rem", marginTop: "0.25rem" }}>Expected games: {status.round_robin_expected_games?.[court.formatType] ?? "—"}</span></label>
                    <label><strong>Players</strong><br /><textarea value={court.namesText} onChange={(event) => patchRrCourt(court.rowId, { namesText: event.target.value })} rows={3} placeholder="Alex, Blair, Casey, Devon" style={inputStyle} /></label>
                  </div>
                </div>
              ))}
            </div>
            <p><button type="button" onClick={() => setRrCourts((current) => [...current, { rowId: randomId("court"), formatType: firstFormat, namesText: "" }])} disabled={rrCourts.length >= 10} style={ghostButtonStyle}>Add court</button></p>
            <label><strong>Custom schedule override</strong><br /><textarea value={rrCustomSchedule} onChange={(event) => setRrCustomSchedule(event.target.value)} rows={3} placeholder="Optional lines like: 1 2 3 4" style={inputStyle} /></label>
            <p><button type="button" onClick={generateRoundRobin} disabled={generating || creatingPlayers || !accessToken} style={buttonStyle}>{generating ? "Generating…" : "Generate schedule"}</button></p>
          </article>

          {newPlayerDrafts.length ? (
            <article style={{ ...cardStyle, borderColor: "#f59e0b", background: "#fffbeb" }}>
              <h2 style={{ marginTop: 0 }}>New players found — create profiles to continue</h2>
              <p style={{ color: "#92400e" }}>Review each new player and set a Starting JUPR.</p>
              <div style={{ display: "grid", gap: "0.5rem" }}>
                {newPlayerDrafts.map((draft, index) => (
                  <div key={`${draft.name}-${index}`} style={{ display: "grid", gridTemplateColumns: "minmax(180px, 1fr) 140px", gap: "0.5rem" }}>
                    <input value={draft.name} onChange={(event) => setNewPlayerDrafts((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, name: event.target.value } : item))} style={inputStyle} />
                    <input value={draft.startingJupr} onChange={(event) => setNewPlayerDrafts((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, startingJupr: event.target.value } : item))} type="number" min={1} max={7} step={0.01} style={inputStyle} />
                  </div>
                ))}
              </div>
              <p><button type="button" onClick={createPlayersAndContinue} disabled={creatingPlayers || generating || !accessToken} style={buttonStyle}>{creatingPlayers ? "Creating…" : "Create Players & Continue"}</button></p>
            </article>
          ) : null}

          {rrSchedule.length ? (
            <article style={cardStyle}>
              <h2 style={{ marginTop: 0 }}>Round-robin scores</h2>
              <p style={{ color: "#475569" }}>Zero-zero games are left unsubmitted, matching the Streamlit flow.</p>
              <div style={{ display: "grid", gap: "1rem" }}>
                {rrSchedule.map((court) => (
                  <div key={`court-${court.court}`} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
                    <h3 style={{ marginTop: 0 }}>Court {court.court} · {court.formatType}</h3>
                    <div style={{ display: "grid", gap: "0.5rem" }}>
                      {court.matches.map((match) => (
                        <div key={match.rowId} style={{ display: "grid", gridTemplateColumns: "minmax(160px, 1fr) 80px 80px minmax(160px, 1fr)", gap: "0.5rem", alignItems: "center" }}>
                          <div><strong>{match.label}</strong><br />{match.t1.map((player) => player.name).join(" / ")}</div>
                          <input value={match.s1} onChange={(event) => patchRrScore(match.rowId, { s1: event.target.value })} type="number" min={0} max={99} style={inputStyle} />
                          <input value={match.s2} onChange={(event) => patchRrScore(match.rowId, { s2: event.target.value })} type="number" min={0} max={99} style={inputStyle} />
                          <div>{match.t2.map((player) => player.name).join(" / ")}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <p><strong>Scored games:</strong> {scoredRrRows.length} / {rrSchedule.reduce((total, court) => total + court.matches.length, 0)}</p>
              <button type="button" onClick={submitRoundRobinScores} disabled={saving || !accessToken || !scoredRrRows.length} style={buttonStyle}>{saving ? "Submitting…" : "Submit scored round-robin games"}</button>
            </article>
          ) : null}
        </>
      ) : null}

      {message ? <p style={{ color: result?.ok ? "#166534" : "#b91c1c" }}>{message}</p> : null}

      {result ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Submission result</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem" }}>
            <div><strong>Inserted</strong><br />{result.result?.inserted ?? 0}</div>
            <div><strong>Match write</strong><br />{result.match_write_committed ? "Committed" : "See response"}</div>
            <div><strong>Rating type</strong><br />{result.feedback?.rating_type || result.result?.match_format || "doubles/overall"}</div>
            <div><strong>Skipped incomplete</strong><br />{result.result?.skipped_incomplete ?? 0}</div>
            <div><strong>Skipped empty</strong><br />{result.result?.skipped_empty ?? 0}</div>
            <div><strong>Skipped unrated</strong><br />{result.result?.skipped_unrated ?? 0}</div>
          </div>
          {result.auto_player_updates ? <div style={{ marginTop: "1rem", padding: "0.75rem", border: "1px solid #cbd5e1", borderRadius: "12px", background: result.auto_player_updates.mode === "error" ? "#fef2f2" : "#f8fafc" }}><strong>Post-batch player-update email</strong><p style={{ marginBottom: 0 }}>Mode: {result.auto_player_updates.mode || "unknown"} · Attempted: {result.auto_player_updates.attempted ?? 0} · Sent: {result.auto_player_updates.sent ?? 0} · Skipped: {result.auto_player_updates.skipped ?? 0} · Errors: {result.auto_player_updates.errors ?? 0}{result.auto_player_updates.email_mode ? ` · Delivery: ${result.auto_player_updates.email_mode}` : ""}</p>{result.auto_player_updates.reason ? <p style={{ color: "#92400e", marginBottom: 0 }}>{result.auto_player_updates.reason}</p> : null}</div> : null}
          {result.recovery ? <p style={{ color: "#475569" }}>{result.recovery.operator_rule} {result.recovery.match_log_route ? <Link href={result.recovery.match_log_route}>Open Match Log</Link> : null}{result.recovery.player_updates_route ? <> · <Link href={result.recovery.player_updates_route}>Open Player Updates</Link></> : null}</p> : null}
          {result.feedback?.affected_players?.length ? (
            <div style={{ overflowX: "auto", marginTop: "1rem" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Player</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Before</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>After</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Change</th></tr></thead>
                <tbody>
                  {result.feedback.affected_players.map((player) => (
                    <tr key={player.id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{player.name}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{ratingLabel(player.rating_before)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{ratingLabel(player.rating_after)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{deltaLabel(player.rating_delta)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
          {result.warnings?.length ? <ul style={{ color: "#92400e" }}>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
        </article>
      ) : null}
    </section>
  );
}
