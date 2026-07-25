"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type { AdminDuplicateDeletePreview, AdminDuplicateGroup, AdminMatchEditOperation, AdminMatchLogMatch, AdminMatchLogPlayer, AdminMatchLogWriteResult } from "@/lib/adminMatchLogApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type MatchLogApplyPanelProps = {
  apiBase: string | null;
  clubId: string;
  applyEnabled: boolean;
  duplicateCleanupEnabled: boolean;
  duplicatePreview?: AdminDuplicateDeletePreview | null;
  duplicateGroups?: AdminDuplicateGroup[];
  matches?: AdminMatchLogMatch[];
  recentOperations?: AdminMatchEditOperation[];
};

type MatchPatch = {
  id: number;
  league?: string;
  date?: string;
  week_tag?: string;
  match_type?: string;
  notes?: string;
  t1_p1?: number;
  t1_p2?: number;
  t2_p1?: number;
  t2_p2?: number;
  score_t1?: number;
  score_t2?: number;
};

type MatchEditState = {
  league: string;
  weekTag: string;
  matchType: string;
  notes: string;
  date: string;
  scoreT1: string;
  scoreT2: string;
  t1p1: string;
  t1p2: string;
  t2p1: string;
  t2p2: string;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const secondaryButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

class ApiCallError extends Error {
  operationId: string | null;

  constructor(message: string, operationId: string | null = null) {
    super(message);
    this.operationId = operationId;
  }
}

function requestKey(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `match-edit-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function apiErrorDetail(payload: unknown, fallback: string): { message: string; operationId: string | null } {
  if (!payload || typeof payload !== "object") return { message: fallback, operationId: null };
  const detail = (payload as { detail?: unknown }).detail;
  if (detail && typeof detail === "object") {
    const record = detail as { message?: unknown; operation_id?: unknown };
    return { message: String(record.message || fallback), operationId: record.operation_id ? String(record.operation_id) : null };
  }
  return { message: detail ? String(detail) : fallback, operationId: null };
}

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function resultSummary(result: AdminMatchLogWriteResult | null): string | null {
  if (!result) return null;
  if (result.mode === "duplicates_cleaned") return `Cleaned ${result.deleted_count ?? 0} duplicate row(s). Replay scope: ${result.recommended_replay_scope ?? "ALL"}.`;
  if (result.mode === "duplicate_no_issue") return `Marked match IDs ${(result.match_ids ?? []).join(", ") || "selected group"} as no issue.`;
  if (result.mode === "applied_and_replayed") return `Atomically applied ${result.updated_count ?? 0} match edit(s) and completed replay job ${result.replay_job_id || "—"}.`;
  if (result.mode === "replay_in_progress") return `Applied ${result.updated_count ?? 0} match edit(s); replay job ${result.replay_job_id || "—"} is still in progress.`;
  if (result.mode === "recovered" || result.mode === "already_recovered") return `Mandatory replay recovery completed for operation ${result.operation_id || "—"}.`;
  if (result.mode === "applied") return `Atomically applied ${result.updated_count ?? 0} non-rating match edit(s).`;
  return "Operation completed.";
}

function groupLabel(group: AdminDuplicateGroup): string {
  return `${group.league || "—"} · ${group.week_tag || "—"} · IDs ${group.ids.join(", ")}`;
}

function playerNames(players: AdminMatchLogPlayer[]): string {
  return players.map((player) => player.name || (player.id ? `#${player.id}` : "—")).join(" / ");
}

function playerOptionLabel(player: AdminMatchLogPlayer): string {
  const name = player.name || (player.id == null ? "Unknown player" : `Player ${player.id}`);
  return player.id == null ? name : `${name} (#${player.id})`;
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 19);
  return date.toISOString().replace("T", " ").slice(0, 16);
}

function toDateTimeInput(value?: string | null): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).replace(" ", "T").slice(0, 16);
  return date.toISOString().slice(0, 16);
}

function emptyEditState(): MatchEditState {
  return {
    league: "",
    weekTag: "",
    matchType: "",
    notes: "",
    date: "",
    scoreT1: "",
    scoreT2: "",
    t1p1: "",
    t1p2: "",
    t2p1: "",
    t2p2: ""
  };
}

function editStateFromMatch(match: AdminMatchLogMatch | null): MatchEditState {
  if (!match) return emptyEditState();
  return {
    league: match.league || "",
    weekTag: match.week_tag || "",
    matchType: match.match_type || "",
    notes: match.notes || "",
    date: toDateTimeInput(match.date),
    scoreT1: String(match.score?.team1 ?? ""),
    scoreT2: String(match.score?.team2 ?? ""),
    t1p1: match.team1?.[0]?.id == null ? "" : String(match.team1[0].id),
    t1p2: match.team1?.[1]?.id == null ? "" : String(match.team1[1].id),
    t2p1: match.team2?.[0]?.id == null ? "" : String(match.team2[0].id),
    t2p2: match.team2?.[1]?.id == null ? "" : String(match.team2[1].id)
  };
}

function integerInput(value: string, label: string): number {
  const cleaned = String(value || "").trim();
  if (!cleaned) throw new Error(`${label} is required.`);
  const parsed = Number(cleaned);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed)) throw new Error(`${label} must be a whole number.`);
  return parsed;
}

function changedText(current: string, original?: string | null): boolean {
  return String(current || "").trim() !== String(original || "").trim();
}

function buildPatch(match: AdminMatchLogMatch, edit: MatchEditState): MatchPatch {
  if (match.id == null) throw new Error("Select a saved match with an ID before staging an edit.");
  const patch: MatchPatch = { id: Number(match.id) };

  if (changedText(edit.league, match.league)) patch.league = edit.league.trim();
  if (changedText(edit.weekTag, match.week_tag)) patch.week_tag = edit.weekTag.trim();
  if (changedText(edit.matchType, match.match_type)) patch.match_type = edit.matchType.trim();
  if (changedText(edit.notes, match.notes)) patch.notes = edit.notes.trim();

  const originalDate = toDateTimeInput(match.date);
  if (edit.date && edit.date !== originalDate) patch.date = `${edit.date}:00Z`;

  const scoreT1 = integerInput(edit.scoreT1, "Team 1 score");
  const scoreT2 = integerInput(edit.scoreT2, "Team 2 score");
  if (scoreT1 !== Number(match.score?.team1 ?? 0)) patch.score_t1 = scoreT1;
  if (scoreT2 !== Number(match.score?.team2 ?? 0)) patch.score_t2 = scoreT2;

  const playerFields: Array<[keyof MatchPatch, string, number | null | undefined, string]> = [
    ["t1_p1", edit.t1p1, match.team1?.[0]?.id, "Team 1 player 1"],
    ["t1_p2", edit.t1p2, match.team1?.[1]?.id, "Team 1 player 2"],
    ["t2_p1", edit.t2p1, match.team2?.[0]?.id, "Team 2 player 1"],
    ["t2_p2", edit.t2p2, match.team2?.[1]?.id, "Team 2 player 2"]
  ];
  for (const [field, value, original, label] of playerFields) {
    if (!String(value || "").trim() && original == null && (field === "t1_p2" || field === "t2_p2")) continue;
    if (!String(value || "").trim() && original != null) throw new Error(`${label} cannot be cleared. Choose a replacement player instead.`);
    const playerId = integerInput(value, label);
    if (playerId !== Number(original ?? -1)) {
      patch[field] = playerId as never;
    }
  }

  return patch;
}

function patchFields(patch: MatchPatch): string[] {
  return Object.keys(patch).filter((key) => key !== "id");
}

function patchScope(patches: MatchPatch[]): { standings: boolean; ratings: boolean } {
  const standingsFields = new Set(["week_tag", "league", "date", "t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2"]);
  const ratingFields = new Set(["league", "date", "match_type", "t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2"]);
  const changed = patches.flatMap((patch) => patchFields(patch));
  return {
    standings: changed.some((field) => standingsFields.has(field)),
    ratings: changed.some((field) => ratingFields.has(field))
  };
}

function collectVisiblePlayers(matches: AdminMatchLogMatch[]): AdminMatchLogPlayer[] {
  const byId = new Map<number, AdminMatchLogPlayer>();
  for (const match of matches) {
    for (const player of [...(match.team1 || []), ...(match.team2 || [])]) {
      if (player.id != null && !byId.has(Number(player.id))) byId.set(Number(player.id), player);
    }
  }
  return Array.from(byId.values()).sort((a, b) => String(a.name || "").localeCompare(String(b.name || "")));
}

function PlayerSelect({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: AdminMatchLogPlayer[] }) {
  const currentInOptions = Boolean(value) && options.some((player) => String(player.id) === String(value));
  return (
    <label><strong>{label}</strong><br />
      <select value={value} onChange={(event) => onChange(event.target.value)} style={inputStyle}>
        <option value="">Select player…</option>
        {value && !currentInOptions ? <option value={value}>Current player #{value}</option> : null}
        {options.map((player) => <option key={String(player.id)} value={String(player.id)}>{playerOptionLabel(player)}</option>)}
      </select>
    </label>
  );
}

function MatchSummary({ match }: { match: AdminMatchLogMatch }) {
  return (
    <div style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem", background: "#f8fafc" }}>
      <strong>#{match.id} · {dateLabel(match.date)}</strong>
      <p style={{ margin: "0.35rem 0", color: "#475569" }}>{match.league || "—"} · {match.week_tag || "—"} · {match.match_type || "—"}</p>
      <p style={{ margin: 0 }}>{playerNames(match.team1)} <strong>{match.score?.display}</strong> {playerNames(match.team2)}</p>
      {match.notes ? <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>Notes: {match.notes}</p> : null}
    </div>
  );
}

export default function MatchLogApplyPanel({ apiBase, clubId, applyEnabled, duplicateCleanupEnabled, duplicatePreview, duplicateGroups = [], matches = [], recentOperations = [] }: MatchLogApplyPanelProps) {
  const router = useRouter();
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const firstMatch = matches.find((match) => match.id != null) || null;
  const [selectedMatchId, setSelectedMatchId] = useState(firstMatch?.id == null ? "" : String(firstMatch.id));
  const [edit, setEdit] = useState<MatchEditState>(() => editStateFromMatch(firstMatch));
  const [stagedPatches, setStagedPatches] = useState<MatchPatch[]>([]);
  const [rosterPlayers, setRosterPlayers] = useState<AdminMatchLogPlayer[]>([]);
  const [rosterMessage, setRosterMessage] = useState<string | null>(null);
  const [correctionNote, setCorrectionNote] = useState("");
  const [noIssueReason, setNoIssueReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AdminMatchLogWriteResult | null>(null);
  const [idempotencyKey, setIdempotencyKey] = useState(requestKey);
  const [recoveryOperationId, setRecoveryOperationId] = useState<string | null>(() => recentOperations.find((operation) => operation.status === "recovery_required")?.id || null);
  const [bulkIds, setBulkIds] = useState<string[]>([]);
  const [bulkLeague, setBulkLeague] = useState("");
  const [bulkWeekMode, setBulkWeekMode] = useState<"unchanged" | "set" | "clear">("unchanged");
  const [bulkWeekTag, setBulkWeekTag] = useState("");
  const [bulkMatchType, setBulkMatchType] = useState("");
  const [bulkNotesMode, setBulkNotesMode] = useState<"unchanged" | "set" | "clear">("unchanged");
  const [bulkNotes, setBulkNotes] = useState("");
  const [bulkShiftDays, setBulkShiftDays] = useState("0");
  const [bulkReplaceSlot, setBulkReplaceSlot] = useState<"" | "t1_p1" | "t1_p2" | "t2_p1" | "t2_p2">("");
  const [bulkReplacementPlayer, setBulkReplacementPlayer] = useState("");
  const selectedMatch = matches.find((match) => match.id != null && String(match.id) === selectedMatchId) || null;
  const visiblePlayerOptions = collectVisiblePlayers(matches);
  const playerOptions = rosterPlayers.length ? rosterPlayers : visiblePlayerOptions;
  const scope = patchScope(stagedPatches);

  useEffect(() => {
    let cancelled = false;
    async function loadRosterPlayers() {
      if (!applyEnabled || !apiBase || !accessToken) {
        setRosterPlayers([]);
        setRosterMessage(null);
        return;
      }
      setRosterMessage("Loading full roster picker…");
      try {
        const response = await fetch(apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/match-log/player-options`), {
          cache: "no-store",
          headers: { accept: "application/json", Authorization: `Bearer ${accessToken}` }
        });
        const payload = await response.json().catch(() => null) as { players?: AdminMatchLogPlayer[]; detail?: unknown } | null;
        if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
        const players = (Array.isArray(payload?.players) ? payload.players : []).filter((player) => player && player.id != null);
        if (!cancelled) {
          setRosterPlayers(players);
          setRosterMessage(players.length ? `Full roster picker loaded (${players.length} players).` : "Roster picker returned no players; using visible-player fallback.");
        }
      } catch (error) {
        if (!cancelled) {
          setRosterPlayers([]);
          setRosterMessage(`Full roster picker unavailable; using visible-player fallback. ${error instanceof Error ? error.message : ""}`.trim());
        }
      }
    }
    void loadRosterPlayers();
    return () => { cancelled = true; };
  }, [accessToken, apiBase, applyEnabled, clubId]);

  function updateEdit<K extends keyof MatchEditState>(key: K, value: MatchEditState[K]) {
    setEdit((current) => ({ ...current, [key]: value }));
  }

  function selectMatch(matchId: string) {
    const nextMatch = matches.find((match) => match.id != null && String(match.id) === matchId) || null;
    setSelectedMatchId(matchId);
    setEdit(editStateFromMatch(nextMatch));
    setMessage(null);
  }

  async function callApi(path: string, method: "PATCH" | "POST", body: unknown) {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before applying Match Log changes.");
    const response = await fetch(apiUrl(apiBase, path), {
      method,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`
      },
      body: JSON.stringify(body)
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = apiErrorDetail(payload, `API error (${response.status})`);
      throw new ApiCallError(detail.message, detail.operationId);
    }
    return payload as AdminMatchLogWriteResult;
  }

  function stageGuidedEdit() {
    setMessage(null);
    setResult(null);
    try {
      if (!selectedMatch) throw new Error("Select a match before staging an edit.");
      const patch = buildPatch(selectedMatch, edit);
      const fields = patchFields(patch);
      if (!fields.length) throw new Error("No changes detected for the selected match.");
      setStagedPatches((current) => [patch, ...current.filter((existing) => existing.id !== patch.id)]);
      setMessage(`Staged edit for match #${patch.id}: ${fields.join(", ")}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to stage match edit.");
    }
  }

  function toggleBulkMatch(matchId: number) {
    const key = String(matchId);
    setBulkIds((current) => current.includes(key) ? current.filter((value) => value !== key) : current.length < 100 ? [...current, key] : current);
  }

  function stageBulkEdits() {
    setMessage(null);
    setResult(null);
    try {
      const selected = matches.filter((match) => match.id != null && bulkIds.includes(String(match.id))).slice(0, 100);
      if (!selected.length) throw new Error("Select at least one visible match for the bulk stage.");
      const shiftDays = Number(bulkShiftDays || 0);
      if (!Number.isInteger(shiftDays)) throw new Error("Date shift must be a whole number of days.");
      if (bulkWeekMode === "set" && !bulkWeekTag.trim()) throw new Error("Enter a week tag before staging the bulk set action.");
      if (bulkNotesMode === "set" && !bulkNotes.trim()) throw new Error("Enter notes before staging the bulk set action.");
      if (bulkReplaceSlot && !bulkReplacementPlayer) throw new Error("Choose a replacement player for the selected slot.");

      const generated = selected.map((match) => {
        const patch: MatchPatch = { id: Number(match.id) };
        if (bulkLeague.trim()) patch.league = bulkLeague.trim();
        if (bulkWeekMode === "set") patch.week_tag = bulkWeekTag.trim();
        if (bulkWeekMode === "clear") patch.week_tag = "";
        if (bulkMatchType.trim()) patch.match_type = bulkMatchType.trim();
        if (bulkNotesMode === "set") patch.notes = bulkNotes.trim();
        if (bulkNotesMode === "clear") patch.notes = "";
        if (shiftDays !== 0) {
          const original = new Date(String(match.date || ""));
          if (Number.isNaN(original.getTime())) throw new Error(`Match #${match.id} has no valid date to shift.`);
          original.setUTCDate(original.getUTCDate() + shiftDays);
          patch.date = original.toISOString();
        }
        if (bulkReplaceSlot) patch[bulkReplaceSlot] = integerInput(bulkReplacementPlayer, "Replacement player");
        return patch;
      });
      if (generated.every((patch) => patchFields(patch).length === 0)) throw new Error("Choose at least one bulk field change.");
      setStagedPatches((current) => {
        const byId = new Map(current.map((patch) => [patch.id, patch]));
        for (const patch of generated) byId.set(patch.id, { ...(byId.get(patch.id) || { id: patch.id }), ...patch });
        return Array.from(byId.values());
      });
      setMessage(`Staged bulk changes for ${generated.length} match(es).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to stage bulk changes.");
    }
  }

  async function submitGuidedPatches(confirmationText: string) {
    setBusy(true);
    setMessage(null);
    setResult(null);
    try {
      if (!stagedPatches.length) throw new Error("Stage at least one match edit first.");
      const payload = await callApi(`/admin/clubs/${encodeURIComponent(clubId)}/match-log/edits`, "PATCH", {
        patches: stagedPatches,
        confirmation_text: confirmationText,
        correction_note: correctionNote,
        source: "next_match_log_guided_editor",
        idempotency_key: idempotencyKey,
        replay_target: "ALL (Full System Reset)"
      });
      setResult(payload);
      setMessage(resultSummary(payload));
      if (payload.ok) {
        setStagedPatches([]);
        setRecoveryOperationId(null);
        setIdempotencyKey(requestKey());
      }
      router.refresh();
    } catch (error) {
      if (error instanceof ApiCallError && error.operationId) setRecoveryOperationId(error.operationId);
      setMessage(error instanceof Error ? error.message : "Unable to apply match edits.");
    } finally {
      setBusy(false);
    }
  }

  async function recoverMandatoryReplay(confirmationText: string) {
    if (!recoveryOperationId) return;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await callApi(`/admin/clubs/${encodeURIComponent(clubId)}/match-log/edits/${encodeURIComponent(recoveryOperationId)}/recover`, "POST", {
        confirmation_text: confirmationText,
        source: "next_match_log_recovery"
      });
      setResult(payload);
      setMessage(`Recovery completed for operation ${recoveryOperationId}.`);
      setRecoveryOperationId(null);
      setStagedPatches([]);
      setIdempotencyKey(requestKey());
      router.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to complete replay recovery.");
    } finally {
      setBusy(false);
    }
  }

  async function cleanupDuplicates(confirmationText: string) {
    const deleteIds = duplicatePreview?.delete_ids ?? [];
    setBusy(true);
    setMessage(null);
    setResult(null);
    try {
      if (!deleteIds.length) throw new Error("No duplicate cleanup IDs are available in the current preview.");
      const payload = await callApi(`/admin/clubs/${encodeURIComponent(clubId)}/match-log/duplicates/cleanup`, "POST", {
        delete_ids: deleteIds,
        confirmation_text: confirmationText,
        source: "next_match_log_duplicate_cleanup_panel"
      });
      setResult(payload);
      setMessage(resultSummary(payload));
      router.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to clean duplicates.");
    } finally {
      setBusy(false);
    }
  }

  async function resolveNoIssue(group: AdminDuplicateGroup, confirmationText: string) {
    setBusy(true);
    setMessage(null);
    setResult(null);
    try {
      const payload = await callApi(`/admin/clubs/${encodeURIComponent(clubId)}/match-log/duplicates/resolve`, "POST", {
        match_ids: group.ids,
        dup_key: group.dup_key,
        reason: noIssueReason,
        confirmation_text: confirmationText,
        source: "next_match_log_duplicate_no_issue_panel"
      });
      setResult(payload);
      setMessage(resultSummary(payload));
      setNoIssueReason("");
      router.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to mark duplicate group as no issue.");
    } finally {
      setBusy(false);
    }
  }

  if (!applyEnabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Apply flow is disabled</h2>
        <p style={{ color: "#475569" }}>
          Match Log writes require <code>JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY=1</code> on FastAPI plus Supabase JWT role authorization.
        </p>
      </article>
    );
  }

  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Apply audited Match Log changes</h2>
      <p style={{ color: "#475569" }}>
        This panel uses guided controls modeled after the Streamlit Match Log editor. It builds safe FastAPI patches for you; no raw JSON editing is required.
      </p>

      <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
        <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
        <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
          {accessToken ? "Ready to send authorized Match Log requests." : sessionLoading ? "Checking admin session…" : "Sign in before applying changes or cleaning duplicates."}
        </p>
        {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
      </div>

      <h3>Guided match editor</h3>
      <p style={{ color: "#475569" }}>
        Select a match from the current filtered results, change fields with form controls, stage the edit, then apply all staged edits together.
      </p>
      <div style={{ display: "grid", gap: "0.75rem" }}>
        <label><strong>Match</strong><br />
          <select value={selectedMatchId} onChange={(event) => selectMatch(event.target.value)} style={inputStyle}>
            <option value="">Select a match…</option>
            {matches.filter((match) => match.id != null).map((match) => (
              <option key={match.id ?? "match"} value={String(match.id)}>
                #{match.id} · {dateLabel(match.date)} · {match.league || "—"} · {playerNames(match.team1)} vs {playerNames(match.team2)} · {match.score?.display || "—"}
              </option>
            ))}
          </select>
        </label>

        {selectedMatch ? <MatchSummary match={selectedMatch} /> : <p style={{ color: "#475569" }}>No match selected.</p>}

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem" }}>
          <label><strong>League</strong><br /><input value={edit.league} onChange={(event) => updateEdit("league", event.target.value)} style={inputStyle} /></label>
          <label><strong>Week tag</strong><br /><input value={edit.weekTag} onChange={(event) => updateEdit("weekTag", event.target.value)} style={inputStyle} placeholder="Blank clears week" /></label>
          <label><strong>Match type</strong><br /><input value={edit.matchType} onChange={(event) => updateEdit("matchType", event.target.value)} style={inputStyle} /></label>
          <label><strong>UTC date/time</strong><br /><input type="datetime-local" value={edit.date} onChange={(event) => updateEdit("date", event.target.value)} style={inputStyle} /></label>
          <label><strong>Team 1 score</strong><br /><input type="number" min="0" step="1" value={edit.scoreT1} onChange={(event) => updateEdit("scoreT1", event.target.value)} style={inputStyle} /></label>
          <label><strong>Team 2 score</strong><br /><input type="number" min="0" step="1" value={edit.scoreT2} onChange={(event) => updateEdit("scoreT2", event.target.value)} style={inputStyle} /></label>
        </div>

        <label><strong>Match notes</strong><br /><textarea value={edit.notes} onChange={(event) => updateEdit("notes", event.target.value)} maxLength={2000} rows={3} style={inputStyle} placeholder="Blank clears notes" /></label>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem" }}>
          <PlayerSelect label="Team 1 player 1" value={edit.t1p1} onChange={(value) => updateEdit("t1p1", value)} options={playerOptions} />
          <PlayerSelect label="Team 1 player 2" value={edit.t1p2} onChange={(value) => updateEdit("t1p2", value)} options={playerOptions} />
          <PlayerSelect label="Team 2 player 1" value={edit.t2p1} onChange={(value) => updateEdit("t2p1", value)} options={playerOptions} />
          <PlayerSelect label="Team 2 player 2" value={edit.t2p2} onChange={(value) => updateEdit("t2p2", value)} options={playerOptions} />
        </div>

        <p style={{ color: rosterPlayers.length ? "#166534" : "#64748b", margin: 0 }}>
          {rosterMessage || (playerOptions.length ? `Player picker using ${playerOptions.length} visible player option(s).` : "No player options are loaded yet.")}
        </p>

        <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: 0 }}>
          <button type="button" onClick={stageGuidedEdit} disabled={busy || !selectedMatch} style={buttonStyle}>Stage guided edit</button>
          <button type="button" onClick={() => selectedMatch ? setEdit(editStateFromMatch(selectedMatch)) : undefined} disabled={busy || !selectedMatch} style={secondaryButtonStyle}>Reset selected fields</button>
        </p>
      </div>

      <section data-testid="match-log-bulk-editor" style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.85rem", marginTop: "1rem", background: "#f8fafc" }}>
        <h3 style={{ marginTop: 0 }}>Bulk stage visible matches</h3>
        <p style={{ color: "#475569" }}>Select up to 100 rows, then set shared fields, clear notes/week tags, shift dates, or replace one player slot. Nothing is written until the staged operation is confirmed below.</p>
        <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <button type="button" onClick={() => setBulkIds(matches.filter((match) => match.id != null).slice(0, 100).map((match) => String(match.id)))} style={secondaryButtonStyle}>Select first 100 visible</button>
          <button type="button" onClick={() => setBulkIds([])} disabled={!bulkIds.length} style={secondaryButtonStyle}>Clear selection</button>
          <span style={{ alignSelf: "center" }}><strong>{bulkIds.length}</strong> selected</span>
        </p>
        <div style={{ maxHeight: "220px", overflowY: "auto", display: "grid", gap: "0.35rem", border: "1px solid #e2e8f0", borderRadius: "8px", padding: "0.6rem", background: "white" }}>
          {matches.filter((match) => match.id != null).map((match) => (
            <label key={String(match.id)} style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start" }}>
              <input type="checkbox" checked={bulkIds.includes(String(match.id))} onChange={() => toggleBulkMatch(Number(match.id))} />
              <span>#{match.id} · {dateLabel(match.date)} · {match.league || "—"} · {playerNames(match.team1)} vs {playerNames(match.team2)} · {match.score?.display || "—"}</span>
            </label>
          ))}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>
          <label><strong>Set league</strong><br /><input value={bulkLeague} onChange={(event) => setBulkLeague(event.target.value)} style={inputStyle} placeholder="Blank = unchanged" /></label>
          <label><strong>Week tag action</strong><br /><select value={bulkWeekMode} onChange={(event) => setBulkWeekMode(event.target.value as "unchanged" | "set" | "clear")} style={inputStyle}><option value="unchanged">No change</option><option value="set">Set</option><option value="clear">Clear</option></select></label>
          <label><strong>New week tag</strong><br /><input value={bulkWeekTag} onChange={(event) => setBulkWeekTag(event.target.value)} disabled={bulkWeekMode !== "set"} style={inputStyle} /></label>
          <label><strong>Set match type</strong><br /><input value={bulkMatchType} onChange={(event) => setBulkMatchType(event.target.value)} style={inputStyle} placeholder="Blank = unchanged" /></label>
          <label><strong>Notes action</strong><br /><select value={bulkNotesMode} onChange={(event) => setBulkNotesMode(event.target.value as "unchanged" | "set" | "clear")} style={inputStyle}><option value="unchanged">No change</option><option value="set">Set</option><option value="clear">Clear</option></select></label>
          <label><strong>Shift UTC date</strong><br /><input type="number" step="1" value={bulkShiftDays} onChange={(event) => setBulkShiftDays(event.target.value)} style={inputStyle} /><small>Whole days; 0 = unchanged.</small></label>
          <label><strong>Replace player slot</strong><br /><select value={bulkReplaceSlot} onChange={(event) => setBulkReplaceSlot(event.target.value as "" | "t1_p1" | "t1_p2" | "t2_p1" | "t2_p2")} style={inputStyle}><option value="">No change</option><option value="t1_p1">Team 1 player 1</option><option value="t1_p2">Team 1 player 2</option><option value="t2_p1">Team 2 player 1</option><option value="t2_p2">Team 2 player 2</option></select></label>
          <PlayerSelect label="Replacement player" value={bulkReplacementPlayer} onChange={setBulkReplacementPlayer} options={playerOptions} />
        </div>
        {bulkNotesMode === "set" ? <label style={{ display: "block", marginTop: "0.75rem" }}><strong>Replacement notes</strong><br /><textarea value={bulkNotes} onChange={(event) => setBulkNotes(event.target.value)} maxLength={2000} rows={3} style={inputStyle} /></label> : null}
        <p><button type="button" onClick={stageBulkEdits} disabled={busy || !bulkIds.length} style={buttonStyle}>Stage bulk changes</button></p>
      </section>

      <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", marginTop: "1rem", background: stagedPatches.length ? "#f8fafc" : "white" }}>
        <h4 style={{ marginTop: 0 }}>Staged edits</h4>
        {stagedPatches.length ? (
          <>
            <ul style={{ paddingLeft: "1.25rem" }}>
              {stagedPatches.map((patch) => (
                <li key={patch.id}>Match #{patch.id}: {patchFields(patch).join(", ")}</li>
              ))}
            </ul>
            <p style={{ color: "#475569" }}>Impact preview: standings={String(scope.standings)}, ratings={String(scope.ratings)}</p>
            {scope.ratings ? <p style={{ color: "#92400e" }}>This operation cannot report success until its durable Replay ALL job succeeds.</p> : <p style={{ color: "#166534" }}>These metadata-only changes do not require rating replay.</p>}
          </>
        ) : <p style={{ color: "#475569" }}>No edits staged yet.</p>}
        <label><strong>Correction note</strong><br /><input value={correctionNote} onChange={(event) => setCorrectionNote(event.target.value)} style={inputStyle} /></label>
        <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <ConfirmAction
            triggerLabel="Apply staged edits"
            title={`Apply ${stagedPatches.length || "the staged"} match edit(s)?`}
            description={scope.ratings ? "This will apply the exact staged edits and must complete a durable Replay ALL job before it can report success." : "This will apply the exact staged metadata edits and write an audit record."}
            confirmLabel="Yes, apply staged edits"
            confirmationText="APPLY"
            disabled={busy || !accessToken || Boolean(recoveryOperationId) || !stagedPatches.length}
            busy={busy}
            onConfirm={submitGuidedPatches}
          />
          <button type="button" onClick={() => setStagedPatches([])} disabled={busy || !stagedPatches.length} style={secondaryButtonStyle}>Clear staged edits</button>
        </p>
      </div>

      {recoveryOperationId ? (
        <div role="alert" data-testid="match-edit-recovery" style={{ border: "2px solid #dc2626", borderRadius: "12px", padding: "0.85rem", marginTop: "1rem", background: "#fef2f2" }}>
          <h4 style={{ marginTop: 0 }}>Mandatory replay recovery required</h4>
          <p>Edits were committed in one database transaction, but replay did not finish. Do not start another correction until operation <code>{recoveryOperationId}</code> is recovered.</p>
          <p>
            <ConfirmAction
              triggerLabel="Complete mandatory replay"
              title="Retry this mandatory replay?"
              description={<>This retries the same durable replay job for operation <code>{recoveryOperationId}</code>. Do not start another correction until it completes.</>}
              confirmLabel="Yes, retry replay"
              confirmationText="RECOVER"
              disabled={busy}
              busy={busy}
              onConfirm={recoverMandatoryReplay}
            />
          </p>
        </div>
      ) : null}

      {recentOperations.length ? (
        <div data-testid="match-edit-operation-history" style={{ overflowX: "auto", marginTop: "1rem" }}>
          <h4>Recent durable edit operations</h4>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "680px" }}>
            <thead><tr>{["Created", "Status", "Replay", "Actor", "Operation"].map((label) => <th key={label} style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1", padding: "0.5rem" }}>{label}</th>)}</tr></thead>
            <tbody>{recentOperations.map((operation) => (
              <tr key={operation.id} data-operation-status={operation.status}>
                <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{operation.created_at ? new Date(operation.created_at).toISOString().slice(0, 19).replace("T", " ") : "—"}</td>
                <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{operation.status}{operation.error_text ? ` · ${operation.error_text}` : ""}</td>
                <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{operation.replay_target || "Not required"}</td>
                <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{operation.actor_email || "—"}</td>
                <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem", fontFamily: "monospace" }}>{operation.id}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      ) : null}

      <hr style={{ border: 0, borderTop: "1px solid #e2e8f0", margin: "1rem 0" }} />

      <h3>Duplicate false positive / no issue</h3>
      <p style={{ color: "#475569" }}>
        Use this when the scanner found a legitimate repeated matchup that should no longer appear as an active cleanup candidate.
      </p>
      {duplicateGroups.length ? (
        <div style={{ display: "grid", gap: "0.75rem" }}>
          <label><strong>Reason for no-issue resolution</strong><br /><input value={noIssueReason} onChange={(event) => setNoIssueReason(event.target.value)} style={inputStyle} /></label>
          {duplicateGroups.map((group) => (
            <div key={group.dup_key} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem", background: "#f8fafc" }}>
              <p style={{ marginTop: 0 }}><strong>{groupLabel(group)}</strong></p>
              <p style={{ color: "#475569" }}>Cleanup candidates: {group.delete_ids.map((id) => `#${id}`).join(", ") || "none"}</p>
              <ConfirmAction
                triggerLabel="Mark this group no issue"
                title="Mark this duplicate group as no issue?"
                description={<>This will resolve match IDs {group.ids.join(", ")} as a legitimate repeated matchup using the reason entered above.</>}
                confirmLabel="Yes, mark no issue"
                confirmationText="NO ISSUE"
                disabled={busy || !accessToken || !noIssueReason.trim()}
                busy={busy}
                onConfirm={(confirmationText) => resolveNoIssue(group, confirmationText)}
              />
            </div>
          ))}
        </div>
      ) : (
        <p style={{ color: "#475569" }}>No active duplicate groups are available to resolve as no issue.</p>
      )}

      <hr style={{ border: 0, borderTop: "1px solid #e2e8f0", margin: "1rem 0" }} />

      <h3>Duplicate cleanup</h3>
      {duplicateCleanupEnabled ? (
        <>
          <p style={{ color: "#475569" }}>
            Current preview would remove {duplicatePreview?.delete_count ?? 0} duplicate row(s): {(duplicatePreview?.delete_ids ?? []).join(", ") || "none"}.
          </p>
          <p>
            <ConfirmAction
              triggerLabel="Clean duplicate rows from preview"
              title={`Delete ${duplicatePreview?.delete_count ?? 0} duplicate row(s)?`}
              description={<>This will delete the exact duplicate row IDs from the current preview: {(duplicatePreview?.delete_ids ?? []).join(", ") || "none"}. Recovery may require Replay History.</>}
              confirmLabel="Yes, delete duplicate rows"
              confirmationText="DELETE"
              tone="danger"
              disabled={busy || !accessToken || !(duplicatePreview?.delete_ids?.length)}
              busy={busy}
              onConfirm={cleanupDuplicates}
            />
          </p>
        </>
      ) : (
        <p style={{ color: "#475569" }}>
          Destructive duplicate cleanup is disabled. Match edits and duplicate no-issue resolution remain available.
        </p>
      )}

      {message ? <p style={{ color: result?.ok ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      {result?.warnings?.length ? (
        <ul style={{ color: "#92400e", paddingLeft: "1.25rem" }}>
          {result.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      ) : null}
    </article>
  );
}
