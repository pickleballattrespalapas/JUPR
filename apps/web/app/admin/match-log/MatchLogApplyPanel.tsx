"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import type { AdminDuplicateDeletePreview, AdminDuplicateGroup, AdminMatchLogMatch, AdminMatchLogPlayer, AdminMatchLogWriteResult } from "@/lib/adminMatchLogApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type MatchLogApplyPanelProps = {
  apiBase: string | null;
  clubId: string;
  applyEnabled: boolean;
  duplicatePreview?: AdminDuplicateDeletePreview | null;
  duplicateGroups?: AdminDuplicateGroup[];
  matches?: AdminMatchLogMatch[];
};

type MatchPatch = {
  id: number;
  league?: string;
  date?: string;
  week_tag?: string;
  match_type?: string;
  is_active?: boolean;
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
  date: string;
  scoreT1: string;
  scoreT2: string;
  isActive: "true" | "false";
  t1p1: string;
  t1p2: string;
  t2p1: string;
  t2p2: string;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const secondaryButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function resultSummary(result: AdminMatchLogWriteResult | null): string | null {
  if (!result) return null;
  if (result.mode === "duplicates_cleaned") return `Cleaned ${result.deleted_count ?? 0} duplicate row(s). Replay scope: ${result.recommended_replay_scope ?? "ALL"}.`;
  if (result.mode === "duplicate_no_issue") return `Marked match IDs ${(result.match_ids ?? []).join(", ") || "selected group"} as no issue.`;
  if (result.mode === "applied") return `Applied ${result.updated_count ?? 0} match edit(s).`;
  return "Operation completed.";
}

function groupLabel(group: AdminDuplicateGroup): string {
  return `${group.league || "—"} · ${group.week_tag || "—"} · IDs ${group.ids.join(", ")}`;
}

function playerNames(players: AdminMatchLogPlayer[]): string {
  return players.map((player) => player.name || (player.id ? `#${player.id}` : "—")).join(" / ");
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
    date: "",
    scoreT1: "",
    scoreT2: "",
    isActive: "true",
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
    date: toDateTimeInput(match.date),
    scoreT1: String(match.score?.team1 ?? ""),
    scoreT2: String(match.score?.team2 ?? ""),
    isActive: match.is_active === false ? "false" : "true",
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

  const originalDate = toDateTimeInput(match.date);
  if (edit.date && edit.date !== originalDate) patch.date = `${edit.date}:00Z`;

  const scoreT1 = integerInput(edit.scoreT1, "Team 1 score");
  const scoreT2 = integerInput(edit.scoreT2, "Team 2 score");
  if (scoreT1 !== Number(match.score?.team1 ?? 0)) patch.score_t1 = scoreT1;
  if (scoreT2 !== Number(match.score?.team2 ?? 0)) patch.score_t2 = scoreT2;

  const isActive = edit.isActive === "true";
  if (isActive !== (match.is_active !== false)) patch.is_active = isActive;

  const playerFields: Array<[keyof MatchPatch, string, number | null | undefined, string]> = [
    ["t1_p1", edit.t1p1, match.team1?.[0]?.id, "Team 1 player 1"],
    ["t1_p2", edit.t1p2, match.team1?.[1]?.id, "Team 1 player 2"],
    ["t2_p1", edit.t2p1, match.team2?.[0]?.id, "Team 2 player 1"],
    ["t2_p2", edit.t2p2, match.team2?.[1]?.id, "Team 2 player 2"]
  ];
  for (const [field, value, original, label] of playerFields) {
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
  const standingsFields = new Set(["week_tag", "league", "date", "is_active", "t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2"]);
  const ratingFields = new Set(["league", "date", "match_type", "is_active", "t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2"]);
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

function MatchSummary({ match }: { match: AdminMatchLogMatch }) {
  return (
    <div style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem", background: "#f8fafc" }}>
      <strong>#{match.id} · {dateLabel(match.date)}</strong>
      <p style={{ margin: "0.35rem 0", color: "#475569" }}>{match.league || "—"} · {match.week_tag || "—"} · {match.match_type || "—"}</p>
      <p style={{ margin: 0 }}>{playerNames(match.team1)} <strong>{match.score?.display}</strong> {playerNames(match.team2)}</p>
    </div>
  );
}

export default function MatchLogApplyPanel({ apiBase, clubId, applyEnabled, duplicatePreview, duplicateGroups = [], matches = [] }: MatchLogApplyPanelProps) {
  const router = useRouter();
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const firstMatch = matches.find((match) => match.id != null) || null;
  const [selectedMatchId, setSelectedMatchId] = useState(firstMatch?.id == null ? "" : String(firstMatch.id));
  const [edit, setEdit] = useState<MatchEditState>(() => editStateFromMatch(firstMatch));
  const [stagedPatches, setStagedPatches] = useState<MatchPatch[]>([]);
  const [correctionNote, setCorrectionNote] = useState("");
  const [applyConfirm, setApplyConfirm] = useState("");
  const [cleanupConfirm, setCleanupConfirm] = useState("");
  const [noIssueReason, setNoIssueReason] = useState("");
  const [noIssueConfirm, setNoIssueConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AdminMatchLogWriteResult | null>(null);
  const selectedMatch = matches.find((match) => match.id != null && String(match.id) === selectedMatchId) || null;
  const playerOptions = collectVisiblePlayers(matches);
  const scope = patchScope(stagedPatches);

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
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
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

  async function submitGuidedPatches() {
    setBusy(true);
    setMessage(null);
    setResult(null);
    try {
      if (!stagedPatches.length) throw new Error("Stage at least one match edit first.");
      const payload = await callApi(`/admin/clubs/${encodeURIComponent(clubId)}/match-log/edits`, "PATCH", {
        patches: stagedPatches,
        confirmation_text: applyConfirm,
        correction_note: correctionNote,
        source: "next_match_log_guided_editor"
      });
      setResult(payload);
      setMessage(resultSummary(payload));
      setStagedPatches([]);
      setApplyConfirm("");
      router.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to apply match edits.");
    } finally {
      setBusy(false);
    }
  }

  async function cleanupDuplicates() {
    const deleteIds = duplicatePreview?.delete_ids ?? [];
    setBusy(true);
    setMessage(null);
    setResult(null);
    try {
      if (!deleteIds.length) throw new Error("No duplicate cleanup IDs are available in the current preview.");
      const payload = await callApi(`/admin/clubs/${encodeURIComponent(clubId)}/match-log/duplicates/cleanup`, "POST", {
        delete_ids: deleteIds,
        confirmation_text: cleanupConfirm,
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

  async function resolveNoIssue(group: AdminDuplicateGroup) {
    setBusy(true);
    setMessage(null);
    setResult(null);
    try {
      const payload = await callApi(`/admin/clubs/${encodeURIComponent(clubId)}/match-log/duplicates/resolve`, "POST", {
        match_ids: group.ids,
        dup_key: group.dup_key,
        reason: noIssueReason,
        confirmation_text: noIssueConfirm,
        source: "next_match_log_duplicate_no_issue_panel"
      });
      setResult(payload);
      setMessage(resultSummary(payload));
      setNoIssueConfirm("");
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
      <datalist id="match-log-visible-player-options">
        {playerOptions.map((player) => <option key={player.id ?? player.name} value={String(player.id ?? "")} label={player.name} />)}
      </datalist>
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
          <label><strong>Status</strong><br />
            <select value={edit.isActive} onChange={(event) => updateEdit("isActive", event.target.value as "true" | "false")} style={inputStyle}>
              <option value="true">Active</option>
              <option value="false">Inactive</option>
            </select>
          </label>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem" }}>
          <label><strong>Team 1 player 1 ID</strong><br /><input list="match-log-visible-player-options" value={edit.t1p1} onChange={(event) => updateEdit("t1p1", event.target.value)} style={inputStyle} /></label>
          <label><strong>Team 1 player 2 ID</strong><br /><input list="match-log-visible-player-options" value={edit.t1p2} onChange={(event) => updateEdit("t1p2", event.target.value)} style={inputStyle} /></label>
          <label><strong>Team 2 player 1 ID</strong><br /><input list="match-log-visible-player-options" value={edit.t2p1} onChange={(event) => updateEdit("t2p1", event.target.value)} style={inputStyle} /></label>
          <label><strong>Team 2 player 2 ID</strong><br /><input list="match-log-visible-player-options" value={edit.t2p2} onChange={(event) => updateEdit("t2p2", event.target.value)} style={inputStyle} /></label>
        </div>

        <p style={{ color: "#64748b", margin: 0 }}>
          Player fields accept IDs and offer visible-player autocomplete. A full roster selector can be added next once the admin API exposes roster options to Next.
        </p>

        <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: 0 }}>
          <button type="button" onClick={stageGuidedEdit} disabled={busy || !selectedMatch} style={buttonStyle}>Stage guided edit</button>
          <button type="button" onClick={() => selectedMatch ? setEdit(editStateFromMatch(selectedMatch)) : undefined} disabled={busy || !selectedMatch} style={secondaryButtonStyle}>Reset selected fields</button>
        </p>
      </div>

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
          </>
        ) : <p style={{ color: "#475569" }}>No edits staged yet.</p>}
        <label><strong>Correction note</strong><br /><input value={correctionNote} onChange={(event) => setCorrectionNote(event.target.value)} style={inputStyle} /></label>
        <label><strong>Type APPLY to confirm edits</strong><br /><input value={applyConfirm} onChange={(event) => setApplyConfirm(event.target.value)} style={inputStyle} /></label>
        <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <button type="button" onClick={submitGuidedPatches} disabled={busy || !accessToken || applyConfirm.trim().toUpperCase() !== "APPLY" || !stagedPatches.length} style={buttonStyle}>
            {busy ? "Working…" : "Apply staged edits"}
          </button>
          <button type="button" onClick={() => setStagedPatches([])} disabled={busy || !stagedPatches.length} style={secondaryButtonStyle}>Clear staged edits</button>
        </p>
      </div>

      <hr style={{ border: 0, borderTop: "1px solid #e2e8f0", margin: "1rem 0" }} />

      <h3>Duplicate false positive / no issue</h3>
      <p style={{ color: "#475569" }}>
        Use this when the scanner found a legitimate repeated matchup that should no longer appear as an active cleanup candidate.
      </p>
      {duplicateGroups.length ? (
        <div style={{ display: "grid", gap: "0.75rem" }}>
          <label><strong>Reason for no-issue resolution</strong><br /><input value={noIssueReason} onChange={(event) => setNoIssueReason(event.target.value)} style={inputStyle} /></label>
          <label><strong>Type NO ISSUE to confirm false-positive resolution</strong><br /><input value={noIssueConfirm} onChange={(event) => setNoIssueConfirm(event.target.value)} style={inputStyle} /></label>
          {duplicateGroups.map((group) => (
            <div key={group.dup_key} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem", background: "#f8fafc" }}>
              <p style={{ marginTop: 0 }}><strong>{groupLabel(group)}</strong></p>
              <p style={{ color: "#475569" }}>Cleanup candidates: {group.delete_ids.map((id) => `#${id}`).join(", ") || "none"}</p>
              <button
                type="button"
                onClick={() => resolveNoIssue(group)}
                disabled={busy || !accessToken || noIssueConfirm.trim().toUpperCase() !== "NO ISSUE" || !noIssueReason.trim()}
                style={buttonStyle}
              >
                {busy ? "Working…" : "Mark this group no issue"}
              </button>
            </div>
          ))}
        </div>
      ) : (
        <p style={{ color: "#475569" }}>No active duplicate groups are available to resolve as no issue.</p>
      )}

      <hr style={{ border: 0, borderTop: "1px solid #e2e8f0", margin: "1rem 0" }} />

      <h3>Duplicate cleanup</h3>
      <p style={{ color: "#475569" }}>
        Current preview would remove {duplicatePreview?.delete_count ?? 0} duplicate row(s): {(duplicatePreview?.delete_ids ?? []).join(", ") || "none"}.
      </p>
      <label><strong>Type DELETE to confirm duplicate cleanup</strong><br /><input value={cleanupConfirm} onChange={(event) => setCleanupConfirm(event.target.value)} style={inputStyle} /></label>
      <p>
        <button type="button" onClick={cleanupDuplicates} disabled={busy || !accessToken || cleanupConfirm.trim().toUpperCase() !== "DELETE" || !(duplicatePreview?.delete_ids?.length)} style={buttonStyle}>
          {busy ? "Working…" : "Clean duplicate rows from preview"}
        </button>
      </p>

      {message ? <p style={{ color: result?.ok ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      {result?.warnings?.length ? (
        <ul style={{ color: "#92400e", paddingLeft: "1.25rem" }}>
          {result.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      ) : null}
    </article>
  );
}
