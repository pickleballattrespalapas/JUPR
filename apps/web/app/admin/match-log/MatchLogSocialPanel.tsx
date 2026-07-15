"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { AdminSocialMatchLogResponse, AdminSocialMatchLogRow, AdminMatchLogWriteResult } from "@/lib/adminMatchLogApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type MatchLogSocialPanelProps = {
  apiBase: string | null;
  clubId: string;
  enabled: boolean;
};

type SocialEditState = {
  eventName: string;
  playedOn: string;
  roundNumber: string;
  courtNumber: string;
  miniRoundNumber: string;
  scoreT1: string;
  scoreT2: string;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const secondaryButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function rowId(row: AdminSocialMatchLogRow | null): string {
  return String(row?.social_match_id ?? row?.id ?? "");
}

function dateInput(value?: string | null): string {
  if (!value) return "";
  const text = String(value);
  if (/^\d{4}-\d{2}-\d{2}/.test(text)) return text.slice(0, 10);
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString().slice(0, 10);
}

function editFromRow(row: AdminSocialMatchLogRow | null): SocialEditState {
  return {
    eventName: row?.event_name || "",
    playedOn: dateInput(row?.played_on || row?.date),
    roundNumber: row?.round_number == null ? "" : String(row.round_number),
    courtNumber: row?.court_number == null ? "" : String(row.court_number),
    miniRoundNumber: row?.mini_round_number == null ? "" : String(row.mini_round_number),
    scoreT1: row?.score_t1 == null ? "" : String(row.score_t1),
    scoreT2: row?.score_t2 == null ? "" : String(row.score_t2)
  };
}

function maybeNumber(value: string): number | undefined {
  const cleaned = String(value || "").trim();
  if (!cleaned) return undefined;
  const parsed = Number(cleaned);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed)) throw new Error("Numeric fields must be whole numbers.");
  return parsed;
}

function buildPatch(edit: SocialEditState): Record<string, unknown> {
  const patch: Record<string, unknown> = {};
  if (edit.eventName.trim()) patch.event_name = edit.eventName.trim();
  if (edit.playedOn) patch.played_on = edit.playedOn;
  const roundNumber = maybeNumber(edit.roundNumber);
  const courtNumber = maybeNumber(edit.courtNumber);
  const miniRoundNumber = maybeNumber(edit.miniRoundNumber);
  const scoreT1 = maybeNumber(edit.scoreT1);
  const scoreT2 = maybeNumber(edit.scoreT2);
  if (roundNumber !== undefined) patch.round_number = roundNumber;
  if (courtNumber !== undefined) patch.court_number = courtNumber;
  if (miniRoundNumber !== undefined) patch.mini_round_number = miniRoundNumber;
  if (scoreT1 !== undefined) patch.score_t1 = scoreT1;
  if (scoreT2 !== undefined) patch.score_t2 = scoreT2;
  return patch;
}

function resultMessage(result: AdminMatchLogWriteResult | null): string | null {
  if (!result?.ok) return null;
  if (result.mode === "social_match_updated") return `Updated Club Social match ${result.social_match_id || "row"}.`;
  if (result.mode === "social_matches_deleted") return `Deleted ${result.deleted_count ?? 0} Club Social row(s).`;
  return "Operation completed.";
}

function playerLabel(row: AdminSocialMatchLogRow): string {
  return `${row.t1_p1 || "—"} / ${row.t1_p2 || "—"} vs ${row.t2_p1 || "—"} / ${row.t2_p2 || "—"}`;
}

export default function MatchLogSocialPanel({ apiBase, clubId, enabled }: MatchLogSocialPanelProps) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [rows, setRows] = useState<AdminSocialMatchLogRow[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [edit, setEdit] = useState<SocialEditState>(() => editFromRow(null));
  const [deleteConfirm, setDeleteConfirm] = useState("");
  const [loadingRows, setLoadingRows] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [result, setResult] = useState<AdminMatchLogWriteResult | null>(null);
  const selectedRow = rows.find((row) => rowId(row) === selectedId) || null;

  async function loadRows() {
    setMessage(null);
    setWarnings([]);
    if (!apiBase) {
      setMessage("API base URL is not configured.");
      return;
    }
    if (!accessToken) {
      setRows([]);
      return;
    }
    setLoadingRows(true);
    try {
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/match-log/social?limit=500`), {
        cache: "no-store",
        headers: { accept: "application/json", Authorization: `Bearer ${accessToken}` }
      });
      const payload = await response.json().catch(() => null) as AdminSocialMatchLogResponse | { detail?: unknown } | null;
      if (!response.ok) throw new Error(String((payload as { detail?: unknown } | null)?.detail || `API error (${response.status})`));
      const nextRows = Array.isArray((payload as AdminSocialMatchLogResponse).rows) ? (payload as AdminSocialMatchLogResponse).rows : [];
      setRows(nextRows);
      setWarnings((payload as AdminSocialMatchLogResponse).warnings || []);
      const first = nextRows[0] || null;
      setSelectedId(rowId(first));
      setEdit(editFromRow(first));
      if (!nextRows.length && !((payload as AdminSocialMatchLogResponse).warnings || []).length) setMessage("No Club Social rows found.");
    } catch (error) {
      setRows([]);
      setMessage(error instanceof Error ? error.message : "Unable to load Club Social rows.");
    } finally {
      setLoadingRows(false);
    }
  }

  useEffect(() => {
    if (enabled && accessToken) void loadRows();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, accessToken, apiBase, clubId]);

  function selectRow(nextId: string) {
    const row = rows.find((item) => rowId(item) === nextId) || null;
    setSelectedId(nextId);
    setEdit(editFromRow(row));
    setDeleteConfirm("");
    setMessage(null);
    setResult(null);
  }

  async function saveRow() {
    if (!selectedRow) return;
    setBusy(true);
    setMessage(null);
    setResult(null);
    try {
      if (!apiBase) throw new Error("API base URL is not configured.");
      if (!accessToken) throw new Error("Sign in at /admin/login before editing Club Social rows.");
      const patch = buildPatch(edit);
      if (!Object.keys(patch).length) throw new Error("No Club Social changes detected.");
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/match-log/social/${encodeURIComponent(rowId(selectedRow))}`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}` },
        body: JSON.stringify({ ...patch, source: "next_match_log_social_editor" })
      });
      const payload = await response.json().catch(() => null) as AdminMatchLogWriteResult | { detail?: unknown } | null;
      if (!response.ok) throw new Error(String((payload as { detail?: unknown } | null)?.detail || `API error (${response.status})`));
      setResult(payload as AdminMatchLogWriteResult);
      setMessage(resultMessage(payload as AdminMatchLogWriteResult));
      await loadRows();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save Club Social row.");
    } finally {
      setBusy(false);
    }
  }

  async function deleteRow() {
    if (!selectedRow) return;
    setBusy(true);
    setMessage(null);
    setResult(null);
    try {
      if (!apiBase) throw new Error("API base URL is not configured.");
      if (!accessToken) throw new Error("Sign in at /admin/login before deleting Club Social rows.");
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/match-log/social/delete`), {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}` },
        body: JSON.stringify({ social_match_ids: [rowId(selectedRow)], confirmation_text: deleteConfirm, source: "next_match_log_social_editor" })
      });
      const payload = await response.json().catch(() => null) as AdminMatchLogWriteResult | { detail?: unknown } | null;
      if (!response.ok) throw new Error(String((payload as { detail?: unknown } | null)?.detail || `API error (${response.status})`));
      setResult(payload as AdminMatchLogWriteResult);
      setMessage(resultMessage(payload as AdminMatchLogWriteResult));
      setDeleteConfirm("");
      await loadRows();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to delete Club Social row.");
    } finally {
      setBusy(false);
    }
  }

  if (!enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Club Social Match Log editor unavailable</h2>
        <p style={{ color: "#475569" }}>Enable the Next Match Log pilot before editing Club Social rows in Next.</p>
      </article>
    );
  }

  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Club Social editor</h2>
      <p style={{ color: "#475569" }}>
        Edit or delete unrated Club Social rows from the Match Log workflow. Rated match history is not changed by this panel.
      </p>
      <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
        <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
        <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
          {accessToken ? "Ready to load and edit Club Social rows." : sessionLoading ? "Checking admin session…" : "Sign in before editing Club Social rows."}
        </p>
        {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
      </div>
      <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <button type="button" onClick={loadRows} disabled={loadingRows || busy || !accessToken} style={secondaryButtonStyle}>{loadingRows ? "Loading…" : "Refresh Club Social rows"}</button>
      </p>
      {warnings.length ? <ul style={{ color: "#92400e", paddingLeft: "1.25rem" }}>{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
      {rows.length ? (
        <div style={{ display: "grid", gap: "0.75rem" }}>
          <label><strong>Club Social row</strong><br />
            <select value={selectedId} onChange={(event) => selectRow(event.target.value)} style={inputStyle}>
              {rows.map((row) => (
                <option key={rowId(row)} value={rowId(row)}>
                  {row.event_name || "Club Social"} · {row.played_on || row.date || "—"} · {playerLabel(row)} · {row.score_t1 ?? 0}-{row.score_t2 ?? 0}
                </option>
              ))}
            </select>
          </label>
          {selectedRow ? (
            <div style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem", background: "#f8fafc" }}>
              <strong>{selectedRow.event_name || "Club Social"}</strong>
              <p style={{ margin: "0.35rem 0", color: "#475569" }}>{playerLabel(selectedRow)} · {selectedRow.score_t1 ?? 0}-{selectedRow.score_t2 ?? 0}</p>
              <p style={{ margin: 0, color: "#64748b" }}>Status: {selectedRow.status || "—"} · Submission: {selectedRow.submission_mode || "—"} · Match key: {selectedRow.match_key || "—"}</p>
            </div>
          ) : null}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem" }}>
            <label><strong>Event name</strong><br /><input value={edit.eventName} onChange={(event) => setEdit((current) => ({ ...current, eventName: event.target.value }))} style={inputStyle} /></label>
            <label><strong>Played on</strong><br /><input type="date" value={edit.playedOn} onChange={(event) => setEdit((current) => ({ ...current, playedOn: event.target.value }))} style={inputStyle} /></label>
            <label><strong>Round</strong><br /><input type="number" step="1" value={edit.roundNumber} onChange={(event) => setEdit((current) => ({ ...current, roundNumber: event.target.value }))} style={inputStyle} /></label>
            <label><strong>Court</strong><br /><input type="number" step="1" value={edit.courtNumber} onChange={(event) => setEdit((current) => ({ ...current, courtNumber: event.target.value }))} style={inputStyle} /></label>
            <label><strong>Mini round</strong><br /><input type="number" step="1" value={edit.miniRoundNumber} onChange={(event) => setEdit((current) => ({ ...current, miniRoundNumber: event.target.value }))} style={inputStyle} /></label>
            <label><strong>Team 1 score</strong><br /><input type="number" min="0" step="1" value={edit.scoreT1} onChange={(event) => setEdit((current) => ({ ...current, scoreT1: event.target.value }))} style={inputStyle} /></label>
            <label><strong>Team 2 score</strong><br /><input type="number" min="0" step="1" value={edit.scoreT2} onChange={(event) => setEdit((current) => ({ ...current, scoreT2: event.target.value }))} style={inputStyle} /></label>
          </div>
          <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <button type="button" onClick={saveRow} disabled={busy || !accessToken || !selectedRow} style={buttonStyle}>{busy ? "Working…" : "Save Club Social row"}</button>
            <button type="button" onClick={() => setEdit(editFromRow(selectedRow))} disabled={busy || !selectedRow} style={secondaryButtonStyle}>Reset fields</button>
          </p>
          <hr style={{ border: 0, borderTop: "1px solid #e2e8f0", margin: "0.5rem 0" }} />
          <h3>Delete Club Social row</h3>
          <p style={{ color: "#475569" }}>This deletes the selected unrated Club Social row only. It does not replay or change rated history.</p>
          <label><strong>Type DELETE to confirm social row deletion</strong><br /><input value={deleteConfirm} onChange={(event) => setDeleteConfirm(event.target.value)} style={inputStyle} /></label>
          <p><button type="button" onClick={deleteRow} disabled={busy || !accessToken || !selectedRow || deleteConfirm.trim().toUpperCase() !== "DELETE"} style={buttonStyle}>{busy ? "Working…" : "Delete selected Club Social row"}</button></p>
        </div>
      ) : <p style={{ color: "#475569" }}>{loadingRows ? "Loading Club Social rows…" : "No Club Social rows loaded."}</p>}
      {message ? <p style={{ color: result?.ok ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      {result?.warnings?.length ? <ul style={{ color: "#92400e", paddingLeft: "1.25rem" }}>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
      {rows.length ? (
        <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "12px", marginTop: "1rem" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "900px" }}>
            <thead><tr style={{ textAlign: "left", background: "#f8fafc" }}><th style={{ padding: "0.5rem" }}>Event</th><th style={{ padding: "0.5rem" }}>Played</th><th style={{ padding: "0.5rem" }}>Players</th><th style={{ padding: "0.5rem" }}>Score</th><th style={{ padding: "0.5rem" }}>Status</th></tr></thead>
            <tbody>{rows.slice(0, 25).map((row) => <tr key={rowId(row)}><td style={{ padding: "0.5rem" }}>{row.event_name || "—"}</td><td style={{ padding: "0.5rem" }}>{row.played_on || row.date || "—"}</td><td style={{ padding: "0.5rem" }}>{playerLabel(row)}</td><td style={{ padding: "0.5rem" }}>{row.score_t1 ?? 0}-{row.score_t2 ?? 0}</td><td style={{ padding: "0.5rem" }}>{row.status || "—"}</td></tr>)}</tbody>
          </table>
        </div>
      ) : null}
    </article>
  );
}
