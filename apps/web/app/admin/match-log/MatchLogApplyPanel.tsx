"use client";

import Link from "next/link";
import { useState } from "react";
import type { AdminDuplicateDeletePreview, AdminMatchLogWriteResult } from "@/lib/adminMatchLogApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type MatchLogApplyPanelProps = {
  apiBase: string | null;
  clubId: string;
  applyEnabled: boolean;
  duplicatePreview?: AdminDuplicateDeletePreview | null;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function resultSummary(result: AdminMatchLogWriteResult | null): string | null {
  if (!result) return null;
  if (result.mode === "duplicates_cleaned") return `Cleaned ${result.deleted_count ?? 0} duplicate row(s). Replay scope: ${result.recommended_replay_scope ?? "ALL"}.`;
  if (result.mode === "applied") return `Applied ${result.updated_count ?? 0} match edit(s).`;
  return "Operation completed.";
}

export default function MatchLogApplyPanel({ apiBase, clubId, applyEnabled, duplicatePreview }: MatchLogApplyPanelProps) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [patchesJson, setPatchesJson] = useState('[\n  {"id": 123, "week_tag": "Week 1"}\n]');
  const [correctionNote, setCorrectionNote] = useState("");
  const [applyConfirm, setApplyConfirm] = useState("");
  const [cleanupConfirm, setCleanupConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AdminMatchLogWriteResult | null>(null);

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

  async function submitPatches() {
    setBusy(true);
    setMessage(null);
    setResult(null);
    try {
      const parsed = JSON.parse(patchesJson);
      if (!Array.isArray(parsed)) throw new Error("Patches JSON must be an array.");
      const payload = await callApi(`/admin/clubs/${encodeURIComponent(clubId)}/match-log/edits`, "PATCH", {
        patches: parsed,
        confirmation_text: applyConfirm,
        correction_note: correctionNote,
        source: "next_match_log_apply_panel"
      });
      setResult(payload);
      setMessage(resultSummary(payload));
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
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to clean duplicates.");
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
        This panel uses the signed-in admin browser session and calls FastAPI write endpoints with a Supabase access token. It does not write directly from the browser to Supabase.
      </p>

      <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
        <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
        <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
          {accessToken ? "Ready to send authorized Match Log requests." : sessionLoading ? "Checking admin session…" : "Sign in before applying changes or cleaning duplicates."}
        </p>
        {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
      </div>

      <div style={{ display: "grid", gap: "0.75rem" }}>
        <label><strong>Correction note</strong><br /><input value={correctionNote} onChange={(event) => setCorrectionNote(event.target.value)} style={inputStyle} /></label>
        <label><strong>Patch JSON</strong><br /><textarea value={patchesJson} onChange={(event) => setPatchesJson(event.target.value)} rows={7} style={{ ...inputStyle, fontFamily: "monospace" }} /></label>
        <label><strong>Type APPLY to confirm edits</strong><br /><input value={applyConfirm} onChange={(event) => setApplyConfirm(event.target.value)} style={inputStyle} /></label>
        <button type="button" onClick={submitPatches} disabled={busy || !accessToken || applyConfirm.trim().toUpperCase() !== "APPLY"} style={buttonStyle}>
          {busy ? "Working…" : "Apply match edits"}
        </button>
      </div>

      <hr style={{ border: 0, borderTop: "1px solid #e2e8f0", margin: "1rem 0" }} />

      <h3>Duplicate cleanup</h3>
      <p style={{ color: "#475569" }}>
        Current preview would review {duplicatePreview?.delete_count ?? 0} duplicate row(s): {(duplicatePreview?.delete_ids ?? []).join(", ") || "none"}.
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
