"use client";

import { useEffect, useMemo, useState } from "react";
import type {
  AdminPlayerUpdatesStatusResponse,
  CommunicationsActionResponse,
  CommunicationsOutboxRow,
  CommunicationsSubscription,
  CommunicationsWorkspaceResponse
} from "@/lib/adminPlayerUpdatesApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";
import { clubDaysAgoIso, clubTodayIso } from "@/lib/clubDate";

type Props = { apiBase: string | null; clubId: string; status: AdminPlayerUpdatesStatusResponse };
type RowRef = { id: string; expected_row_version: number };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const dangerButtonStyle = { ...buttonStyle, background: "#991b1b", borderColor: "#991b1b" };

function apiUrl(base: string, path: string): string { return `${base.replace(/\/$/, "")}${path}`; }
function operationKey(): string { return crypto.randomUUID(); }
function rowLabel(row: CommunicationsOutboxRow): string {
  return `${row.player_name || `Player #${row.player_id}`} · ${row.week_start} → ${row.week_end} · ${row.send_status}`;
}

export default function PlayerUpdatesPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [startDate, setStartDate] = useState(() => clubDaysAgoIso(7));
  const [endDate, setEndDate] = useState(() => clubTodayIso());
  const [workspace, setWorkspace] = useState<CommunicationsWorkspaceResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [messageSeverity, setMessageSeverity] = useState<"success" | "error" | null>(null);
  const [selectedOutbox, setSelectedOutbox] = useState<string[]>([]);
  const [outboxFilter, setOutboxFilter] = useState("all");
  const [queuePlayerId, setQueuePlayerId] = useState("");
  const [onlyMatches, setOnlyMatches] = useState(true);
  const [queueConfirm, setQueueConfirm] = useState("");
  const [sendConfirm, setSendConfirm] = useState("");
  const [retryConfirm, setRetryConfirm] = useState("");
  const [deleteConfirm, setDeleteConfirm] = useState("");
  const [preview, setPreview] = useState<Record<string, unknown> | null>(null);
  const [selectedSubscriptionId, setSelectedSubscriptionId] = useState("");
  const [replacementEmail, setReplacementEmail] = useState("");
  const [replacementNote, setReplacementNote] = useState("");
  const [replacementConfirm, setReplacementConfirm] = useState("");
  const [deactivateConfirm, setDeactivateConfirm] = useState("");
  const [queueOperationKey, setQueueOperationKey] = useState("");
  const [sendOperationKey, setSendOperationKey] = useState("");
  const [replacementOperationKey, setReplacementOperationKey] = useState("");

  const activeSubscriptions = useMemo(
    () => (workspace?.subscriptions || []).filter((row) => row.request_status === "active"),
    [workspace]
  );
  const selectedSubscription = activeSubscriptions.find((row) => row.id === selectedSubscriptionId) || null;
  const visibleOutbox = useMemo(
    () => (workspace?.outbox || []).filter((row) => outboxFilter === "all" || row.send_status === outboxFilter),
    [workspace, outboxFilter]
  );
  const selectedRows = (workspace?.outbox || []).filter((row) => selectedOutbox.includes(row.id));
  const includesUncertainSending = selectedRows.some((row) => row.send_status === "sending");

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Player Updates Admin.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null) as { detail?: unknown } | null;
    if (!response.ok) {
      if (response.status === 409) throw new Error(`${String(payload?.detail || "This data changed.")} Reload the workspace before trying again.`);
      throw new Error(String(payload?.detail || `API error (${response.status})`));
    }
    return payload as T;
  }

  async function loadWorkspace(silent = false) {
    if (!accessToken || !status.enabled) return;
    if (!silent) { setBusy(true); setMessage(null); setMessageSeverity(null); }
    try {
      const query = new URLSearchParams({ start_date: startDate, end_date: endDate, limit: "1000" });
      const payload = await requestJson<CommunicationsWorkspaceResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/player-updates/workspace?${query}`);
      setWorkspace(payload);
      setSelectedOutbox((current) => current.filter((id) => payload.outbox.some((row) => row.id === id)));
      if (!selectedSubscriptionId && payload.subscriptions.some((row) => row.request_status === "active")) {
        setSelectedSubscriptionId(payload.subscriptions.find((row) => row.request_status === "active")?.id || "");
      }
      if (!silent) { setMessage(`Loaded ${payload.subscriptions.length} subscriptions, ${payload.digests.length} digests, and ${payload.outbox.length} outbox rows.`); setMessageSeverity("success"); }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load communications workspace.");
      setMessageSeverity("error");
    } finally {
      if (!silent) setBusy(false);
    }
  }

  useEffect(() => {
    if (accessToken && status.enabled) void loadWorkspace(true);
    // The operator controls date-range reloading explicitly after initial load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken, status.enabled]);

  function refs(rows: CommunicationsOutboxRow[]): RowRef[] {
    return rows.map((row) => ({ id: row.id, expected_row_version: Number(row.row_version || 1) }));
  }

  async function runAction(path: string, body: Record<string, unknown>, success: (result: CommunicationsActionResponse) => string) {
    setBusy(true); setMessage(null); setMessageSeverity(null);
    try {
      const result = await requestJson<CommunicationsActionResponse>(path, { method: "POST", body: JSON.stringify(body) });
      setMessage(success(result));
      setMessageSeverity(Number(result.errors || 0) + Number(result.stale || 0) + Number(result.uncertain || 0) > 0 ? "error" : "success");
      setSelectedOutbox([]);
      await loadWorkspace(true);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to complete communications action.");
      setMessageSeverity("error");
    } finally { setBusy(false); }
  }

  async function previewDigest() {
    if (!queuePlayerId) { setMessage("Enter an active player ID to preview a digest."); setMessageSeverity("error"); return; }
    setBusy(true); setMessage(null); setMessageSeverity(null);
    try {
      const payload = await requestJson<{ digest: Record<string, unknown> }>(`/admin/clubs/${encodeURIComponent(clubId)}/player-updates/digests/preview`, {
        method: "POST",
        body: JSON.stringify({ player_id: Number(queuePlayerId), start_date: startDate, end_date: endDate })
      });
      setPreview(payload.digest);
      setMessage("Preview generated without saving or queueing anything.");
      setMessageSeverity("success");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to preview digest."); setMessageSeverity("error"); }
    finally { setBusy(false); }
  }

  async function queueDigests() {
    const key = queueOperationKey || operationKey();
    setQueueOperationKey(key);
    await runAction(
      `/admin/clubs/${encodeURIComponent(clubId)}/player-updates/digests/queue`,
      { start_date: startDate, end_date: endDate, player_id: queuePlayerId ? Number(queuePlayerId) : null, only_players_with_matches: onlyMatches, confirmation_text: queueConfirm, operation_key: key, source: "next_player_updates_queue" },
      (result) => { setQueueConfirm(""); setQueueOperationKey(""); return `Queue operation ${String(result.operation_key || key)} completed.`; }
    );
  }

  async function sendSelected() {
    const pending = selectedRows.filter((row) => row.send_status === "pending");
    const key = sendOperationKey || operationKey();
    setSendOperationKey(key);
    await runAction(
      `/admin/clubs/${encodeURIComponent(clubId)}/player-updates/outbox/send`,
      { items: refs(pending), confirmation_text: sendConfirm, operation_key: key, source: "next_player_updates_outbox_send" },
      (result) => { setSendConfirm(""); setSendOperationKey(""); return `Attempted ${String(result.attempted || 0)}; sent ${String(result.sent || 0)}, skipped ${String(result.skipped || 0)}, errors ${String(result.errors || 0)}, stale ${String(result.stale || 0)}, uncertain ${String(result.uncertain || 0)}.`; }
    );
  }

  async function retrySelected() {
    const retryable = selectedRows.filter((row) => row.send_status === "error" || row.send_status === "sending");
    await runAction(
      `/admin/clubs/${encodeURIComponent(clubId)}/player-updates/outbox/retry`,
      { items: refs(retryable), confirmation_text: retryConfirm, source: "next_player_updates_outbox_retry" },
      (result) => { setRetryConfirm(""); return `Reset ${String(result.reset_to_pending || 0)} row(s) to pending; stale ${String(result.stale || 0)}.`; }
    );
  }

  async function deleteSelected() {
    const pending = selectedRows.filter((row) => row.send_status === "pending");
    await runAction(
      `/admin/clubs/${encodeURIComponent(clubId)}/player-updates/outbox/delete`,
      { items: refs(pending), confirmation_text: deleteConfirm, source: "next_player_updates_outbox_delete" },
      (result) => { setDeleteConfirm(""); return `Deleted ${String(result.deleted || 0)} pending row(s); stale ${String(result.stale || 0)}.`; }
    );
  }

  async function replaceSubscriber() {
    if (!selectedSubscription) { setMessage("Select an active subscription first."); setMessageSeverity("error"); return; }
    const key = replacementOperationKey || operationKey();
    setReplacementOperationKey(key);
    await runAction(
      `/admin/clubs/${encodeURIComponent(clubId)}/player-updates/subscriptions/${encodeURIComponent(selectedSubscription.id)}/replace`,
      { expected_row_version: selectedSubscription.row_version, new_email: replacementEmail, request_note: replacementNote, confirmation_text: replacementConfirm, operation_key: key, source: "next_player_updates_replace" },
      () => { setReplacementEmail(""); setReplacementNote(""); setReplacementConfirm(""); setReplacementOperationKey(""); return "Verified subscriber replaced atomically; the prior row remains in history as unsubscribed."; }
    );
  }

  async function deactivateSubscriber() {
    if (!selectedSubscription) { setMessage("Select an active subscription first."); setMessageSeverity("error"); return; }
    await runAction(
      `/admin/clubs/${encodeURIComponent(clubId)}/player-updates/subscriptions/${encodeURIComponent(selectedSubscription.id)}/deactivate`,
      { expected_row_version: selectedSubscription.row_version, confirmation_text: deactivateConfirm, source: "next_player_updates_deactivate" },
      () => { setDeactivateConfirm(""); return "Verified subscription deactivated; delivery history was retained."; }
    );
  }

  if (!status.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2>Player Updates Admin is disabled</h2><p>{status.warnings?.[0]}</p></article>;

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: accessToken ? "#f0fdf4" : "#fffbeb" }}>
        <h2 style={{ marginTop: 0 }}>Admin session and delivery guard</h2>
        <p><strong>{accessToken ? adminSessionLabel(session) : "Admin session required"}</strong></p>
        <p style={{ color: "#475569" }}>Database access stays in FastAPI with the Supabase service role. Browser code receives only these authenticated API projections.</p>
        {sessionLoading ? <p>Checking session…</p> : null}{sessionMessage ? <p style={{ color: "#b91c1c" }}>{sessionMessage}</p> : null}
        {status.warnings?.length ? <ul style={{ color: "#92400e" }}>{status.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>1. Load exact-range workspace</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Start date<br /><input type="date" value={startDate} onChange={(event) => { setStartDate(event.target.value); setQueueOperationKey(""); }} style={inputStyle} /></label>
          <label>End date<br /><input type="date" value={endDate} onChange={(event) => { setEndDate(event.target.value); setQueueOperationKey(""); }} style={inputStyle} /></label>
          <button type="button" onClick={() => loadWorkspace()} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Working…" : "Reload workspace"}</button>
        </div>
        {workspace ? <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.75rem", marginTop: "1rem" }}>
          <div><strong>Active</strong><br />{workspace.subscription_counts.active || 0}</div><div><strong>Pending review</strong><br />{workspace.subscription_counts.pending_admin_review || 0}</div><div><strong>Digests</strong><br />{workspace.digests.length}</div><div><strong>Pending mail</strong><br />{workspace.outbox_counts.pending || 0}</div><div><strong>Sending</strong><br />{workspace.outbox_counts.sending || 0}</div><div><strong>Errors</strong><br />{workspace.outbox_counts.error || 0}</div>
        </div> : null}
        {message ? <p role={messageSeverity === "error" ? "alert" : "status"} style={{ color: messageSeverity === "error" ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>2. Preview or queue digests</h2>
        <p style={{ color: "#475569" }}>Preview is read-only. Queueing persists the digest and an idempotent outbox row; it does not send email.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Player ID (blank queues eligible active subscriptions)<br /><input value={queuePlayerId} onChange={(event) => { setQueuePlayerId(event.target.value.replace(/\D/g, "")); setQueueOperationKey(""); }} style={inputStyle} /></label>
          <label><input type="checkbox" checked={onlyMatches} onChange={(event) => { setOnlyMatches(event.target.checked); setQueueOperationKey(""); }} /> Only players with matches in range</label>
          <button type="button" onClick={previewDigest} disabled={busy || !queuePlayerId} style={ghostButtonStyle}>Preview one player</button>
          <label>Type QUEUE PLAYER UPDATES<br /><input value={queueConfirm} onChange={(event) => setQueueConfirm(event.target.value)} style={inputStyle} /></label>
          <button type="button" onClick={queueDigests} disabled={busy || queueConfirm.trim().toUpperCase() !== "QUEUE PLAYER UPDATES"} style={buttonStyle}>Queue digests</button>
        </div>
        {preview ? <details open style={{ marginTop: "1rem" }}><summary><strong>Read-only digest preview</strong></summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", background: "#f8fafc", padding: "0.75rem" }}>{JSON.stringify(preview, null, 2)}</pre></details> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>3. Active subscriptions and replacement history</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Active subscription<br /><select value={selectedSubscriptionId} onChange={(event) => setSelectedSubscriptionId(event.target.value)} style={inputStyle}><option value="">Select…</option>{activeSubscriptions.map((row) => <option key={row.id} value={row.id}>{row.player_name || `Player #${row.player_id}`} · {row.email}</option>)}</select></label>
          <label>Replacement email<br /><input type="email" required maxLength={320} value={replacementEmail} onChange={(event) => { setReplacementEmail(event.target.value); setReplacementOperationKey(""); }} style={inputStyle} /></label>
          <label>Replacement note<br /><input value={replacementNote} onChange={(event) => { setReplacementNote(event.target.value); setReplacementOperationKey(""); }} style={inputStyle} /></label>
          <label>Type REPLACE VERIFIED SUBSCRIBER<br /><input value={replacementConfirm} onChange={(event) => setReplacementConfirm(event.target.value)} style={inputStyle} /></label>
          <button type="button" onClick={replaceSubscriber} disabled={busy || !selectedSubscription || !replacementEmail || replacementConfirm.trim().toUpperCase() !== "REPLACE VERIFIED SUBSCRIBER"} style={buttonStyle}>Replace atomically</button>
          <label>Type UNSUBSCRIBE VERIFIED SUBSCRIBER<br /><input value={deactivateConfirm} onChange={(event) => setDeactivateConfirm(event.target.value)} style={inputStyle} /></label>
          <button type="button" onClick={deactivateSubscriber} disabled={busy || !selectedSubscription || deactivateConfirm.trim().toUpperCase() !== "UNSUBSCRIBE VERIFIED SUBSCRIBER"} style={dangerButtonStyle}>Deactivate</button>
        </div>
        <details style={{ marginTop: "1rem" }}><summary><strong>Subscription history ({workspace?.subscriptions.length || 0})</strong></summary><div style={{ overflowX: "auto" }}><table><thead><tr><th>Player</th><th>Email</th><th>Status</th><th>Verified</th><th>Updated</th><th>Version</th></tr></thead><tbody>{(workspace?.subscriptions || []).map((row: CommunicationsSubscription) => <tr key={row.id}><td>{row.player_name || row.player_id}</td><td>{row.email}</td><td>{row.request_status}</td><td>{row.verified_at?.slice(0, 16) || "—"}</td><td>{row.updated_at?.slice(0, 16) || "—"}</td><td>{row.row_version || 1}</td></tr>)}</tbody></table></div></details>
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>4. Outbox queue and immutable delivery history</h2>
        <p style={{ color: "#475569" }}>Sending claims pending rows first. A row left in <code>sending</code> has uncertain delivery and uses a stronger retry phrase to prevent accidental duplicates.</p>
        <label>Status filter<br /><select value={outboxFilter} onChange={(event) => setOutboxFilter(event.target.value)} style={{ ...inputStyle, maxWidth: "240px" }}><option value="all">All history</option><option value="pending">Pending</option><option value="sending">Sending / uncertain</option><option value="sent">Sent</option><option value="skipped">Skipped</option><option value="error">Error</option></select></label>
        <div style={{ display: "grid", gap: "0.45rem", marginTop: "0.75rem", maxHeight: "420px", overflow: "auto" }}>
          {visibleOutbox.map((row) => <label key={row.id} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.65rem", background: selectedOutbox.includes(row.id) ? "#eff6ff" : "white" }}><input type="checkbox" checked={selectedOutbox.includes(row.id)} onChange={(event) => setSelectedOutbox((current) => event.target.checked ? [...new Set([...current, row.id])] : current.filter((id) => id !== row.id))} /> <strong>{rowLabel(row)}</strong><br /><span style={{ color: "#64748b", marginLeft: "1.25rem" }}>{row.email} · attempts {row.attempt_count || 0} · mode {row.delivery_mode || "—"}{row.error_text ? ` · ${row.error_text}` : ""}</span></label>)}
          {!visibleOutbox.length ? <p style={{ color: "#64748b" }}>No outbox rows match this filter.</p> : null}
        </div>
        <p><strong>Selected:</strong> {selectedRows.length} · pending {selectedRows.filter((row) => row.send_status === "pending").length} · retryable {selectedRows.filter((row) => row.send_status === "error" || row.send_status === "sending").length}</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Type SEND PLAYER UPDATES<br /><input value={sendConfirm} onChange={(event) => setSendConfirm(event.target.value)} style={inputStyle} /></label><button type="button" onClick={sendSelected} disabled={busy || !selectedRows.some((row) => row.send_status === "pending") || sendConfirm.trim().toUpperCase() !== "SEND PLAYER UPDATES"} style={buttonStyle}>Send selected pending</button>
          <label>Type {includesUncertainSending ? "RETRY UNCERTAIN EMAILS" : "RETRY PLAYER UPDATES"}<br /><input value={retryConfirm} onChange={(event) => setRetryConfirm(event.target.value)} style={inputStyle} /></label><button type="button" onClick={retrySelected} disabled={busy || !selectedRows.some((row) => row.send_status === "error" || row.send_status === "sending") || retryConfirm.trim().toUpperCase() !== (includesUncertainSending ? "RETRY UNCERTAIN EMAILS" : "RETRY PLAYER UPDATES")} style={ghostButtonStyle}>Reset selected to pending</button>
          <label>Type DELETE QUEUED UPDATES<br /><input value={deleteConfirm} onChange={(event) => setDeleteConfirm(event.target.value)} style={inputStyle} /></label><button type="button" onClick={deleteSelected} disabled={busy || !selectedRows.some((row) => row.send_status === "pending") || deleteConfirm.trim().toUpperCase() !== "DELETE QUEUED UPDATES"} style={dangerButtonStyle}>Delete selected pending</button>
        </div>
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Saved digest history</h2>
        <div style={{ overflowX: "auto" }}><table><thead><tr><th>Player</th><th>Range</th><th>Updated</th><th>Version</th></tr></thead><tbody>{(workspace?.digests || []).map((row) => <tr key={row.id}><td>{row.player_name || row.player_id}</td><td>{row.week_start} → {row.week_end}</td><td>{row.updated_at?.slice(0, 16) || "—"}</td><td>{row.row_version || 1}</td></tr>)}</tbody></table></div>
      </article>
    </section>
  );
}
