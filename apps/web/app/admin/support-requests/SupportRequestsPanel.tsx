"use client";

import { useState } from "react";
import type { AdminSupportRequest, AdminSupportRequestsListResponse, AdminSupportRequestsStatus, AdminSupportRequestUpdateResponse } from "@/lib/adminSupportRequestsApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = { apiBase: string | null; clubId: string; status: AdminSupportRequestsStatus };

type RequestEdit = { status: string; adminNote: string; confirm: string };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const STATUS_OPTIONS = ["new", "in_review", "resolved", "dismissed"];
const TYPE_OPTIONS = ["", "data_correction", "profile_privacy", "general_support"];

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function typeLabel(value: string): string {
  return value.replace(/_/g, " ") || "all types";
}

function statusStyle(value: string) {
  if (value === "resolved") return { background: "#dcfce7", borderColor: "#bbf7d0" };
  if (value === "dismissed") return { background: "#f1f5f9", borderColor: "#cbd5e1" };
  if (value === "in_review") return { background: "#dbeafe", borderColor: "#bfdbfe" };
  return { background: "#fef3c7", borderColor: "#fde68a" };
}

function SupportRequestCard({ request, selected, onSelect }: { request: AdminSupportRequest; selected: boolean; onSelect: () => void }) {
  return (
    <article style={{ ...cardStyle, borderColor: selected ? "#2563eb" : "#e2e8f0" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "flex-start", flexWrap: "wrap" }}>
        <div>
          <h3 style={{ margin: "0 0 0.25rem", fontSize: "1rem" }}>{request.subject || "Untitled request"}</h3>
          <p style={{ margin: 0, color: "#64748b" }}>{typeLabel(request.request_type)} · {request.created_at ? String(request.created_at).slice(0, 16) : "no date"}</p>
        </div>
        <span style={{ border: "1px solid", borderRadius: "999px", padding: "0.12rem 0.5rem", fontSize: "0.78rem", ...statusStyle(request.status) }}>{request.status.replace(/_/g, " ")}</span>
      </div>
      <p style={{ color: "#334155" }}>{request.description}</p>
      <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.5rem", color: "#475569" }}>
        <div><dt style={{ fontWeight: 800 }}>Requester</dt><dd style={{ margin: 0 }}>{request.requester_name}<br />{request.requester_email}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Player</dt><dd style={{ margin: 0 }}>{request.player_name || "—"}{request.player_id ? ` (#${request.player_id})` : ""}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Match / Tournament</dt><dd style={{ margin: 0 }}>{request.match_id || "—"} / {request.tournament_id || "—"}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Reviewed</dt><dd style={{ margin: 0 }}>{request.reviewed_by || "—"}<br />{request.reviewed_at ? String(request.reviewed_at).slice(0, 16) : ""}</dd></div>
      </dl>
      {request.requested_action ? <p style={{ color: "#475569" }}><strong>Requested action:</strong> {request.requested_action}</p> : null}
      {request.evidence_url ? <p><a href={request.evidence_url} target="_blank" rel="noreferrer">Open evidence link</a></p> : null}
      {request.admin_note ? <p style={{ color: "#475569" }}><strong>Admin note:</strong> {request.admin_note}</p> : null}
      <button type="button" onClick={onSelect} style={selected ? buttonStyle : ghostButtonStyle}>{selected ? "Selected" : "Review request"}</button>
    </article>
  );
}

export default function SupportRequestsPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [statusFilter, setStatusFilter] = useState("new");
  const [typeFilter, setTypeFilter] = useState("");
  const [requests, setRequests] = useState<AdminSupportRequest[]>([]);
  const [summary, setSummary] = useState<AdminSupportRequestsListResponse["summary"] | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [edit, setEdit] = useState<RequestEdit>({ status: "in_review", adminNote: "", confirm: "" });
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const selected = requests.find((request) => request.id === selectedId) || null;

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Support Requests.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function loadRequests() {
    setBusy(true);
    setMessage(null);
    try {
      const params = new URLSearchParams();
      if (statusFilter) params.set("status", statusFilter);
      if (typeFilter) params.set("request_type", typeFilter);
      const payload = await requestJson<AdminSupportRequestsListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/support-requests${params.toString() ? `?${params.toString()}` : ""}`);
      setRequests(payload.requests || []);
      setSummary(payload.summary || null);
      setSelectedId("");
      setMessage(`Loaded ${payload.requests?.length ?? 0} request(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load support requests.");
    } finally {
      setBusy(false);
    }
  }

  function selectRequest(request: AdminSupportRequest) {
    setSelectedId(request.id);
    setEdit({ status: request.status === "new" ? "in_review" : request.status, adminNote: request.admin_note || "", confirm: "" });
    setMessage(null);
  }

  async function saveStatus() {
    if (!selected) {
      setMessage("Select a request before saving.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminSupportRequestUpdateResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/support-requests/${encodeURIComponent(selected.id)}`, {
        method: "PATCH",
        body: JSON.stringify({ status: edit.status, admin_note: edit.adminNote, confirmation_text: edit.confirm, source: "next_admin_support_requests" })
      });
      setRequests((current) => current.map((request) => request.id === selected.id ? payload.request : request));
      setSelectedId(payload.request.id);
      setEdit({ status: payload.request.status, adminNote: payload.request.admin_note || "", confirm: "" });
      setMessage(`Request updated to ${payload.request.status}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to update request.");
    } finally {
      setBusy(false);
    }
  }

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Support Requests is disabled</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the support request review flag on FastAPI."}</p>
      </article>
    );
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2>
        <p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p style={{ color: "#64748b" }}>{sessionMessage}</p> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Filters</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
          <label>Status<br /><select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)} style={inputStyle}><option value="">all statuses</option>{STATUS_OPTIONS.map((option) => <option key={option} value={option}>{option.replace(/_/g, " ")}</option>)}</select></label>
          <label>Type<br /><select value={typeFilter} onChange={(event) => setTypeFilter(event.target.value)} style={inputStyle}>{TYPE_OPTIONS.map((option) => <option key={option || "all"} value={option}>{typeLabel(option)}</option>)}</select></label>
        </div>
        <button type="button" onClick={loadRequests} disabled={busy || !accessToken} style={{ ...buttonStyle, marginTop: "0.75rem" }}>{busy ? "Loading…" : "Load requests"}</button>
        {summary ? <p style={{ color: "#475569" }}>Loaded {summary.total} · statuses {JSON.stringify(summary.by_status)} · types {JSON.stringify(summary.by_type)}</p> : null}
      </article>

      {message ? <p style={{ color: message.toLowerCase().includes("error") || message.toLowerCase().includes("unable") ? "#b91c1c" : "#166534" }}>{message}</p> : null}

      {selected ? (
        <article style={{ ...cardStyle, borderColor: "#2563eb" }}>
          <h2 style={{ marginTop: 0 }}>Review selected request</h2>
          <p style={{ color: "#475569" }}>Use this panel only to track review state. Apply actual corrections through Match Log, Player Editor, Tournament Admin, or Replay History.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
            <label>Status<br /><select value={edit.status} onChange={(event) => setEdit((current) => ({ ...current, status: event.target.value }))} style={inputStyle}>{STATUS_OPTIONS.map((option) => <option key={option} value={option}>{option.replace(/_/g, " ")}</option>)}</select></label>
            <label>Confirmation<br /><input value={edit.confirm} onChange={(event) => setEdit((current) => ({ ...current, confirm: event.target.value }))} placeholder="SAVE REQUEST STATUS" style={inputStyle} /></label>
          </div>
          <label>Admin note<br /><textarea value={edit.adminNote} onChange={(event) => setEdit((current) => ({ ...current, adminNote: event.target.value }))} rows={4} style={inputStyle} /></label>
          <button type="button" onClick={saveStatus} disabled={busy} style={{ ...buttonStyle, marginTop: "0.75rem" }}>{busy ? "Saving…" : "Save request status"}</button>
        </article>
      ) : null}

      <div style={{ display: "grid", gap: "0.75rem" }}>
        {requests.map((request) => <SupportRequestCard key={request.id} request={request} selected={request.id === selectedId} onSelect={() => selectRequest(request)} />)}
      </div>
    </div>
  );
}
