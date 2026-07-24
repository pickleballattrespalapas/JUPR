"use client";

import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type { AdminSupportRequest, AdminSupportRequestsListResponse, AdminSupportRequestsStatus, AdminSupportRequestUpdateResponse } from "@/lib/adminSupportRequestsApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = { apiBase: string | null; clubId: string; status: AdminSupportRequestsStatus };

type RequestEdit = {
  status: string;
  adminNote: string;
  identityStatus: string;
  fulfillmentStatus: string;
  resolutionAction: string;
  resolutionEvidence: string;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const STATUS_OPTIONS = ["new", "in_review", "resolved", "dismissed"];
const TYPE_OPTIONS = ["", "data_correction", "profile_privacy", "general_support"];
const IDENTITY_OPTIONS = ["pending", "verified", "rejected"];
const FULFILLMENT_OPTIONS = ["pending", "in_progress", "completed", "declined"];
const RESOLUTION_OPTIONS = ["none", "alias", "hide", "anonymize", "contact_update", "other"];

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

function SupportRequestCard({ request, selected, disabled, onSelect }: { request: AdminSupportRequest; selected: boolean; disabled: boolean; onSelect: () => void }) {
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
      {request.request_type === "profile_privacy" ? (
        <p style={{ color: "#475569" }}>
          <strong>Identity:</strong> {(request.identity_status || "not_required").replace(/_/g, " ")} · <strong>Fulfillment:</strong> {(request.fulfillment_status || "not_required").replace(/_/g, " ")} · <strong>Action:</strong> {(request.resolution_action || "none").replace(/_/g, " ")}
        </p>
      ) : null}
      {request.evidence_url ? <p><a href={request.evidence_url} target="_blank" rel="noreferrer">Open evidence link</a></p> : null}
      {request.admin_note ? <p style={{ color: "#475569" }}><strong>Admin note:</strong> {request.admin_note}</p> : null}
      <button type="button" onClick={onSelect} disabled={disabled} style={selected ? buttonStyle : ghostButtonStyle}>{selected ? "Selected" : "Review request"}</button>
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
  const [edit, setEdit] = useState<RequestEdit>({
    status: "in_review",
    adminNote: "",
    identityStatus: "pending",
    fulfillmentStatus: "pending",
    resolutionAction: "none",
    resolutionEvidence: ""
  });
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const requestsRequest = useLatestRequestGuard(accessToken, clearProtectedSupportRequests);
  const selected = requests.find((request) => request.id === selectedId) || null;

  function resetRequestEdit() {
    setEdit({
      status: "in_review",
      adminNote: "",
      identityStatus: "pending",
      fulfillmentStatus: "pending",
      resolutionAction: "none",
      resolutionEvidence: ""
    });
  }

  function clearProtectedSupportRequests() {
    setRequests([]);
    setSummary(null);
    setSelectedId("");
    resetRequestEdit();
    setBusy(false);
    setMessage(null);
  }

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
    const generation = requestsRequest.begin();
    setRequests([]);
    setSummary(null);
    setSelectedId("");
    resetRequestEdit();
    setBusy(true);
    setMessage(null);
    try {
      const params = new URLSearchParams();
      if (statusFilter) params.set("status", statusFilter);
      if (typeFilter) params.set("request_type", typeFilter);
      const payload = await requestJson<AdminSupportRequestsListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/support-requests${params.toString() ? `?${params.toString()}` : ""}`);
      if (!requestsRequest.isCurrent(generation)) return;
      setRequests(payload.requests || []);
      setSummary(payload.summary || null);
      setMessage(`Loaded ${payload.requests?.length ?? 0} request(s).`);
    } catch (error) {
      if (requestsRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load support requests.");
    } finally {
      if (requestsRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function selectRequest(request: AdminSupportRequest) {
    setSelectedId(request.id);
    setEdit({
      status: request.status === "new" ? "in_review" : request.status,
      adminNote: request.admin_note || "",
      identityStatus: request.identity_status || "pending",
      fulfillmentStatus: request.fulfillment_status || "pending",
      resolutionAction: request.resolution_action || "none",
      resolutionEvidence: request.resolution_evidence || ""
    });
    setMessage(null);
  }

  async function saveStatus(confirmationText: string) {
    if (!selected) {
      setMessage("Select a request before saving.");
      return;
    }
    const generation = requestsRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminSupportRequestUpdateResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/support-requests/${encodeURIComponent(selected.id)}`, {
        method: "PATCH",
        body: JSON.stringify({
          status: edit.status,
          admin_note: edit.adminNote.trim() || null,
          expected_updated_at: selected.updated_at,
          identity_status: selected.request_type === "profile_privacy" ? edit.identityStatus : undefined,
          fulfillment_status: selected.request_type === "profile_privacy" ? edit.fulfillmentStatus : undefined,
          resolution_action: selected.request_type === "profile_privacy" ? edit.resolutionAction : undefined,
          resolution_evidence: selected.request_type === "profile_privacy" ? edit.resolutionEvidence : undefined,
          confirmation_text: confirmationText,
          source: "next_admin_support_requests"
        })
      });
      if (!requestsRequest.isCurrent(generation)) return;
      setRequests((current) => current.map((request) => request.id === selected.id ? payload.request : request));
      setSelectedId(payload.request.id);
      setEdit({
        status: payload.request.status,
        adminNote: payload.request.admin_note || "",
        identityStatus: payload.request.identity_status || "pending",
        fulfillmentStatus: payload.request.fulfillment_status || "pending",
        resolutionAction: payload.request.resolution_action || "none",
        resolutionEvidence: payload.request.resolution_evidence || ""
      });
      setMessage(`Request updated to ${payload.request.status}.`);
    } catch (error) {
      if (requestsRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to update request.");
    } finally {
      if (requestsRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(
    status.enabled ? accessToken : "",
    loadRequests,
    `${statusFilter}:${typeFilter}`
  );

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
          <label>Status<br /><select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)} disabled={busy} style={inputStyle}><option value="">all statuses</option>{STATUS_OPTIONS.map((option) => <option key={option} value={option}>{option.replace(/_/g, " ")}</option>)}</select></label>
          <label>Type<br /><select value={typeFilter} onChange={(event) => setTypeFilter(event.target.value)} disabled={busy} style={inputStyle}>{TYPE_OPTIONS.map((option) => <option key={option || "all"} value={option}>{typeLabel(option)}</option>)}</select></label>
        </div>
        <button type="button" onClick={loadRequests} disabled={busy || !accessToken} style={{ ...buttonStyle, marginTop: "0.75rem" }}>{busy ? "Refreshing…" : "Refresh requests"}</button>
        {summary ? <p style={{ color: "#475569" }}>Loaded {summary.total} · statuses {JSON.stringify(summary.by_status)} · types {JSON.stringify(summary.by_type)}</p> : null}
      </article>

      {message ? <p style={{ color: message.toLowerCase().includes("error") || message.toLowerCase().includes("unable") ? "#b91c1c" : "#166534" }}>{message}</p> : null}

      {selected ? (
        <article style={{ ...cardStyle, borderColor: "#2563eb" }}>
          <h2 style={{ marginTop: 0 }}>Review selected request</h2>
          <p style={{ color: "#475569" }}>Use this panel only to track review state. Apply actual corrections through Match Log, Player Editor, Tournament Admin, or Replay History.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
            <label>Status<br /><select value={edit.status} onChange={(event) => setEdit((current) => ({ ...current, status: event.target.value }))} disabled={busy} style={inputStyle}>{STATUS_OPTIONS.map((option) => <option key={option} value={option}>{option.replace(/_/g, " ")}</option>)}</select></label>
          </div>
          {selected.request_type === "profile_privacy" ? (
            <div style={{ ...cardStyle, marginTop: "0.75rem", background: "#f8fafc" }}>
              <h3 style={{ marginTop: 0 }}>Privacy fulfillment checklist</h3>
              <p style={{ color: "#475569" }}>Verify identity, apply the approved alias/hide/anonymize action through the authorized player workflow, inspect every public projection, and record non-sensitive evidence here. This queue never changes a public profile itself.</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
                <label>Identity verification<br /><select value={edit.identityStatus} onChange={(event) => setEdit((current) => ({ ...current, identityStatus: event.target.value }))} style={inputStyle}>{IDENTITY_OPTIONS.map((option) => <option key={option} value={option}>{option.replace(/_/g, " ")}</option>)}</select></label>
                <label>Fulfillment<br /><select value={edit.fulfillmentStatus} onChange={(event) => setEdit((current) => ({ ...current, fulfillmentStatus: event.target.value }))} style={inputStyle}>{FULFILLMENT_OPTIONS.map((option) => <option key={option} value={option}>{option.replace(/_/g, " ")}</option>)}</select></label>
                <label>Approved action<br /><select value={edit.resolutionAction} onChange={(event) => setEdit((current) => ({ ...current, resolutionAction: event.target.value }))} style={inputStyle}>{RESOLUTION_OPTIONS.map((option) => <option key={option} value={option}>{option.replace(/_/g, " ")}</option>)}</select></label>
              </div>
              <label>Fulfillment evidence<br /><textarea value={edit.resolutionEvidence} onChange={(event) => setEdit((current) => ({ ...current, resolutionEvidence: event.target.value }))} disabled={busy} rows={3} placeholder="Identity method, authorized workflow used, and public projections checked; do not paste identity documents." style={inputStyle} /></label>
            </div>
          ) : null}
          <label>Admin note (optional)<br /><textarea value={edit.adminNote} onChange={(event) => setEdit((current) => ({ ...current, adminNote: event.target.value }))} disabled={busy} rows={4} style={inputStyle} /></label>
          <div style={{ marginTop: "0.75rem" }}>
            <ConfirmAction
              triggerLabel="Save request status"
              title={edit.status === "dismissed" ? "Dismiss this request?" : "Save this request status?"}
              description={<>This will change <strong>{selected.subject || "the selected request"}</strong> from {selected.status.replace(/_/g, " ")} to {edit.status.replace(/_/g, " ")}.{edit.adminNote.trim() ? " It will also record the optional admin note." : ""}{selected.request_type === "profile_privacy" ? " It also saves the selected identity, fulfillment, resolution, and non-sensitive evidence fields." : ""}</>}
              confirmLabel={edit.status === "dismissed" ? "Yes, dismiss request" : "Yes, save status"}
              confirmationText="SAVE REQUEST STATUS"
              tone={edit.status === "dismissed" ? "danger" : "default"}
              disabled={busy}
              busy={busy}
              onConfirm={saveStatus}
            />
          </div>
        </article>
      ) : null}

      <div style={{ display: "grid", gap: "0.75rem" }}>
        {requests.map((request) => <SupportRequestCard key={request.id} request={request} selected={request.id === selectedId} disabled={busy} onSelect={() => selectRequest(request)} />)}
      </div>
    </div>
  );
}
