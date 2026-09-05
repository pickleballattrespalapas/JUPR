"use client";

import { useCallback, useEffect, useState } from "react";

type PartnerRequestSelection = {
  display_name?: string | null;
  division_name?: string | null;
};

type PartnerRequestRow = {
  id: string;
  status?: string | null;
  requester?: PartnerRequestSelection | null;
  target?: PartnerRequestSelection | null;
  created_at?: string | null;
  responded_at?: string | null;
  direction?: "incoming" | "outgoing" | string | null;
  available_actions?: Array<"accept" | "decline" | "cancel" | string>;
};

type PartnerRequestPayload = {
  ok: boolean;
  incoming?: PartnerRequestRow[];
  outgoing?: PartnerRequestRow[];
  summary?: { incoming?: number; outgoing?: number; pending_incoming?: number; pending_outgoing?: number };
};

type WritePayload = {
  ok: boolean;
  message?: string | null;
  status?: string | null;
  partner_request_id?: string | null;
  team_link_id?: string | null;
  idempotent?: boolean | null;
  cancelled_request_ids?: string[];
};

type PartnerRequestReviewPanelProps = {
  apiBase: string | null;
  clubSlug: string;
  tournamentId: string;
  registrationSlug?: string | null;
  editToken: string;
  focusRequestId?: string | null;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "#f8fafc" };
const buttonStyle = { padding: "0.55rem 0.85rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function requestLabel(row: PartnerRequestRow): string {
  const requester = row.requester?.display_name || "Another player";
  const target = row.target?.display_name || "another player";
  const division = row.requester?.division_name || row.target?.division_name || "this division";
  if (row.direction === "incoming") return `${requester} wants to partner with you in ${division}.`;
  if (row.direction === "outgoing") return `You asked ${target} to partner with you in ${division}.`;
  return `${requester} and ${target} · ${division}`;
}

function requestStatusLabel(status?: string | null): string {
  switch (String(status || "").toUpperCase()) {
    case "PENDING":
      return "Pending";
    case "ACCEPTED":
      return "Paired";
    case "DECLINED":
      return "Declined";
    case "CANCELLED":
    case "CANCELED":
      return "Canceled";
    default:
      return "Updated";
  }
}

function actionSuccessMessage(action: "accept" | "decline" | "cancel", idempotent?: boolean | null): string {
  if (idempotent) return "This request was already updated.";
  if (action === "accept") return "You’re now partners for this event.";
  if (action === "decline") return "Request declined.";
  return "Request canceled.";
}

export default function PartnerRequestReviewPanel({ apiBase, clubSlug, tournamentId, registrationSlug, editToken, focusRequestId }: PartnerRequestReviewPanelProps) {
  const [incoming, setIncoming] = useState<PartnerRequestRow[]>([]);
  const [outgoing, setOutgoing] = useState<PartnerRequestRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [confirmAction, setConfirmAction] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadRequests = useCallback(async () => {
    setError(null);
    if (!apiBase) {
      setError("Partner requests are unavailable right now. Please try again shortly.");
      return;
    }
    setLoading(true);
    try {
      const query = new URLSearchParams({ edit_token: editToken, tournament_id: tournamentId });
      if (registrationSlug) query.set("registration_slug", registrationSlug);
      const response = await fetch(apiUrl(apiBase, `/clubs/${encodeURIComponent(clubSlug)}/tournament-registration/pairing-requests?${query.toString()}`), { cache: "no-store" });
      const payload = await response.json().catch(() => null) as PartnerRequestPayload | null;
      if (!response.ok) throw new Error("Partner requests are unavailable right now.");
      setIncoming((payload as PartnerRequestPayload).incoming || []);
      setOutgoing((payload as PartnerRequestPayload).outgoing || []);
    } catch {
      setIncoming([]);
      setOutgoing([]);
      setError("We couldn’t load your partner requests. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [apiBase, clubSlug, editToken, registrationSlug, tournamentId]);

  async function transitionRequest(row: PartnerRequestRow, action: "accept" | "decline" | "cancel") {
    if (!apiBase || !row.id) return;
    const actionKey = `${row.id}:${action}`;
    if (confirmAction !== actionKey) {
      setConfirmAction(actionKey);
      setMessage(null);
      setError(null);
      return;
    }
    setPendingAction(actionKey);
    setMessage(null);
    setError(null);
    try {
      const response = await fetch(apiUrl(apiBase, `/clubs/${encodeURIComponent(clubSlug)}/tournament-registration/pairing-requests/${encodeURIComponent(row.id)}/${action}`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tournament_id: tournamentId, registration_slug: registrationSlug || null, edit_token: editToken })
      });
      const payload = await response.json().catch(() => null) as WritePayload | null;
      if (!response.ok) throw new Error("We couldn’t update this partner request.");
      await loadRequests();
      setMessage(actionSuccessMessage(action, payload?.idempotent));
      setConfirmAction(null);
    } catch {
      setError("We couldn’t update this partner request. Please try again.");
    } finally {
      setPendingAction(null);
    }
  }

  useEffect(() => {
    if (editToken) void loadRequests();
  }, [editToken, loadRequests]);

  if (!editToken) return null;

  const pendingIncoming = incoming.filter((row) => String(row.status || "").toUpperCase() === "PENDING");
  const relevantIncoming = focusRequestId
    ? [...pendingIncoming.filter((row) => row.id === focusRequestId), ...pendingIncoming.filter((row) => row.id !== focusRequestId)]
    : pendingIncoming;
  const pendingOutgoing = outgoing.filter((row) => String(row.status || "").toUpperCase() === "PENDING");
  const completed = [...incoming, ...outgoing].filter((row, index, rows) =>
    String(row.status || "").toUpperCase() !== "PENDING" && rows.findIndex((candidate) => candidate.id === row.id) === index
  );

  function actionButton(row: PartnerRequestRow, action: "accept" | "decline" | "cancel", label: string) {
    const actionKey = `${row.id}:${action}`;
    const confirming = confirmAction === actionKey;
    const pending = pendingAction === actionKey;
    const confirmLabel = action === "accept" ? "Confirm pairing" : action === "decline" ? "Confirm decline" : "Confirm cancellation";
    return (
      <span key={action} style={{ display: "inline-flex", gap: "0.35rem", alignItems: "center" }}>
        <button type="button" onClick={() => void transitionRequest(row, action)} disabled={Boolean(pendingAction)} style={{ ...buttonStyle, background: action === "accept" ? "#166534" : action === "decline" ? "#9f1239" : "#475569" }}>
          {pending ? `${label}…` : confirming ? confirmLabel : label}
        </button>
        {confirming && !pending ? <button type="button" onClick={() => setConfirmAction(null)} style={{ ...buttonStyle, background: "white", color: "#0f172a" }}>Back</button> : null}
      </span>
    );
  }

  return (
    <article style={{ ...cardStyle, marginBottom: "1rem", borderColor: focusRequestId ? "#93c5fd" : "#e2e8f0" }}>
      <h2 style={{ marginTop: 0 }}>Your partner requests</h2>
      <p style={{ color: "#475569" }}>
        Accept a request to become partners for this event, or decline if it isn’t a match.
      </p>
      <p><button type="button" onClick={() => void loadRequests()} disabled={loading || !apiBase} style={buttonStyle}>{loading ? "Loading…" : "Refresh requests"}</button></p>
      {relevantIncoming.length ? (
        <div style={{ display: "grid", gap: "0.75rem" }}>
          {relevantIncoming.map((row) => (
            <div key={row.id} id={`request-${row.id}`} style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.75rem", background: row.id === focusRequestId ? "#eff6ff" : "white" }}>
              <strong>{requestLabel(row)}</strong>
              <p style={{ color: "#475569", margin: "0.35rem 0" }}>Status: {requestStatusLabel(row.status)}</p>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                {actionButton(row, "accept", "Accept & pair")}
                {actionButton(row, "decline", "Decline")}
              </div>
            </div>
          ))}
        </div>
      ) : <p style={{ color: "#64748b" }}>You don’t have any new partner requests.</p>}
      {pendingOutgoing.length ? (
        <div style={{ display: "grid", gap: "0.75rem", marginTop: "1rem" }}>
          <h3 style={{ marginBottom: 0 }}>Requests you sent</h3>
          {pendingOutgoing.map((row) => (
            <div key={row.id} style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.75rem", background: "white" }}>
              <strong>{requestLabel(row)}</strong>
              <p style={{ color: "#475569", margin: "0.35rem 0" }}>Status: {requestStatusLabel(row.status)}</p>
              {actionButton(row, "cancel", "Cancel request")}
            </div>
          ))}
        </div>
      ) : <p style={{ color: "#64748b" }}>You haven’t sent any open requests.</p>}
      {completed.length ? <p style={{ color: "#64748b" }}>Recent requests: {completed.map((row) => `${row.target?.display_name || row.requester?.display_name || "Partner"} — ${requestStatusLabel(row.status)}`).join(", ")}.</p> : null}
      {message ? <p role="status" style={{ color: "#166534", marginBottom: 0 }}>{message}</p> : null}
      {error ? <p role="alert" style={{ color: "#b91c1c", marginBottom: 0 }}>{error}</p> : null}
    </article>
  );
}
