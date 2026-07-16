"use client";

import { useEffect, useState } from "react";

type PartnerRequestSelection = {
  selection_id?: string | null;
  display_name?: string | null;
  division_name?: string | null;
};

type PartnerRequestRow = {
  id: string;
  status?: string | null;
  event_option_id?: string | null;
  requester_selection_id?: string | null;
  target_selection_id?: string | null;
  requester?: PartnerRequestSelection | null;
  target?: PartnerRequestSelection | null;
  created_at?: string | null;
  direction?: "incoming" | "outgoing" | string | null;
};

type PartnerRequestPayload = {
  ok: boolean;
  incoming?: PartnerRequestRow[];
  outgoing?: PartnerRequestRow[];
  summary?: { incoming?: number; outgoing?: number; pending_incoming?: number; pending_outgoing?: number };
};

type WritePayload = { ok: boolean; message?: string | null; status?: string | null; partner_request_id?: string | null; team_link_id?: string | null };

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
  const requester = row.requester?.display_name || "Requester";
  const target = row.target?.display_name || "you";
  const division = row.requester?.division_name || row.target?.division_name || "Division";
  return `${requester} → ${target} · ${division}`;
}

export default function PartnerRequestReviewPanel({ apiBase, clubSlug, tournamentId, registrationSlug, editToken, focusRequestId }: PartnerRequestReviewPanelProps) {
  const [incoming, setIncoming] = useState<PartnerRequestRow[]>([]);
  const [outgoing, setOutgoing] = useState<PartnerRequestRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function loadRequests() {
    setMessage(null);
    setError(null);
    if (!apiBase) {
      setError("API base URL is not configured.");
      return;
    }
    setLoading(true);
    try {
      const query = new URLSearchParams({ edit_token: editToken, tournament_id: tournamentId });
      if (registrationSlug) query.set("registration_slug", registrationSlug);
      const response = await fetch(apiUrl(apiBase, `/clubs/${encodeURIComponent(clubSlug)}/tournament-registration/pairing-requests?${query.toString()}`), { cache: "no-store" });
      const payload = await response.json().catch(() => null) as PartnerRequestPayload | { detail?: unknown } | null;
      if (!response.ok) throw new Error(String((payload as { detail?: unknown } | null)?.detail || `API error (${response.status})`));
      setIncoming((payload as PartnerRequestPayload).incoming || []);
      setOutgoing((payload as PartnerRequestPayload).outgoing || []);
    } catch (err) {
      setIncoming([]);
      setOutgoing([]);
      setError(err instanceof Error ? err.message : "Unable to load partner requests.");
    } finally {
      setLoading(false);
    }
  }

  async function acceptRequest(row: PartnerRequestRow) {
    if (!apiBase || !row.id) return;
    setPendingId(row.id);
    setMessage(null);
    setError(null);
    try {
      const response = await fetch(apiUrl(apiBase, `/clubs/${encodeURIComponent(clubSlug)}/tournament-registration/pairing-requests/${encodeURIComponent(row.id)}/accept`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tournament_id: tournamentId, registration_slug: registrationSlug || null, edit_token: editToken })
      });
      const payload = await response.json().catch(() => null) as WritePayload | { detail?: unknown } | null;
      if (!response.ok) throw new Error(String((payload as { detail?: unknown } | null)?.detail || `API error (${response.status})`));
      setMessage((payload as WritePayload)?.message || "Partner request accepted. You are now paired.");
      await loadRequests();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to accept partner request.");
    } finally {
      setPendingId(null);
    }
  }

  useEffect(() => {
    if (editToken) void loadRequests();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editToken, tournamentId, registrationSlug, apiBase, clubSlug]);

  if (!editToken) return null;

  const pendingIncoming = incoming.filter((row) => String(row.status || "").toUpperCase() === "PENDING");
  const relevantIncoming = focusRequestId
    ? [...pendingIncoming.filter((row) => row.id === focusRequestId), ...pendingIncoming.filter((row) => row.id !== focusRequestId)]
    : pendingIncoming;
  const pendingOutgoing = outgoing.filter((row) => String(row.status || "").toUpperCase() === "PENDING");

  return (
    <article style={{ ...cardStyle, marginBottom: "1rem", borderColor: focusRequestId ? "#93c5fd" : "#e2e8f0" }}>
      <h2 style={{ marginTop: 0 }}>Your partner requests</h2>
      <p style={{ color: "#475569" }}>
        Review incoming requests sent to your registration. Accepting a request immediately confirms the pairing for that division and removes both players from the open partner board.
      </p>
      <p><button type="button" onClick={loadRequests} disabled={loading || !apiBase} style={buttonStyle}>{loading ? "Loading…" : "Refresh requests"}</button></p>
      {relevantIncoming.length ? (
        <div style={{ display: "grid", gap: "0.75rem" }}>
          {relevantIncoming.map((row) => (
            <div key={row.id} id={`request-${row.id}`} style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.75rem", background: row.id === focusRequestId ? "#eff6ff" : "white" }}>
              <strong>{requestLabel(row)}</strong>
              <p style={{ color: "#475569", margin: "0.35rem 0" }}>Status: {row.status || "PENDING"}</p>
              <button type="button" onClick={() => acceptRequest(row)} disabled={pendingId === row.id} style={buttonStyle}>{pendingId === row.id ? "Accepting…" : "Accept & pair automatically"}</button>
            </div>
          ))}
        </div>
      ) : <p style={{ color: "#64748b" }}>No pending incoming partner requests for this registration.</p>}
      {pendingOutgoing.length ? <p style={{ color: "#475569" }}>Pending requests you sent: {pendingOutgoing.map((row) => row.target?.display_name || row.id).join(", ")}.</p> : null}
      {message ? <p style={{ color: "#166534", marginBottom: 0 }}>{message}</p> : null}
      {error ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{error}</p> : null}
    </article>
  );
}
