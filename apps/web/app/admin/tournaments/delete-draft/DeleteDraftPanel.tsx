"use client";

import Link from "next/link";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess } from "@/components/interaction";
import type { AdminTournament, AdminTournamentListResponse, AdminTournamentStatusResponse, AdminTournamentWriteResponse } from "@/lib/adminTournamentApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentStatusResponse;
  initialTournamentId: string;
  initialTournamentName: string;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const ghostButtonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "white", color: "#0f172a", fontWeight: 800 };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function usageText(value?: Record<string, number>): string {
  const entries = Object.entries(value || {});
  return entries.length ? entries.map(([key, count]) => `${key}: ${count}`).join(" · ") : "—";
}

export default function DeleteDraftPanel({ apiBase, clubId, status, initialTournamentId, initialTournamentName }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [tournaments, setTournaments] = useState<AdminTournament[]>([]);
  const [selectedTournamentId, setSelectedTournamentId] = useState(initialTournamentId);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const listRequest = useLatestRequestGuard(accessToken, clearProtectedDraftState);
  const actionRequest = useLatestRequestGuard(accessToken);
  const selectedTournament = tournaments.find((row) => row.id === selectedTournamentId) || null;

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Tournament Admin.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  function clearProtectedDraftState() {
    setBusy(false); setMessage(null); setTournaments([]); setSelectedTournamentId(initialTournamentId);
  }

  async function loadTournaments() {
    const generation = listRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments?include_archived=true`);
      if (!listRequest.isCurrent(generation)) return;
      const drafts = (payload.tournaments || []).filter((row) => row.status === "DRAFT");
      setTournaments(drafts);
      setSelectedTournamentId((current) => drafts.some((row) => row.id === current) ? current : initialTournamentId);
      setMessage(drafts.length ? `Loaded ${drafts.length} draft tournament(s).` : "No empty draft tournaments are available.");
    } catch (error) {
      if (listRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load tournaments.");
    } finally {
      if (listRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function deleteDraft(confirmationText: string) {
    if (!selectedTournamentId) {
      setMessage("Select a draft tournament first.");
      throw new Error("Select a draft tournament first.");
    }
    const generation = actionRequest.begin();
    const requestedTournamentId = selectedTournamentId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/delete-draft`,
        { method: "POST", body: JSON.stringify({ expected_updated_at: selectedTournament?.updated_at, confirmation_text: confirmationText, source: "next_tournament_admin_delete_draft_page" }) }
      );
      const completion = actionSuccess("Draft deleted", `Draft ${payload.tournament_id} was permanently deleted.`);
      if (!actionRequest.isCurrent(generation)) return completion;
      setTournaments((current) => current.filter((row) => row.id !== requestedTournamentId));
      setSelectedTournamentId("");
      setMessage(`Deleted draft ${payload.tournament_id}. Usage summary: ${usageText(payload.usage_summary)}.`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to delete draft tournament.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadTournaments);

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Next Tournament Admin is disabled</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the Tournament Admin pilot flag on FastAPI."}</p>
      </article>
    );
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, borderColor: "#fecaca", background: "#fff7ed" }}>
        <h2 style={{ marginTop: 0 }}>Delete empty draft tournament</h2>
        <p style={{ color: "#7c2d12" }}>This only succeeds for DRAFT tournament shells with no registrations, selections, draws, teams, games, or podium rows.</p>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>{accessToken ? "Ready to send guarded draft deletion requests." : sessionLoading ? "Checking admin session…" : "Sign in before loading draft tournaments."}</p>
          {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
          {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
        </div>
        <button type="button" onClick={loadTournaments} disabled={busy || !accessToken} style={ghostButtonStyle}>{busy ? "Refreshing…" : "Refresh draft tournaments"}</button>
      </article>

      {tournaments.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Confirm deletion</h2>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) auto", gap: "0.75rem", alignItems: "end" }}>
            <p><strong>{selectedTournament?.name || initialTournamentName}</strong></p>
            <ConfirmAction triggerLabel="Delete draft" title="Delete this draft tournament?" description={`This permanently deletes ${selectedTournament?.name || "the selected draft"}. It can proceed only if no registrations, selections, draws, teams, games, or podium rows exist.`} confirmLabel="Yes, delete draft" confirmationText="DELETE DRAFT" tone="danger" disabled={!selectedTournamentId || !selectedTournament?.updated_at} busy={busy} onConfirm={deleteDraft} />
          </div>
          {selectedTournament ? <p style={{ color: "#64748b" }}>Selected: <strong>{selectedTournament.name}</strong></p> : null}
        </article>
      ) : <article style={cardStyle}><p style={{ color: "#64748b" }}>{busy ? "Loading draft tournaments…" : "No empty draft tournaments are available."}</p></article>}

      {message ? <p role="status" style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("error") || message.toLowerCase().includes("sign in") || message.toLowerCase().includes("reload") || message.toLowerCase().includes("changed") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
