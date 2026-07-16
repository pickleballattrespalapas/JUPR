"use client";

import { useMemo, useState } from "react";
import type { PublicRegistrationEditSelection, PublicTournamentNeedsPartnerEntry } from "@/lib/tournamentRegistrationApi";

type PairingInterestPanelProps = {
  apiBase: string | null;
  clubSlug: string;
  tournamentId: string;
  registrationSlug?: string | null;
  editToken: string;
  requesterSelections: PublicRegistrationEditSelection[];
  boardEntries: PublicTournamentNeedsPartnerEntry[];
};

const selectStyle = { width: "100%", padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

export default function PairingInterestPanel({ apiBase, clubSlug, tournamentId, registrationSlug, editToken, requesterSelections, boardEntries }: PairingInterestPanelProps) {
  const [selectionByEntry, setSelectionByEntry] = useState<Record<string, string>>({});
  const [pendingKey, setPendingKey] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selectionsByEvent = useMemo(() => {
    const map = new Map<string, PublicRegistrationEditSelection[]>();
    for (const selection of requesterSelections) {
      const eventId = String(selection.event_option_id || "");
      if (!eventId) continue;
      map.set(eventId, [...(map.get(eventId) || []), selection]);
    }
    return map;
  }, [requesterSelections]);

  async function sendInterest(entry: PublicTournamentNeedsPartnerEntry) {
    setMessage(null);
    setError(null);
    if (!apiBase) {
      setError("API base URL is not configured.");
      return;
    }
    const entryKey = String(entry.selection_id || "");
    const requesterSelectionId = selectionByEntry[entryKey] || selectionsByEvent.get(String(entry.event_option_id || ""))?.[0]?.id || "";
    if (!entryKey || !requesterSelectionId) {
      setError("Choose one of your registrations for this division before sending a request.");
      return;
    }
    setPendingKey(entryKey);
    try {
      const response = await fetch(apiUrl(apiBase, `/clubs/${encodeURIComponent(clubSlug)}/tournament-registration/pairing-interest`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tournament_id: tournamentId,
          registration_slug: registrationSlug || null,
          edit_token: editToken,
          requester_selection_id: requesterSelectionId,
          board_entry_selection_id: entryKey
        })
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
      setMessage(payload?.message || "Pairing request sent. If the other player accepts, JUPR will automatically pair both registrations.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to send pairing request.");
    } finally {
      setPendingKey(null);
    }
  }

  if (!editToken || !requesterSelections.length) {
    return null;
  }

  return (
    <div style={{ display: "grid", gap: "0.75rem" }}>
      {boardEntries.map((entry) => {
        const entryKey = String(entry.selection_id || "");
        const possibleSelections = selectionsByEvent.get(String(entry.event_option_id || "")) || [];
        if (!entryKey || !possibleSelections.length) return null;
        return (
          <div key={entryKey} style={{ borderTop: "1px solid #e2e8f0", paddingTop: "0.75rem", marginTop: "0.75rem" }}>
            <label>
              Your registration for this division<br />
              <select value={selectionByEntry[entryKey] || possibleSelections[0]?.id || ""} onChange={(event) => setSelectionByEntry((current) => ({ ...current, [entryKey]: event.target.value }))} style={selectStyle}>
                {possibleSelections.map((selection) => <option key={selection.id} value={selection.id}>{selection.partner_mode || "Registration"}</option>)}
              </select>
            </label>
            <button type="button" onClick={() => sendInterest(entry)} disabled={pendingKey === entryKey} style={{ marginTop: "0.5rem", padding: "0.55rem 0.8rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 }}>
              {pendingKey === entryKey ? "Sending…" : "Request pairing"}
            </button>
          </div>
        );
      })}
      {message ? <p style={{ color: "#166534", margin: 0 }}>{message}</p> : null}
      {error ? <p style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
    </div>
  );
}
