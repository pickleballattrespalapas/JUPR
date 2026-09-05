"use client";

import { useState } from "react";
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

function partnerModeLabel(partnerMode?: string | null): string {
  switch (String(partnerMode || "").toUpperCase()) {
    case "HAS_PARTNER":
      return "Has a partner";
    case "NEEDS_PARTNER":
      return "Looking for a partner";
    default:
      return "Your entry";
  }
}

export default function PairingInterestPanel({ apiBase, clubSlug, tournamentId, registrationSlug, editToken, requesterSelections, boardEntries }: PairingInterestPanelProps) {
  const [selectionByEntry, setSelectionByEntry] = useState<Record<string, string>>({});
  const [pendingKey, setPendingKey] = useState<string | null>(null);
  const [sentEntries, setSentEntries] = useState<Record<string, boolean>>({});
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function sendInterest(entry: PublicTournamentNeedsPartnerEntry) {
    setMessage(null);
    setError(null);
    if (!apiBase) {
      setError("Partner requests are unavailable right now. Please try again shortly.");
      return;
    }
    const entryKey = String(entry.board_entry_key || "");
    const requesterSelectionId = selectionByEntry[entryKey] || requesterSelections[0]?.id || "";
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
          board_entry_key: entryKey
        })
      });
      const payload = await response.json().catch(() => null) as { idempotent?: boolean } | null;
      if (!response.ok) throw new Error("Partner requests are unavailable right now. Please try again.");
      setMessage(payload?.idempotent
        ? "You’ve already sent this request."
        : "Request sent. If they accept, you’ll be partners for this event.");
      setSentEntries((current) => ({ ...current, [entryKey]: true }));
    } catch {
      setError("We couldn’t send your request. Please try again.");
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
        const entryKey = String(entry.board_entry_key || "");
        const possibleSelections = requesterSelections;
        if (!entryKey || !possibleSelections.length) return null;
        return (
          <div key={entryKey} style={{ borderTop: "1px solid #e2e8f0", paddingTop: "0.75rem", marginTop: "0.75rem" }}>
            <label>
              Your registration for this division<br />
              <select value={selectionByEntry[entryKey] || possibleSelections[0]?.id || ""} onChange={(event) => setSelectionByEntry((current) => ({ ...current, [entryKey]: event.target.value }))} style={selectStyle}>
                {possibleSelections.map((selection, index) => (
                  <option key={selection.id} value={selection.id}>
                    {possibleSelections.length > 1 ? `Entry ${index + 1} · ` : ""}{partnerModeLabel(selection.partner_mode)}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" onClick={() => sendInterest(entry)} disabled={pendingKey === entryKey || Boolean(sentEntries[entryKey])} style={{ marginTop: "0.5rem", padding: "0.55rem 0.8rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 }}>
              {pendingKey === entryKey ? "Sending…" : sentEntries[entryKey] ? "Request sent" : "Ask to partner"}
            </button>
          </div>
        );
      })}
      {message ? <p role="status" style={{ color: "#166534", margin: 0 }}>{message}</p> : null}
      {error ? <p role="alert" style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
    </div>
  );
}
