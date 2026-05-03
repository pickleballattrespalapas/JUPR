"use client";

import { useState } from "react";

type MatchPayload = {
  league: string;
  t1_p1: string;
  t1_p2: string;
  t2_p1: string;
  t2_p2: string;
  s1: number;
  s2: number;
};

export default function ScoreEntryPage({ params }: { params: { clubId: string } }) {
  const [match, setMatch] = useState<MatchPayload>({
    league: "",
    t1_p1: "",
    t1_p2: "",
    t2_p1: "",
    t2_p2: "",
    s1: 11,
    s2: 0,
  });
  const [status, setStatus] = useState<string>("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setStatus("");
    try {
      const response = await fetch(`/api/admin/clubs/${params.clubId}/matches/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ matches: [match], source: "next_admin_score_entry" }),
      });
      const body = await response.json();
      if (!response.ok) {
        setStatus(`Error: ${body?.error || body?.detail || "Unable to submit"}`);
      } else {
        const inserted = body?.result?.inserted ?? 0;
        const skipped = body?.result?.skipped_incomplete ?? 0;
        setStatus(`Submitted. Inserted: ${inserted}, skipped: ${skipped}`);
      }
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ maxWidth: 700, margin: "2rem auto", padding: "0 1rem" }}>
      <h1>Score Entry</h1>
      <p>Club: {params.clubId}</p>
      <form onSubmit={onSubmit} style={{ display: "grid", gap: "0.75rem" }}>
        <label>
          League
          <input value={match.league} onChange={(e) => setMatch({ ...match, league: e.target.value })} required />
        </label>
        <label>
          Team 1 Player 1 (ID or name)
          <input value={match.t1_p1} onChange={(e) => setMatch({ ...match, t1_p1: e.target.value })} required />
        </label>
        <label>
          Team 1 Player 2 (ID or name)
          <input value={match.t1_p2} onChange={(e) => setMatch({ ...match, t1_p2: e.target.value })} required />
        </label>
        <label>
          Team 2 Player 1 (ID or name)
          <input value={match.t2_p1} onChange={(e) => setMatch({ ...match, t2_p1: e.target.value })} required />
        </label>
        <label>
          Team 2 Player 2 (ID or name)
          <input value={match.t2_p2} onChange={(e) => setMatch({ ...match, t2_p2: e.target.value })} required />
        </label>
        <label>
          Team 1 score
          <input type="number" value={match.s1} onChange={(e) => setMatch({ ...match, s1: Number(e.target.value) })} min={0} required />
        </label>
        <label>
          Team 2 score
          <input type="number" value={match.s2} onChange={(e) => setMatch({ ...match, s2: Number(e.target.value) })} min={0} required />
        </label>
        <button type="submit" disabled={busy}>{busy ? "Submitting..." : "Submit"}</button>
      </form>
      {status ? <p>{status}</p> : null}
    </main>
  );
}
