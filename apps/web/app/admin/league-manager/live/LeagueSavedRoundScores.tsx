"use client";

import { useState } from "react";
import { LeagueRoundScoreReview } from "./LeagueRoundScoreReview";
import { savedRoundScoreGames } from "./leagueRoundScores";

type SavedRound = {
  round_number: number; round_label?: string | null; status: string; match_date?: string | null;
  matches_json?: Array<Record<string, unknown>> | null; preview_json?: unknown; operation_key?: string | null;
  score_readback_status?: string; score_readback_message?: string | null;
};

export function LeagueSavedRoundScores({ rounds, playerNames, busy, onRefresh }: {
  rounds: SavedRound[]; playerNames: Map<number, string>; busy: boolean; onRefresh: () => void;
}) {
  const [selectedRound, setSelectedRound] = useState<number | null>(null);
  const saved = rounds.filter(round => round.status === "submitted").sort((a, b) => b.round_number - a.round_number);
  const round = saved.find(row => row.round_number === selectedRound) || saved[0];
  if (!round) return null;
  return <details style={{ border: "1px solid #93c5fd", borderRadius: 12, padding: "1rem", background: "#eff6ff" }}>
    <summary style={{ cursor: "pointer", fontWeight: 800 }}>View or correct previous-round scores</summary>
    <div style={{ display: "flex", gap: ".75rem", flexWrap: "wrap", alignItems: "center", margin: ".75rem 0" }}>
      <label>Round <select value={round.round_number} onChange={event => setSelectedRound(Number(event.target.value))}>{saved.map(row => <option key={row.round_number} value={row.round_number}>{row.round_label || `Round ${row.round_number}`} · {row.match_date || ""}</option>)}</select></label>
      <button type="button" onClick={onRefresh} disabled={busy}>{busy ? "Refreshing…" : "Refresh saved scores"}</button>
    </div>
    <p>Find a court or player to see the recorded games. Corrections open that game in Match Log in a separate tab; return here and refresh after saving.</p>
    {round.score_readback_message ? <p role="alert" style={{ color: "#92400e" }}>{round.score_readback_message}</p> : null}
    {round.operation_key ? <p style={{ color: "#92400e" }}>Movement for this round has already been applied. Correcting a score updates the record; it does not rearrange courts that are already playing.</p> : <p style={{ color: "#475569" }}>After correcting a score, refresh and recalculate the movement board before applying it.</p>}
    <LeagueRoundScoreReview key={round.round_number} games={savedRoundScoreGames(round, playerNames)} />
  </details>;
}
