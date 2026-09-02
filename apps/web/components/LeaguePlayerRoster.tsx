"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { LeagueResultsStanding } from "@/lib/api";

type SortKey = "player" | "rating" | "record" | "games" | "win_pct" | "improvement";

const thStyle = { textAlign: "left" as const, padding: "0.6rem", borderBottom: "1px solid #cbd5e1", fontSize: "0.8rem", color: "#475569" };
const tdStyle = { padding: "0.6rem", borderBottom: "1px solid #e2e8f0" };

function number(value: number | null | undefined): number {
  return value == null || Number.isNaN(Number(value)) ? Number.NEGATIVE_INFINITY : Number(value);
}

function ratingLabel(value?: number | null): string {
  return value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(3);
}

function percentLabel(value?: number | null): string {
  return value == null || Number.isNaN(Number(value)) ? "—" : `${Number(value).toFixed(1)}%`;
}

function deltaLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(3)}`;
}

function compareRows(left: LeagueResultsStanding, right: LeagueResultsStanding, sort: SortKey): number {
  if (sort === "player") return left.player_name.localeCompare(right.player_name);
  if (sort === "record") {
    const leftPct = number(left.win_pct);
    const rightPct = number(right.win_pct);
    return rightPct - leftPct || number(right.wins) - number(left.wins);
  }
  const values: Record<Exclude<SortKey, "player" | "record">, keyof LeagueResultsStanding> = {
    rating: "rating_jupr",
    games: "matches_played",
    win_pct: "win_pct",
    improvement: "rating_delta_jupr"
  };
  return number(right[values[sort]] as number | null | undefined) - number(left[values[sort]] as number | null | undefined);
}

export default function LeaguePlayerRoster({ standings, clubSlug }: { standings: LeagueResultsStanding[]; clubSlug: string }) {
  const [sort, setSort] = useState<SortKey>("player");
  const rows = useMemo(
    () => [...standings].sort((left, right) => compareRows(left, right, sort) || left.player_name.localeCompare(right.player_name)),
    [sort, standings]
  );

  if (!rows.length) return <p>No public players are available for this league yet.</p>;

  return (
    <div data-testid="league-player-roster">
      <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.75rem", fontWeight: 700 }}>
        Sort roster by
        <select aria-label="Sort player roster" value={sort} onChange={(event) => setSort(event.target.value as SortKey)} style={{ padding: "0.35rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" }}>
          <option value="player">Player</option>
          <option value="rating">League rating</option>
          <option value="record">Record</option>
          <option value="games">Games</option>
          <option value="win_pct">Win percentage</option>
          <option value="improvement">Improvement</option>
        </select>
      </label>
      <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "14px", background: "white" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "720px" }}>
          <thead><tr><th style={thStyle}>Player</th><th style={thStyle}>League rating</th><th style={thStyle}>Record</th><th style={thStyle}>Games</th><th style={thStyle}>Win %</th><th style={thStyle}>Improvement</th></tr></thead>
          <tbody>
            {rows.map((row) => (
              <tr key={String(row.player_id)}>
                <td style={tdStyle}><Link href={`/clubs/${clubSlug}/players/${row.player_id}`}>{row.player_name}</Link></td>
                <td style={tdStyle}>{ratingLabel(row.rating_jupr)}</td>
                <td style={tdStyle}>{row.wins ?? 0}-{row.losses ?? 0}</td>
                <td style={tdStyle}>{row.matches_played ?? 0}</td>
                <td style={tdStyle}>{percentLabel(row.win_pct)}</td>
                <td style={tdStyle}>{deltaLabel(row.rating_delta_jupr)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
