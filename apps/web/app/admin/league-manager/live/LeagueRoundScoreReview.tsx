"use client";

import { useState } from "react";
import { roundCourtStandings, type RoundScoreGame } from "./leagueRoundScores";

export function LeagueRoundScoreReview({ games, disabled = false, onEdit }: {
  games: RoundScoreGame[];
  disabled?: boolean;
  onEdit?: (game: RoundScoreGame) => void;
}) {
  const [courtFilter, setCourtFilter] = useState("");
  const [playerFilter, setPlayerFilter] = useState("");
  const courts = [...new Set(games.map(game => game.court))].sort((a, b) => a - b);
  const query = playerFilter.trim().toLocaleLowerCase();
  const includesPlayer = (game: RoundScoreGame) => !query || [...game.team1, ...game.team2].some(player => player.name.toLocaleLowerCase().includes(query));
  const visibleCourts = courts.filter(court => (!courtFilter || String(court) === courtFilter) && games.some(game => game.court === court && includesPlayer(game)));
  return <section aria-label="Round scores by court">
    <div style={{ display: "flex", gap: ".75rem", flexWrap: "wrap", alignItems: "end", marginBottom: ".75rem" }}>
      <label>Court<br /><select aria-label="Filter scores by court" value={courtFilter} onChange={event => setCourtFilter(event.target.value)}><option value="">All courts</option>{courts.map(court => <option key={court} value={court}>Court {court}</option>)}</select></label>
      <label>Find player<br /><input type="search" value={playerFilter} onChange={event => setPlayerFilter(event.target.value)} placeholder="e.g. Shannon" /></label>
      {courtFilter || playerFilter ? <button type="button" onClick={() => { setCourtFilter(""); setPlayerFilter(""); }}>Show all courts</button> : null}
    </div>
    {!visibleCourts.length ? <p>No games match this court or player.</p> : null}
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 340px), 1fr))", gap: ".75rem" }}>
      {visibleCourts.map(court => {
        const courtGames = games.filter(game => game.court === court);
        const standings = roundCourtStandings(courtGames);
        const filteredGames = courtGames.filter(includesPlayer);
        return <section key={court} style={{ border: "1px solid #cbd5e1", borderRadius: 12, padding: ".75rem", background: "white" }}>
          <h4 style={{ margin: "0 0 .5rem" }}>Court {court}</h4>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <caption style={{ textAlign: "left", color: "#475569", fontSize: ".85rem" }}>Full round record</caption>
            <thead><tr><th align="left">Player</th><th align="center">W–L</th><th align="right">+/−</th></tr></thead>
            <tbody>{standings.map(row => <tr key={row.id} style={{ background: query && row.name.toLocaleLowerCase().includes(query) ? "#fef3c7" : undefined }}>
              <td>{row.name}</td><td align="center"><strong>{row.wins}–{row.losses}</strong></td><td align="right">{row.differential > 0 ? "+" : ""}{row.differential}</td>
            </tr>)}</tbody>
          </table>
          <details open={Boolean(query || courtFilter || courtGames.some(game => game.unusualReason))}>
            <summary style={{ cursor: "pointer", padding: ".6rem 0", fontWeight: 700 }}>View {filteredGames.length} game{filteredGames.length === 1 ? "" : "s"}{query ? " for this player" : ""}</summary>
            {filteredGames.map(game => <div key={game.key} style={{ borderTop: "1px solid #e2e8f0", padding: ".6rem 0" }}>
              <div style={{ color: "#475569", fontSize: ".85rem" }}>{game.gameLabel}{game.corrected ? " · Corrected" : ""}</div>
              <div><strong>{game.team1.map(player => player.name).join(" / ")}</strong> · {game.notPlayed ? "—" : game.scoreT1 ?? "—"}</div>
              <div><strong>{game.team2.map(player => player.name).join(" / ")}</strong> · {game.notPlayed ? "—" : game.scoreT2 ?? "—"}</div>
              {game.notPlayed ? <small>Not counted</small> : null}
              {game.note ? <small style={{ display: "block" }}>{game.note}</small> : null}
              {game.unusualReason ? <p role="alert" style={{ color: "#92400e" }}>{game.unusualReason}</p> : null}
              {onEdit ? <button type="button" disabled={disabled} onClick={() => onEdit(game)}>Edit this game</button> : game.correctionUrl ? <a href={game.correctionUrl} target="_blank" rel="noopener noreferrer">Correct this score ↗</a> : null}
            </div>)}
          </details>
        </section>;
      })}
    </div>
  </section>;
}
