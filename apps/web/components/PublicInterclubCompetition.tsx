"use client";

import { Fragment, useId, useState } from "react";
import type { CompetitionResult, PublicLeague, PublicMeet, ResultPlayer } from "@/lib/interclubPublic";
import { gameResultNote, hasResultPlayer, resultGameWinner, resultPairingNames, resultPlayers, sortSkillLevels } from "@/lib/interclubResultViews";
import SearchablePlayerSelect from "./SearchablePlayerSelect";
import styles from "./InterclubResults.module.css";

export function CompetitionStandings({ league, onClubSelect }: { league: PublicLeague; onClubSelect?: (club: string) => void }) {
  const [division, setDivision] = useState("");
  const names = Object.fromEntries(league.document.clubs.map(club => [club.id, club.name]));
  const group = league.standings.find(group => group.division === division);
  const qualification = league.qualification?.[division];
  const cup = league.club_cup;
  const options = sortSkillLevels([...new Set([...league.document.divisions, ...league.standings.map(group => group.division)])]);
  const label = division ? `${division} standings` : "Club Cup standings";
  const clubLabel = (id: string, fallback?: string) => onClubSelect ? <button type="button" className={styles.clubButton} onClick={() => onClubSelect(id)} aria-label={`View results for ${names[id] || fallback || id}`}>{names[id] || fallback || id}</button> : names[id] || fallback || id;
  return <section className={styles.workspace} aria-label="Season standings">
    <div className={styles.heading}><div><h2>{division ? `${division} skill level` : "Overall Club Cup"}</h2><p className={styles.meta}>{division ? "3 points for a matchup win · 1 each for a split" : "Season points across all skill levels"}</p></div>
      <label className={styles.field}>Standings<select aria-label="Standings" value={division} onChange={event => setDivision(event.target.value)}><option value="">Overall Club Cup</option>{options.map(value => <option key={value} value={value}>{value} skill level</option>)}</select></label>
    </div>
    <div className={styles.scroll} role="region" aria-label={label} tabIndex={0}><table className={styles.table} aria-label={label}>
      <thead><tr><th scope="col">Place</th><th scope="col">Club</th>{division ? <><th scope="col">Points</th><th scope="col">Meets</th><th scope="col">Pairings won</th><th scope="col">Games won</th><th scope="col">Point difference</th></> : <th scope="col">Total points</th>}</tr></thead>
      <tbody>{division ? (group?.rows || []).map((row, index) => <tr key={row.club_id}><td>{row.tied ? "T" : ""}{row.position || index + 1}</td><th scope="row">{clubLabel(row.club_id, row.name)}</th><td className={styles.points}>{row.points ?? 0}</td><td>{row.meets_played ?? 0}</td><td>{row.pairings_won ?? 0}</td><td>{row.games_won}</td><td>{(row.point_differential ?? 0) > 0 ? "+" : ""}{row.point_differential ?? 0}</td></tr>) : (cup?.standings || []).map((row, index) => <tr key={row.club_id}><td>{row.tied ? "T" : ""}{row.position || index + 1}</td><th scope="row">{clubLabel(row.club_id, row.name)}<small className={styles.breakdown}>{row.regular_points} regular season · {row.championship_points} finals bonus</small></th><td className={styles.points}>{row.points}</td></tr>)}</tbody>
    </table></div>
    {!(division ? group?.rows.length : cup?.standings.length) && <p className={styles.empty}>No official results yet.</p>}
    {!division && (cup?.status === "complete" && cup.champions.length ? <p><strong>{cup.champions.length > 1 ? "Joint Club Cup champions" : "Club Cup champion"}:</strong> {cup.champions.map(id => names[id] || id).join(" · ")}</p> : <p className={styles.meta}>Standings are provisional. Each skill-level final adds 6 points for the winner and 3 for the runner-up.</p>)}
    {division && <p className={styles.meta}>Ties: pairings won, games won, point difference, then head-to-head. Weather draws follow the league’s partial-result rules.</p>}
    {division && qualification?.status === "playoff_required" && <p><strong>Qualifying playoff needed:</strong> {Array.isArray(qualification.playoff_required) ? qualification.playoff_required.map(id => names[id] || id).join(" · ") : "Tied clubs"}. A playoff decides the championship places; no club advances on alphabetical order.</p>}
    {division && qualification?.status === "insufficient_entries" && <p>At least two clubs need a regular-season appearance before a final can be set.</p>}
    {division && !!qualification?.qualifiers?.length && <p className={styles.meta}><strong>Current top two:</strong> {qualification.qualifiers.map(id => names[id] || id).join(" · ")}</p>}
  </section>;
}

function meetLabel(meet: PublicMeet | undefined, index: number, zone: string, names: Record<string, string>) {
  if (!meet) return "Meet results";
  const date = new Intl.DateTimeFormat("en-US", { dateStyle: "medium", timeZone: zone }).format(new Date(meet.starts_at));
  return `Meet ${index + 1} · ${date}${meet.host_club_id && names[meet.host_club_id] ? ` · ${names[meet.host_club_id]}` : ""}`;
}

export function CompetitionResults({ results, names, meets = [], timezone = "America/Mazatlan", players = [], initialClub = "", singleMeet = false }: {
  results: CompetitionResult[]; names: Record<string, string>; meets?: PublicMeet[]; timezone?: string; players?: ResultPlayer[]; initialClub?: string; singleMeet?: boolean;
}) {
  const [meet, setMeet] = useState(""), [club, setClub] = useState(initialClub), [division, setDivision] = useState(""), [player, setPlayer] = useState("");
  const playerInputId = useId();
  const orderedMeets = [...meets].sort((a, b) => a.starts_at.localeCompare(b.starts_at));
  const meetNames = Object.fromEntries(orderedMeets.map((meet, index) => [meet.id, meetLabel(meet, index, timezone, names)]));
  const catalog = Object.fromEntries(players.map(player => [player.id, player]));
  const base = results.filter(row => (!meet || row.meet_id === meet) && (!club || row.club_a === club || row.club_b === club) && (!division || row.division === division));
  const availableIds = new Set(base.flatMap(row => [...row.pairings.flatMap(pairing => pairing.games.flatMap(resultPlayers)), ...(row.tiebreak ? resultPlayers(row.tiebreak) : [])]));
  const playerOptions = players.filter(row => availableIds.has(row.id) && (!club || row.club_id === club)).sort((a, b) => a.name.localeCompare(b.name));
  const filtered = base.filter(row => hasResultPlayer(row, player));
  const visibleMeetIds = [...new Set(filtered.map(row => row.meet_id))].sort((a, b) => {
    const left = orderedMeets.findIndex(meet => meet.id === a), right = orderedMeets.findIndex(meet => meet.id === b);
    return right - left || a.localeCompare(b);
  });
  const gameCount = filtered.reduce((sum, row) => sum + row.pairings.reduce((n, pairing) => n + pairing.games.filter(game => !player || resultPlayers(game).includes(player)).length, 0), 0);
  const reset = () => { setMeet(""); setClub(""); setDivision(""); setPlayer(""); };
  const fieldChanged = (change: () => void) => { change(); setPlayer(""); };
  const participantNames = (ids: string[] | undefined) => (ids || []).map(id => catalog[id]?.name || "Player").join(" / ");
  if (!results.length) return <p className={styles.empty}>Results will appear after the organizer approves and publishes them.</p>;
  return <section className={styles.workspace} aria-label="Browse match results">
    <div className={`${styles.filters} ${singleMeet ? styles.meetFilters : ""}`}>
      {!singleMeet && <label className={styles.field}>Meet<select aria-label="Filter results by meet" value={meet} onChange={event => fieldChanged(() => setMeet(event.target.value))}><option value="">All meets</option>{orderedMeets.filter(meet => results.some(row => row.meet_id === meet.id)).map(meet => <option key={meet.id} value={meet.id}>{meetNames[meet.id]}</option>)}</select></label>}
      <label className={styles.field}>Club<select aria-label="Filter results by club" value={club} onChange={event => fieldChanged(() => setClub(event.target.value))}><option value="">All clubs</option>{Object.entries(names).sort((a, b) => a[1].localeCompare(b[1])).map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label>
      <label className={styles.field}>Skill level<select aria-label="Filter results by skill level" value={division} onChange={event => fieldChanged(() => setDivision(event.target.value))}><option value="">All skill levels</option>{sortSkillLevels([...new Set(results.map(row => row.division))]).map(value => <option key={value} value={value}>{value}</option>)}</select></label>
      <label className={styles.field} htmlFor={playerInputId}>Player<SearchablePlayerSelect id={playerInputId} aria-label="Filter results by player" disabled={!players.length} value={player} onValueChange={setPlayer}><option value="">All players</option>{playerOptions.map(row => <option key={row.id} value={row.id}>{row.name} · {names[row.club_id] || row.club_id}</option>)}</SearchablePlayerSelect></label>
    </div>
    <div className={styles.count}><p className={styles.meta} role="status">{filtered.length} {filtered.length === 1 ? "matchup" : "matchups"} · {gameCount} doubles {gameCount === 1 ? "game" : "games"}{player ? ` involving ${catalog[player]?.name || "this player"}` : ""}</p>{(meet || club || division || player) && <button type="button" className={styles.reset} onClick={reset}>Clear filters</button>}</div>
    {!players.length && <p className={styles.meta}>Player details are not available for these results yet.</p>}
    {!filtered.length && <p className={styles.empty}>No results match these filters. Try another selection or clear the filters.</p>}
    {visibleMeetIds.map(meetId => <section key={meetId} aria-label={meetNames[meetId] || "Meet results"}>
      {!singleMeet && <h3 className={styles.meetHeading}>{meetNames[meetId] || "Meet results"}</h3>}
      <div className={styles.matches}>{filtered.filter(row => row.meet_id === meetId).sort((a, b) => a.division.localeCompare(b.division, undefined, { numeric: true }) || (names[a.club_a] || a.club_a).localeCompare(names[b.club_a] || b.club_a) || a.id.localeCompare(b.id)).map(row => {
        const games = row.pairings.flatMap(pairing => pairing.games);
        const winsA = games.filter(game => resultGameWinner(game) === "a").length, winsB = games.filter(game => resultGameWinner(game) === "b").length;
        const outcome = row.outcome;
        const tieVisible = row.tiebreak?.status === "completed" && (!player || resultPlayers(row.tiebreak).includes(player));
        return <details key={`${row.meet_id}:${row.phase}:${row.id}`} className={styles.match} open={player ? true : undefined}>
          <summary aria-label={`${row.division} · ${names[row.club_a] || row.club_a} vs ${names[row.club_b] || row.club_b}`}><span><span className={styles.meta}>{row.division} · {row.phase === "final" ? "Championship final" : row.phase === "qualifier" ? "Qualifying playoff" : "Regular season"}</span><span className={styles.matchTitle}>{names[row.club_a] || row.club_a} <span className={styles.meta}>vs</span> {names[row.club_b] || row.club_b}</span><span className={styles.scoreLine}>{outcome ? `${outcome.pairings_a}–${outcome.pairings_b} pairings · ` : ""}{winsA}–{winsB} games{outcome && row.phase === "regular" ? ` · ${outcome.points_a}–${outcome.points_b} standings points` : ""}{outcome?.winner === "draw" ? " · Draw" : ""}{outcome?.winner === "double_forfeit" ? " · Both clubs forfeited" : ""}</span></span></summary>
          <div className={styles.details}>
            {row.weather === "finalized_partial" && <p className={styles.meta}>Finalized using available results; no further play could be scheduled.</p>}
            {row.pairings.map(pairing => {
              const visibleGames = pairing.games.map((game, index) => ({ game, index })).filter(({ game }) => !player || resultPlayers(game).includes(player));
              if (!visibleGames.length) return null;
              return <section key={pairing.kind}><h4>{resultPairingNames[pairing.kind] || pairing.kind}</h4><table className={styles.gameTable} aria-label={`${resultPairingNames[pairing.kind] || pairing.kind} scores`}>
                <thead><tr><th scope="col">Game</th><th scope="col">{names[row.club_a] || row.club_a}</th><th scope="col">{names[row.club_b] || row.club_b}</th></tr></thead>
                <tbody>{visibleGames.map(({ game, index }) => <Fragment key={index}><tr><th scope="row">{index + 1}</th>{(["a", "b"] as const).map(side => <td key={side} className={resultGameWinner(game) === side ? styles.winner : undefined}><strong>{game[side] ?? "—"}</strong><small>{participantNames(game[`players_${side}`])}</small></td>)}</tr>{gameResultNote(game, names, row) && <tr><td colSpan={3} className={styles.note}>{game.a !== null && game.b !== null ? `${game.a}–${game.b} · ` : ""}{gameResultNote(game, names, row)}</td></tr>}</Fragment>)}</tbody>
              </table></section>;
            })}
            {tieVisible && row.tiebreak && <p><strong>Rotating singles:</strong> {row.tiebreak.a}–{row.tiebreak.b}.<br />{participantNames(row.tiebreak.players_a)} vs {participantNames(row.tiebreak.players_b)}<br /><span className={styles.meta}>This tiebreak does not affect individual ratings.</span></p>}
          </div>
        </details>;
      })}</div>
    </section>)}
  </section>;
}
