"use client";

import { useState } from "react";
import { CompetitionDocument, CompetitionEncounter, CompetitionGame, CompetitionPairing, CompetitionPlayer, GameStatus, MeetCompetition, activeSinglesPlayers, fromLocalInput, gameStatusLabels, matchesSkillLevel, pairingLabels, playerNames, singlesCourt, toLocalInput } from "@/lib/interclubCompetition";
import styles from "./competition.module.css";

type Props = { document: CompetitionDocument; detail: MeetCompetition; players: Map<string, CompetitionPlayer>; clubName: (id: string) => string; disabled: boolean; onChange: (document: CompetitionDocument) => void };

export default function ScoreEditor({ document, detail, players, clubName, disabled, onChange }: Props) {
  const [division, setDivision] = useState("");
  function changeEncounter(id: string, patch: Partial<CompetitionEncounter>) {
    onChange({ ...document, encounters: document.encounters.map(encounter => encounter.id === id ? { ...encounter, ...patch } : encounter) });
  }
  function changePairing(encounter: CompetitionEncounter, id: string, patch: Partial<CompetitionPairing>) {
    changeEncounter(encounter.id, { pairings: encounter.pairings.map(pairing => pairing.id === id ? { ...pairing, ...patch } : pairing) });
  }
  function changeGame(encounter: CompetitionEncounter, pairing: CompetitionPairing, id: string, patch: Partial<CompetitionGame>) {
    changePairing(encounter, pairing.id, { games: pairing.games.map(game => game.id === id ? { ...game, ...patch } : game) });
  }
  function changeStartingPair(encounter: CompetitionEncounter, pairing: CompetitionPairing, side: "a" | "b", ids: string[]) {
    const clubId = encounter[`club_${side}`];
    onChange({ ...document, encounters: document.encounters.map(row => {
      if (row.division !== encounter.division) return row;
      const target = row.club_a === clubId ? "a" : row.club_b === clubId ? "b" : null;
      return target ? { ...row, pairings: row.pairings.map(line => line.kind === pairing.kind ? { ...line, [`players_${target}`]: ids } : line) } : row;
    }) });
  }
  const divisions = Array.from(new Set(document.encounters.map(encounter => encounter.division)));
  return <section className={styles.section} aria-labelledby="score-entry-heading">
    <div className={styles.toolbar}><div><h2 id="score-entry-heading">Enter the official score sheets</h2><p>Enter all results, save the draft, then submit the complete meet for organizer approval.</p></div>
      {divisions.length > 1 && <label>Show skill level<select value={division} onChange={event => setDivision(event.target.value)}><option value="">All skill levels</option>{divisions.map(value => <option key={value}>{value}</option>)}</select></label>}
    </div>
    <div className={styles.notice}>All three games are played in regular-season pairings. Only completed doubles games affect ratings. Enter the actual stopped score for an injury retirement; leave scores empty for an unplayed forfeit.</div>
    {document.encounters.filter(encounter => !division || encounter.division === division).map(encounter => <article key={encounter.id} className={styles.card}>
      <p className={styles.eyebrow}>Skill level {encounter.division} · Rotation {encounter.rotation}</p>
      <h3>{clubName(encounter.club_a)} <span className={styles.muted}>vs</span> {clubName(encounter.club_b)}</h3>
      {encounter.pairings.map(pairing => <section key={pairing.id} className={styles.pairing} aria-label={`${pairingLabels[pairing.kind]} ${encounter.division}`}>
        <h4>{pairingLabels[pairing.kind]}{pairing.court ? ` · Court ${pairing.court}` : ""}</h4>
        <p>{playerNames(pairing.players_a, players)} <strong>vs</strong> {playerNames(pairing.players_b, players)}</p>
        {!disabled && !document.encounters.some(row => row.pairings.some(line => line.games.some(game => game.status !== "pending"))) && <details className={styles.lineup}><summary>Arrange the starting pairing</summary><p>Choose from the approved four-player team. Changes apply to this club’s pairing against every opponent at this meet. The complete lineup is checked when you save.</p>
          <div className={styles.twoColumns}>{(["a", "b"] as const).map(side => {
            const roster = detail.teams.find(team => team.club_id === encounter[`club_${side}`] && team.division === encounter.division)?.roster || [];
            return <PlayerSelect key={side} label={clubName(encounter[`club_${side}`])} ids={pairing[`players_${side}`]} options={roster} disabled={disabled} onChange={ids => changeStartingPair(encounter, pairing, side, ids)} />;
          })}</div>
        </details>}
        <fieldset disabled={disabled} className={styles.gameFields}><legend className={styles.srOnly}>{pairingLabels[pairing.kind]} scores</legend>
          {pairing.games.map((game, index) => <div key={game.id} className={styles.game}>
            <div className={styles.gameRow}><strong>Game {index + 1}</strong>
              <label>Status<select aria-label={`${pairingLabels[pairing.kind]} game ${index + 1} status`} value={game.status} onChange={event => {
                const status = event.target.value as GameStatus;
                changeGame(encounter, pairing, game.id, { status, ...(status === "forfeit" || status === "double_forfeit" || status === "unplayed" ? { a: null, b: null } : {}), ...(["completed", "pending", "unplayed", "double_forfeit"].includes(status) ? { winner: null } : {}) });
              }}>{Object.entries(gameStatusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              <label>{clubName(encounter.club_a)}<input aria-label={`${pairing.id} game ${index + 1} club A score`} type="number" min={0} step={1} value={game.a ?? ""} disabled={["forfeit", "double_forfeit", "unplayed"].includes(game.status)} onChange={event => changeGame(encounter, pairing, game.id, { a: event.target.value === "" ? null : Number(event.target.value), ...(game.status === "completed" ? { winner: null } : {}) })} /></label>
              <label>{clubName(encounter.club_b)}<input aria-label={`${pairing.id} game ${index + 1} club B score`} type="number" min={0} step={1} value={game.b ?? ""} disabled={["forfeit", "double_forfeit", "unplayed"].includes(game.status)} onChange={event => changeGame(encounter, pairing, game.id, { b: event.target.value === "" ? null : Number(event.target.value), ...(game.status === "completed" ? { winner: null } : {}) })} /></label>
              {["retired", "forfeit"].includes(game.status) && <label>Game awarded to<select value={game.winner || ""} onChange={event => changeGame(encounter, pairing, game.id, { winner: event.target.value as "a" | "b" || null })}><option value="">Choose winner</option><option value="a">{clubName(encounter.club_a)}</option><option value="b">{clubName(encounter.club_b)}</option></select></label>}
              {["completed", "retired"].includes(game.status) && <label>Actual time played (your device’s time)<input required type="datetime-local" value={toLocalInput(game.played_at)} onChange={event => changeGame(encounter, pairing, game.id, { played_at: fromLocalInput(event.target.value) })} /></label>}
            </div>
            <details><summary>Time played and injury replacement</summary>
              {!['completed', 'retired'].includes(game.status) && <p>Choose a played-game status to record its actual play time. Scheduled time is never used as a confirmed game time.</p>}
              <p>After play begins, a player may be replaced only because of injury, between games. A retirement concedes the interrupted game; an eligible replacement can play the next game.</p>
              <div className={styles.twoColumns}>{(["a", "b"] as const).map(side => {
                const options = (detail.eligible_players?.[encounter[`club_${side}`]] || detail.teams.find(team => team.club_id === encounter[`club_${side}`] && team.division === encounter.division)?.roster || []).filter(player => matchesSkillLevel(player, encounter.division));
                return <PlayerSelect key={side} label={`${clubName(encounter[`club_${side}`])} actual players`} ids={game[`players_${side}`].length ? game[`players_${side}`] : pairing[`players_${side}`]} options={options} disabled={disabled} onChange={ids => changeGame(encounter, pairing, game.id, { [`players_${side}`]: ids })} />;
              })}</div>
              <label>Injury reason<textarea rows={2} value={game.injury_reason || ""} onChange={event => changeGame(encounter, pairing, game.id, { injury_reason: event.target.value || null })} placeholder="Who was injured and when the replacement entered" /></label>
            </details>
          </div>)}
        </fieldset>
      </section>)}
      {document.phase !== "regular" && <section className={styles.pairing}>
        <h4>Rotating singles — only at 2–2</h4>
        <p>{singlesCourt(encounter.division)} singles. Rally scoring to 21, win by two, no cap. Both teams rotate after every four rallies, repeating their fixed order. This game never affects individual ratings.</p>
        <fieldset disabled={disabled} className={styles.gameFields}><legend className={styles.srOnly}>Rotating singles tiebreak</legend>
          <label className={styles.check}><input type="checkbox" checked={!!encounter.tiebreak} onChange={event => changeEncounter(encounter.id, { tiebreak: event.target.checked ? { status: "pending", a: null, b: null, order_a: activeSinglesPlayers(encounter, "a"), order_b: activeSinglesPlayers(encounter, "b") } : null })} />A singles tiebreak is required</label>
          {encounter.tiebreak && <><div className={styles.gameRow}>
            <label>Status<select value={encounter.tiebreak.status} onChange={event => changeEncounter(encounter.id, { tiebreak: { ...encounter.tiebreak!, status: event.target.value as "pending" | "completed" } })}><option value="pending">Not completed</option><option value="completed">Completed</option></select></label>
            {(["a", "b"] as const).map(side => <label key={side}>{clubName(encounter[`club_${side}`])}<input type="number" min={0} step={1} value={encounter.tiebreak![side] ?? ""} onChange={event => changeEncounter(encounter.id, { tiebreak: { ...encounter.tiebreak!, [side]: event.target.value === "" ? null : Number(event.target.value) } })} /></label>)}
          </div><div className={styles.twoColumns}>{(["a", "b"] as const).map(side => <PlayerSelect key={side} label={`${clubName(encounter[`club_${side}`])} rotation order`} ids={encounter.tiebreak![`order_${side}`]} options={activeSinglesPlayers(encounter, side).map(id => players.get(id) || { entry_id: id, name: "Player unavailable" })} disabled={disabled} onChange={ids => changeEncounter(encounter.id, { tiebreak: { ...encounter.tiebreak!, [`order_${side}`]: ids } })} />)}</div></>}
        </fieldset>
      </section>}
    </article>)}
  </section>;
}

function PlayerSelect({ label, ids, options, disabled, onChange }: { label: string; ids: string[]; options: CompetitionPlayer[]; disabled: boolean; onChange: (ids: string[]) => void }) {
  return <fieldset className={styles.players} disabled={disabled}><legend>{label}</legend>{ids.map((id, index) => <label key={index}>Player {index + 1}<select value={id} onChange={event => onChange(ids.map((old, i) => i === index ? event.target.value : old))}>
    {!options.some(player => player.entry_id === id) && <option value={id}>Scheduled player</option>}
    {options.map(player => <option key={player.entry_id} value={player.entry_id}>{player.name}{(player.eligibility_rating ?? player.rating) != null ? ` · ${(player.eligibility_rating ?? player.rating)!.toFixed(3)}` : ""}</option>)}
  </select></label>)}</fieldset>;
}
