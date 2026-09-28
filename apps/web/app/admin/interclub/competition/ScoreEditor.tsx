"use client";

import SearchablePlayerSelect from "@/components/SearchablePlayerSelect";

import { useEffect, useRef, useState, type FocusEvent, type KeyboardEvent } from "react";
import { CompetitionDocument, CompetitionEncounter, CompetitionGame, CompetitionPairing, CompetitionPlayer, GameStatus, MeetCompetition, activeSinglesPlayers, automaticGameStatus, gameStatusLabels, isFinalScore, matchesSkillLevel, pairingLabels, playerNames, scheduledEncounters, scheduleRoundLabel, singlesCourt } from "@/lib/interclubCompetition";
import styles from "./competition.module.css";
import InjurySubstitutionEditor from "./InjurySubstitutionEditor";

type Props = { document: CompetitionDocument; detail: MeetCompetition; players: Map<string, CompetitionPlayer>; clubName: (id: string) => string; disabled: boolean; onChange: (document: CompetitionDocument) => void; divisionFilter?: string; onDivisionFilterChange?: (division: string) => void; onScoreEntryEnd?: () => boolean; focusGame?: { id: string; substitution: boolean; sequence: number } | null; gameError?: string; onSubstitution?: (document: CompetitionDocument) => void };

export default function ScoreEditor({ document, detail, players, clubName, disabled, onChange, divisionFilter, onDivisionFilterChange, onScoreEntryEnd, focusGame, gameError, onSubstitution }: Props) {
  const [localDivision, setLocalDivision] = useState("");
  const scoreEntry = useRef<HTMLElement | null>(null);
  const division = divisionFilter ?? localDivision, setDivision = onDivisionFilterChange ?? setLocalDivision;
  useEffect(() => {
    if (!focusGame) return;
    const game = scoreEntry.current?.ownerDocument.getElementById(`interclub-game-${focusGame.id}`);
    if (!game) return;
    const options = game.querySelector<HTMLDetailsElement>(focusGame.substitution ? "[data-substitution]" : "details");
    if (options) options.open = true;
    game.scrollIntoView({ block: "center" }); game.focus({ preventScroll: true });
  }, [focusGame, division]);
  function navigateScores(event: KeyboardEvent<HTMLInputElement>) {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key === "Escape") {
      const options = event.currentTarget.closest("[data-score-game]")?.querySelector("details");
      if (options) { event.preventDefault(); options.open = true; options.querySelector("summary")?.focus(); }
      return;
    }
    if (event.key !== "Tab") return;
    const inputs = Array.from(scoreEntry.current?.querySelectorAll<HTMLInputElement>("input[data-interclub-score]") || []).filter(input => !input.matches(":disabled"));
    const next = inputs[inputs.indexOf(event.currentTarget) + (event.shiftKey ? -1 : 1)];
    if (next) { event.preventDefault(); next.focus(); }
    else if (!event.shiftKey && onScoreEntryEnd?.()) event.preventDefault();
  }
  const scoreInput = { "data-interclub-score": true, inputMode: "numeric" as const, onKeyDown: navigateScores, onFocus: (event: FocusEvent<HTMLInputElement>) => event.currentTarget.select() };
  function changeEncounter(id: string, patch: Partial<CompetitionEncounter>) {
    onChange({ ...document, encounters: document.encounters.map(encounter => encounter.id === id ? { ...encounter, ...patch } : encounter) });
  }
  function changePairing(encounter: CompetitionEncounter, id: string, patch: Partial<CompetitionPairing>) {
    changeEncounter(encounter.id, { pairings: encounter.pairings.map(pairing => pairing.id === id ? { ...pairing, ...patch } : pairing) });
  }
  function changeGame(encounter: CompetitionEncounter, pairing: CompetitionPairing, id: string, patch: Partial<CompetitionGame>) {
    changePairing(encounter, pairing.id, { games: pairing.games.map(game => game.id === id ? automaticGameStatus({ ...game, ...patch }) : game) });
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
  const canChangePartners = !disabled && document.encounters.every(encounter => encounter.pairings.every(pairing => pairing.games.every(game =>
    game.status === "pending" && game.a === null && game.b === null ||
    ["forfeit", "double_forfeit"].includes(game.status) && (!pairing.players_a.length || !pairing.players_b.length))));
  return <section ref={scoreEntry} className={styles.section} aria-labelledby="score-entry-heading">
    <div className={styles.toolbar}><div><h2 id="score-entry-heading">Enter the official score sheets</h2><p>Enter all results, save the draft, then submit the complete meet for organizer approval.</p></div>
      {divisions.length > 1 && <label>Show skill level<select value={division} onChange={event => setDivision(event.target.value)}><option value="">All skill levels</option>{divisions.map(value => <option key={value}>{value}</option>)}</select></label>}
    </div>
    <div className={styles.notice}><strong>Score → Tab → score → Tab → next game.</strong> Both final scores mark a game completed automatically. Shift+Tab moves back; Esc reaches game options. Game dates are recorded automatically from your device when you enter scores.</div>
    <p>The players are already set. For an injury, choose “Substitute a player” on the replacement’s first game. The change carries through the remaining games automatically.</p>
    {scheduledEncounters(document).filter(encounter => !division || encounter.division === division).map(encounter => <article key={encounter.id} className={styles.card}>
      <p className={styles.eyebrow}>Skill level {encounter.division} · {scheduleRoundLabel(document)} {encounter.rotation}</p>
      <h3>{clubName(encounter.club_a)} <span className={styles.muted}>vs</span> {clubName(encounter.club_b)}</h3>
      {encounter.pairings.map(pairing => <section key={pairing.id} className={styles.pairing} aria-label={`${pairingLabels[pairing.kind]} ${encounter.division}`}>
        <h4>{pairingLabels[pairing.kind]}{pairing.court ? ` · Court ${pairing.court}` : ""}</h4>
        <p><strong>Starting players:</strong> {playerNames(pairing.players_a, players)} <strong>vs</strong> {playerNames(pairing.players_b, players)}</p>
        {canChangePartners && pairing.kind.startsWith("mixed") && <details className={styles.lineup}><summary>Change mixed partners before play</summary><p>The four team players are already selected. Use this only to change who partners with whom in Mixed A and Mixed B. Keep each player in one mixed pair. Changes apply against every opponent at this meet and are checked when you save.</p>
          <div className={styles.twoColumns}>{(["a", "b"] as const).map(side => {
            const roster = detail.teams.find(team => team.club_id === encounter[`club_${side}`] && team.division === encounter.division)?.roster || [];
            return <PlayerSelect key={side} label={clubName(encounter[`club_${side}`])} ids={pairing[`players_${side}`]} options={roster} disabled={disabled} onChange={ids => changeStartingPair(encounter, pairing, side, ids)} />;
          })}</div>
        </details>}
        <fieldset disabled={disabled} className={styles.gameFields}><legend className={styles.srOnly}>{pairingLabels[pairing.kind]} scores</legend>
          {pairing.games.map((game, index) => <div key={game.id} id={`interclub-game-${game.id}`} data-score-game tabIndex={-1} className={`${styles.game}${focusGame?.id === game.id ? ` ${styles.highlightedGame}` : ""}`}>
            {focusGame?.id === game.id && gameError && <p className={styles.error} role="alert">{gameError}</p>}
            {!!(game.players_a.length || game.players_b.length) && <p><strong>Players for Game {index + 1}:</strong> {playerNames(game.players_a.length ? game.players_a : pairing.players_a, players)} <strong>vs</strong> {playerNames(game.players_b.length ? game.players_b : pairing.players_b, players)}</p>}
            <div className={styles.gameRow}><strong>Game {index + 1}</strong>
              <label>{clubName(encounter.club_a)}<input {...scoreInput} aria-label={`${pairing.id} game ${index + 1} club A score`} type="number" min={0} step={1} value={game.a ?? ""} disabled={["forfeit", "double_forfeit", "unplayed"].includes(game.status)} onChange={event => changeGame(encounter, pairing, game.id, { a: event.target.value === "" ? null : Number(event.target.value), ...(game.status === "completed" ? { winner: null } : {}) })} /></label>
              <label>{clubName(encounter.club_b)}<input {...scoreInput} aria-label={`${pairing.id} game ${index + 1} club B score`} type="number" min={0} step={1} value={game.b ?? ""} disabled={["forfeit", "double_forfeit", "unplayed"].includes(game.status)} onChange={event => changeGame(encounter, pairing, game.id, { b: event.target.value === "" ? null : Number(event.target.value), ...(game.status === "completed" ? { winner: null } : {}) })} /></label>
              <span className={styles.muted}>{game.status === "completed" ? "Completed" : game.status === "pending" ? game.a === null && game.b === null ? "Awaiting scores" : "Score incomplete" : gameStatusLabels[game.status]}</span>
              {["retired", "forfeit"].includes(game.status) && <label>Game awarded to<select value={game.winner || ""} onChange={event => changeGame(encounter, pairing, game.id, { winner: event.target.value as "a" | "b" || null })}><option value="">Choose winner</option><option value="a">{clubName(encounter.club_a)}</option><option value="b">{clubName(encounter.club_b)}</option></select></label>}
            </div>
            {game.status === "pending" && game.a !== null && game.b !== null && <p className={styles.warning}>A final score must reach 11 and win by two. Beyond 11, the winning margin must be exactly two. If play stopped because of injury, select Injury retirement below.</p>}
            <details><summary>No score or injury</summary>
              <label>Game outcome<select aria-label={`${pairingLabels[pairing.kind]} game ${index + 1} status`} value={["pending", "completed"].includes(game.status) ? "automatic" : game.status} onChange={event => {
                const status: GameStatus = event.target.value === "automatic" ? "pending" : event.target.value as GameStatus;
                changeGame(encounter, pairing, game.id, { status, winner: null, ...(["forfeit", "double_forfeit", "unplayed"].includes(status) ? { a: null, b: null, played_at: null } : {}) });
              }}><option value="automatic">Use entered scores automatically</option>{(["retired", "forfeit", "double_forfeit", "unplayed"] as const).map(value => <option key={value} value={value}>{gameStatusLabels[value]}</option>)}</select></label>
              <p>Leave scores empty for an unplayed game. For an injury retirement, keep the stopped score and choose the winning club.</p>
            </details>
            <details data-substitution><summary>Substitute a player · Game {index + 1}</summary>
              <InjurySubstitutionEditor document={document} row={{ encounter, pairing, game, number: index + 1 }} detail={detail} players={players} clubName={clubName} disabled={disabled} onChange={onSubstitution || onChange} />
              <details><summary>Correct a previously entered lineup</summary>
              <p>Use this only to fix a mistaken player entry on this game. For an injury replacement, use the controls above to update all remaining games together.</p>
              <div className={styles.twoColumns}>{(["a", "b"] as const).map(side => {
                const options = (detail.eligible_players?.[encounter[`club_${side}`]] || detail.teams.find(team => team.club_id === encounter[`club_${side}`] && team.division === encounter.division)?.roster || []).filter(player => matchesSkillLevel(player, encounter.division));
                return <PlayerSelect key={side} label={`${clubName(encounter[`club_${side}`])} actual players`} ids={game[`players_${side}`].length ? game[`players_${side}`] : pairing[`players_${side}`]} options={options} disabled={disabled} onChange={ids => changeGame(encounter, pairing, game.id, { [`players_${side}`]: ids })} />;
              })}</div>
              <label>Injury reason<textarea rows={2} value={game.injury_reason || ""} onChange={event => changeGame(encounter, pairing, game.id, { injury_reason: event.target.value || null })} placeholder="Injury" /></label>
              </details>
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
            {(["a", "b"] as const).map(side => <label key={side}>{clubName(encounter[`club_${side}`])}<input {...scoreInput} aria-label={`${encounter.id} singles club ${side.toUpperCase()} score`} type="number" min={0} step={1} value={encounter.tiebreak![side] ?? ""} onChange={event => {
              const tie = { ...encounter.tiebreak!, [side]: event.target.value === "" ? null : Number(event.target.value) };
              changeEncounter(encounter.id, { tiebreak: { ...tie, status: isFinalScore(tie.a, tie.b, 21) ? "completed" : "pending" } });
            }} /></label>)}
            <span className={styles.muted}>{encounter.tiebreak.status === "completed" ? "Completed" : "Enter both final scores to complete the tiebreak"}</span>
          </div><div className={styles.twoColumns}>{(["a", "b"] as const).map(side => <PlayerSelect key={side} label={`${clubName(encounter[`club_${side}`])} rotation order`} ids={encounter.tiebreak![`order_${side}`]} options={activeSinglesPlayers(encounter, side).map(id => players.get(id) || { entry_id: id, name: "Player unavailable" })} disabled={disabled} onChange={ids => changeEncounter(encounter.id, { tiebreak: { ...encounter.tiebreak!, [`order_${side}`]: ids } })} />)}</div></>}
        </fieldset>
      </section>}
    </article>)}
  </section>;
}

function PlayerSelect({ label, ids, options, disabled, onChange }: { label: string; ids: string[]; options: CompetitionPlayer[]; disabled: boolean; onChange: (ids: string[]) => void }) {
  return <fieldset className={styles.players} disabled={disabled}><legend>{label}</legend>{ids.map((id, index) => <label key={index}>Player {index + 1}<SearchablePlayerSelect aria-label={["Player",String(index + 1)].join(" ")} value={id} onValueChange={playerValue => onChange(ids.map((old, i) => i === index ? playerValue : old))}>
    {!options.some(player => player.entry_id === id) && <option value={id}>Scheduled player</option>}
    {options.map(player => <option key={player.entry_id} value={player.entry_id}>{player.name}{(player.eligibility_rating ?? player.rating) != null ? ` · ${(player.eligibility_rating ?? player.rating)!.toFixed(3)}` : ""}</option>)}
  </SearchablePlayerSelect></label>)}</fieldset>;
}
