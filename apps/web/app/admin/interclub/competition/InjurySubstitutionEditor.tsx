"use client";

import { useState } from "react";
import SearchablePlayerSelect from "@/components/SearchablePlayerSelect";
import type { CompetitionDocument, CompetitionPlayer, MeetCompetition } from "@/lib/interclubCompetition";
import { GameRow, gamePlayers, replacementOptions, substituteForRemainingGames } from "@/lib/interclubSubstitutions";
import styles from "./competition.module.css";

export default function InjurySubstitutionEditor({ document, row, detail, players, clubName, disabled, onChange }: {
  document: CompetitionDocument; row: GameRow; detail: MeetCompetition; players: Map<string, CompetitionPlayer>;
  clubName: (id: string) => string; disabled: boolean; onChange: (document: CompetitionDocument) => void;
}) {
  const [selection, setSelection] = useState(""), [incoming, setIncoming] = useState(""), [note, setNote] = useState("");
  const [error, setError] = useState("");
  const side = selection.slice(0, 1) as "a" | "b", outgoing = selection.slice(2);
  const options = outgoing ? replacementOptions(document, row, side, outgoing,
    detail.eligible_players?.[row.encounter[`club_${side}`]] || detail.teams.find(team => team.club_id === row.encounter[`club_${side}`] && team.division === row.encounter.division)?.roster || [], players) : [];
  return <div>
    <p>Replace a player only because of injury, between games. Start on the first game the replacement plays; keep the injured player on any game they retired from.</p>
    <div className={styles.twoColumns}>
      <label>Injured player<select value={selection} disabled={disabled} onChange={event => { setSelection(event.target.value); setIncoming(""); setError(""); }}>
        <option value="">Choose the injured player</option>
        {(["a", "b"] as const).flatMap(value => gamePlayers(row, value).map(id => <option key={`${value}:${id}`} value={`${value}:${id}`}>{players.get(id)?.name || "Scheduled player"} · {clubName(row.encounter[`club_${value}`])}</option>))}
      </select></label>
      <label>Replacement<SearchablePlayerSelect aria-label="Replacement" value={incoming} disabled={disabled || !outgoing} onValueChange={setIncoming}>
        <option value="">Choose a replacement</option>{options.map(player => <option key={player.entry_id} value={player.entry_id}>{player.name}</option>)}
      </SearchablePlayerSelect></label>
    </div>
    {outgoing && !options.length && <p className={styles.warning}>No available eligible replacement was found in this club’s season pool.</p>}
    <p>The replacement stays in for this and all later games at this skill level. Injury is recorded automatically.</p>
    <details><summary>Add an injury note (optional)</summary><label>Injury note<textarea maxLength={1000} value={note} onChange={event => setNote(event.target.value)} rows={2} /></label></details>
    {error && <p role="alert" className={styles.error}>{error}</p>}
    <button type="button" className={styles.primary} disabled={disabled || !options.some(player => player.entry_id === incoming)} onClick={() => {
      try { onChange(substituteForRemainingGames(document, { gameId: row.game.id, side, outgoing, incoming }, note).document); setSelection(""); setIncoming(""); setNote(""); setError(""); }
      catch (cause) { setError(cause instanceof Error ? cause.message : "Check the replacement and try again."); }
    }}>Apply substitution to remaining games</button>
  </div>;
}
