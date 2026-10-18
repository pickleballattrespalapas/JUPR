"use client";

import { useState } from "react";
import SearchablePlayerSelect from "@/components/SearchablePlayerSelect";
import type { AdminMatchUploaderRoundRobinMatch } from "@/lib/adminMatchUploaderApi";
import { actualGamePlayers, gameEditError, gameHasSubstitution, scheduledGamePlayers, type LeagueLiveGameEdit } from "./leagueLiveGameEdits";

const inputStyle = { width: "100%", minWidth: 0, boxSizing: "border-box" as const, padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
type Props = {
  match: AdminMatchUploaderRoundRobinMatch;
  gameNumber: number;
  edit?: LeagueLiveGameEdit;
  players: Array<{ id: number; name: string }>;
  disabled: boolean;
  allowUnplayed: boolean;
  onChange: (edit: LeagueLiveGameEdit) => void;
};
export function LeagueLiveGameEditor({ match, gameNumber, edit = {}, players, disabled, allowUnplayed, onChange }: Props) {
  const [expanded, setExpanded] = useState(false);
  const ids = actualGamePlayers(match, edit);
  const changed = gameHasSubstitution(match, edit);
  const error = gameEditError(match, edit);
  const prefix = `Court ${match.court} ${match.label} Game ${gameNumber}`;
  const names = new Map(players.map(player => [Number(player.id), player.name]));
  const team = (teamIds: number[]) => teamIds.map(id => names.get(id) || `Player ${id}`).join(" / ");
  return <div style={{ gridColumn: "1 / -1", display: "grid", gap: "0.45rem" }}>
    {allowUnplayed ? <label><input type="checkbox" aria-label={`${prefix} Not played or unfinished`} checked={Boolean(edit.notPlayed)} disabled={disabled} onChange={event => onChange({ ...edit, notPlayed: event.target.checked })} /> Not played / unfinished</label> : null}
    {edit.notPlayed ? <label>Reason this game was not completed *<input aria-label={`${prefix} Unplayed reason`} maxLength={500} value={edit.unplayedReason || ""} disabled={disabled} onChange={event => onChange({ ...edit, unplayedReason: event.target.value })} placeholder="For example: stopped due to heat" style={inputStyle} /></label> : <details open={expanded || changed} onToggle={event => setExpanded(event.currentTarget.open)}>
      <summary>Who played this game?{changed ? " · substitution recorded" : " · change a player"}</summary>
      {expanded || changed ? <>
      <p style={{ margin: "0.35rem 0" }}>{team(ids.slice(0, 2))} vs {team(ids.slice(2))}</p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.5rem" }}>
        {ids.map((id, index) => <label key={index}>Team {index < 2 ? 1 : 2}, player {index % 2 + 1}<SearchablePlayerSelect aria-label={`${prefix} Team ${index < 2 ? 1 : 2} player ${index % 2 + 1}`} value={String(id)} disabled={disabled} onValueChange={value => onChange({ ...edit, playerIds: ids.map((current, position) => position === index ? Number(value) : current) })} style={inputStyle}>{players.map(player => <option key={player.id} value={String(player.id)}>{player.name}</option>)}</SearchablePlayerSelect></label>)}
      </div>
      {changed ? <><label>Substitution reason *<input aria-label={`${prefix} Substitution reason`} maxLength={500} value={edit.substitutionReason || ""} disabled={disabled} onChange={event => onChange({ ...edit, substitutionReason: event.target.value })} placeholder="For example: staff filled in for game two" style={inputStyle} /></label><button type="button" disabled={disabled} onClick={() => onChange({ ...edit, playerIds: scheduledGamePlayers(match), substitutionReason: "" })}>Restore scheduled players</button></> : null}
      <small>This changes only this game. Other games keep their scheduled players.</small>
      </> : null}
    </details>}
    {error ? <small role="alert" style={{ color: "#b45309" }}>{error}</small> : null}
  </div>;
}
