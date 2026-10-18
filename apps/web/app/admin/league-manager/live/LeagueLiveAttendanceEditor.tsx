"use client";

import { useState } from "react";
import SearchablePlayerSelect from "@/components/SearchablePlayerSelect";

type Player = { id: number; name: string };
type Props = {
  roster: Player[];
  availablePlayers: Player[];
  activePlayerIds: number[];
  courtSizes: string;
  disabled: boolean;
  onActivePlayersChange: (ids: number[]) => void;
  onCourtSizesChange: (value: string) => void;
};
const inputStyle = { width: "100%", minWidth: 0, boxSizing: "border-box" as const, padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
export function LeagueLiveAttendanceEditor({ roster, availablePlayers, activePlayerIds, courtSizes, disabled, onActivePlayersChange, onCourtSizesChange }: Props) {
  const [incomingId, setIncomingId] = useState("");
  const [filter, setFilter] = useState("");
  const selected = new Set(activePlayerIds);
  const byId = new Map([...roster, ...availablePlayers.filter(player => selected.has(player.id))].map(player => [player.id, player]));
  const rows = [...byId.values()].sort((a, b) => a.name.localeCompare(b.name));
  const addable = availablePlayers.filter(player => !selected.has(player.id));
  return <section style={{ display: "grid", gap: "0.65rem" }}>
    <h3 style={{ margin: 0 }}>Who is playing the next round?</h3>
    <p style={{ margin: 0 }}>Uncheck anyone sitting out, and check them again when they return. Add late arrivals here. Previous games stay attached to the people who played them.</p>
    <strong>{selected.size} playing · {rows.filter(player => !selected.has(player.id)).length} sitting out</strong>
    <label>Find a session player<input aria-label="Find a session player" value={filter} onChange={event => setFilter(event.target.value)} style={inputStyle} /></label>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.4rem", maxHeight: "24rem", overflowY: "auto" }}>
      {rows.filter(player => player.name.toLocaleLowerCase().includes(filter.toLocaleLowerCase())).map(player => <label key={player.id} style={{ padding: "0.4rem", borderRadius: "6px", background: selected.has(player.id) ? "#f0fdf4" : "#f1f5f9" }}><input aria-label={`Playing next round: ${player.name}`} type="checkbox" checked={selected.has(player.id)} disabled={disabled} onChange={event => onActivePlayersChange(event.target.checked ? [...activePlayerIds, player.id] : activePlayerIds.filter(id => id !== player.id))} /> {player.name}{selected.has(player.id) ? "" : " · sitting out"}</label>)}
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "minmax(180px, 1fr) auto", gap: "0.5rem", alignItems: "end" }}>
      <label>Add a late arrival or returning club player<SearchablePlayerSelect aria-label="Add next-round player" value={incomingId} onValueChange={setIncomingId} disabled={disabled} style={inputStyle}><option value="">Select player…</option>{addable.map(player => <option key={player.id} value={String(player.id)}>{player.name}</option>)}</SearchablePlayerSelect></label>
      <button type="button" disabled={disabled || !addable.some(player => player.id === Number(incomingId))} onClick={() => { onActivePlayersChange([...activePlayerIds, Number(incomingId)]); setIncomingId(""); }}>Add player</button>
    </div>
    <label>Next-round court sizes<input aria-label="Next-round court sizes" value={courtSizes} disabled={disabled} onChange={event => onCourtSizesChange(event.target.value)} placeholder="Automatic — or enter 4,4,4 for three courts" style={inputStyle} /></label>
    <small>Leave blank to fit the selected players automatically. Enter one 4 or 5 per court to choose the layout. With 32 players, eight 4-player courts; with 28, seven. Refresh the court board to review placement and make any swaps.</small>
  </section>;
}
