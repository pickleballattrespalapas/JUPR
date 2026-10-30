"use client";

type Participant = { id: string; name: string; inactive_from_round?: number | null; roster_order?: number };

export default function GeneratorDepartures({ participants, nextRound, selected, onChange, disabled }: {
  participants: Participant[];
  nextRound: number;
  selected: string[];
  onChange: (ids: string[]) => void;
  disabled: boolean;
}) {
  const available = participants.filter(player => !player.inactive_from_round || player.inactive_from_round > nextRound)
    .sort((left, right) => Number(left.roster_order || 0) - Number(right.roster_order || 0));
  return <fieldset disabled={disabled} style={{ border: "1px solid #cbd5e1", borderRadius: 8, padding: ".75rem" }}>
    <legend>Players leaving</legend>
    <p style={{ marginTop: 0 }}>Select everyone leaving. Apply once to update the next round together.</p>
    <div style={{ display: "grid", gap: ".4rem", maxHeight: 240, overflowY: "auto" }}>
      {available.map(player => <label key={player.id} style={{ display: "flex", gap: ".5rem", alignItems: "center" }}>
        <input type="checkbox" aria-label={`Remove ${player.name}`} checked={selected.includes(player.id)}
          onChange={event => onChange(event.target.checked ? [...selected, player.id] : selected.filter(id => id !== player.id))} />
        {player.name}
      </label>)}
    </div>
    <p style={{ marginBottom: 0 }}>{selected.length} selected</p>
  </fieldset>;
}
