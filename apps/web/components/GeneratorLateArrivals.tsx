"use client";

import { useState } from "react";

type Arrival = {
  id: string;
  name: string;
  active_from_round?: number;
  inactive_from_round?: number | null;
  inactive_rounds?: number[];
  substitutes_for?: string;
};

type Props = {
  participants: Arrival[];
  roundNumber: number;
  totalRounds: number;
  generatorKind: string;
  playFormat: string;
  roundStatus: string;
  courtCount?: number;
  usedCourts: number[];
  busy: boolean;
  onStart: (participantIds: string[], court: number) => Promise<void>;
};

export default function GeneratorLateArrivals({ participants, roundNumber, totalRounds,
  generatorKind, playFormat, roundStatus, courtCount = 0, usedCourts, busy, onStart }: Props) {
  const [selection, setSelection] = useState<string[] | null>(null);
  const [chosenCourt, setChosenCourt] = useState<number | null>(null);
  const arrivals = participants.filter(player =>
    player.active_from_round === roundNumber + 1 && !player.substitutes_for &&
    (player.inactive_from_round == null || player.inactive_from_round > roundNumber + 1) &&
    !player.inactive_rounds?.includes(roundNumber + 1)
  );
  const hasNextRound = generatorKind === "round_robin" || roundNumber < totalRounds;
  const required = playFormat === "singles" ? 2 : 4;
  const selected = (selection ?? arrivals.slice(0, required).map(player => player.id))
    .filter(id => arrivals.some(player => player.id === id));
  const courts = Array.from({ length: Math.min(20, Math.max(courtCount, ...usedCourts, 0) + 1) }, (_, i) => i + 1)
    .filter(court => !usedCourts.includes(court));
  const court = chosenCourt != null && courts.includes(chosenCourt) ? chosenCourt : courts[0];
  const canStart = generatorKind === "round_robin" && ["singles", "doubles"].includes(playFormat) &&
    roundStatus === "active" && arrivals.length >= required && courts.length > 0;
  if (!arrivals.length) return null;

  return (
    <section aria-label="Late arrivals" style={{ marginTop: "1rem", padding: "1rem", borderRadius: 10, background: "#eff6ff" }}>
      <h3 style={{ marginTop: 0 }}>
        {hasNextRound ? `Joining Round ${roundNumber + 1}` : "Waiting late arrivals"}
      </h3>
      <p>{arrivals.map(player => player.name).join(", ")}</p>
      {canStart ? (
        <>
          <p>Have a spare court? These arrivals can play together now, leaving every game already on court in place.
            {hasNextRound ? " They will mix in with everyone from the next round." : " This is the final round."}</p>
          <fieldset disabled={busy} style={{ border: 0, padding: 0, margin: "0 0 1rem" }}>
            <legend>Choose {required} late arrivals</legend>
            {arrivals.map(player => (
              <label key={player.id} style={{ display: "block", padding: "0.25rem 0" }}>
                <input type="checkbox" checked={selected.includes(player.id)} onChange={event =>
                  setSelection(event.target.checked ? [...selected, player.id] : selected.filter(id => id !== player.id))
                } /> {player.name}
              </label>
            ))}
          </fieldset>
          <label>
            Spare court{" "}
            <select aria-label="Spare court" disabled={busy} value={court} onChange={event => setChosenCourt(Number(event.target.value))}>
              {courts.map(number => <option key={number} value={number}>Court {number}</option>)}
            </select>
          </label>
          <p>Make sure this court is available for the rest of the session.</p>
          <button type="button" disabled={busy || selected.length !== required} onClick={async () => {
            await onStart(selected, court);
            setSelection(null);
          }} style={{ border: 0, borderRadius: 24, padding: "0.65rem 1rem", background: "#0f172a", color: "white", fontWeight: 800 }}>
            Start late-arrival game on Court {court}
          </button>
        </>
      ) : (
        <p>{hasNextRound ? "Current games stay as they are." : "There are no more scheduled rounds."}
          {generatorKind === "round_robin" && ["singles", "doubles"].includes(playFormat) && roundStatus === "active" && arrivals.length < required
            ? ` Add ${required - arrivals.length} more late ${required - arrivals.length === 1 ? "arrival" : "arrivals"} to offer a separate game on a spare court.` : ""}</p>
      )}
    </section>
  );
}
