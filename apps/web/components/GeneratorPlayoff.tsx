"use client";

import { useId, useState } from "react";
import Link from "@/components/PublicClubLink";

export type PlayoffFormat = "groups_of_four" | "top_eight";
type Seed = { participantId: string; name: string; seed: number };
type Match = {
  id?: string; label?: string; sideA?: string[]; sideB?: string[];
  scoreA?: number | null; scoreB?: number | null; sourceMatchIds?: string[];
};
export type GeneratorPlayoffOptions = {
  seeds: Seed[]; sourceRound: number;
  formats: Array<{ format: PlayoffFormat; reason: string | null; matches: Match[]; sitOutParticipantIds: string[] }>;
};
export type GeneratorPlayoffEvent = {
  playoff?: { format: PlayoffFormat; seeds: Seed[]; sitOutParticipantIds: string[] };
  rounds?: Array<{ number: number; status: string; stage?: string; label?: string; matches?: Match[] }>;
};
type Props = {
  event: GeneratorPlayoffEvent; options?: GeneratorPlayoffOptions | null;
  canManage: boolean; locked: boolean; busy: boolean;
  onStart: (format: PlayoffFormat) => Promise<void>;
  roundHref: (round: number) => string;
};

const cardStyle = { border: "1px solid #cbd5e1", borderRadius: "14px", padding: "1rem", background: "white", minWidth: 0 };
const buttonStyle = { border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#0f172a", color: "white", fontWeight: 800, cursor: "pointer" };

export default function GeneratorPlayoff({ event, options, canManage, locked, busy, onStart, roundHref }: Props) {
  const [open, setOpen] = useState(false);
  const [format, setFormat] = useState<PlayoffFormat>("groups_of_four");
  const panelId = useId();
  const bracket = event.playoff;
  const seeds = bracket?.seeds || options?.seeds || [];
  const names = new Map(seeds.map(seed => [seed.participantId, `#${seed.seed} ${seed.name}`]));
  const team = (ids?: string[]) => (ids || []).map(id => names.get(id) || id).join(" / ");
  const selected = options?.formats.find(option => option.format === format);

  if (bracket) {
    const rounds = (event.rounds || []).filter(round => round.stage === "playoff");
    return <article style={cardStyle} aria-label="Playoff bracket">
      <h2 style={{ marginTop: 0 }}>Playoff</h2>
      <p style={{ color: "#475569" }}>Teams and seeds are fixed from the round-robin standings.</p>
      <div style={{ display: "grid", gap: "0.75rem" }}>
        {rounds.map(round => <section key={round.number}>
          <h3 style={{ margin: "0 0 0.4rem" }}><Link href={roundHref(round.number)}>{round.label || `Round ${round.number}`}</Link></h3>
          {(round.matches || []).map((match, index) => {
            const scored = round.status === "saved" && match.scoreA != null && match.scoreB != null;
            const winner = scored ? team(Number(match.scoreA) > Number(match.scoreB) ? match.sideA : match.sideB) : "";
            return <p key={match.id || index} style={{ margin: "0.35rem 0" }}>
              <strong>{match.label}: </strong>{team(match.sideA) || "Winner of semifinal 1"} vs {team(match.sideB) || "Winner of semifinal 2"}
              {scored ? <> · {match.scoreA}–{match.scoreB} · <strong>Winner: {winner}</strong></> : null}
            </p>;
          })}
        </section>)}
      </div>
      {bracket.sitOutParticipantIds.length ? <p style={{ color: "#475569" }}>Sitting out: {team(bracket.sitOutParticipantIds)}.</p> : null}
    </article>;
  }
  if (!canManage || locked || !options) return null;

  return <article style={cardStyle}>
    <button type="button" aria-expanded={open} aria-controls={panelId} onClick={() => setOpen(value => !value)} disabled={busy} style={buttonStyle}>Playoff</button>
    {open ? <div id={panelId} style={{ marginTop: "1rem" }}>
      <div style={{ display: "grid", gap: "0.4rem", maxWidth: "32rem", fontWeight: 750 }}>
        <label htmlFor={`${panelId}-format`}>Playoff format</label>
        <select id={`${panelId}-format`} value={format} disabled={busy} onChange={event => setFormat(event.target.value as PlayoffFormat)} style={{ minWidth: 0, width: "100%", padding: "0.6rem", borderRadius: "8px", border: "1px solid #cbd5e1", font: "inherit" }}>
          <option value="groups_of_four">Games in groups of four — 1/4 vs 2/3, 5/8 vs 6/7…</option>
          <option value="top_eight">Top 8 — two semifinals, then a final</option>
        </select>
      </div>
      <p style={{ color: "#475569" }}>{format === "groups_of_four"
        ? "One game for each complete group of four. Players in an incomplete group sit out."
        : "1/8 vs 4/5 and 2/7 vs 3/6. The winning teams play each other in the final. Other players sit out."}</p>
      {(selected?.matches || []).map((match, index) => <p key={index}><strong>{match.label}: </strong>{team(match.sideA)} vs {team(match.sideB)}</p>)}
      {selected?.sitOutParticipantIds.length ? <p>Sitting out: {team(selected.sitOutParticipantIds)}.</p> : null}
      <p style={{ color: "#475569" }}>Starting the playoff ends the round robin after Round {options.sourceRound}. Only available players with a completed game are seeded.</p>
      {selected?.reason ? <p role="status">{selected.reason}</p> : null}
      <button type="button" disabled={busy || !selected || Boolean(selected.reason)} onClick={() => void onStart(format)} style={{ ...buttonStyle, opacity: selected?.reason ? 0.55 : 1 }}>{busy ? "Starting…" : "Start playoff"}</button>
    </div> : null}
  </article>;
}
