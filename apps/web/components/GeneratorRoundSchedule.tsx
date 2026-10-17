"use client";

type Match = { court?: number; sideA?: string[]; sideB?: string[]; teamA?: string[]; teamB?: string[]; scoreA?: number | null; scoreB?: number | null };
type Round = { number: number; status: string; matches?: Match[]; courts?: Array<{ matches?: Match[] }>; byeParticipantIds?: string[] };

export default function GeneratorRoundSchedule({ rounds, participants, selectedRound, currentRound, disabled, unsaved, perfect, adaptive, moneyball, onOpen }: {
  rounds: Round[];
  participants: Array<{ id: string; name: string }>;
  selectedRound: number;
  currentRound: number;
  disabled: boolean;
  unsaved: boolean;
  perfect: boolean;
  adaptive: boolean;
  moneyball?: boolean;
  onOpen: (number: number) => void;
}) {
  const names = new Map(participants.map(player => [player.id, player.name]));
  const label = (ids: string[]) => ids.map(id => names.get(id) || id).join(" + ");
  return (
    <section aria-label="Round-Robin schedule" style={{ background: "white", border: "1px solid #e2e8f0", borderRadius: 14, padding: "1rem" }}>
      <h2 style={{ marginTop: 0 }}>Round schedule</h2>
      <p style={{ color: "#475569" }}>
        {moneyball ? "Fixed Moneyball rotation: complete every scheduled game to decide the winner. Choose any scheduled round; substitutions disqualify both players from winning."
          : adaptive ? "Adaptive rotation: balance assigned byes first, then partner and opponent coverage."
          : perfect ? "Perfect rotation: partner everyone once and face everyone twice. Compatible roster changes preserve this priority. Choose any round; skipping keeps the scheduled matchups."
          : "Choose any scheduled round. Matchups stay fixed until the roster changes or you add extra rounds."}
      </p>
      {unsaved ? <p style={{ color: "#92400e" }}>Save or cancel your score entries before changing rounds.</p> : null}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 280px), 1fr))", gap: "0.65rem" }}>
        {rounds.map(round => {
          const matches = round.matches || round.courts?.flatMap(court => court.matches || []) || [];
          const status = round.status === "saved" || round.status === "played" ? "Played" : round.status === "skipped" ? "Skipped" : round.number === currentRound && round.status === "active" ? "Current" : "Scheduled";
          return (
            <article key={round.number} style={{ border: `2px solid ${round.number === selectedRound ? "#2563eb" : "#e2e8f0"}`, borderRadius: 10, padding: "0.7rem" }}>
              <button type="button" onClick={() => onOpen(round.number)} disabled={disabled} aria-current={round.number === selectedRound ? "step" : undefined} aria-label={`Open round ${round.number}`} style={{ font: "inherit", fontWeight: 800, color: "#1d4ed8", background: "none", border: 0, padding: 0, cursor: "pointer" }}>
                Round {round.number}
              </button>
              <span style={{ float: "right", color: status === "Played" ? "#166534" : status === "Skipped" ? "#92400e" : "#475569", fontSize: "0.85rem" }}>{status}</span>
              {matches.map((match, index) => <p key={index} style={{ margin: "0.5rem 0 0", fontSize: "0.85rem" }}>
                <strong>Court {match.court || index + 1}:</strong> {label(match.sideA || match.teamA || [])} vs. {label(match.sideB || match.teamB || [])}
                {match.scoreA != null && match.scoreB != null ? ` · ${match.scoreA}–${match.scoreB}` : ""}
              </p>)}
              {round.byeParticipantIds?.length ? <p style={{ margin: "0.4rem 0 0", fontSize: "0.85rem", color: "#64748b" }}><strong>Bye:</strong> {label(round.byeParticipantIds)}</p> : null}
            </article>
          );
        })}
      </div>
    </section>
  );
}
