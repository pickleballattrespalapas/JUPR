import { CompetitionDocument, pairingLabels, regularCourtBlockInstructions, scheduledEncounters, scheduleRoundLabel } from "@/lib/interclubCompetition";
import styles from "./competition.module.css";

type Props = { document: CompetitionDocument; clubName: (id: string) => string };

export function CourtAssignments({ document, clubName }: Props) {
  const rows = scheduledEncounters(document).flatMap(encounter => encounter.pairings.map(pairing => ({ encounter, pairing })))
    .sort((a, b) => a.encounter.rotation - b.encounter.rotation || (a.pairing.court ?? 101) - (b.pairing.court ?? 101));
  return <div className={styles.tableScroll}><table className={styles.table}>
    <caption className={styles.srOnly}>Court assignments by {scheduleRoundLabel(document).toLowerCase()}</caption>
    <thead><tr><th scope="col">{scheduleRoundLabel(document)}</th><th scope="col">Court</th><th scope="col">Skill</th><th scope="col">Club matchup</th><th scope="col">Pairing</th><th scope="col">Games</th></tr></thead>
    <tbody>{rows.map(({ encounter, pairing }) => <tr key={pairing.id}>
      <td>{encounter.rotation}</td><td>{pairing.court ?? (!pairing.players_a.length || !pairing.players_b.length ? "No court · Forfeit" : "Assign: ____")}</td><td>{encounter.division}</td>
      <td>{clubName(encounter.club_a)} vs {clubName(encounter.club_b)}</td><td>{pairingLabels[pairing.kind]}</td><td>{pairing.games.length === 1 ? "1" : `1-${pairing.games.length}`}</td>
    </tr>)}</tbody>
  </table></div>;
}

export default function CourtSchedule({ document, clubName }: Props) {
  const waves = new Set(document.encounters.map(encounter => encounter.rotation)).size;
  return <section className={styles.card} aria-label="Court schedule">
    <h2>Court schedule</h2>
    <p><strong>{document.schedule_mode === "staggered" ? "Staggered starts" : "Simultaneous starts"} · {waves} {scheduleRoundLabel(document).toLowerCase()}{waves === 1 ? "" : "s"}</strong></p>
    {document.phase === "regular" && <p><strong>{regularCourtBlockInstructions}</strong></p>}
    {document.schedule_mode === "staggered" && <p>Each wave is a full three-game block. Start the next wave after every pairing in the current wave finishes. Exact start times depend on match length.</p>}
    <CourtAssignments document={document} clubName={clubName} />
  </section>;
}
