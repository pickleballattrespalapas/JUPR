"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { CompetitionDocument, CompetitionPlayer, pairingLabels, phaseLabels, playerNames, scheduledEncounters, scheduleRoundLabel, singlesCourt } from "@/lib/interclubCompetition";
import { CourtAssignments } from "./CourtSchedule";
import { InterclubMeet } from "@/lib/interclubRegistration";
import styles from "./competition.module.css";

type Props = {
  document: CompetitionDocument; meet: InterclubMeet; seasonName: string; timezone: string; revision: number; players: Map<string, CompetitionPlayer>; clubName: (id: string) => string;
};
export default function PrintPacket(props: Props) {
  const [target, setTarget] = useState<HTMLElement | null>(null);
  useEffect(() => {
    const portal = window.document.createElement("div"); portal.className = styles.printPortal;
    window.document.body.appendChild(portal); window.document.body.classList.add(styles.printBody); setTarget(portal);
    return () => { portal.remove(); window.document.body.classList.remove(styles.printBody); };
  }, []);
  return target ? createPortal(<PrintPacketContent {...props} />, target) : null;
}

export function PrintPacketContent({ document, meet, seasonName, timezone, revision, players, clubName }: Props) {
  const when = (iso: string) => new Date(iso).toLocaleString("en-US", { timeZone: timezone, dateStyle: "medium", timeStyle: "short" });
  return <div className={styles.printRoot} aria-hidden="true">
    <section className={styles.printPage}>
      <p>Southern BCS · Paper meet packet · Revision {revision}</p>
      <h1>{seasonName}</h1><h2>{phaseLabels[document.phase]} · {when(meet.starts_at)}</h2>
      <p>Host: {clubName(meet.host_club_id)} · Times: {timezone} · Roster deadline: {when(meet.roster_deadline)}</p>
      <h3>Court assignments</h3>
      {document.schedule_mode === "staggered" && <p><strong>Staggered starts:</strong> complete all three games in each pairing before moving to the next wave. Start each wave after the previous wave finishes; exact times depend on match length.</p>}
      <CourtAssignments document={document} clubName={clubName} />
      <h3>At the courts</h3>
      <ul>
        <li>{document.phase === "regular" ? "Play all three games in every doubles pairing. Each player plays 3, 6 or 9 games for a two-, three- or four-club field." : "Play women’s doubles, men’s doubles and both mixed doubles games. At 2–2, play the rotating singles tiebreak."}</li>
        <li>Doubles use side-out scoring to 11, win by two, no cap. Record both scores and the time played.</li>
        <li>Both clubs check the players and scores, then sign this sheet. Return all sheets together to the meet organizer.</li>
        <li>Injury: concede only the interrupted game; record the actual stopped score and the winning club. An eligible substitute may enter between games only. Record their name and injury reason.</li>
        <li>{document.phase === "regular" ? "Missing pairing: mark its three games “forfeit.” Do not invent 11–0 scores. The other pairing still plays." : "Missing doubles pairing: mark its game “forfeit” and identify the winning club. Do not invent an 11–0 score. Continue the other doubles games."}</li>
        <li>{document.phase === "regular" ? "Weather delay: record the stopped score and server; resume from that position. If the meet is rescheduled, unfinished three-game pairings restart; completed pairings stand." : "Weather delay: record the stopped score and server; resume from that position. Ask the organizer to arrange completion of the full MLP matchup."}</li>
      </ul>
      <p>Organizer: __________________________ Phone: __________________________</p>
      <p>Started: __________ Weather pause: __________ Resumed: __________ Ended: __________</p>
    </section>
    {scheduledEncounters(document).map(encounter => <section key={encounter.id} className={`${styles.printPage} ${document.phase !== "regular" ? styles.printMlp : ""}`}>
      <p>{seasonName} · {when(meet.starts_at)} · Revision {revision} · {scheduleRoundLabel(document)} {encounter.rotation}</p>
      <h2>Skill {encounter.division}: {clubName(encounter.club_a)} vs {clubName(encounter.club_b)}</h2>
      <p>Side-out scoring to 11 · Win by two · No cap{document.phase === "regular" ? " · Play all three games" : " · One game per doubles pairing"}</p>
      {encounter.pairings.map(pairing => <section key={pairing.id} className={styles.printPairing}>
        <h3>{pairingLabels[pairing.kind]} · Court {pairing.court || "____"}</h3>
        <p><strong>A · {clubName(encounter.club_a)}:</strong> {playerNames(pairing.players_a, players)}<br /><strong>B · {clubName(encounter.club_b)}:</strong> {playerNames(pairing.players_b, players)}</p>
        {pairing.eligibility_deadline && <p>Eligibility locked: {when(pairing.eligibility_deadline)}</p>}
        <table><thead><tr><th>Game</th><th>A score</th><th>B score</th><th>Time played</th><th>Complete / injury / forfeit / unplayed</th><th>Winner</th></tr></thead><tbody>
          {pairing.games.map((game, index) => <tr key={game.id}><td>{index + 1}</td><td>{game.a ?? ""}</td><td>{game.b ?? ""}</td><td>{["completed", "retired"].includes(game.status) && game.played_at ? when(game.played_at) : ""}</td><td>{{ pending: "", completed: "Completed", retired: "Injury", forfeit: "Forfeit", double_forfeit: "Both forfeit", unplayed: "Unplayed" }[game.status]}</td><td>{game.winner?.toUpperCase() || ""}</td></tr>)}
        </tbody></table>
        {pairing.games.filter(game => game.players_a.length || game.players_b.length || game.injury_reason).map((game) => <p key={game.id}>Game {pairing.games.indexOf(game) + 1} actual players — A: {playerNames(game.players_a.length ? game.players_a : pairing.players_a, players)}; B: {playerNames(game.players_b.length ? game.players_b : pairing.players_b, players)}.{game.injury_reason ? ` Injury note: ${game.injury_reason}` : ""}</p>)}
        <p>Injury / replacement player and game: _______________________________________________________</p>
      </section>)}
      {document.phase !== "regular" && <section className={styles.printPairing}>
        <h3>At 2–2: {singlesCourt(encounter.division).toLowerCase()} rotating singles</h3>
        <p>Rally scoring to 21 · Win by two · No cap · Both teams rotate after every four rallies · No individual rating effect</p>
        <p>A fixed order: 1. ______________ 2. ______________ 3. ______________ 4. ______________</p>
        <p>B fixed order: 1. ______________ 2. ______________ 3. ______________ 4. ______________</p>
        <p>Final score A: ______ B: ______ Winning club: ___________________________________</p>
      </section>}
      <p>Weather pause details (score, server and position): _____________________________________________</p>
      <p>_______________________________________________________________________________________</p>
      <p>Verified by club A: ______________________ Club B: ______________________ Date: ______________</p>
      <small>Matchup reference: {encounter.id}</small>
    </section>)}
  </div>;
}
