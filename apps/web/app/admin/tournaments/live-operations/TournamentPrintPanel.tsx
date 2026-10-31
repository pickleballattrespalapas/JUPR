"use client";

import { useEffect, useRef, useState } from "react";
import type { AdminTournamentDayWorkspaceSnapshot } from "@/lib/adminTournamentDayOpsApi";
import type { TournamentPrintScope } from "@/lib/tournamentPrintPacket";
import styles from "./TournamentPrintPanel.module.css";

export default function TournamentPrintPanel({ snapshot }: { snapshot: AdminTournamentDayWorkspaceSnapshot }) {
  const [selected, setSelected] = useState<string[]>(snapshot.draws.map(draw => draw.id));
  const [unplayedOnly, setUnplayedOnly] = useState(false);
  const [includeResults, setIncludeResults] = useState(false);
  const [blankCards, setBlankCards] = useState(2);
  const [cardLayout, setCardLayout] = useState<"cut-stack" | "consecutive">("cut-stack");
  const [busy, setBusy] = useState<TournamentPrintScope | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const current = useRef(true);
  const exporting = useRef(false);
  useEffect(() => { current.current = true; return () => { current.current = false; }; }, []);
  const drawIds = selected.filter(id => snapshot.draws.some(draw => draw.id === id));
  const games = snapshot.games.filter(game => drawIds.includes(game.draw_id));
  const cardCount = games.filter(game => !unplayedOnly || (!game.finalized_at && game.state !== "COMPLETED")).length + blankCards;
  const missingSchedules = snapshot.draws.filter(draw => drawIds.includes(draw.id) && !games.some(game => game.draw_id === draw.id));
  const noPlayoffs = snapshot.draws.filter(draw => drawIds.includes(draw.id) && !games.some(game => game.draw_id === draw.id && game.stage === "PLAYOFF"));

  async function download(scope: TournamentPrintScope) {
    if (exporting.current || !drawIds.length) return;
    exporting.current = true; setBusy(scope); setError(""); setMessage("");
    try {
      // Use the same loaded snapshot as the board; this never creates games,
      // changes scores, or fetches another roster.
      const { buildTournamentPrintPdf } = await import("@/lib/tournamentPrintPacket");
      const { pdf, filename } = await buildTournamentPrintPdf(snapshot, { drawIds, unplayedOnly, includeResults, blankCards, cardLayout }, scope);
      if (!current.current) return;
      pdf.save(filename);
      setMessage(`${pdf.getNumberOfPages()}-page PDF downloaded. Print single-sided on Letter paper at Actual size (100%).`);
    } catch (cause) {
      if (current.current) setError(cause instanceof Error ? cause.message : "The PDF could not be created. Try again.");
    } finally {
      exporting.current = false;
      if (current.current) setBusy(null);
    }
  }

  return (
    <details className={styles.panel}>
      <summary>Print tournament paper backups</summary>
      <p>Save PDFs before play so you can run from paper without internet. Scorecards print two per Letter page; cut across the middle for 8.5 × 5.5 inch clipboards.</p>
      <fieldset disabled={Boolean(busy)} className={styles.draws}>
        <legend>Divisions / pools for {snapshot.day_scope.selected_day.label}</legend>
        <div className={styles.selectionActions}>
          <button type="button" onClick={() => setSelected(snapshot.draws.map(draw => draw.id))}>Select all</button>
          <button type="button" onClick={() => setSelected([])}>Clear selection</button>
        </div>
        {snapshot.draws.map(draw => (
          <label key={draw.id}>
            <input type="checkbox" checked={drawIds.includes(draw.id)} onChange={event => setSelected(previous => event.target.checked ? [...previous, draw.id] : previous.filter(id => id !== draw.id))} />
            {draw.name}
          </label>
        ))}
        {!snapshot.draws.length ? <p>Create your divisions and generate their match schedules first.</p> : null}
      </fieldset>
      <div className={styles.options}>
        <label>Match cards
          <select disabled={Boolean(busy)} value={unplayedOnly ? "remaining" : "all"} onChange={event => setUnplayedOnly(event.target.value === "remaining")}>
            <option value="all">All saved matches</option><option value="remaining">Unplayed matches only</option>
          </select>
        </label>
        <label>Scores and totals
          <select disabled={Boolean(busy)} value={includeResults ? "saved" : "blank"} onChange={event => setIncludeResults(event.target.value === "saved")}>
            <option value="blank">Blank for handwriting</option><option value="saved">Include saved results</option>
          </select>
        </label>
        <label>Spare blank cards
          <select disabled={Boolean(busy)} value={blankCards} onChange={event => setBlankCards(Number(event.target.value))}>
            {[0,2,4,6,10,20].map(count => <option key={count} value={count}>{count}</option>)}
          </select>
        </label>
        <label>Card page layout
          <select disabled={Boolean(busy)} value={cardLayout} onChange={event => setCardLayout(event.target.value as "cut-stack" | "consecutive")}>
            <option value="cut-stack">Cut stack: top pile, then bottom pile</option>
            <option value="consecutive">Consecutive: top, then bottom of each page</option>
          </select>
        </label>
      </div>
      <p className={styles.help}>Cards follow current court assignments and the ready queue, then the saved rounds across selected divisions, with playoff feeder matches first. Future dispatch can change as courts finish. {cardLayout === "cut-stack" ? "Keep the printed pages in page order, cut the stack in half, then put the top-half pile on the bottom-half pile." : "Take the top card, then the bottom card from each page in page order."}</p>
      <p className={styles.help}>The admin chart keeps the saved round-robin schedule on the left and each team’s For / Against columns on the right, with wins/losses, totals, differential and seed below. Large draws continue across sheets. Admin sheets always include the full saved schedule.</p>
      {missingSchedules.length ? <p className={styles.notice}>No saved matches yet for {missingSchedules.map(draw => draw.name).join(", ")}. Generate those schedules before printing their match cards.</p> : null}
      {noPlayoffs.length ? <p className={styles.help}>Playoff matches have not been generated for {noPlayoffs.map(draw => draw.name).join(", ")}. Their playoff cards and bracket worksheets will be available after generation. Spare blank cards can be used for later matches.</p> : null}
      <p className={styles.help}>{drawIds.length} selected · {cardCount} scorecards · {Math.ceil(cardCount/2)} scorecard pages. Every court field is blank so the desk can write the court number when sending a match out. Reprint after changing teams or the schedule.</p>
      <div className={styles.actions}>
        <button type="button" disabled={Boolean(busy) || !drawIds.length || !cardCount} onClick={() => void download("cards")}>{busy === "cards" ? "Creating scorecards…" : "Download match cards PDF"}</button>
        <button type="button" disabled={Boolean(busy) || !drawIds.length} onClick={() => void download("admin")}>{busy === "admin" ? "Creating admin sheets…" : "Download admin sheets PDF"}</button>
        <button type="button" disabled={Boolean(busy) || !drawIds.length} onClick={() => void download("packet")}>{busy === "packet" ? "Creating packet…" : "Download full paper packet"}</button>
      </div>
      {message ? <p role="status">{message}</p> : null}
      {error ? <p role="alert" className={styles.notice}>{error}</p> : null}
    </details>
  );
}
