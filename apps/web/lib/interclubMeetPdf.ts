import type { jsPDF } from "jspdf";
import type { InterclubMeet } from "./interclubRegistration";
import { CompetitionDocument, CompetitionPlayer, pairingLabels, phaseLabels, playerNames, regularCourtBlockInstructions, scheduledEncounters, scheduleRoundLabel, singlesCourt } from "./interclubCompetition";

export type MeetPdfScope = "schedule" | "packet";
export type MeetPdfOptions = {
  document: CompetitionDocument; meet: InterclubMeet; seasonName: string; timezone: string;
  revision: number; players: Map<string, CompetitionPlayer>; clubName: (id: string) => string;
};

// Standard PDF fonts support the accented Latin names used by these clubs.
// Normalize typographic punctuation before measuring so wrapping matches output.
function printable(value: string): string {
  return value.normalize("NFC").replace(/[\u2018\u2019]/g, "'").replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2010-\u2015]/g, "-").replace(/\u2026/g, "...").replace(/\u00a0/g, " ");
}

class PacketLayout {
  private y = 0;
  private heading = "";
  private started = false;
  readonly margin = 36;
  readonly width: number;
  readonly bottom: number;
  constructor(readonly pdf: jsPDF, private options: MeetPdfOptions, private when: (iso: string) => string) {
    this.width = pdf.internal.pageSize.getWidth() - this.margin * 2;
    this.bottom = pdf.internal.pageSize.getHeight() - 42;
  }
  private lines(value: string, width: number, size: number, bold = false): string[] {
    this.pdf.setFont("helvetica", bold ? "bold" : "normal"); this.pdf.setFontSize(size);
    return this.pdf.splitTextToSize(printable(value || " "), width);
  }
  page(heading: string) {
    if (this.started) this.pdf.addPage();
    this.started = true; this.heading = heading; this.y = this.margin;
    for (const [value, size, bold] of [
      [this.options.seasonName, 16, true],
      [`${phaseLabels[this.options.document.phase]} | ${this.when(this.options.meet.starts_at)} | Revision ${this.options.revision}`, 9, false],
      [heading, 12, true],
    ] as const) {
      const lines = this.lines(value, this.width, size, bold);
      this.pdf.text(lines, this.margin, this.y); this.y += lines.length * size * 1.2 + 8;
    }
    this.pdf.setDrawColor(185); this.pdf.line(this.margin, this.y - 3, this.margin + this.width, this.y - 3);
    this.y += 9;
  }
  private ensure(height: number) { if (this.y + height > this.bottom) this.page(`${this.heading.replace(/ \(continued\)$/, "")} (continued)`); }
  text(value: string, size = 10, bold = false) {
    const leading = size * 1.3;
    for (const line of this.lines(value, this.width, size, bold)) {
      this.ensure(leading);
      this.pdf.setFont("helvetica", bold ? "bold" : "normal"); this.pdf.setFontSize(size);
      this.pdf.text(line, this.margin, this.y); this.y += leading;
    }
    this.y += 7;
  }
  table(headers: string[], rows: string[][], widths: number[], size = 9) {
    const leading = size * 1.25;
    const drawRow = (cells: string[][], count: number, header = false) => {
      const height = count * leading + 12;
      let x = this.margin;
      this.pdf.setFont("helvetica", header ? "bold" : "normal"); this.pdf.setFontSize(size);
      cells.forEach((cell, index) => {
        this.pdf.setDrawColor(180); const shade = header ? 237 : 255; this.pdf.setFillColor(shade, shade, shade);
        this.pdf.rect(x, this.y - size, widths[index], height, "FD");
        if (cell.length) this.pdf.text(cell, x + 6, this.y, { lineHeightFactor: 1.25 });
        x += widths[index];
      });
      this.y += height;
    };
    const head = headers.map((cell, index) => this.lines(cell, widths[index] - 12, size, true));
    const headLines = Math.max(...head.map(cell => cell.length));
    const drawHead = () => drawRow(head, headLines, true);
    this.ensure(headLines * leading + 12 + leading + 12); drawHead();
    for (const row of rows) {
      const cells = row.map((cell, index) => this.lines(cell, widths[index] - 12, size));
      const count = Math.max(...cells.map(cell => cell.length));
      if (this.y + count * leading + 12 > this.bottom) { this.page(this.heading); drawHead(); }
      // A long name or note is continued intact instead of clipped or shrunk.
      let offset = 0;
      while (offset < count) {
        const capacity = Math.max(1, Math.floor((this.bottom - this.y - 12) / leading));
        const take = Math.min(capacity, count - offset);
        drawRow(cells.map(cell => cell.slice(offset, offset + take)), take);
        offset += take;
        if (offset < count) { this.page(this.heading); drawHead(); }
      }
    }
    this.y += 12;
  }
  finish(scope: MeetPdfScope) {
    const pages = this.pdf.getNumberOfPages();
    for (let page = 1; page <= pages; page++) {
      this.pdf.setPage(page); this.pdf.setFont("helvetica", "normal"); this.pdf.setFontSize(8); this.pdf.setTextColor(90);
      this.pdf.text(`${scope === "schedule" ? "Court schedule" : "Full meet packet"} | Saved revision ${this.options.revision}`, this.margin, this.bottom + 18);
      this.pdf.text(`Page ${page} of ${pages}`, this.margin + this.width, this.bottom + 18, { align: "right" });
    }
  }
}

export async function buildInterclubMeetPdf(options: MeetPdfOptions, scope: MeetPdfScope) {
  const { jsPDF } = await import("jspdf");
  const pdf = new jsPDF({ orientation: "portrait", unit: "pt", format: "a4", compress: true });
  const { document, meet, seasonName, timezone, revision, players, clubName } = options;
  const when = (iso: string) => new Date(iso).toLocaleString("en-US", { timeZone: timezone, dateStyle: "medium", timeStyle: "short" });
  const layout = new PacketLayout(pdf, options, when);
  const encounters = scheduledEncounters(document), round = scheduleRoundLabel(document);
  pdf.setProperties({ title: `${seasonName} - ${scope === "schedule" ? "Court schedule" : "Meet packet"}`, subject: `Saved meet revision ${revision}`, creator: "Pickleball Club Sandwich" });
  layout.page("Court schedule");
  layout.text(`Host: ${clubName(meet.host_club_id)} | Timezone: ${timezone}`, 9);
  layout.text(`Roster deadline: ${when(meet.roster_deadline)}`, 9);
  if (document.phase === "regular") layout.text(regularCourtBlockInstructions, 9, true);
  if (document.schedule_mode === "staggered") layout.text("Staggered starts: each wave is a full three-game block. Start the next wave after every pairing in the current wave finishes. Exact times depend on match length.", 9);
  const assignments = encounters.flatMap(encounter => encounter.pairings.map(pairing => ({ encounter, pairing })))
    .sort((a, b) => a.encounter.rotation - b.encounter.rotation || (a.pairing.court ?? 101) - (b.pairing.court ?? 101));
  layout.table([round, "Court", "Skill", "Club matchup", "Pairing", "Games"], assignments.map(({ encounter, pairing }) => [
    String(encounter.rotation), pairing.court != null ? String(pairing.court) : !pairing.players_a.length || !pairing.players_b.length ? "Forfeit" : "Assign",
    encounter.division, `${clubName(encounter.club_a)} vs ${clubName(encounter.club_b)}`, pairingLabels[pairing.kind], pairing.games.length === 1 ? "1" : `1-${pairing.games.length}`,
  ]), [40, 40, 34, layout.width - 266, 100, 52]);
  if (!assignments.length) layout.text("No pairings have been generated for this meet.");
  if (scope === "packet") {
    layout.page("At the courts");
    const regular = document.phase === "regular";
    for (const instruction of [
      regular ? "Play all three games in every doubles pairing. Each player plays 3, 6 or 9 games for a two-, three- or four-club field." : "Play women's doubles, men's doubles and both mixed doubles games. At 2-2, play the rotating singles tiebreak.",
      "Doubles use side-out scoring to 11, win by two, no cap. Record both scores and the time played.",
      "Both clubs check the players and scores, then sign the sheet. Return all sheets together to the meet organizer.",
      "Injury: concede only the interrupted game; record the actual stopped score and winning club. An eligible substitute may enter between games only. Record their name and injury reason.",
      regular ? "Missing pairing: mark its three games as forfeit. Do not invent 11-0 scores. The other pairing still plays." : "Missing doubles pairing: mark its game as forfeit and identify the winning club. Do not invent an 11-0 score. Continue the other doubles games.",
      regular ? "Weather delay: record the stopped score and server; resume from that position. If rescheduled, unfinished three-game pairings restart; completed pairings stand." : "Weather delay: record the stopped score and server; resume from that position. Ask the organizer to arrange completion of the full MLP matchup.",
    ]) layout.text(instruction);
    layout.text("Organizer: ____________________ Phone: ____________________");
    layout.text("Started: ______ Weather pause: ______ Resumed: ______ Ended: ______");
    for (const encounter of encounters) {
      layout.page(`${round} ${encounter.rotation} | Skill ${encounter.division} | Score sheet`);
      layout.text(`${clubName(encounter.club_a)} vs ${clubName(encounter.club_b)}`, 13, true);
      layout.text(`Side-out to 11 | Win by two | No cap | ${regular ? "Play all three games" : "One game per doubles pairing"}`, 9);
      for (const pairing of encounter.pairings) {
        layout.text(`${pairingLabels[pairing.kind]} | Court ${pairing.court ?? "____"}${regular ? " | Games 1-3 on this court" : ""}`, 11, true);
        layout.text(`A - ${clubName(encounter.club_a)}: ${playerNames(pairing.players_a, players)}`, 9);
        layout.text(`B - ${clubName(encounter.club_b)}: ${playerNames(pairing.players_b, players)}`, 9);
        if (pairing.eligibility_deadline) layout.text(`Eligibility locked: ${when(pairing.eligibility_deadline)}`, 8);
        layout.table(["Game", "A", "B", "Time played", "Outcome", "Winner"], pairing.games.map((game, index) => [
          String(index + 1), game.a == null ? "" : String(game.a), game.b == null ? "" : String(game.b),
          ["completed", "retired"].includes(game.status) && game.played_at ? when(game.played_at) : "",
          { pending: "", completed: "Completed", retired: "Injury", forfeit: "Forfeit", double_forfeit: "Both forfeit", unplayed: "Unplayed" }[game.status], game.winner?.toUpperCase() || "",
        ]), [44, 32, 32, 128, layout.width - 290, 54], 8.5);
        pairing.games.forEach((game, index) => {
          if (game.players_a.length || game.players_b.length || game.injury_reason) layout.text(`Game ${index + 1} actual players - A: ${playerNames(game.players_a.length ? game.players_a : pairing.players_a, players)}; B: ${playerNames(game.players_b.length ? game.players_b : pairing.players_b, players)}.${game.injury_reason ? ` Injury note: ${game.injury_reason}` : ""}`, 9);
        });
        layout.text("Injury / replacement player and game: __________________________________", 9);
      }
      if (!regular) {
        layout.text(`At 2-2: ${singlesCourt(encounter.division)} rotating singles`, 11, true);
        layout.text("Rally to 21 | Win by two | No cap | Both teams rotate every four rallies | No individual rating effect", 9);
        for (const side of ["a", "b"] as const) layout.text(`${side.toUpperCase()} fixed order: ${encounter.tiebreak?.[`order_${side}`]?.length ? playerNames(encounter.tiebreak[`order_${side}`], players) : "1. ________ 2. ________ 3. ________ 4. ________"}`, 9);
        layout.text(`Tiebreak score - A: ${encounter.tiebreak?.a ?? "______"} B: ${encounter.tiebreak?.b ?? "______"} | ${encounter.tiebreak?.status === "completed" ? "Completed" : "Winning club: ____________________"}`, 9);
      }
      layout.text("Weather pause (score, server, position): _________________________________", 9);
      layout.text("Verified by club A: ______________ Club B: ______________ Date: __________", 9);
      layout.text(`Matchup reference: ${encounter.id}`, 8);
    }
  }
  layout.finish(scope);
  const date = new Intl.DateTimeFormat("en-CA", { timeZone: timezone, year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date(meet.starts_at));
  const name = seasonName.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-zA-Z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0,70) || "interclub";
  return { pdf, filename: `${name}-${date}-${meet.id.slice(0,8)}-${scope}-r${revision}.pdf` };
}
