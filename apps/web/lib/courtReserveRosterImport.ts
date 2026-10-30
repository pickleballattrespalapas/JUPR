export type CourtReservePlayer = { id: number; name: string };
export type CourtReserveReportRow = {
  rowNumber: number;
  name: string;
  requestedRating: string;
};
export type CourtReserveMatchedRow = CourtReserveReportRow & {
  playerId: number | null;
  matchStatus: "matched" | "ambiguous" | "unmatched";
  candidateIds: number[];
};
export type CourtReserveRosterSelection = {
  playerIds: number[];
  requestedRatings: Record<string, string>;
};

export const MAX_COURTRESERVE_FILE_BYTES = 2 * 1024 * 1024;
const MAX_REPORT_ROWS = 500;

/** Only CourtReserve's known terminal primary-player marker is removed. */
export function cleanCourtReserveName(value: string): string {
  return value.normalize("NFKC").trim().replace(/\s*\(P\)\s*$/i, "").trim().replace(/\s+/g, " ");
}

export function normalizeCourtReserveName(value: string): string {
  return cleanCourtReserveName(value).normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase("en");
}

function headerKey(value: string): string {
  return value.replace(/^\uFEFF/, "").normalize("NFKC").toLowerCase().replace(/[^a-z0-9]/g, "");
}

const NAME_HEADERS = new Set(["fullname", "name", "playername", "participantname", "membername", "registrantname"]);
const RATING_HEADERS = new Set(["enteryourskilllevel", "skilllevel", "rating", "requestedrating", "selfrating", "selfreportedrating", "estimatedrating", "estimatedskilllevel"]);

/** Quoted cells may contain delimiters, line breaks and escaped quotes. */
function readDelimited(text: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [], cell = "", quoted = false, closedQuote = false;
  const finishCell = () => { row.push(cell.trim()); cell = ""; closedQuote = false; };
  const finishRow = () => {
    finishCell();
    if (row.some(value => value !== "")) rows.push(row);
    row = [];
    if (rows.length > MAX_REPORT_ROWS + 20) throw new Error(`Import up to ${MAX_REPORT_ROWS} players at a time.`);
  };
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (char === '"') { quoted = false; closedQuote = true; }
      else cell += char;
      continue;
    }
    if (char === '"') {
      if (cell.trim() || closedQuote) throw new Error("A quoted cell could not be read. Export the report as CSV again or paste its cells.");
      cell = ""; quoted = true;
    } else if (char === delimiter) finishCell();
    else if (char === "\n" || char === "\r") {
      if (char === "\r" && text[i + 1] === "\n") i++;
      finishRow();
    } else {
      if (closedQuote && char.trim()) throw new Error("Unexpected text after a quoted cell. Check the report format.");
      if (!closedQuote) cell += char;
    }
  }
  if (quoted) throw new Error("The report has an unfinished quoted cell. Export the report as CSV again or paste its cells.");
  if (cell || row.length || closedQuote) finishRow();
  return rows;
}

export function parseCourtReserveReport(raw: string): CourtReserveReportRow[] {
  if (!raw.trim()) throw new Error("Choose a CSV report or paste the report's name and skill columns, including their headings.");
  if (raw.length > MAX_COURTRESERVE_FILE_BYTES) throw new Error("The report is too large. Use a CSV file smaller than 2 MB.");
  const text = raw.replace(/^\uFEFF/, "");
  let parsed: string[][] | null = null, headerIndex = -1, nameIndex = -1, ratingIndex = -1;
  let parsingError: Error | null = null;
  for (const delimiter of ["\t", ",", ";"]) {
    try {
      const candidate = readDelimited(text, delimiter);
      for (let i = 0; i < Math.min(20, candidate.length); i++) {
        const headers = candidate[i].map(headerKey);
        const names = headers.flatMap((name, index) => NAME_HEADERS.has(name) ? [index] : []);
        const ratings = headers.flatMap((name, index) => RATING_HEADERS.has(name) ? [index] : []);
        if (names.length === 1 && ratings.length === 1) {
          parsed = candidate; headerIndex = i; nameIndex = names[0]; ratingIndex = ratings[0]; break;
        }
      }
      if (parsed) break;
    } catch (error) { parsingError = error instanceof Error ? error : null; }
  }
  if (!parsed) throw parsingError || new Error('Include the "Full Name" and "Enter Your Skill Level" columns and their headings. CSV and copied spreadsheet cells are supported.');
  const rows: CourtReserveReportRow[] = [];
  for (let i = headerIndex + 1; i < parsed.length; i++) {
    const cells = parsed[i];
    const name = cleanCourtReserveName(cells[nameIndex] || "");
    const requestedRating = String(cells[ratingIndex] || "").trim();
    if (!name && !requestedRating) continue;
    if (!name) throw new Error(`Report row ${i + 1} has a skill level but no player name.`);
    if (name.length > 200 || requestedRating.length > 120) throw new Error(`Report row ${i + 1} contains an unusually long name or skill value. Check its columns.`);
    rows.push({ rowNumber: i + 1, name, requestedRating });
  }
  if (!rows.length) throw new Error("The report has headings but no players.");
  if (rows.length > MAX_REPORT_ROWS) throw new Error(`Import up to ${MAX_REPORT_ROWS} players at a time.`);
  return rows;
}

export function matchCourtReserveRows(rows: CourtReserveReportRow[], players: CourtReservePlayer[]): CourtReserveMatchedRow[] {
  const byName = new Map<string, CourtReservePlayer[]>();
  for (const player of players) {
    if (!Number.isSafeInteger(player.id) || player.id <= 0) continue;
    const key = normalizeCourtReserveName(player.name);
    if (!key) continue;
    const candidates = byName.get(key) || [];
    if (!candidates.some(candidate => candidate.id === player.id)) candidates.push(player);
    byName.set(key, candidates);
  }
  return rows.map(row => {
    const candidates = byName.get(normalizeCourtReserveName(row.name)) || [];
    return { ...row, playerId: candidates.length === 1 ? candidates[0].id : null,
      matchStatus: candidates.length === 1 ? "matched" : candidates.length > 1 ? "ambiguous" : "unmatched",
      candidateIds: candidates.map(player => player.id) };
  });
}

/** Selection is explicit for unresolved names; duplicate identities are never imported. */
export function buildCourtReserveSelection(rows: CourtReserveReportRow[], choices: Record<number, number | null>, skipped: Set<number> = new Set()): CourtReserveRosterSelection {
  const playerIds: number[] = [], requestedRatings: Record<string, string> = {};
  for (const row of rows) {
    if (skipped.has(row.rowNumber)) continue;
    const id = choices[row.rowNumber];
    if (!Number.isSafeInteger(id) || Number(id) <= 0) throw new Error(`Choose a PCS profile for ${row.name}, or skip that row.`);
    if (playerIds.includes(Number(id))) throw new Error("One PCS player is selected more than once. Choose the correct profiles or skip the duplicate row.");
    playerIds.push(Number(id));
    if (row.requestedRating) requestedRatings[String(id)] = row.requestedRating;
  }
  if (!playerIds.length) throw new Error("Select at least one player to import.");
  return { playerIds, requestedRatings };
}
