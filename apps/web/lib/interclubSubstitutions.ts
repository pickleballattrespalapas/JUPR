import { CompetitionDocument, CompetitionEncounter, CompetitionGame, CompetitionPairing, CompetitionPlayer, activeSinglesPlayers, matchesSkillLevel, pairingLabels, scheduleRoundLabel } from "./interclubCompetition";

export type GameRow = { encounter: CompetitionEncounter; pairing: CompetitionPairing; game: CompetitionGame; number: number };
export type InjurySubstitution = { gameId: string; side: "a" | "b"; outgoing: string; incoming: string; previousIncoming?: string };
export type SubstitutionReview = InjurySubstitution & { row: GameRow; reason: string; missingReason: boolean; returningGames: string[] };
const kindOrder = ["women", "men", "mixed_a", "mixed_b"];
const gender = (value?: string) => /^(f|female|woman|women)$/i.test(value || "") ? "f" : /^(m|male|man|men)$/i.test(value || "") ? "m" : "";
const cutoff = (pairing: CompetitionPairing) => pairing.eligibility_deadline ? Date.parse(pairing.eligibility_deadline) : null;
const scope = (row: GameRow, side: "a" | "b") => JSON.stringify([row.encounter[`club_${side}`], row.encounter.division, cutoff(row.pairing)]);
export const gamePlayers = (row: GameRow, side: "a" | "b") => row.game[`players_${side}`].length ? row.game[`players_${side}`] : row.pairing[`players_${side}`];

// Same chronological order as official validation, independent of storage order.
export function competitionGameRows(document: CompetitionDocument): GameRow[] {
  return [...document.encounters].sort((a, b) => a.rotation - b.rotation || a.id.localeCompare(b.id)).flatMap(encounter =>
    [...encounter.pairings].sort((a, b) => kindOrder.indexOf(a.kind) - kindOrder.indexOf(b.kind)).flatMap(pairing =>
      pairing.games.map((game, index) => ({ encounter, pairing, game, number: index + 1 }))));
}
export function gameLocation(document: CompetitionDocument, row: GameRow): string {
  return `Skill level ${row.encounter.division} · ${scheduleRoundLabel(document)} ${row.encounter.rotation}${row.pairing.court ? ` · Court ${row.pairing.court}` : ""} · ${pairingLabels[row.pairing.kind]} · Game ${row.number}`;
}
export function gameFromError(document: CompetitionDocument, error: string): string | undefined {
  const normalize = (value: string) => value.replaceAll("’", "'").replaceAll("Mixed doubles", "Mixed");
  return competitionGameRows(document).find(row => normalize(error).startsWith(`${normalize(gameLocation(document, row))}:`))?.game.id;
}

// Find each real change once. A later accidental return is attached to the
// original substitution, so the operator can repair the whole chain at once.
export function reviewSubstitutions(document: CompetitionDocument): SubstitutionReview[] {
  const current = new Map<string, string[]>(), removed = new Map<string, Map<string, SubstitutionReview>>();
  const changes: SubstitutionReview[] = [];
  for (const row of competitionGameRows(document)) {
    if (["forfeit", "double_forfeit", "unplayed"].includes(row.game.status)) continue;
    for (const side of ["a", "b"] as const) {
      const key = scope(row, side), lineupKey = `${key}:${row.pairing.kind}`;
      const actual = gamePlayers(row, side), previous = current.get(lineupKey) || row.pairing[`players_${side}`];
      const injuries = removed.get(key) || new Map<string, SubstitutionReview>();
      removed.set(key, injuries);
      const returns = actual.filter(id => injuries.has(id));
      if (returns.length) {
        for (const id of returns) injuries.get(id)!.returningGames.push(row.game.id);
        continue;
      }
      const outgoing = previous.filter(id => !actual.includes(id)), incoming = actual.filter(id => !previous.includes(id));
      if (outgoing.length === 1 && incoming.length === 1) {
        const change = { gameId: row.game.id, side, outgoing: outgoing[0], incoming: incoming[0], row,
          reason: row.game.injury_reason?.trim() || "", missingReason: !row.game.injury_reason?.trim(), returningGames: [] };
        changes.push(change); injuries.set(outgoing[0], change);
      }
      current.set(lineupKey, actual);
    }
  }
  return changes;
}

export function substituteForRemainingGames(document: CompetitionDocument, change: InjurySubstitution, note = "Injury"): { document: CompetitionDocument; changedGames: number } {
  const rows = competitionGameRows(document), start = rows.findIndex(row => row.game.id === change.gameId);
  if (start < 0 || change.outgoing === change.incoming) throw new Error("Choose a different replacement player.");
  const source = rows[start], sourceScope = scope(source, change.side);
  const actual = gamePlayers(source, change.side);
  if (!actual.includes(change.outgoing) && !actual.includes(change.incoming) && !actual.includes(change.previousIncoming || "")) throw new Error("The playing lineup changed. Choose the injured player again.");
  if (actual.includes(change.outgoing) && actual.includes(change.incoming)) throw new Error("The replacement is already playing in this pair.");
  const laterChanges = reviewSubstitutions(document), replacements = new Map([[change.outgoing, change.incoming]]);
  const edits = new Map<string, CompetitionGame>();
  const replacement = (id: string) => {
    const visited = new Set<string>();
    while (replacements.has(id) && !visited.has(id)) { visited.add(id); id = replacements.get(id)!; }
    return id;
  };
  let reason = note.trim() || "Injury";
  for (const row of rows.slice(start)) {
    const side = row.encounter.club_a === source.encounter[`club_${change.side}`] ? "a" : row.encounter.club_b === source.encounter[`club_${change.side}`] ? "b" : null;
    if (!side || scope(row, side) !== sourceScope || ["forfeit", "double_forfeit", "unplayed"].includes(row.game.status)) continue;
    // Preserve a subsequent injury replacement instead of bringing its injured
    // predecessor back when repairing an older, partially recorded change.
    for (const later of laterChanges.filter(item => item.gameId === row.game.id && item.side === side && item.gameId !== change.gameId)) {
      if ([...replacements.values()].includes(later.outgoing) || later.outgoing === change.previousIncoming) {
        replacements.set(later.outgoing === change.previousIncoming ? change.incoming : later.outgoing, later.incoming); reason = later.reason || "Injury";
      }
    }
    const before = gamePlayers(row, side);
    const correctsOriginal = row.pairing.kind === source.pairing.kind || row.pairing[`players_${side}`].includes(change.outgoing);
    const after = before.map(id => replacement(correctsOriginal && id === change.previousIncoming ? change.incoming : id));
    if (new Set(after).size !== after.length) throw new Error("That replacement is already in a later pair. Choose another eligible player.");
    if (after.some((id, index) => id !== before[index]) || row.game.id === change.gameId) {
      edits.set(row.game.id, { ...row.game, [`players_${side}`]: after, injury_reason: row.game.injury_reason?.trim() || reason });
    }
  }
  return { changedGames: edits.size, document: { ...document, encounters: document.encounters.map(encounter => {
    const updated = { ...encounter, pairings: encounter.pairings.map(pairing => ({ ...pairing, games: pairing.games.map(game => edits.get(game.id) || game) })) };
    if (encounter.tiebreak) {
      updated.tiebreak = { ...encounter.tiebreak };
      for (const side of ["a", "b"] as const) {
        const before = activeSinglesPlayers(encounter, side), after = activeSinglesPlayers(updated, side);
        const removed = before.filter(id => !after.includes(id)), added = after.filter(id => !before.includes(id));
        if (removed.length === 1 && added.length === 1) updated.tiebreak[`order_${side}`] = encounter.tiebreak[`order_${side}`].map(id => id === removed[0] ? added[0] : id);
      }
    }
    return updated;
  }) } };
}

export function replacementOptions(document: CompetitionDocument, row: GameRow, side: "a" | "b", outgoing: string, options: CompetitionPlayer[], players: Map<string, CompetitionPlayer>): CompetitionPlayer[] {
  const expectedGender = row.pairing.kind === "women" ? "f" : row.pairing.kind === "men" ? "m" : gender(players.get(outgoing)?.gender);
  const rows = competitionGameRows(document);
  const start = rows.findIndex(item => item.game.id === row.game.id);
  const affected = rows.slice(start).filter(other => other.encounter.division === row.encounter.division && cutoff(other.pairing) === cutoff(row.pairing) &&
    ["a", "b"].some(value => other.encounter[`club_${value as "a" | "b"}`] === row.encounter[`club_${side}`] && (gamePlayers(other, value as "a" | "b").includes(outgoing) || other.pairing[`players_${value as "a" | "b"}`].includes(outgoing))));
  const injured = new Set(reviewSubstitutions(document).filter(change => scope(change.row, change.side) === scope(row, side)).map(change => change.outgoing));
  return options.filter(player => player.entry_id !== outgoing && !injured.has(player.entry_id) && matchesSkillLevel(player, row.encounter.division) && (!expectedGender || gender(player.gender) === expectedGender) &&
    !(document.phase === "regular" && rows.some(other => affected.some(target => target.encounter.rotation === other.encounter.rotation && target.pairing.id !== other.pairing.id) &&
      ["a", "b"].some(s => gamePlayers(other, s as "a" | "b").includes(player.entry_id)))) &&
    !gamePlayers(row, side).includes(player.entry_id));
}

export function substitutionEligibilityProblem(change: SubstitutionReview, options: CompetitionPlayer[], players: Map<string, CompetitionPlayer>): string {
  const player = options.find(item => item.entry_id === change.incoming);
  if (!player || !matchesSkillLevel(player, change.row.encounter.division)) return "Choose a replacement from this club’s eligible pool for this skill level.";
  const expected = change.row.pairing.kind === "women" ? "f" : change.row.pairing.kind === "men" ? "m" : gender(players.get(change.outgoing)?.gender);
  if (expected && gender(player.gender) !== expected) return `This replacement is not eligible for ${pairingLabels[change.row.pairing.kind].toLowerCase()}. Choose an eligible replacement below.`;
  return "";
}
