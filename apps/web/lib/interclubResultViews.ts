import type { CompetitionDocument, CompetitionPlayer } from "./interclubCompetition";
import type { CompetitionResult, ResultGame, ResultPlayer } from "./interclubPublic";

export const resultPairingNames: Record<string, string> = { women: "Women’s doubles", men: "Men’s doubles", mixed_a: "Mixed doubles A", mixed_b: "Mixed doubles B" };
export const sortSkillLevels = (values: string[]) => [...values].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
export const resultPlayers = (game: { players_a?: string[]; players_b?: string[] }) => [...(game.players_a || []), ...(game.players_b || [])];
export const hasResultPlayer = (row: CompetitionResult, player: string) => !player || row.pairings.some(pairing => pairing.games.some(game => resultPlayers(game).includes(player))) || row.tiebreak?.status === "completed" && resultPlayers(row.tiebreak).includes(player);
export function resultGameWinner(game: ResultGame): "a" | "b" | null {
  if (game.status === "completed" && game.a !== null && game.b !== null) return game.a > game.b ? "a" : game.b > game.a ? "b" : null;
  return game.status === "forfeit" || game.status === "retired" ? game.winner : null;
}
export function gameResultNote(game: ResultGame, names: Record<string, string>, row: CompetitionResult) {
  const winner = names[game.winner === "a" ? row.club_a : row.club_b] || "Winning club";
  if (game.status === "retired") return `Injury retirement; ${winner} awarded the game`;
  if (game.status === "forfeit") return `Forfeit — ${winner} wins`;
  if (game.status === "double_forfeit") return "Both clubs forfeited — a game loss for each club";
  if (game.status === "unplayed") return "Not played — weather";
  if (game.status === "pending") return "Result not entered";
  return "";
}
export function documentResults(document: CompetitionDocument): CompetitionResult[] {
  return document.encounters.map(encounter => ({ id: encounter.id, meet_id: document.meet_id, phase: document.phase,
    weather: document.weather, division: encounter.division, club_a: encounter.club_a, club_b: encounter.club_b,
    pairings: encounter.pairings.map(pairing => ({ kind: pairing.kind, games: pairing.games.map(game => ({
      status: game.status, a: game.a, b: game.b, winner: game.winner,
      players_a: ["completed", "retired"].includes(game.status) ? game.players_a.length ? game.players_a : pairing.players_a : [],
      players_b: ["completed", "retired"].includes(game.status) ? game.players_b.length ? game.players_b : pairing.players_b : [],
    })) })),
    tiebreak: encounter.tiebreak ? { ...encounter.tiebreak,
      players_a: encounter.tiebreak.status === "completed" ? encounter.tiebreak.order_a : [],
      players_b: encounter.tiebreak.status === "completed" ? encounter.tiebreak.order_b : [] } : null,
  }));
}

export function documentResultPlayers(document: CompetitionDocument, players: Map<string, CompetitionPlayer>): ResultPlayer[] {
  const catalog = new Map<string, ResultPlayer>();
  for (const row of documentResults(document)) {
    for (const game of [...row.pairings.flatMap(pairing => pairing.games), ...(row.tiebreak ? [row.tiebreak] : [])]) {
      for (const side of ["a", "b"] as const) {
        for (const id of game[`players_${side}`] || []) {
          const player = players.get(id);
          if (player) catalog.set(id, { id, name: player.name, club_id: row[`club_${side}`] });
        }
      }
    }
  }
  return [...catalog.values()];
}
