import type { AdminMatchUploaderRoundRobinMatch } from "@/lib/adminMatchUploaderApi";

export type LeagueLiveGameEdit = {
  playerIds?: number[];
  substitutionReason?: string;
  notPlayed?: boolean;
  unplayedReason?: string;
};
export type LeagueLiveGameEdits = Record<string, LeagueLiveGameEdit>;

export function scheduledGamePlayers(match: AdminMatchUploaderRoundRobinMatch): number[] {
  return [match.t1_p1, match.t1_p2, match.t2_p1, match.t2_p2].map(Number);
}
export function actualGamePlayers(match: AdminMatchUploaderRoundRobinMatch, edit?: LeagueLiveGameEdit): number[] {
  return edit?.playerIds || scheduledGamePlayers(match);
}
export function gameHasSubstitution(match: AdminMatchUploaderRoundRobinMatch, edit?: LeagueLiveGameEdit): boolean {
  return actualGamePlayers(match, edit).some((id, index) => id !== scheduledGamePlayers(match)[index]);
}
export function gameEditError(match: AdminMatchUploaderRoundRobinMatch, edit?: LeagueLiveGameEdit): string | null {
  if (edit?.notPlayed) return edit.unplayedReason?.trim() ? null : "Give a reason this game was not completed.";
  const ids = actualGamePlayers(match, edit);
  if (ids.length !== 4 || ids.some(id => !Number.isInteger(id) || id <= 0) || new Set(ids).size !== 4) return "Choose four different players for this game.";
  if (gameHasSubstitution(match, edit) && !edit?.substitutionReason?.trim()) return "Give a reason for this game's substitution.";
  return null;
}
