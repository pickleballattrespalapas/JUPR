export type RoundScorePlayer = { id: number; name: string };
export type RoundScoreGame = {
  key: string;
  court: number;
  gameLabel: string;
  team1: RoundScorePlayer[];
  team2: RoundScorePlayer[];
  scoreT1: number | null;
  scoreT2: number | null;
  notPlayed?: boolean;
  note?: string | null;
  unusualReason?: string | null;
  correctionUrl?: string;
  officialMatchId?: number;
  corrected?: boolean;
};
export type RoundScoreStanding = RoundScorePlayer & { wins: number; losses: number; differential: number };

export function courtScoreCorrectionUrl(games: RoundScoreGame[]): string | undefined {
  const recorded = games.filter(game => !game.notPlayed || game.officialMatchId != null);
  if (recorded.length < 2 || recorded.some(game => !Number.isSafeInteger(game.officialMatchId) || Number(game.officialMatchId) <= 0)) return undefined;
  const ids = [...new Set(recorded.map(game => game.officialMatchId))].join(",");
  const query = new URLSearchParams({ match_ids: ids, selected_ids: ids });
  return `/admin/match-log/bulk?${query.toString()}`;
}

export function roundCourtStandings(games: RoundScoreGame[]): RoundScoreStanding[] {
  const players = new Map<number, RoundScoreStanding>();
  for (const game of games) {
    for (const player of [...game.team1, ...game.team2]) {
      if (!players.has(player.id)) players.set(player.id, { ...player, wins: 0, losses: 0, differential: 0 });
    }
    if (game.notPlayed || game.scoreT1 == null || game.scoreT2 == null || game.scoreT1 === game.scoreT2) continue;
    for (const [index, team] of [game.team1, game.team2].entries()) {
      const margin = (game.scoreT1 - game.scoreT2) * (index === 0 ? 1 : -1);
      for (const player of team) {
        const row = players.get(player.id)!;
        row.wins += Number(margin > 0);
        row.losses += Number(margin < 0);
        row.differential += margin;
      }
    }
  }
  return [...players.values()].sort((a, b) => b.wins - a.wins || b.differential - a.differential || a.name.localeCompare(b.name));
}

export function savedRoundScoreGames(
  round: { matches_json?: Array<Record<string, unknown>> | null; preview_json?: unknown; score_readback_status?: string },
  playerNames: Map<number, string>,
): RoundScoreGame[] {
  const preview = (round.preview_json || {}) as { courts?: Array<{ matches?: Array<{ row_id: string; label?: string; t1?: RoundScorePlayer[]; t2?: RoundScorePlayer[] }> }>; unplayed_games?: Array<{ series_key: string; court: number; game_number: number; reason?: string }> };
  const slots = (preview.courts || []).flatMap(court => court.matches || []);
  const names = new Map(playerNames);
  for (const slot of slots) for (const player of [...(slot.t1 || []), ...(slot.t2 || [])]) if (!names.has(player.id)) names.set(player.id, player.name);
  const player = (id: unknown): RoundScorePlayer => ({ id: Number(id), name: names.get(Number(id)) || `Player ${id}` });
  const games: RoundScoreGame[] = (round.matches_json || []).map((row, index) => {
    const slot = slots.find(candidate => candidate.row_id === row.series_key);
    const query = new URLSearchParams();
    if (row.official_match_id != null) query.set("match_id", String(row.official_match_id));
    else if (row.context_id) { query.set("context_type", "league_live_session"); query.set("context_id", String(row.context_id)); }
    const canCorrect = Boolean(query.size) && !["unavailable", "snapshot"].includes(round.score_readback_status || "snapshot");
    const score = (value: unknown) => value == null || value === "" || !Number.isFinite(Number(value)) ? null : Number(value);
    return {
      key: String(row.context_id || `${row.series_key || index}:${row.game_number || 1}`),
      court: Number(row.court),
      gameLabel: `${slot?.label || `Game ${index + 1}`}${Number(row.series_games) > 1 ? ` · Game ${row.game_number}` : ""}`,
      team1: [player(row.t1_p1), player(row.t1_p2)],
      team2: [player(row.t2_p1), player(row.t2_p2)],
      scoreT1: score(row.score_t1), scoreT2: score(row.score_t2),
      notPlayed: Boolean(row.score_excluded),
      note: row.score_excluded ? "Excluded from official results" : String(row.substitution_reason || ""),
      corrected: Boolean(row.score_corrected),
      officialMatchId: canCorrect && row.official_match_id != null ? Number(row.official_match_id) : undefined,
      correctionUrl: canCorrect ? `/admin/match-log/edit?${query.toString()}` : undefined,
    };
  });
  for (const skipped of preview.unplayed_games || []) {
    const slot = slots.find(candidate => candidate.row_id === skipped.series_key);
    games.push({ key: `unplayed:${skipped.series_key}:${skipped.game_number}`, court: skipped.court,
      gameLabel: `${slot?.label || "Game"} · Game ${skipped.game_number}`, team1: slot?.t1 || [], team2: slot?.t2 || [],
      scoreT1: null, scoreT2: null, notPlayed: true, note: skipped.reason });
  }
  return games;
}
