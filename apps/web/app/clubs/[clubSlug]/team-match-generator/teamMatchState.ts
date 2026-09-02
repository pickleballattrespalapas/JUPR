export type Team = {
  id: string;
  name: string;
  women: [string, string];
  men: [string, string];
};

export type RegulationGameKey = "women" | "men" | "mixed-1" | "mixed-2";

export type RegulationGame = {
  key: RegulationGameKey;
  label: string;
  sideA: string[];
  sideB: string[];
  scoreA: number | null;
  scoreB: number | null;
  submitted: boolean;
  submittedAt?: string | null;
};

export type MixedLineup = {
  woman: string;
  man: string;
};

export type DreamBreaker = {
  orderA: string[];
  orderB: string[];
  scoreA: number | null;
  scoreB: number | null;
  submitted: boolean;
  submittedAt?: string | null;
};

export type TeamMatchup = {
  id: string;
  round: number;
  teamAId: string;
  teamBId: string;
  games: RegulationGame[];
  mixedLineups: {
    teamA: MixedLineup | null;
    teamB: MixedLineup | null;
  };
  dreamBreaker: DreamBreaker;
};

export type TeamMatchSession = {
  schemaVersion: 2;
  id: string;
  title: string;
  teams: Team[];
  matchups: TeamMatchup[];
  activeMatchupId: string | null;
  createdAt: string;
  updatedAt: string;
};

export type MatchupStep =
  | { kind: "game"; gameKey: RegulationGameKey }
  | { kind: "mixed_lineups" }
  | { kind: "dreambreaker_lineups" }
  | { kind: "dreambreaker_score" }
  | { kind: "complete" };

export type MatchupResult = {
  complete: boolean;
  winnerTeamId: string | null;
  regulationWinsA: number;
  regulationWinsB: number;
  decidedByDreamBreaker: boolean;
};

export type TeamStanding = {
  id: string;
  name: string;
  teamWins: number;
  teamLosses: number;
  gameWins: number;
  gameLosses: number;
  pointDiff: number;
  played: number;
  dreamBreakerWins: number;
};

export function createId(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function blankTeam(index: number): Team {
  return {
    id: createId("team"),
    name: `Team ${index + 1}`,
    women: ["", ""],
    men: ["", ""]
  };
}

export function normalizedName(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

export function teamPlayers(team: Team): string[] {
  return [...team.women, ...team.men].map((name) => name.trim());
}

export function teamReady(team: Team): boolean {
  const names = teamPlayers(team).filter(Boolean);
  return (
    team.name.trim().length > 0 &&
    names.length === 4 &&
    new Set(names.map(normalizedName)).size === 4
  );
}

export function duplicateRosterNames(teams: Team[]): string[] {
  const labels = new Map<string, string>();
  const duplicates = new Set<string>();
  for (const team of teams) {
    for (const player of teamPlayers(team)) {
      const key = normalizedName(player);
      if (!key) continue;
      if (labels.has(key)) duplicates.add(labels.get(key) || player);
      else labels.set(key, player);
    }
  }
  return [...duplicates].sort((a, b) => a.localeCompare(b));
}

function emptyGame(
  key: RegulationGameKey,
  label: string,
  sideA: string[],
  sideB: string[]
): RegulationGame {
  return {
    key,
    label,
    sideA,
    sideB,
    scoreA: null,
    scoreB: null,
    submitted: false,
    submittedAt: null
  };
}

function buildRegulationGames(teamA: Team, teamB: Team): RegulationGame[] {
  return [
    emptyGame("women", "Women’s Doubles", [...teamA.women], [...teamB.women]),
    emptyGame("men", "Men’s Doubles", [...teamA.men], [...teamB.men]),
    emptyGame("mixed-1", "Mixed Doubles 1", [], []),
    emptyGame("mixed-2", "Mixed Doubles 2", [], [])
  ];
}

function roundRobinPairs(teamIds: string[]): Array<{ round: number; a: string; b: string }> {
  const ids = [...teamIds];
  if (ids.length % 2 === 1) ids.push("BYE");
  const rounds = ids.length - 1;
  const half = ids.length / 2;
  const rotation = [...ids];
  const result: Array<{ round: number; a: string; b: string }> = [];

  for (let round = 1; round <= rounds; round += 1) {
    for (let index = 0; index < half; index += 1) {
      const a = rotation[index];
      const b = rotation[rotation.length - 1 - index];
      if (a !== "BYE" && b !== "BYE") result.push({ round, a, b });
    }
    const fixed = rotation[0];
    const rest = rotation.slice(1);
    rest.unshift(rest.pop() as string);
    rotation.splice(0, rotation.length, fixed, ...rest);
  }
  return result;
}

export function createSession(title: string, teams: Team[]): TeamMatchSession {
  const teamMap = new Map(teams.map((team) => [team.id, team]));
  const matchups = roundRobinPairs(teams.map((team) => team.id)).map((pair) => {
    const teamA = teamMap.get(pair.a) as Team;
    const teamB = teamMap.get(pair.b) as Team;
    return {
      id: createId("matchup"),
      round: pair.round,
      teamAId: pair.a,
      teamBId: pair.b,
      games: buildRegulationGames(teamA, teamB),
      mixedLineups: { teamA: null, teamB: null },
      dreamBreaker: {
        orderA: [],
        orderB: [],
        scoreA: null,
        scoreB: null,
        submitted: false,
        submittedAt: null
      }
    } satisfies TeamMatchup;
  });
  const now = new Date().toISOString();
  return {
    schemaVersion: 2,
    id: createId("team-session"),
    title: title.trim() || "Team Match",
    teams,
    matchups,
    activeMatchupId: matchups[0]?.id || null,
    createdAt: now,
    updatedAt: now
  };
}

export function mixedPairs(team: Team, lineup: MixedLineup): [[string, string], [string, string]] {
  const firstWoman = team.women.find((name) => name === lineup.woman) || team.women[0];
  const firstMan = team.men.find((name) => name === lineup.man) || team.men[0];
  const secondWoman = team.women.find((name) => name !== firstWoman) || team.women[1];
  const secondMan = team.men.find((name) => name !== firstMan) || team.men[1];
  return [
    [firstWoman, firstMan],
    [secondWoman, secondMan]
  ];
}

export function applyMixedLineups(
  matchup: TeamMatchup,
  teamA: Team,
  teamB: Team,
  lineupA: MixedLineup,
  lineupB: MixedLineup
): TeamMatchup {
  const pairsA = mixedPairs(teamA, lineupA);
  const pairsB = mixedPairs(teamB, lineupB);
  return {
    ...matchup,
    mixedLineups: { teamA: lineupA, teamB: lineupB },
    games: matchup.games.map((game) => {
      if (game.key === "mixed-1") {
        return { ...game, sideA: [...pairsA[0]], sideB: [...pairsB[0]] };
      }
      if (game.key === "mixed-2") {
        return { ...game, sideA: [...pairsA[1]], sideB: [...pairsB[1]] };
      }
      return game;
    })
  };
}

export function gameByKey(matchup: TeamMatchup, key: RegulationGameKey): RegulationGame {
  const game = matchup.games.find((candidate) => candidate.key === key);
  if (!game) throw new Error(`Missing ${key} game.`);
  return game;
}

export function regulationWins(matchup: TeamMatchup): { a: number; b: number; submitted: number } {
  let a = 0;
  let b = 0;
  let submitted = 0;
  for (const game of matchup.games) {
    if (!game.submitted || game.scoreA === null || game.scoreB === null || game.scoreA === game.scoreB) continue;
    submitted += 1;
    if (game.scoreA > game.scoreB) a += 1;
    else b += 1;
  }
  return { a, b, submitted };
}

export function validDreamBreakerOrder(order: string[], team: Team): boolean {
  if (order.length !== 4) return false;
  const roster = teamPlayers(team).map(normalizedName).sort();
  const selected = order.map(normalizedName).sort();
  return selected.length === 4 && new Set(selected).size === 4 && selected.every((value, index) => value === roster[index]);
}

export function matchupStep(matchup: TeamMatchup, teamA: Team, teamB: Team): MatchupStep {
  if (!gameByKey(matchup, "women").submitted) return { kind: "game", gameKey: "women" };
  if (!gameByKey(matchup, "men").submitted) return { kind: "game", gameKey: "men" };
  if (!matchup.mixedLineups.teamA || !matchup.mixedLineups.teamB) return { kind: "mixed_lineups" };
  if (!gameByKey(matchup, "mixed-1").submitted) return { kind: "game", gameKey: "mixed-1" };
  if (!gameByKey(matchup, "mixed-2").submitted) return { kind: "game", gameKey: "mixed-2" };

  const wins = regulationWins(matchup);
  if (wins.a !== wins.b) return { kind: "complete" };
  if (
    !validDreamBreakerOrder(matchup.dreamBreaker.orderA, teamA) ||
    !validDreamBreakerOrder(matchup.dreamBreaker.orderB, teamB)
  ) {
    return { kind: "dreambreaker_lineups" };
  }
  if (!matchup.dreamBreaker.submitted) return { kind: "dreambreaker_score" };
  return { kind: "complete" };
}

export function matchupResult(matchup: TeamMatchup): MatchupResult {
  const wins = regulationWins(matchup);
  if (wins.submitted < 4) {
    return {
      complete: false,
      winnerTeamId: null,
      regulationWinsA: wins.a,
      regulationWinsB: wins.b,
      decidedByDreamBreaker: false
    };
  }
  if (wins.a > wins.b) {
    return {
      complete: true,
      winnerTeamId: matchup.teamAId,
      regulationWinsA: wins.a,
      regulationWinsB: wins.b,
      decidedByDreamBreaker: false
    };
  }
  if (wins.b > wins.a) {
    return {
      complete: true,
      winnerTeamId: matchup.teamBId,
      regulationWinsA: wins.a,
      regulationWinsB: wins.b,
      decidedByDreamBreaker: false
    };
  }
  const breaker = matchup.dreamBreaker;
  if (
    !breaker.submitted ||
    breaker.scoreA === null ||
    breaker.scoreB === null ||
    breaker.scoreA === breaker.scoreB
  ) {
    return {
      complete: false,
      winnerTeamId: null,
      regulationWinsA: wins.a,
      regulationWinsB: wins.b,
      decidedByDreamBreaker: true
    };
  }
  return {
    complete: true,
    winnerTeamId: breaker.scoreA > breaker.scoreB ? matchup.teamAId : matchup.teamBId,
    regulationWinsA: wins.a,
    regulationWinsB: wins.b,
    decidedByDreamBreaker: true
  };
}

export function standingsForSession(session: TeamMatchSession): TeamStanding[] {
  const rows = new Map(
    session.teams.map((team) => [
      team.id,
      {
        id: team.id,
        name: team.name,
        teamWins: 0,
        teamLosses: 0,
        gameWins: 0,
        gameLosses: 0,
        pointDiff: 0,
        played: 0,
        dreamBreakerWins: 0
      } satisfies TeamStanding
    ])
  );

  for (const matchup of session.matchups) {
    const sideA = rows.get(matchup.teamAId);
    const sideB = rows.get(matchup.teamBId);
    if (!sideA || !sideB) continue;

    for (const game of matchup.games) {
      if (!game.submitted || game.scoreA === null || game.scoreB === null || game.scoreA === game.scoreB) continue;
      sideA.pointDiff += game.scoreA - game.scoreB;
      sideB.pointDiff += game.scoreB - game.scoreA;
      if (game.scoreA > game.scoreB) {
        sideA.gameWins += 1;
        sideB.gameLosses += 1;
      } else {
        sideB.gameWins += 1;
        sideA.gameLosses += 1;
      }
    }

    const result = matchupResult(matchup);
    if (!result.complete || !result.winnerTeamId) continue;
    sideA.played += 1;
    sideB.played += 1;
    if (result.winnerTeamId === matchup.teamAId) {
      sideA.teamWins += 1;
      sideB.teamLosses += 1;
      if (result.decidedByDreamBreaker) sideA.dreamBreakerWins += 1;
    } else {
      sideB.teamWins += 1;
      sideA.teamLosses += 1;
      if (result.decidedByDreamBreaker) sideB.dreamBreakerWins += 1;
    }
  }

  return [...rows.values()].sort((a, b) =>
    b.teamWins - a.teamWins ||
    b.gameWins - a.gameWins ||
    b.pointDiff - a.pointDiff ||
    a.name.localeCompare(b.name)
  );
}

export function teamMatchDraftKey(clubId: string): string {
  return `team-match-generator:${clubId}:draft-v2`;
}

export function legacyTeamMatchDraftKey(clubId: string): string {
  return `team-match-generator:${clubId}`;
}

export function teamMatchSessionKey(clubId: string, sessionId: string): string {
  return `team-match-generator:${clubId}:session:${sessionId}`;
}

export function lastTeamMatchSessionKey(clubId: string): string {
  return `team-match-generator:${clubId}:last-session`;
}

export function readSession(clubId: string, sessionId: string): TeamMatchSession | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(teamMatchSessionKey(clubId, sessionId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as TeamMatchSession;
    if (parsed.schemaVersion !== 2 || parsed.id !== sessionId || !Array.isArray(parsed.teams) || !Array.isArray(parsed.matchups)) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function writeSession(clubId: string, session: TeamMatchSession): void {
  if (typeof window === "undefined") return;
  const next = { ...session, updatedAt: new Date().toISOString() };
  window.localStorage.setItem(teamMatchSessionKey(clubId, session.id), JSON.stringify(next));
  window.localStorage.setItem(lastTeamMatchSessionKey(clubId), session.id);
}
