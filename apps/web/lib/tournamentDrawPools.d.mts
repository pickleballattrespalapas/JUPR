import type { TournamentDrawPoolConfiguration } from "./adminTournamentApi";
export type PoolEditorTeam = { id: string; label?: string; seed?: number | null; team_number?: number | null };
export function seededPoolConfiguration(teams: PoolEditorTeam[], poolCount?: number): TournamentDrawPoolConfiguration;
export function inspectPoolConfiguration(configuration: TournamentDrawPoolConfiguration | null, teams: PoolEditorTeam[]): {
  ok: boolean; errors: string[]; poolMatches: { poolId: string; matches: number; gamesPerTeam: number }[];
  totalMatches: number; totalQualifiers: number; uneven: boolean;
};
export function moveTeamToPool(configuration: TournamentDrawPoolConfiguration, teamId: string, poolId: string): TournamentDrawPoolConfiguration;
