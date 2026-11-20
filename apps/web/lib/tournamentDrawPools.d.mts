import type { TournamentDrawPoolConfiguration } from "./adminTournamentApi";
export type PoolEditorTeam = { id: string; label?: string; seed?: number | null; team_number?: number | null };
export type PoolTeamLabel = { poolId: string; poolName: string; number: number; seed: number | null };
export function poolTeamLabels(configuration: TournamentDrawPoolConfiguration | null | undefined, teams?: PoolEditorTeam[]): Map<string, PoolTeamLabel>;
export function seededPoolConfiguration(teams: PoolEditorTeam[], poolCount?: number): TournamentDrawPoolConfiguration;
export function inspectPoolConfiguration(configuration: TournamentDrawPoolConfiguration | null, teams: PoolEditorTeam[]): {
  ok: boolean; errors: string[]; poolMatches: { poolId: string; matches: number; gamesPerTeam: number }[];
  totalMatches: number; totalQualifiers: number; uneven: boolean;
};
export function moveTeamToPool(configuration: TournamentDrawPoolConfiguration, teamId: string, poolId: string): TournamentDrawPoolConfiguration;
