export function publicTeamLeagueHref(clubSlug: string, leagueName: string): string {
  return `/clubs/${encodeURIComponent(clubSlug)}/team-leagues/${encodeURIComponent(leagueName)}`;
}
