import TournamentLiveRoute from "../../live-operations/TournamentLiveRoute";

type Props = { searchParams?: Record<string, string | string[] | undefined> };
export default function TournamentResultsImportPage({ searchParams }: Props) {
  return <TournamentLiveRoute searchParams={searchParams} view="results" phase="publish" kicker="Tournament Manager / Publish" title="review results" description="Review division and draw cards, completed and missing scores, standings, podium readiness, corrections, and publish blockers." />;
}
