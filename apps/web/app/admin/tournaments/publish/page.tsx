import TournamentLiveRoute from "../live-operations/TournamentLiveRoute";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentPublishPhasePage({ searchParams }: Props) {
  return <TournamentLiveRoute searchParams={searchParams} view="publish-overview" phase="publish" kicker="Tournament Manager / Publish" title="publish" description="Review results, import external results separately, publish only ready divisions, and finish tournament closeout." />;
}
