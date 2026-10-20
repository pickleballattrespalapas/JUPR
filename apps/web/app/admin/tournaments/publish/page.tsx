import TournamentLiveRoute from "../live-operations/TournamentLiveRoute";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };

export default async function TournamentPublishPhasePage(props: Props) {
  const searchParams = await props.searchParams;
  return <TournamentLiveRoute searchParams={searchParams} view="publish-overview" phase="publish" kicker="Tournament Manager / Publish" title="publish" description="Review results, import external results separately, publish only ready divisions, and finish tournament closeout." />;
}
