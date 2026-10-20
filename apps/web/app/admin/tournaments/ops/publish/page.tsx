import TournamentLiveRoute from "../../live-operations/TournamentLiveRoute";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };
export default async function TournamentOfficialPublishPage(props: Props) {
  const searchParams = await props.searchParams;
  return <TournamentLiveRoute searchParams={searchParams} view="publish" phase="publish" kicker="Tournament Manager / Publish" title="publish divisions" description="Compare tournament readiness with runtime capability, inspect exact blockers, and publish only a fully reviewed division." />;
}
