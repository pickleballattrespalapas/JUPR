import TournamentLiveRoute from "../../live-operations/TournamentLiveRoute";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };
export default async function TournamentCloseoutPage(props: Props) {
  const searchParams = await props.searchParams;
  return <TournamentLiveRoute searchParams={searchParams} view="closeout" phase="publish" kicker="Tournament Manager / Publish" title="tournament closeout" description="Review live completion cards and archive only after every server-enforced closeout prerequisite passes." />;
}
