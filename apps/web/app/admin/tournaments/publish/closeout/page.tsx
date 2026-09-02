import TournamentLiveRoute from "../../live-operations/TournamentLiveRoute";

type Props = { searchParams?: Record<string, string | string[] | undefined> };
export default function TournamentCloseoutPage({ searchParams }: Props) {
  return <TournamentLiveRoute searchParams={searchParams} view="closeout" phase="publish" kicker="Tournament Manager / Publish" title="tournament closeout" description="Review live completion cards and archive only after every server-enforced closeout prerequisite passes." />;
}
