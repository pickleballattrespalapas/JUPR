import TournamentLiveRoute from "../../live-operations/TournamentLiveRoute";

type Props = { searchParams?: Record<string, string | string[] | undefined> };
export default function TournamentDrawOperationsPage({ searchParams }: Props) {
  return <TournamentLiveRoute searchParams={searchParams} view="draws" phase="live" kicker="Tournament Manager / Live Operations" title="draws and schedule" description="Review prepared rounds, court slots, and playoff progression for the selected draw." />;
}
