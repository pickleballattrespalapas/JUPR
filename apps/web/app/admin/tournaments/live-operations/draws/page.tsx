import TournamentLiveRoute from "../TournamentLiveRoute";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentDrawsSchedulePage({ searchParams }: Props) {
  return <TournamentLiveRoute searchParams={searchParams} view="draws" phase="live" kicker="Tournament Manager / Live Operations" title="draws and schedule" description="Review prepared rounds and court slots, then progress the selected draw only when its prerequisites are complete." />;
}
