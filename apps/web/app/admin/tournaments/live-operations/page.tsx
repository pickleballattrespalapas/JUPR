import TournamentLiveRoute from "./TournamentLiveRoute";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentLiveOperationsPhasePage({ searchParams }: Props) {
  return <TournamentLiveRoute searchParams={searchParams} view="overview" phase="live" kicker="Tournament Manager / Live Operations" title="live operations" description="Move through tournament-day work in focused modules while preserving the selected tournament and draw." />;
}
