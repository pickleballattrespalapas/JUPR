import TournamentDayWorkspaceRoute from "./TournamentDayWorkspaceRoute";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentLiveOperationsPhasePage({ searchParams }: Props) {
  return <TournamentDayWorkspaceRoute searchParams={searchParams} />;
}
