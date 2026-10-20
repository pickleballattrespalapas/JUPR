import TournamentDayWorkspaceRoute from "./TournamentDayWorkspaceRoute";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };

export default async function TournamentLiveOperationsPhasePage(props: Props) {
  const searchParams = await props.searchParams;
  return <TournamentDayWorkspaceRoute searchParams={searchParams} />;
}
