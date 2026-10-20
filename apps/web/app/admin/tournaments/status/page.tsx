import TournamentLiveRoute from "../live-operations/TournamentLiveRoute";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };

export default async function AdminTournamentStatusPage(props: Props) {
  const searchParams = await props.searchParams;
  return <TournamentLiveRoute searchParams={searchParams} view="status" phase="live" kicker="Tournament Manager / Live Operations" title="status and recovery" description="Inspect actual recoverable score and publication operations, reconciliation state, and audit evidence without exposing archive as a recovery shortcut." />;
}
