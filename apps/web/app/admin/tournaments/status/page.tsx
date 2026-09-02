import TournamentLiveRoute from "../live-operations/TournamentLiveRoute";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function AdminTournamentStatusPage({ searchParams }: Props) {
  return <TournamentLiveRoute searchParams={searchParams} view="status" phase="live" kicker="Tournament Manager / Live Operations" title="status and recovery" description="Inspect actual recoverable score and publication operations, reconciliation state, and audit evidence without exposing archive as a recovery shortcut." />;
}
