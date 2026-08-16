import TournamentLiveRoute from "../TournamentLiveRoute";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentCorrectionsRecoveryPage({ searchParams }: Props) {
  return <TournamentLiveRoute searchParams={searchParams} view="corrections" phase="live" kicker="Tournament Manager / Live Operations" title="corrections and recovery" description="Review scored games, confirm before-and-after corrections, and reconcile failed or uncertain draw operations." />;
}
