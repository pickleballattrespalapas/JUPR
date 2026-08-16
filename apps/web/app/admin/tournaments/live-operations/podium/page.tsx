import TournamentLiveRoute from "../TournamentLiveRoute";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentPodiumDraftPage({ searchParams }: Props) {
  return <TournamentLiveRoute searchParams={searchParams} view="podium" phase="live" kicker="Tournament Manager / Live Operations" title="podium draft" description="Generate placements, record explicit review evidence for the current draw, and award only a reviewed podium." />;
}
