import { Suspense } from "react";
import TournamentAdminNav from "@/components/TournamentAdminNav";
import TournamentLiveRoute from "../tournaments/live-operations/TournamentLiveRoute";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function AdminTournamentLivePage({ searchParams }: Props) {
  return (
    <>
      <Suspense fallback={<div aria-hidden="true" style={{ minHeight: "42px", marginBottom: "1rem" }} />}>
        <TournamentAdminNav />
      </Suspense>
      <TournamentLiveRoute searchParams={searchParams} view="scoring" phase="live" kicker="Tournament Manager / Live Operations" title="live scoring" description="Enter and review human-readable matchup scores for the selected draw, one game at a time." />
    </>
  );
}
