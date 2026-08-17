import { redirect } from "next/navigation";
import { Suspense } from "react";
import TournamentPhaseNav from "@/components/TournamentPhaseNav";
import { getAdminTournamentApiBaseUrl } from "@/lib/adminTournamentApi";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";
import TournamentCheckInPanel from "./TournamentCheckInPanel";

type Props = {
  searchParams?: Record<string, string | string[] | undefined>;
};

export default function TournamentCheckInPage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");
  const rawDay = searchParams?.day_id;
  const initialDayId = String(Array.isArray(rawDay) ? rawDay[0] || "" : rawDay || "").trim();

  return (
    <section style={{ minWidth: 0 }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager / Live Operations
      </p>
      <h1 style={{ marginTop: 0 }}>
        {context.tournamentName || "Tournament"} preflight and check-in
      </h1>
      <p style={{ color: "#475569", maxWidth: "72ch" }}>
        Choose a tournament day, confirm who is physically attending, verify the
        attending player&apos;s waiver, and resolve that day&apos;s blockers from durable data.
      </p>
      <TournamentPhaseNav phase="live" />
      <Suspense fallback={<p>Loading tournament-day check-in…</p>}>
        <TournamentCheckInPanel
          apiBase={getAdminTournamentApiBaseUrl()}
          clubId="tres_palapas"
          initialDayId={initialDayId}
          tournamentId={context.tournamentId}
        />
      </Suspense>
    </section>
  );
}
