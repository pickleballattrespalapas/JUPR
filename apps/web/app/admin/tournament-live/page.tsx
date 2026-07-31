import { Suspense } from "react";
import { redirect } from "next/navigation";
import TournamentAdminNav from "@/components/TournamentAdminNav";
import { getAdminTournamentApiBaseUrl, getAdminTournamentLiveStatus } from "@/lib/adminTournamentApi";
import SelectedTournamentPanelScope from "../tournaments/SelectedTournamentPanelScope";
import TournamentLivePanel from "./TournamentLivePanel";

type Props = { searchParams?: Record<string, string | string[] | undefined> };
function first(value: string | string[] | undefined): string { return Array.isArray(value) ? String(value[0] || "") : String(value || ""); }

export default async function AdminTournamentLivePage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");

  const clubId = "tres_palapas";
  const { data: status, error: statusError } = await getAdminTournamentLiveStatus(clubId);

  return (
    <>
      <Suspense fallback={<div aria-hidden="true" style={{ minHeight: "42px", marginBottom: "1rem" }} />}>
        <TournamentAdminNav />
      </Suspense>
      <section>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          Tournament Manager
        </p>
        <h1 style={{ marginTop: 0 }}>{tournamentName || "Tournament"} live runner</h1>
        <p style={{ color: "#334155", maxWidth: "850px" }}>
          Run the selected tournament draw, scoring, progression, publication evidence, and recovery from one draw-scoped control room.
        </p>
        {statusError ? <p role="alert" style={{ color: "#b91c1c" }}>Tournament Live is unavailable. {statusError}</p> : null}
        {status ? (
          <SelectedTournamentPanelScope tournamentId={tournamentId} tournamentName={tournamentName || null}>
            <TournamentLivePanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={status} />
          </SelectedTournamentPanelScope>
        ) : null}
      </section>
    </>
  );
}
