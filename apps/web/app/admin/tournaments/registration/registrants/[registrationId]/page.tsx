import { redirect } from "next/navigation";
import TournamentPhaseNav from "@/components/TournamentPhaseNav";
import {
  getAdminTournamentApiBaseUrl,
  getAdminTournamentStatus
} from "@/lib/adminTournamentApi";
import { readTournamentRouteContext, tournamentRouteHref } from "@/lib/tournamentRouteContext";
import TournamentRegistrantEditPanel from "./TournamentRegistrantEditPanel";

type Props = {
  params: { registrationId: string };
  searchParams?: Record<string, string | string[] | undefined>;
};
export default async function TournamentRegistrantEditPage({ params, searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  const registrationId = decodeURIComponent(String(params.registrationId || "")).trim();
  if (!context.tournamentId) redirect("/admin/tournaments");
  if (!registrationId) redirect(tournamentRouteHref("/admin/tournaments/registration/registrants", context));
  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Tournament Manager / Registration</p>
      <h1 style={{ marginTop: 0 }}>Edit registration</h1>
      <TournamentPhaseNav phase="registration" />
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Registration editing is unavailable. {error}</p> : null}
      {data ? <TournamentRegistrantEditPanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} tournamentId={context.tournamentId} tournamentName={context.tournamentName || context.tournamentId} drawId={context.drawId} registrationId={registrationId} /> : null}
    </section>
  );
}
