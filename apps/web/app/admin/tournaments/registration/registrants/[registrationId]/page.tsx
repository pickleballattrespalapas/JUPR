import SectionLoadError from "@/components/SectionLoadError";
import { requireAdminWorkspace } from "@/lib/adminWorkspaceServer";
import { redirect } from "next/navigation";
import TournamentPhaseNav from "@/components/TournamentPhaseNav";
import {
  getAdminTournamentApiBaseUrl,
  getAdminTournamentStatus
} from "@/lib/adminTournamentApi";
import { readTournamentRouteContext, tournamentRouteHref } from "@/lib/tournamentRouteContext";
import TournamentRegistrantEditPanel from "./TournamentRegistrantEditPanel";

type Props = {
  params: Promise<{ registrationId: string }>;
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
};
export default async function TournamentRegistrantEditPage(props: Props) {
  const searchParams = await props.searchParams;
  const params = await props.params;
  const context = readTournamentRouteContext(searchParams);
  const registrationId = decodeURIComponent(String(params.registrationId || "")).trim();
  if (!context.tournamentId) redirect("/admin/tournaments");
  if (!registrationId) redirect(tournamentRouteHref("/admin/tournaments/registration/registrants", context));
  const { clubId } = await requireAdminWorkspace();
  const { data, error } = await getAdminTournamentStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Tournament Manager / Registration</p>
      <h1 style={{ marginTop: 0 }}>Edit registration</h1>
      <TournamentPhaseNav phase="registration" />
      {error ? <SectionLoadError title="Tournament Manager temporarily unavailable" message="Your tournament data couldn’t load. Try again, or use the navigation to open another section." /> : null}
      {data ? <TournamentRegistrantEditPanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} tournamentId={context.tournamentId} tournamentName={context.tournamentName || context.tournamentId} drawId={context.drawId} registrationId={registrationId} /> : null}
    </section>
  );
}
