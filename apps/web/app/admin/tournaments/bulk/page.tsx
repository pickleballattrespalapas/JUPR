import SectionLoadError from "@/components/SectionLoadError";
import { requireAdminWorkspace } from "@/lib/adminWorkspaceServer";
import { redirect } from "next/navigation";
import { getAdminTournamentApiBaseUrl, getAdminTournamentStatus } from "@/lib/adminTournamentApi";
import BulkRegistrationPanel from "./BulkRegistrationPanel";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };

export default async function AdminTournamentBulkRegistrationPage(props: Props) {
  const searchParams = await props.searchParams;
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");

  const { clubId } = await requireAdminWorkspace();
  const { data, error } = await getAdminTournamentStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager
      </p>
      <h1 style={{ marginTop: 0 }}>{context.tournamentName || "Tournament"} bulk actions</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Apply one reviewed registration, offline-payment, or admin-note update to multiple registrations.
      </p>
      {error ? <SectionLoadError title="Tournament Manager temporarily unavailable" message="Your tournament data couldn’t load. Try again, or use the navigation to open another section." /> : null}
      {data ? (
        <BulkRegistrationPanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} initialTournamentId={context.tournamentId} initialTournamentName={context.tournamentName || context.tournamentId} />
      ) : null}
    </section>
  );
}
