import { getAdminTournamentApiBaseUrl, getAdminTournamentStatus } from "@/lib/adminTournamentApi";
import TournamentCreatePanel from "./TournamentCreatePanel";

export default async function AdminTournamentCreatePage() {
  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager
      </p>
      <h1 style={{ marginTop: 0 }}>Create tournament</h1>
      <p style={{ color: "#334155", maxWidth: "720px" }}>
        Create a draft shell. Setup, registration, operations, and publishing remain separate reviewed modules.
      </p>
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Tournament Manager is unavailable. {error}</p> : null}
      {data ? <TournamentCreatePanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} /> : null}
    </section>
  );
}
