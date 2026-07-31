import Link from "next/link";
import { getAdminTournamentApiBaseUrl, getAdminTournamentStatus } from "@/lib/adminTournamentApi";
import TournamentAdminPanel from "./TournamentAdminPanel";

export default async function AdminTournamentsPage() {
  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager
      </p>
      <h1 style={{ marginTop: 0 }}>Tournament Manager</h1>
      <p style={{ color: "#334155", maxWidth: "720px" }}>
        Create a new tournament or open an existing tournament.
      </p>

      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Tournament Manager is unavailable. {error}</p> : null}
      {data ? <TournamentAdminPanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} /> : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href="/admin/tools">Admin Tools</Link> · <Link href="/admin">Operations cockpit</Link>
      </p>
    </section>
  );
}
