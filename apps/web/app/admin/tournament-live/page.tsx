import Link from "next/link";
import TournamentAdminNav from "@/components/TournamentAdminNav";
import { getAdminTournamentApiBaseUrl, getAdminTournamentLiveStatus } from "@/lib/adminTournamentApi";
import TournamentLivePanel from "./TournamentLivePanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function AdminTournamentLivePage() {
  const clubId = "tres_palapas";
  const { data: status, error: statusError } = await getAdminTournamentLiveStatus(clubId);

  return (
    <>
      <TournamentAdminNav />
      <section>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          Tournament Live
        </p>
        <h1 style={{ marginTop: 0 }}>Tournament Live runner</h1>
        <p style={{ color: "#334155", maxWidth: "880px" }}>
          A draw-scoped control room for running a prepared tournament during play. FastAPI/Python owns scoring and progression; the browser submits reviewed commands and displays durable recovery evidence. This is explicitly separate from the one-off JUPR Live product.
        </p>

        {statusError ? <p style={{ color: "#b91c1c" }}>Tournament Admin status is unavailable. {statusError}</p> : null}
        {!status ? (
          <article style={cardStyle}>Tournament Live status is temporarily unavailable. Use the Streamlit fallback and do not attempt a write.</article>
        ) : (
          <TournamentLivePanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={status} />
        )}

        <p style={{ marginTop: "1rem" }}>
          <Link href="/admin/player-updates">Player Updates</Link> · <Link href="/admin">Operations cockpit</Link>
        </p>
      </section>
    </>
  );
}
