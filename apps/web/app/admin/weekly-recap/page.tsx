import Link from "next/link";
import { getAdminWeeklyRecapApiBaseUrl, getAdminWeeklyRecapStatus } from "@/lib/adminWeeklyRecapApi";
import WeeklyRecapAdminPanel from "./WeeklyRecapAdminPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

type PageProps = { searchParams?: { week_start?: string; print?: string } };

export default async function AdminWeeklyRecapPage({ searchParams }: PageProps) {
  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminWeeklyRecapStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Weekly Recap Admin
      </p>
      <h1 style={{ marginTop: 0 }}>Weekly Recap Admin</h1>
      <p style={{ color: "#334155", maxWidth: "880px" }}>
        Generate, edit, preview, publish, and unpublish weekly recaps through guarded FastAPI/Python services while keeping Streamlit available as the fallback reference.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Weekly Recap Admin status is unavailable. {error}</p> : null}

      {status ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
          <article style={cardStyle}><strong>Status</strong><br />{status.status.replace(/_/g, " ")}</article>
          <article style={cardStyle}><strong>Recap data</strong><br />{status.enabled ? "Available after admin sign-in" : "Guarded off"}</article>
          <article style={cardStyle}><strong>Gate</strong><br /><code>manage_matches</code></article>
        </div>
      ) : null}

      {status ? <WeeklyRecapAdminPanel apiBase={getAdminWeeklyRecapApiBaseUrl()} clubId={clubId} status={status} initialWeekStart={searchParams?.week_start || ""} printMode={["1", "true", "yes"].includes(String(searchParams?.print || "").toLowerCase())} /> : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href="/clubs/tres-palapas/weekly-recap">Public weekly recap</Link> · <Link href="/admin/player-updates">Player Updates</Link> · <Link href="/admin">Operations cockpit</Link>
      </p>
    </section>
  );
}
