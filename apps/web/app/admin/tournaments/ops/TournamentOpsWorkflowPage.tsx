import Link from "next/link";
import { getAdminTournamentApiBaseUrl, getAdminTournamentStatus } from "@/lib/adminTournamentApi";
import TournamentOpsPanel, { type OpsWorkflow } from "./TournamentOpsPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

type Props = {
  workflow: OpsWorkflow;
  kicker: string;
  title: string;
  description: string;
};

export default async function TournamentOpsWorkflowPage({ workflow, kicker, title, description }: Props) {
  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>{kicker}</p>
      <h1 style={{ marginTop: 0 }}>{title}</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>{description}</p>

      {error ? <p style={{ color: "#b91c1c" }}>Tournament Admin status is temporarily unavailable. {error}</p> : null}

      {data ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Status</strong><br />{data.enabled ? data.status : "Disabled"}</article>
            <article style={cardStyle}><strong>Tournaments</strong><br />{data.tournament_count ?? "—"}</article>
            <article style={cardStyle}><strong>API</strong><br />{data.tournaments_endpoint ? "Configured" : "Guarded off"}</article>
          </div>
          <TournamentOpsPanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} workflow={workflow} />
        </>
      ) : null}

      <p style={{ marginTop: "1rem", display: "flex", gap: "1rem", flexWrap: "wrap" }}>
        <Link href="/admin/tournaments">Tournament Admin</Link>
        <Link href="/admin/tournaments/bulk">Bulk registration actions</Link>
        <Link href="/admin">Operations cockpit</Link>
      </p>
    </section>
  );
}
