import { getAdminTournamentApiBaseUrl, getAdminTournamentStatus } from "@/lib/adminTournamentApi";
import SelectedTournamentPanelScope from "../SelectedTournamentPanelScope";
import TournamentOpsPanel, { type OpsWorkflow } from "./TournamentOpsPanel";

type Props = {
  workflow: OpsWorkflow;
  kicker: string;
  title: string;
  description: string;
  tournamentId: string;
  tournamentName?: string | null;
};

export default async function TournamentOpsWorkflowPage({ workflow, kicker, title, description, tournamentId, tournamentName }: Props) {
  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>{kicker}</p>
      <h1 style={{ marginTop: 0 }}>{tournamentName ? `${tournamentName} — ${title}` : title}</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>{description}</p>
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Tournament Operations are unavailable. {error}</p> : null}
      {data ? (
        <SelectedTournamentPanelScope tournamentId={tournamentId} tournamentName={tournamentName || null}>
          <TournamentOpsPanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} workflow={workflow} />
        </SelectedTournamentPanelScope>
      ) : null}
    </section>
  );
}
