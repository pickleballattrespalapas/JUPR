import { redirect } from "next/navigation";
import SelectedTournamentPanelScope from "../SelectedTournamentPanelScope";
import TournamentCommercePanel from "./TournamentCommercePanel";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

export default function AdminTournamentCommercePage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager
      </p>
      <h1 style={{ marginTop: 0 }}>{tournamentName || "Tournament"} extras and fulfillment</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        Configure extras and bundles, then track offline payment, pickup, fulfillment, and recovery without rewriting prior orders.
      </p>
      <SelectedTournamentPanelScope tournamentId={tournamentId} tournamentName={tournamentName || null}>
        <TournamentCommercePanel clubId="tres_palapas" />
      </SelectedTournamentPanelScope>
    </section>
  );
}
