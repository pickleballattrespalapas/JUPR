import { redirect } from "next/navigation";
import TournamentPhaseNav from "@/components/TournamentPhaseNav";
import TournamentCommercePanel from "./TournamentCommercePanel";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function AdminTournamentCommercePage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager / Registration
      </p>
      <h1 style={{ marginTop: 0 }}>{context.tournamentName || "Tournament"} payments, extras, and fulfillment</h1>
      <TournamentPhaseNav phase="registration" />
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        Configure extras and bundles, then track offline payment, pickup, fulfillment, and recovery without rewriting prior orders.
      </p>
      <TournamentCommercePanel
        clubId="tres_palapas"
        tournamentId={context.tournamentId}
        tournamentName={context.tournamentName || context.tournamentId}
      />
    </section>
  );
}
