import Link from "next/link";
import { redirect } from "next/navigation";
import TournamentPhaseNav from "@/components/TournamentPhaseNav";

type Props = { searchParams?: Record<string, string | string[] | undefined> };
function first(value: string | string[] | undefined): string { return Array.isArray(value) ? String(value[0] || "") : String(value || ""); }
function href(path: string, tournament: string, name: string): string { const params = new URLSearchParams({ tournament }); if (name) params.set("name", name); return `${path}?${params.toString()}`; }
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default function TournamentCheckInPage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Tournament Manager / Live Operations</p>
      <h1 style={{ marginTop: 0 }}>{tournamentName || "Tournament"} preflight and check-in</h1>
      <TournamentPhaseNav phase="live" />
      <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
        <h2 style={{ marginTop: 0 }}>Preflight checklist</h2>
        <p style={{ color: "#475569" }}>Resolve these items before the first court is called.</p>
        <ul>
          <li>Registration is closed or intentionally left open.</li>
          <li>Partners and teams are complete.</li>
          <li>Offline payments and waivers have been reviewed.</li>
          <li>Draws, courts, times, and staff assignments are ready.</li>
          <li>Late arrivals, withdrawals, substitutes, and waitlist promotions have a clear process.</li>
        </ul>
      </article>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.85rem", marginTop: "1rem" }}>
        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Registrant check-in</h2><p style={{ color: "#475569" }}>Use the registration list to confirm player identity, events, teams, payments, and extras.</p><Link href={href("/admin/tournaments/registration/registrants", tournamentId, tournamentName)}>Open registrants</Link></article>
        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Partners and substitutes</h2><p style={{ color: "#475569" }}>Review incomplete pairs, team rosters, and approved substitutes.</p><Link href={href("/admin/tournaments/registration/partners", tournamentId, tournamentName)}>Open partners and teams</Link></article>
        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Draw readiness</h2><p style={{ color: "#475569" }}>Generate or review draws only after check-in changes are resolved.</p><Link href={href("/admin/tournaments/ops/draws", tournamentId, tournamentName)}>Open draws and schedule</Link></article>
      </div>
    </section>
  );
}
