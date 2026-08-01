import Link from "next/link";
import { redirect } from "next/navigation";
import TournamentPhaseNav from "@/components/TournamentPhaseNav";

type Props = { searchParams?: Record<string, string | string[] | undefined> };
function first(value: string | string[] | undefined): string { return Array.isArray(value) ? String(value[0] || "") : String(value || ""); }
function href(path: string, tournament: string, name: string): string { const params = new URLSearchParams({ tournament }); if (name) params.set("name", name); return `${path}?${params.toString()}`; }
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default function TournamentCloseoutPage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Tournament Manager / Publish</p>
      <h1 style={{ marginTop: 0 }}>{tournamentName || "Tournament"} closeout</h1>
      <TournamentPhaseNav phase="publish" />
      <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
        <h2 style={{ marginTop: 0 }}>Closeout checklist</h2>
        <ul>
          <li>Every division is either published or explicitly blocked.</li>
          <li>Official matches and required replay evidence are complete.</li>
          <li>Podiums and awards are verified.</li>
          <li>Participant communication is reviewed; staging email remains dry-run.</li>
          <li>Extras and fulfillment exceptions are resolved.</li>
          <li>Public results are reviewed before the tournament is closed or archived.</li>
        </ul>
      </article>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.85rem", marginTop: "1rem" }}>
        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Publication evidence</h2><p style={{ color: "#475569" }}>Review official match publication and replay-safe operation evidence.</p><Link href={href("/admin/tournaments/ops/publish", tournamentId, tournamentName)}>Open division publishing</Link></article>
        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Fulfillment report</h2><p style={{ color: "#475569" }}>Review unfulfilled extras, pickup corrections, and offline payment exceptions.</p><Link href={href("/admin/tournaments/commerce", tournamentId, tournamentName)}>Open payments and fulfillment</Link></article>
        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Lifecycle and archive</h2><p style={{ color: "#475569" }}>Close or archive only after results, communications, and fulfillment are complete.</p><Link href={href("/admin/tournaments/status", tournamentId, tournamentName)}>Open status and recovery</Link></article>
      </div>
    </section>
  );
}
