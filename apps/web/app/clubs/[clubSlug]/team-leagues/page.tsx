import Link from "@/components/PublicClubLink";
import { getPublicTeamLeagues } from "@/lib/teamLeagueApi";
import TeamLeagueCards from "@/components/TeamLeagueCards";

type Props = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};
type LeagueView = "active" | "past";

function selectedView(searchParams: Props["searchParams"]): LeagueView {
  const raw = searchParams?.view;
  const value = Array.isArray(raw) ? raw[0] : raw;
  return value === "past" ? "past" : "active";
}

export default async function TeamLeaguesPage({ params, searchParams }: Props) {
  const view = selectedView(searchParams);
  const { data, error } = await getPublicTeamLeagues(params.clubSlug, view);
  const base = `/clubs/${encodeURIComponent(params.clubSlug)}/team-leagues`;
  return (
    <section>
      <p style={{ color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Season team leagues
      </p>
      <h1>Team leagues</h1>
      <p style={{ color: "#475569", maxWidth: "760px" }}>
        {view === "past"
          ? "See final standings and results from past team leagues."
          : "Register a team, view the schedule, and follow the season."}
      </p>
      <nav aria-label="Team league collections" data-testid="public-team-league-view-toggle" style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
        {(["active", "past"] as LeagueView[]).map((option) => {
          const current = option === view;
          return (
            <Link
              key={option}
              href={option === "past" ? `${base}?view=past` : base}
              aria-current={current ? "page" : undefined}
              style={{ border: `1px solid ${current ? "#2563eb" : "#cbd5e1"}`, borderRadius: "999px", padding: "0.5rem 0.85rem", background: current ? "#dbeafe" : "white", color: current ? "#1d4ed8" : "#0f172a", textDecoration: "none", fontWeight: current ? 800 : 650 }}
            >
              {option === "active" ? "Active leagues" : "Past leagues"}
            </Link>
          );
        })}
      </nav>
      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      <TeamLeagueCards clubSlug={params.clubSlug} leagues={data?.leagues || []} past={view === "past"} />
      {!error && !data?.leagues?.length ? (
        <p>{view === "past" ? "No past team leagues yet." : "No active team leagues right now."}</p>
      ) : null}
      <p style={{ marginTop: "1rem" }}>
        <Link href={`/clubs/${params.clubSlug}/leagues`}>All leagues</Link> · <Link href={`/clubs/${params.clubSlug}`}>Club home</Link>
      </p>
    </section>
  );
}
