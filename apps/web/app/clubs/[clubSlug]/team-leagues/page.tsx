import Link from "next/link";
import { getPublicTeamLeagues } from "@/lib/teamLeagueApi";

type Props = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};
type LeagueView = "active" | "past";
const card = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

function categoryLabel(category: string): string {
  return ({ mens: "Men's", womens: "Women's", mixed: "Mixed", open: "Open" } as Record<string, string>)[category] || "Open";
}

function leagueStatusLabel(status: string): string {
  return ({
    registration_open: "Registration open",
    registration_closed: "Registration closed",
    scheduled: "Schedule published",
    active: "Season in progress",
    playoffs: "Playoffs",
    complete: "Complete"
  } as Record<string, string>)[status] || "League available";
}

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
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem" }}>
        {(data?.leagues || []).map((league) => (
          <article key={league.league_name} style={card}>
            <h2 style={{ marginTop: 0 }}>{league.league_name}</h2>
            <p>{league.venue || "Venue to be announced"}</p>
            <p>{categoryLabel(league.team_category)} · {league.team_size} players per team{league.max_alternates ? ` · up to ${league.max_alternates} alternate${league.max_alternates === 1 ? "" : "s"}` : ""}</p>
            <p>
              {league.registration_open
                ? "Registration open"
                : league.registration_configured_open && !league.online_team_registration_supported
                  ? "Contact staff to register"
                : leagueStatusLabel(league.status)}
              {" · "}
              {league.allow_substitutes ? "Substitutes allowed" : "No substitutes"}
            </p>
            <Link href={`/clubs/${params.clubSlug}/team-leagues/${encodeURIComponent(league.league_name)}`}>
              Open league
            </Link>
          </article>
        ))}
      </div>
      {!error && !data?.leagues?.length ? (
        <p>{view === "past" ? "No past team leagues yet." : "No active team leagues right now."}</p>
      ) : null}
      <p style={{ marginTop: "1rem" }}>
        <Link href={`/clubs/${params.clubSlug}`}>Club home</Link>
      </p>
    </section>
  );
}
