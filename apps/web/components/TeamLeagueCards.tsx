import Link from "@/components/PublicClubLink";
import type { TeamLeagueSettings } from "@/lib/teamLeagueApi";
import { publicTeamLeagueHref } from "@/lib/teamLeagueLinks";

function categoryLabel(category: string): string {
  return ({ mens: "Men's", womens: "Women's", mixed: "Mixed", open: "Open" } as Record<string, string>)[category] || "Open";
}

function statusLabel(league: TeamLeagueSettings): string {
  if (league.registration_open) return "Registration open";
  if (league.registration_configured_open && !league.online_team_registration_supported) return "Contact staff to register";
  return ({ registration_closed: "Registration closed", scheduled: "Schedule published", active: "Season in progress", playoffs: "Playoffs", complete: "Complete" } as Record<string, string>)[league.status] || "Registration closed";
}

export default function TeamLeagueCards({ clubSlug, leagues, past = false }: {
  clubSlug: string;
  leagues: TeamLeagueSettings[];
  past?: boolean;
}) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 240px), 1fr))", gap: "1rem" }}>
      {leagues.map((league) => (
        <article key={league.league_name} style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white", minWidth: 0, overflowWrap: "anywhere" }}>
          <h2 style={{ marginTop: 0 }}>{league.league_name}</h2>
          <p>{league.venue || "Venue to be announced"}</p>
          <p>{categoryLabel(league.team_category)} · {league.team_size} players per team{league.max_alternates ? ` · up to ${league.max_alternates} alternate${league.max_alternates === 1 ? "" : "s"}` : ""}</p>
          <p>{statusLabel(league)} · {league.allow_substitutes ? "Substitutes allowed" : "No substitutes"}</p>
          <Link href={publicTeamLeagueHref(clubSlug, league.league_name)} style={{ display: "inline-block", border: "1px solid #0f172a", borderRadius: "999px", padding: "0.6rem 0.9rem", background: "#0f172a", color: "white", textDecoration: "none", fontWeight: 800 }}>
            {past ? "View league results" : league.registration_open ? "Register" : "View league"}
          </Link>
        </article>
      ))}
    </div>
  );
}
