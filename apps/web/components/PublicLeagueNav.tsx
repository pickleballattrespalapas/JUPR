import Link from "@/components/PublicClubLink";
import styles from "./PublicLeagueNav.module.css";

export type PublicLeagueModule = "home" | "overall" | "weekly" | "player" | "history";

type Props = {
  clubSlug: string;
  leagueName: string;
  active: PublicLeagueModule;
  leagueView?: "active" | "past";
  team?: boolean;
};

function safeLeagueName(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

export function publicLeagueHomeHref(clubSlug: string, leagueName: string): string {
  return `/clubs/${clubSlug}/leagues/${encodeURIComponent(
    safeLeagueName(leagueName)
  )}`;
}

export function publicLeagueResultsHref(
  clubSlug: string,
  leagueName: string,
  section: Exclude<PublicLeagueModule, "home">
): string {
  const routes: Record<Exclude<PublicLeagueModule, "home">, string> = {
    overall: "standings",
    weekly: "weekly-history",
    player: "players",
    history: "history"
  };
  return `${publicLeagueHomeHref(clubSlug, leagueName)}/${routes[section]}`;
}

export default function PublicLeagueNav({
  clubSlug,
  leagueName,
  active,
  leagueView = "active",
  team = false
}: Props) {
  const cleanName = safeLeagueName(leagueName);
  const items: Array<[PublicLeagueModule, string, string]> = team ? [
    ["home", "League Home", `/clubs/${clubSlug}/team-leagues/${encodeURIComponent(cleanName)}`],
    ["history", "History", publicLeagueResultsHref(clubSlug, cleanName, "history")]
  ] : [
    ["home", "League Home", publicLeagueHomeHref(clubSlug, cleanName)],
    ["overall", "Standings", publicLeagueResultsHref(clubSlug, cleanName, "overall")],
    ["weekly", "Weekly History", publicLeagueResultsHref(clubSlug, cleanName, "weekly")],
    ["player", "Player Summaries", publicLeagueResultsHref(clubSlug, cleanName, "player")],
    ["history", "History", publicLeagueResultsHref(clubSlug, cleanName, "history")]
  ];

  return (
    <div className={styles.shell}>
      <div className={styles.contextRow}>
        <p className={styles.context}>{cleanName}</p>
        <Link
          href={`/clubs/${clubSlug}/${team ? "team-leagues" : "leagues"}${!team && leagueView === "past" ? "?view=past" : ""}`}
          className={styles.backLink}
        >
          {leagueView === "past" ? "Past leagues" : "All leagues"}
        </Link>
      </div>
      <nav className={styles.nav} aria-label={`${cleanName} league navigation`}>
        {items.map(([module, label, href]) => (
          <Link
            key={module}
            href={href}
            aria-current={active === module ? "page" : undefined}
            className={`${styles.link} ${active === module ? styles.active : ""}`}
          >
            {label}
          </Link>
        ))}
      </nav>
    </div>
  );
}
