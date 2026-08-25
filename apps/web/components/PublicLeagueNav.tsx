import Link from "next/link";
import styles from "./PublicLeagueNav.module.css";

export type PublicLeagueModule = "home" | "overall" | "weekly" | "player";

type Props = {
  clubSlug: string;
  leagueName: string;
  active: PublicLeagueModule;
  leagueView?: "active" | "past";
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
    player: "players"
  };
  return `${publicLeagueHomeHref(clubSlug, leagueName)}/${routes[section]}`;
}

export default function PublicLeagueNav({
  clubSlug,
  leagueName,
  active,
  leagueView = "active"
}: Props) {
  const cleanName = safeLeagueName(leagueName);
  const items: Array<[PublicLeagueModule, string, string]> = [
    ["home", "League Home", publicLeagueHomeHref(clubSlug, cleanName)],
    ["overall", "Standings", publicLeagueResultsHref(clubSlug, cleanName, "overall")],
    ["weekly", "Weekly History", publicLeagueResultsHref(clubSlug, cleanName, "weekly")],
    ["player", "Player Summaries", publicLeagueResultsHref(clubSlug, cleanName, "player")]
  ];

  return (
    <div className={styles.shell}>
      <div className={styles.contextRow}>
        <p className={styles.context}>{cleanName} league pages</p>
        <Link
          href={`/clubs/${clubSlug}/leagues${leagueView === "past" ? "?view=past" : ""}`}
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
