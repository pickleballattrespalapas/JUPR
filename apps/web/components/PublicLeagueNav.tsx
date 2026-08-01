import Link from "next/link";
import styles from "./PublicLeagueNav.module.css";

export type PublicLeagueModule = "home" | "overall" | "weekly" | "player";

type Props = {
  clubSlug: string;
  leagueName: string;
  active: PublicLeagueModule;
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
  const query = new URLSearchParams({
    league: safeLeagueName(leagueName)
  });
  if (section !== "overall") query.set("section", section);
  return `/clubs/${clubSlug}/league-results?${query.toString()}`;
}

export default function PublicLeagueNav({
  clubSlug,
  leagueName,
  active
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
        <Link href={`/clubs/${clubSlug}/leagues`} className={styles.backLink}>
          All leagues
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
