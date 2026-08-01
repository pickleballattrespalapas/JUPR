import Link from "next/link";
import styles from "./PublicTournamentNav.module.css";

export type PublicTournamentModule =
  | "overview"
  | "registration"
  | "roster"
  | "partner-board";

type Props = {
  clubSlug: string;
  tournamentName?: string | null;
  tournamentId?: string | null;
  registrationSlug?: string | null;
  active: PublicTournamentModule;
};

function tournamentQuery(
  tournamentId?: string | null,
  registrationSlug?: string | null
): string {
  const query = new URLSearchParams();
  if (registrationSlug) query.set("tournament", registrationSlug);
  else if (tournamentId) query.set("tournament_id", tournamentId);
  const text = query.toString();
  return text ? `?${text}` : "";
}

function normalizedModule(value: string): PublicTournamentModule {
  if (
    value === "registration" ||
    value === "roster" ||
    value === "partner-board"
  ) {
    return value;
  }
  return "overview";
}

export function publicTournamentHref(
  clubSlug: string,
  module: PublicTournamentModule | string,
  tournamentId?: string | null,
  registrationSlug?: string | null
): string {
  const base = `/clubs/${clubSlug}`;
  const paths: Record<PublicTournamentModule, string> = {
    overview: `${base}/tournaments`,
    registration: `${base}/tournament-registration`,
    roster: `${base}/tournament-roster`,
    "partner-board": `${base}/tournament-partner-board`
  };
  const selectedModule = normalizedModule(module);
  return `${paths[selectedModule]}${tournamentQuery(
    tournamentId,
    registrationSlug
  )}`;
}

export default function PublicTournamentNav({
  clubSlug,
  tournamentName,
  tournamentId,
  registrationSlug,
  active
}: Props) {
  const items: Array<[PublicTournamentModule, string]> = [
    ["overview", "Tournament Home"],
    ["registration", "Register"],
    ["roster", "Roster"],
    ["partner-board", "Partner Board"]
  ];

  return (
    <div className={styles.shell}>
      <p className={styles.context}>
        {tournamentName
          ? `${tournamentName} public tournament pages`
          : "Public tournament pages"}
      </p>
      <nav className={styles.nav} aria-label="Tournament public navigation">
        {items.map(([module, label]) => (
          <Link
            key={module}
            href={publicTournamentHref(
              clubSlug,
              module,
              tournamentId,
              registrationSlug
            )}
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
