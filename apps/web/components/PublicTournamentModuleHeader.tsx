import Link from "next/link";
import PublicTournamentSponsors from "./PublicTournamentSponsors";
import PublicTournamentNav, {
  publicTournamentHref,
  type PublicTournamentModule
} from "./PublicTournamentNav";

type Props = {
  clubSlug: string;
  tournamentName: string;
  tournamentId?: string | null;
  registrationSlug?: string | null;
  active: PublicTournamentModule;
  kicker: string;
  description: string;
};

export default function PublicTournamentModuleHeader({
  clubSlug,
  tournamentName,
  tournamentId,
  registrationSlug,
  active,
  kicker,
  description
}: Props) {
  return (
    <>
      <p style={{ margin: "0 0 0.65rem" }}>
        <Link
          href={publicTournamentHref(
            clubSlug,
            "overview",
            tournamentId,
            registrationSlug
          )}
          style={{ fontWeight: 800 }}
        >
          ← Tournament Home
        </Link>
      </p>
      <p
        style={{
          margin: "0 0 0.5rem",
          color: "#2563eb",
          fontWeight: 800,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          fontSize: "0.78rem"
        }}
      >
        {kicker}
      </p>
      <PublicTournamentSponsors clubSlug={clubSlug} tournamentId={tournamentId} placement="header" title={tournamentName} />
      <p style={{ color: "#334155", maxWidth: "820px" }}>{description}</p>
      <PublicTournamentNav
        clubSlug={clubSlug}
        tournamentName={tournamentName}
        tournamentId={tournamentId}
        registrationSlug={registrationSlug}
        active={active}
      />
    </>
  );
}
