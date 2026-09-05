import Link from "next/link";
import { getClubMatchExplorerContext, getClubPlayers } from "@/lib/api";
import MatchExplorerForm from "./MatchExplorerForm";

type MatchExplorerPageProps = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

function firstParam(searchParams: MatchExplorerPageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function apiBase(): string {
  // Use the same-origin Next route handler so browser previews do not depend on cross-origin FastAPI access.
  return "/api";
}

export default async function MatchExplorerPage({ params, searchParams }: MatchExplorerPageProps) {
  const { clubSlug } = params;
  const [playersResult, contextResult] = await Promise.all([
    getClubPlayers(clubSlug),
    getClubMatchExplorerContext(clubSlug)
  ]);
  const club = playersResult.data?.club ?? contextResult.data?.club;
  const players = playersResult.data?.players ?? [];
  const contexts = contextResult.data?.contexts?.length ? contextResult.data.contexts : ["OVERALL"];
  const error = playersResult.error || contextResult.error;
  const initialSelection = {
    context: firstParam(searchParams, "ctx") ?? firstParam(searchParams, "league") ?? firstParam(searchParams, "context"),
    me: firstParam(searchParams, "me"),
    partner: firstParam(searchParams, "partner"),
    opp1: firstParam(searchParams, "opp1"),
    opp2: firstParam(searchParams, "opp2"),
    scoreYou: firstParam(searchParams, "sy") ?? firstParam(searchParams, "score_you"),
    scoreOpp: firstParam(searchParams, "so") ?? firstParam(searchParams, "score_opp")
  };

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Match Explorer
      </p>
      <h1 style={{ marginTop: 0 }}>{club?.name ?? clubSlug} matchup preview</h1>
      <p style={{ color: "#334155", maxWidth: "760px" }}>
        Choose four players to see each team&apos;s win odds and how the result could affect their ratings. Nothing is saved.
      </p>

      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Match Explorer is unavailable right now. Please try again shortly.</p> : null}
      {!error && players.length < 4 ? <p>At least four active players are needed to preview a doubles matchup.</p> : null}

      {players.length >= 4 ? (
        <MatchExplorerForm apiBase={apiBase()} clubSlug={clubSlug} players={players} contexts={contexts} initialSelection={initialSelection} />
      ) : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href={`/clubs/${clubSlug}/players`}>Open player directory</Link>
        <span style={{ color: "#64748b" }}> · </span>
        <Link href={`/clubs/${clubSlug}/matches`}>View match history</Link>
      </p>
    </section>
  );
}
