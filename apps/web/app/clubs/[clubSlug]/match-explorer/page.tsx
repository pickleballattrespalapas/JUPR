import Link from "next/link";
import { getClubMatchExplorerContext, getClubPlayers } from "@/lib/api";
import MatchExplorerForm from "./MatchExplorerForm";

type MatchExplorerPageProps = {
  params: { clubSlug: string };
};

function apiBase(): string | null {
  return process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || process.env.JUPR_API_BASE_URL || null;
}

export default async function MatchExplorerPage({ params }: MatchExplorerPageProps) {
  const { clubSlug } = params;
  const [playersResult, contextResult] = await Promise.all([
    getClubPlayers(clubSlug),
    getClubMatchExplorerContext(clubSlug)
  ]);
  const club = playersResult.data?.club ?? contextResult.data?.club;
  const players = playersResult.data?.players ?? [];
  const contexts = contextResult.data?.contexts?.length ? contextResult.data.contexts : ["OVERALL"];
  const error = playersResult.error || contextResult.error;

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Match Explorer
      </p>
      <h1 style={{ marginTop: 0 }}>{club?.name ?? clubSlug} matchup preview</h1>
      <p style={{ color: "#334155", maxWidth: "760px" }}>
        Preview doubles win odds and projected JUPR movement before anything is saved. This page calls the public FastAPI preview service and keeps the rating formula in Python.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Match Explorer data is temporarily unavailable. {error}</p> : null}
      {!error && players.length < 4 ? <p>At least four public players are required to preview a doubles matchup.</p> : null}

      {players.length >= 4 ? (
        <MatchExplorerForm apiBase={apiBase()} clubSlug={clubSlug} players={players} contexts={contexts} />
      ) : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href={`/clubs/${clubSlug}/players`}>Open player directory</Link>
        <span style={{ color: "#64748b" }}> · </span>
        <Link href={`/clubs/${clubSlug}/matches`}>View match history</Link>
      </p>
    </section>
  );
}
