import PublicGeneratorStandings from "@/app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings";

type Props = { params: { clubSlug: string; sessionKey: string } };

function apiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export default function PublicRoundRobinStandingsPage({ params }: Props) {
  return (
    <PublicGeneratorStandings
      apiBase={apiBase()}
      clubId={params.clubSlug}
      sessionKey={params.sessionKey}
    />
  );
}
