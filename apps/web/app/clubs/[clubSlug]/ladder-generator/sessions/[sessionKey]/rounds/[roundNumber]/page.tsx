import PublicGeneratorRoundRunner from "@/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner";

type Props = {
  params: Promise<{ clubSlug: string; sessionKey: string; roundNumber: string }>;
};

function apiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export default async function PublicLadderRoundPage(props: Props) {
  const params = await props.params;
  return (
    <PublicGeneratorRoundRunner
      apiBase={apiBase()}
      clubId={params.clubSlug}
      generatorKind="ladder"
      sessionKey={params.sessionKey}
      roundNumber={Math.max(1, Number(params.roundNumber) || 1)}
    />
  );
}
