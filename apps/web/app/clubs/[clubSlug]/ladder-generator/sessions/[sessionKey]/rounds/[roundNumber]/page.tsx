import PublicGeneratorRoundRunner from "@/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner";

type Props = {
  params: { clubSlug: string; sessionKey: string; roundNumber: string };
};

export default function PublicLadderRoundPage({ params }: Props) {
  return (
    <PublicGeneratorRoundRunner
      apiBase="/api"
      clubId={params.clubSlug}
      generatorKind="ladder"
      sessionKey={params.sessionKey}
      roundNumber={Math.max(1, Number(params.roundNumber) || 1)}
    />
  );
}
