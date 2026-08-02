import PublicGeneratorRoundRunner from "@/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner";

type Props = {
  params: { clubSlug: string; sessionKey: string; roundNumber: string };
};

export default function PublicRoundRobinRoundPage({ params }: Props) {
  return (
    <PublicGeneratorRoundRunner
      apiBase="/api"
      clubId={params.clubSlug}
      generatorKind="round_robin"
      sessionKey={params.sessionKey}
      roundNumber={Math.max(1, Number(params.roundNumber) || 1)}
    />
  );
}
