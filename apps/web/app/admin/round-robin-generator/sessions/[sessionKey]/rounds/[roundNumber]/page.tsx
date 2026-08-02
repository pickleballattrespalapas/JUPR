import GeneratorRoundRunner from "@/app/admin/play-generators/GeneratorRoundRunner";

type Props = {
  params: { sessionKey: string; roundNumber: string };
};

function apiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export default function RoundRobinRoundPage({ params }: Props) {
  return (
    <GeneratorRoundRunner
      apiBase={apiBase()}
      clubId="tres_palapas"
      generatorKind="round_robin"
      sessionKey={params.sessionKey}
      roundNumber={Math.max(1, Number(params.roundNumber) || 1)}
    />
  );
}
