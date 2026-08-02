import GeneratorStandings from "@/app/admin/play-generators/GeneratorStandings";

type Props = { params: { sessionKey: string } };

function apiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export default function RoundRobinStandingsPage({ params }: Props) {
  return <GeneratorStandings apiBase={apiBase()} clubId="tres_palapas" sessionKey={params.sessionKey} />;
}
