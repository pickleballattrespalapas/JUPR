import { requireAdminWorkspace } from "@/lib/adminWorkspaceServer";
import GeneratorStandings from "@/app/admin/play-generators/GeneratorStandings";

type Props = { params: { sessionKey: string } };

function apiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export default function RoundRobinStandingsPage({ params }: Props) {
  const { clubId } = requireAdminWorkspace();
  return <GeneratorStandings apiBase={apiBase()} clubId={clubId} sessionKey={params.sessionKey} />;
}
