import { requireAdminWorkspace } from "@/lib/adminWorkspaceServer";
import GeneratorStandings from "@/app/admin/play-generators/GeneratorStandings";

type Props = { params: Promise<{ sessionKey: string }> };

function apiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export default async function RoundRobinStandingsPage(props: Props) {
  const params = await props.params;
  const { clubId } = await requireAdminWorkspace();
  return <GeneratorStandings apiBase={apiBase()} clubId={clubId} sessionKey={params.sessionKey} />;
}
