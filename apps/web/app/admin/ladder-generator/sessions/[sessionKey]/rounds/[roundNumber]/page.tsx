import { requireAdminWorkspace } from "@/lib/adminWorkspaceServer";
import GeneratorRoundRunner from "@/app/admin/play-generators/GeneratorRoundRunner";

type Props = {
  params: Promise<{ sessionKey: string; roundNumber: string }>;
};

function apiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export default async function LadderRoundPage(props: Props) {
  const params = await props.params;
  const { clubId } = await requireAdminWorkspace();
  return (
    <GeneratorRoundRunner
      apiBase={apiBase()}
      clubId={clubId}
      generatorKind="ladder"
      sessionKey={params.sessionKey}
      roundNumber={Math.max(1, Number(params.roundNumber) || 1)}
    />
  );
}
