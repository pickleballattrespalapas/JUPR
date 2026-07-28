import MatchLogWorkspace from "../MatchLogWorkspace";
import type { MatchLogSearchParams } from "../MatchLogWorkspace";

export default function MatchLogReplayPage({
  searchParams,
}: {
  searchParams?: MatchLogSearchParams;
}) {
  return <MatchLogWorkspace mode="replay" searchParams={searchParams} />;
}
