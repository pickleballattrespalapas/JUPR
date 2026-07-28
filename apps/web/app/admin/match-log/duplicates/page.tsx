import MatchLogWorkspace from "../MatchLogWorkspace";
import type { MatchLogSearchParams } from "../MatchLogWorkspace";

export default function MatchLogDuplicatesPage({
  searchParams,
}: {
  searchParams?: MatchLogSearchParams;
}) {
  return <MatchLogWorkspace mode="duplicates" searchParams={searchParams} />;
}
