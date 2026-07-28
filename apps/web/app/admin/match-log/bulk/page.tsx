import MatchLogWorkspace from "../MatchLogWorkspace";
import type { MatchLogSearchParams } from "../MatchLogWorkspace";

export default function MatchLogBulkPage({
  searchParams,
}: {
  searchParams?: MatchLogSearchParams;
}) {
  return <MatchLogWorkspace mode="bulk" searchParams={searchParams} />;
}
