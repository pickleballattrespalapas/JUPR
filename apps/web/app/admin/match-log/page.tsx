import MatchLogWorkspace from "./MatchLogWorkspace";
import type { MatchLogSearchParams } from "./MatchLogWorkspace";

export default function AdminMatchLogPage({
  searchParams,
}: {
  searchParams?: MatchLogSearchParams;
}) {
  return <MatchLogWorkspace mode="review" searchParams={searchParams} />;
}
