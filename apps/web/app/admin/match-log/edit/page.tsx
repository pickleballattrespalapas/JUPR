import MatchLogWorkspace from "../MatchLogWorkspace";
import type { MatchLogSearchParams } from "../MatchLogWorkspace";

export default function MatchLogEditPage({
  searchParams,
}: {
  searchParams?: MatchLogSearchParams;
}) {
  return <MatchLogWorkspace mode="edit" searchParams={searchParams} />;
}
