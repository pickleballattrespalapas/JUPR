import MatchLogWorkspace from "../MatchLogWorkspace";
import type { MatchLogSearchParams } from "../MatchLogWorkspace";

export default async function MatchLogBulkPage(
  props: {
    searchParams?: Promise<MatchLogSearchParams>;
  }
) {
  const searchParams = await props.searchParams;
  return <MatchLogWorkspace mode="bulk" searchParams={searchParams} />;
}
