import MatchLogWorkspace from "./MatchLogWorkspace";
import type { MatchLogSearchParams } from "./MatchLogWorkspace";

export default async function AdminMatchLogPage(
  props: {
    searchParams?: Promise<MatchLogSearchParams>;
  }
) {
  const searchParams = await props.searchParams;
  return <MatchLogWorkspace mode="review" searchParams={searchParams} />;
}
