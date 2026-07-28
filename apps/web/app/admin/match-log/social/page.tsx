import MatchLogWorkspace from "../MatchLogWorkspace";
import type { MatchLogSearchParams } from "../MatchLogWorkspace";

export default function MatchLogSocialPage({
  searchParams,
}: {
  searchParams?: MatchLogSearchParams;
}) {
  return <MatchLogWorkspace mode="social" searchParams={searchParams} />;
}
