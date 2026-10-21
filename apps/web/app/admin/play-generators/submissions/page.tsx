import { requireAdminWorkspace } from "@/lib/adminWorkspaceServer";
import GeneratorSubmissions from "./GeneratorSubmissions";

export default async function GeneratorSubmissionsPage() {
  const { clubId } = await requireAdminWorkspace();
  return <section>
    <h1>Generator submissions</h1>
    <p>Review the players, scores, and Rated/Unrated setting before approving results. Approved games count toward player stats and weekly recaps.</p>
    <GeneratorSubmissions clubId={clubId} apiBase={process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || ""} />
  </section>;
}
