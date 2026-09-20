import { requireAdminWorkspace } from "@/lib/adminWorkspaceServer";
import GeneratorSubmissions from "./GeneratorSubmissions";

export default function GeneratorSubmissionsPage() {
  const { clubId } = requireAdminWorkspace();
  return <section>
    <h1>Generator submissions</h1>
    <p>Review the players and scores, then approve or reject the results. The organizer’s choice of rated or unrated was fixed before play. Both count toward player stats and weekly recaps.</p>
    <GeneratorSubmissions clubId={clubId} apiBase={process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || ""} />
  </section>;
}
