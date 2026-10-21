import { requireAdminWorkspace } from "@/lib/adminWorkspaceServer";
import RoundRobinHubPanel from "./RoundRobinHubPanel";

type SearchParams = Record<string, string | string[] | undefined>;
function first(value: string | string[] | undefined): string { return String(Array.isArray(value) ? value[0] || "" : value || ""); }

export default async function RoundRobinHubPage({ searchParams }: { searchParams?: Promise<SearchParams> }) {
  const { clubId } = await requireAdminWorkspace();
  const query = await searchParams || {};
  return <section style={{ minWidth: 0 }}>
    <p style={{ margin: "0 0 .5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: ".08em", fontSize: ".78rem" }}>Competition tools</p>
    <h1 style={{ marginTop: 0 }}>Round Robin Hub</h1>
    <p style={{ color: "#475569", maxWidth: "58rem" }}>Run several round robins from one court board. Each group keeps its own format, players, scores, and standings.</p>
    <RoundRobinHubPanel apiBase={process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null}
      clubId={clubId} initialHubId={first(query.hub)} initialAddSessionKey={first(query.add_session)} initialView={first(query.view)} />
  </section>;
}
