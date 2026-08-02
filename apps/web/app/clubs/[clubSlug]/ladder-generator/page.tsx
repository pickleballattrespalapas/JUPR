import PublicGeneratorWorkspace from "@/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace";

type Props = { params: { clubSlug: string } };
type StatusResponse = { enabled: boolean; writes_enabled?: boolean; status: string; warnings?: string[] };

function serverApiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function loadStatus(clubSlug: string): Promise<StatusResponse | null> {
  const base = serverApiBase();
  if (!base) return null;
  try {
    const response = await fetch(`${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubSlug)}/play-generators/status`, { cache: "no-store" });
    return response.ok ? ((await response.json()) as StatusResponse) : null;
  } catch {
    return null;
  }
}

export default async function PublicGeneratorPage({ params }: Props) {
  const status = await loadStatus(params.clubSlug);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Public play tools
      </p>
      <h1 style={{ marginTop: 0 }}>Ladder Generator</h1>
      <p style={{ color: "#334155", maxWidth: "900px" }}>Preview Round 1, record results, and generate each later court assignment adaptively from the completed or skipped round.</p>
      <p style={{ color: "#475569", maxWidth: "900px" }}>
        Public sessions are unrated. Keep the organizer link private; share the clean page URL as the view-only scoreboard.
      </p>
      <PublicGeneratorWorkspace generatorKind="ladder" apiBase="/api" clubId={params.clubSlug} status={status} />
    </section>
  );
}
