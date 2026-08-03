import GeneratorWorkspace from "@/app/admin/play-generators/GeneratorWorkspace";

type StatusResponse = {
  enabled: boolean;
  writes_enabled?: boolean;
  status: string;
  warnings?: string[];
};

function apiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function loadStatus(clubId: string): Promise<StatusResponse | null> {
  const base = apiBase();
  if (!base) return null;
  try {
    const response = await fetch(
      `${base.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/play-generators/status`,
      { cache: "no-store" }
    );
    return response.ok ? ((await response.json()) as StatusResponse) : null;
  } catch {
    return null;
  }
}

export default async function RoundRobinGeneratorPage() {
  const clubId = "tres_palapas";
  const status = await loadStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Competition tools
      </p>
      <h1 style={{ marginTop: 0 }}>Round-Robin Generator</h1>
      <p style={{ color: "#334155", maxWidth: "900px" }}>
        Build singles, doubles, or mixed doubles-and-singles schedules; preview every planned matchup and bye, print the schedule, then run one round at a time.
      </p>
      <GeneratorWorkspace
        generatorKind="round_robin"
        apiBase={apiBase()}
        clubId={clubId}
        status={status}
      />
    </section>
  );
}
