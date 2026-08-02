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

export default async function LadderGeneratorPage() {
  const clubId = "tres_palapas";
  const status = await loadStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Competition tools
      </p>
      <h1 style={{ marginTop: 0 }}>Ladder Generator</h1>
      <p style={{ color: "#334155", maxWidth: "900px" }}>
        Build a singles or doubles ladder, preview Round 1, then generate each later round adaptively from saved results.
      </p>
      <GeneratorWorkspace
        generatorKind="ladder"
        apiBase={apiBase()}
        clubId={clubId}
        status={status}
      />
    </section>
  );
}
