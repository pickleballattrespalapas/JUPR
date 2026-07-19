import Link from "next/link";
import { redirect } from "next/navigation";
import { getClub } from "@/lib/api";
import { isNextAdminScoreEntryEnabled } from "@/lib/scoreEntry";

type LegacyScoreEntryPageProps = {
  params: { clubId: string };
};

export default async function LegacyScoreEntryPage({ params }: LegacyScoreEntryPageProps) {
  if (!isNextAdminScoreEntryEnabled()) {
    return (
      <section>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          Admin score entry
        </p>
        <h1 style={{ marginTop: 0 }}>Score entry is disabled</h1>
        <p style={{ color: "#475569" }}>
          This guarded workflow is available only in explicitly enabled staff environments.
        </p>
        <p><Link href="/admin">Return to the operations cockpit</Link></p>
      </section>
    );
  }

  const { data, error } = await getClub(params.clubId);
  const clubSlug = String(data?.slug || "").trim();

  if (clubSlug) {
    redirect(`/clubs/${encodeURIComponent(clubSlug)}/admin/score-entry`);
  }

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin score entry
      </p>
      <h1 style={{ marginTop: 0 }}>Unable to resolve this club</h1>
      <p style={{ color: "#475569" }}>
        The older club-ID score-entry address now resolves to the auth-aware club route. No score was submitted.
      </p>
      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      <p><Link href="/admin">Return to the operations cockpit</Link></p>
    </section>
  );
}
