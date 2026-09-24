import { requireAdminWorkspace } from "@/lib/adminWorkspaceServer";
import Link from "next/link";
import { getAdminWeeklyRecapApiBaseUrl, getAdminWeeklyRecapStatus } from "@/lib/adminWeeklyRecapApi";
import WeeklyRecapAdminPanel from "./WeeklyRecapAdminPanel";

type PageProps = { searchParams?: { week_start?: string; print?: string } };

export default async function AdminWeeklyRecapPage({ searchParams }: PageProps) {
  const { clubId, clubSlug } = requireAdminWorkspace();
  const { data: status, error } = await getAdminWeeklyRecapStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Weekly Recap Admin
      </p>
      <h1 style={{ marginTop: 0 }}>Weekly Recap Admin</h1>
      <p style={{ color: "#334155", maxWidth: "880px" }}>
        Create and edit weekly recaps, preview them before publishing, or delete drafts you no longer need.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Weekly Recap Admin status is unavailable. {error}</p> : null}

      {status ? <WeeklyRecapAdminPanel apiBase={getAdminWeeklyRecapApiBaseUrl()} clubId={clubId} status={status} initialWeekStart={searchParams?.week_start || ""} printMode={["1", "true", "yes"].includes(String(searchParams?.print || "").toLowerCase())} /> : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href={`/clubs/${encodeURIComponent(clubSlug)}/weekly-recap`}>Public weekly recap</Link> · <Link href="/admin/player-updates">Player Updates</Link> · <Link href="/admin">Admin Home</Link>
      </p>
    </section>
  );
}
