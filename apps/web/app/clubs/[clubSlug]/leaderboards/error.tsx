"use client";

import Link from "next/link";
import { useParams } from "next/navigation";

export default function LeaderboardsError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  const params = useParams<{ clubSlug?: string }>();
  const clubSlug = String(params?.clubSlug || "tres-palapas");
  return (
    <section role="alert" data-testid="leaderboard-route-error-state">
      <p style={{ color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Leaderboards</p>
      <h1>We couldn&apos;t load the leaderboards.</h1>
      <p>Try loading the leaderboards again or return to the club page.</p>
      <p style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
        <button type="button" onClick={reset} style={{ border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#0f172a", color: "white", fontWeight: 800 }}>Try again</button>
        <Link href={`/clubs/${encodeURIComponent(clubSlug)}`}>Return to club home</Link>
      </p>
    </section>
  );
}
