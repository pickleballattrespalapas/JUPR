import Link from "next/link";
import { getClubBadgeCodex } from "@/lib/api";
import type { PublicBadge } from "@/lib/api";

type BadgeCodexPageProps = {
  params: { clubSlug: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const badgeIcons: Record<string, string> = {
  participant: "🎟️",
  dedicated_participant_50: "🧭",
  lifetime_participant_200: "🏅",
  mountain_climber: "🧗",
  breakthrough: "🚀",
  above_expectations: "⭐",
  clutch_performer: "⚡",
  dominant_run: "🔥",
  high_output: "💥",
  battle_tested: "🛡️",
  consistency: "🎯",
  giant_slayer: "🗡️",
  upset_champion: "👑",
  league_champion: "🥇",
  league_runner_up: "🥈",
  league_third_place: "🥉",
  tournament_champion: "🥇",
  tournament_runner_up: "🥈",
  tournament_third_place: "🥉",
  podium: "🏅"
};

function iconForBadge(badge: PublicBadge): string {
  return badgeIcons[String(badge.badge_id)] ?? "🏆";
}

function prestigeLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "0";
  return String(Number(value));
}

function dateLabel(value?: string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10);
  return date.toISOString().slice(0, 10);
}

function BadgeCard({ badge, clubSlug }: { badge: PublicBadge; clubSlug: string }) {
  const stateLabel = badge.state && badge.state !== "live" ? badge.state : null;
  return (
    <article style={{ ...cardStyle, display: "flex", flexDirection: "column", gap: "0.65rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "0.75rem" }}>
        <div>
          <div style={{ fontSize: "2rem", lineHeight: 1 }}>{iconForBadge(badge)}</div>
          <h3 style={{ margin: "0.35rem 0 0", lineHeight: 1.2 }}>{badge.name}</h3>
        </div>
        {stateLabel ? <span style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.15rem 0.45rem", fontSize: "0.75rem", color: "#475569" }}>{stateLabel}</span> : null}
      </div>
      <p style={{ margin: 0, color: "#475569" }}>{badge.description || badge.requirements || "Requirements TBD"}</p>
      {badge.description && badge.requirements ? <p style={{ margin: 0, color: "#64748b", fontSize: "0.9rem" }}><strong>Unlock:</strong> {badge.requirements}</p> : null}
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", fontSize: "0.82rem", color: "#475569" }}>
        <span style={{ border: "1px solid #e2e8f0", borderRadius: "999px", padding: "0.2rem 0.5rem" }}>Prestige {prestigeLabel(badge.prestige)}</span>
        <span style={{ border: "1px solid #e2e8f0", borderRadius: "999px", padding: "0.2rem 0.5rem" }}>{badge.earners_count ?? 0} earners</span>
        {badge.rarity ? <span style={{ border: "1px solid #e2e8f0", borderRadius: "999px", padding: "0.2rem 0.5rem" }}>{badge.rarity}</span> : null}
      </div>
      <div style={{ borderTop: "1px solid #e2e8f0", paddingTop: "0.6rem" }}>
        <strong style={{ fontSize: "0.9rem" }}>Recent earners</strong>
        {!badge.recent_earners?.length ? <p style={{ color: "#64748b", margin: "0.35rem 0 0" }}>No one has earned this badge yet.</p> : null}
        <ul style={{ margin: "0.35rem 0 0", paddingLeft: "1.25rem" }}>
          {(badge.recent_earners ?? []).map((earner) => (
            <li key={`${badge.badge_id}-${earner.player_id}`}>
              <Link href={`/clubs/${clubSlug}/players/${earner.player_id}`}>{earner.player_name}</Link>
              {dateLabel(earner.earned_at) ? <span style={{ color: "#64748b" }}> · {dateLabel(earner.earned_at)}</span> : null}
            </li>
          ))}
        </ul>
      </div>
    </article>
  );
}

export default async function BadgeCodexPage({ params }: BadgeCodexPageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClubBadgeCodex(clubSlug);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Badge Codex
      </p>
      <h1 style={{ marginTop: 0 }}>{data?.club.name ?? clubSlug} badge codex</h1>
      <p style={{ color: "#334155", maxWidth: "760px" }}>
        A public ledger of JUPR badges, reels, trophies, and unlock paths. This page is read-only and shows public-safe badge metadata and recent earners.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Badge Codex is temporarily unavailable. {error}</p> : null}
      {!error && !data?.sections?.length ? <p>No public badges are available yet.</p> : null}

      {data ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
          <article style={cardStyle}><strong>Badges</strong><br />{data.summary.badge_count}</article>
          <article style={cardStyle}><strong>Earned badge types</strong><br />{data.summary.earned_badge_count}</article>
          <article style={cardStyle}><strong>Unclaimed badge types</strong><br />{data.summary.unclaimed_badge_count}</article>
          <article style={cardStyle}><strong>Total badge earners</strong><br />{data.summary.total_unique_earners_by_badge}</article>
        </div>
      ) : null}

      {data?.sections?.map((section) => (
        <section key={section.name} style={{ marginTop: "1.5rem" }}>
          <h2>{section.name}</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
            {section.badges.map((badge) => <BadgeCard key={badge.badge_id} badge={badge} clubSlug={clubSlug} />)}
          </div>
        </section>
      ))}
    </section>
  );
}
