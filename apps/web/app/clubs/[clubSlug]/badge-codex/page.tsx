import Link from "next/link";
import { getClubBadgeCodex } from "@/lib/badgeApi";
import type { BadgeCodexSection, PublicBadge } from "@/lib/badgeApi";

type BadgeCodexPageProps = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

type TrophyRoomEntry = {
  player_id: string | number;
  player_name: string;
  badge_count: number;
  latest_earned_at?: string | null;
  badges: string[];
};

const DEFAULT_BADGE_LIMIT = 12;
const LOAD_MORE_STEP = 12;

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

function firstParam(searchParams: BadgeCodexPageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function iconForBadge(badge: PublicBadge): string {
  return badgeIcons[String(badge.icon_key || badge.badge_id)] ?? "🏆";
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

function parseLimit(value: string | null): number | "all" {
  if (value === "all") return "all";
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_BADGE_LIMIT;
  return Math.max(1, Math.round(parsed));
}

function pageHref({
  clubSlug,
  section,
  badge,
  limit,
  anchor
}: {
  clubSlug: string;
  section?: string | null;
  badge?: string | null;
  limit?: number | "all" | null;
  anchor?: string | null;
}): string {
  const params = new URLSearchParams();
  if (section) params.set("section", section);
  if (badge) params.set("badge", badge);
  if (limit != null) params.set("limit", String(limit));
  const query = params.toString();
  return `/clubs/${clubSlug}/badge-codex${query ? `?${query}` : ""}${anchor ? `#${anchor}` : ""}`;
}

function badgeAnchor(badge: PublicBadge): string {
  return `badge-${encodeURIComponent(String(badge.badge_id))}`;
}

function buildTrophyRoom(sections: BadgeCodexSection[]): TrophyRoomEntry[] {
  const byPlayer = new Map<string, TrophyRoomEntry>();
  for (const section of sections) {
    for (const badge of section.badges) {
      for (const earner of badge.recent_earners ?? []) {
        const key = String(earner.player_id);
        const current = byPlayer.get(key) ?? {
          player_id: earner.player_id,
          player_name: earner.player_name,
          badge_count: 0,
          latest_earned_at: null,
          badges: []
        };
        current.badge_count += 1;
        if (!current.badges.includes(badge.name)) current.badges.push(badge.name);
        if (earner.earned_at && (!current.latest_earned_at || String(earner.earned_at) > String(current.latest_earned_at))) {
          current.latest_earned_at = earner.earned_at;
        }
        byPlayer.set(key, current);
      }
    }
  }
  return Array.from(byPlayer.values()).sort((a, b) => {
    if (b.badge_count !== a.badge_count) return b.badge_count - a.badge_count;
    return String(b.latest_earned_at ?? "").localeCompare(String(a.latest_earned_at ?? ""));
  });
}

function BadgeCard({ badge, clubSlug, selected }: { badge: PublicBadge; clubSlug: string; selected: boolean }) {
  const stateLabel = badge.state && badge.state !== "live" ? badge.state : null;
  return (
    <article id={badgeAnchor(badge)} style={{ ...cardStyle, display: "flex", flexDirection: "column", gap: "0.65rem", borderColor: selected ? "#2563eb" : "#e2e8f0", boxShadow: selected ? "0 0 0 3px rgba(37,99,235,0.12)" : "none" }}>
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
            <li key={`${badge.badge_id}-${earner.player_id}-${earner.earned_at ?? ""}`}>
              <Link href={`/clubs/${clubSlug}/players/${earner.player_id}`}>{earner.player_name}</Link>
              {dateLabel(earner.earned_at) ? <span style={{ color: "#64748b" }}> · {dateLabel(earner.earned_at)}</span> : null}
            </li>
          ))}
        </ul>
      </div>
      <div style={{ marginTop: "auto", borderTop: "1px solid #f1f5f9", paddingTop: "0.55rem" }}>
        <Link href={pageHref({ clubSlug, badge: badge.badge_id, anchor: badgeAnchor(badge), limit: "all" })}>Link directly to this badge</Link>
      </div>
    </article>
  );
}

function TrophyRoom({ entries, clubSlug }: { entries: TrophyRoomEntry[]; clubSlug: string }) {
  return (
    <section style={{ marginTop: "1.5rem" }}>
      <h2>Recent trophy room</h2>
      <p style={{ color: "#475569", maxWidth: "760px" }}>
        A public-safe rollup of recent badge earners from the codex feed. Open a player to see their full public profile.
      </p>
      {!entries.length ? <p style={{ color: "#64748b" }}>No recent trophy-room activity yet.</p> : null}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem" }}>
        {entries.slice(0, 12).map((entry) => (
          <article key={String(entry.player_id)} style={cardStyle}>
            <h3 style={{ marginTop: 0 }}><Link href={`/clubs/${clubSlug}/players/${entry.player_id}`}>{entry.player_name}</Link></h3>
            <p style={{ margin: "0 0 0.35rem", color: "#475569" }}>{entry.badge_count} recent badge{entry.badge_count === 1 ? "" : "s"}</p>
            {dateLabel(entry.latest_earned_at) ? <p style={{ margin: "0 0 0.35rem", color: "#64748b" }}>Latest: {dateLabel(entry.latest_earned_at)}</p> : null}
            <p style={{ marginBottom: 0 }}>{entry.badges.slice(0, 4).join(" · ")}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

export default async function BadgeCodexPage({ params, searchParams }: BadgeCodexPageProps) {
  const { clubSlug } = params;
  const selectedSection = firstParam(searchParams, "section");
  const selectedBadge = firstParam(searchParams, "badge");
  const limit = parseLimit(firstParam(searchParams, "limit"));
  const { data, error } = await getClubBadgeCodex(clubSlug);
  const sections = data?.sections ?? [];
  const visibleSections = selectedSection ? sections.filter((section) => section.name === selectedSection) : sections;
  const trophyRoom = buildTrophyRoom(sections);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Badge Codex
      </p>
      <h1 style={{ marginTop: 0 }}>{data?.club.name ?? clubSlug} badge codex</h1>
      <p style={{ color: "#334155", maxWidth: "760px" }}>
        A public ledger of club badges, reels, trophies, and unlock paths. This page is read-only and shows public-safe badge metadata and recent earners.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Badge Codex is temporarily unavailable. {error}</p> : null}
      {!error && !data?.sections?.length ? <p>No public badges are available yet.</p> : null}

      {data ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Badges</strong><br />{data.summary.badge_count}</article>
            <article style={cardStyle}><strong>Earned badge types</strong><br />{data.summary.earned_badge_count}</article>
            <article style={cardStyle}><strong>Unclaimed badge types</strong><br />{data.summary.unclaimed_badge_count}</article>
            <article style={cardStyle}><strong>Total badge earners</strong><br />{data.summary.total_unique_earners_by_badge}</article>
          </div>

          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
            <Link href={pageHref({ clubSlug, limit })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: !selectedSection ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: !selectedSection ? 800 : 600 }}>
              All sections
            </Link>
            {sections.map((section) => {
              const active = section.name === selectedSection;
              return (
                <Link key={section.name} href={pageHref({ clubSlug, section: section.name, limit })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                  {section.name}
                </Link>
              );
            })}
            <Link href={pageHref({ clubSlug, section: selectedSection, limit: "all" })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: limit === "all" ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: 800 }}>
              Show all badges
            </Link>
          </div>

          <TrophyRoom entries={trophyRoom} clubSlug={clubSlug} />
        </>
      ) : null}

      {visibleSections.map((section) => {
        const shownBadges = limit === "all" ? section.badges : section.badges.slice(0, limit);
        const canLoadMore = limit !== "all" && section.badges.length > shownBadges.length;
        const nextLimit = limit === "all" ? "all" : limit + LOAD_MORE_STEP;
        return (
          <section key={section.name} style={{ marginTop: "1.5rem" }}>
            <h2>{section.name}</h2>
            <p style={{ color: "#64748b" }}>{section.badges.length} badge type{section.badges.length === 1 ? "" : "s"} in this section.</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
              {shownBadges.map((badge) => <BadgeCard key={badge.badge_id} badge={badge} clubSlug={clubSlug} selected={String(badge.badge_id) === String(selectedBadge)} />)}
            </div>
            {canLoadMore ? (
              <p style={{ marginTop: "1rem" }}>
                <Link href={pageHref({ clubSlug, section: selectedSection, badge: selectedBadge, limit: nextLimit })}>Load more badges</Link>
                <span style={{ color: "#64748b" }}> · showing {shownBadges.length} of {section.badges.length}</span>
              </p>
            ) : null}
          </section>
        );
      })}
    </section>
  );
}
