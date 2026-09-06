import Link from "next/link";
import { getClubBadgeCodex, getClubBadgeEarners } from "@/lib/badgeApi";
import type {
  BadgeCatalogBucket,
  BadgeCodexSection,
  BadgeEarnersResponse,
  BadgeTrophyRoomEntry,
  PublicBadge
} from "@/lib/badgeApi";

type BadgeCodexPageProps = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

const CATEGORY_ORDER = ["Participation", "Improvement", "Partnerships", "Match Achievements", "Trophies"];
const DEFAULT_BADGE_LIMIT = 12;
const BADGE_LOAD_MORE_STEP = 12;
const DEFAULT_EARNERS_LIMIT = 25;
const EARNERS_LOAD_MORE_STEP = 25;

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

function dateLabel(value?: string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10);
  return date.toISOString().slice(0, 10);
}

function humanizeLabel(value?: string | null): string {
  const text = String(value || "").trim().replace(/[_-]+/g, " ");
  if (!text) return "Other";
  return text.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function badgeAvailabilityLabel(badge: PublicBadge): string {
  const lifecycle = String(badge.lifecycle_state || badge.state || "").trim().toLowerCase();
  const status = String(badge.badge_status || "").trim().toLowerCase();
  const timing = String(badge.badge_award_timing || "").trim().toLowerCase();
  if (["draft", "planned", "upcoming"].includes(lifecycle)) return "Coming soon";
  if (["deprecated", "retired"].includes(lifecycle) || ["deprecated", "retired"].includes(status)) return "Retired";
  if (lifecycle === "frozen" || ["tracked", "disabled", "frozen"].includes(status) || timing === "disabled") return "Not currently awarded";
  return "Available now";
}

function badgeScopeLabel(value?: string | null): string {
  const scope = String(value || "").trim().toLowerCase();
  if (scope === "overall") return "All results";
  if (scope === "lifetime") return "Club career";
  if (scope === "week") return "One week";
  if (scope === "month") return "One month";
  if (scope === "season") return "This season";
  if (scope === "league") return "This league";
  if (scope === "tournament") return "This tournament";
  if (scope === "event") return "One event";
  if (scope === "match") return "One match";
  if (scope === "club") return "Club play";
  if (scope === "opponent") return "Club career";
  return "Other club activity";
}

function badgeTimingLabel(value?: string | null): string {
  const timing = String(value || "live").trim().toLowerCase();
  if (timing === "live") return "As results are posted";
  if (timing === "manual" || timing === "curated") return "By club staff";
  if (timing === "on_league_close") return "When the league ends";
  if (timing === "seasonal") return "When the season ends";
  if (timing === "disabled") return "No longer awarded";
  return "When the requirements are met";
}

function bucketLabel(value: string): string {
  if (value === "Live Now") return "Earn as you play";
  if (value === "Seasonal / League Close") return "League and season awards";
  if (value === "Manual / Curated") return "Special awards";
  if (value === "Tracked / Disabled") return "Not currently awarded";
  return humanizeLabel(value);
}

function bucketDescription(value: string): string {
  if (value === "Live Now") return "Earned automatically when posted results meet the requirements.";
  if (value === "Seasonal / League Close") return "Awarded after final league or season results are posted.";
  if (value === "Manual / Curated") return "Awarded by club staff for tournament finishes, sportsmanship, or community contributions.";
  if (value === "Tracked / Disabled") return "These badges aren’t currently awarded, but previous earners remain listed.";
  return "Browse badges in this group.";
}

function badgeRequirements(badge: PublicBadge): string {
  const requirements = String(badge.requirements || "").trim();
  if (!requirements || /^requirements?\s*(tbd|coming soon)?\.?$/i.test(requirements)) return "Details coming soon.";
  return requirements;
}

function parsePositiveInt(value: string | null, fallback: number, maximum: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return Math.max(1, Math.min(maximum, Math.round(parsed)));
}

function parseOffset(value: string | null): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) return 0;
  return Math.round(parsed);
}

function parseBadgeLimit(value: string | null): number | "all" {
  if (value === "all") return "all";
  return parsePositiveInt(value, DEFAULT_BADGE_LIMIT, 500);
}

type PageHrefOptions = {
  clubSlug: string;
  bucket?: string | null;
  category?: string | null;
  scope?: string | null;
  badge?: string | null;
  badgeLimit?: number | "all" | null;
  earners?: string | null;
  earnersOffset?: number | null;
  earnersLimit?: number | null;
  anchor?: string | null;
};

function pageHref({
  clubSlug,
  bucket,
  category,
  scope,
  badge,
  badgeLimit,
  earners,
  earnersOffset,
  earnersLimit,
  anchor
}: PageHrefOptions): string {
  const params = new URLSearchParams();
  if (bucket) params.set("bucket", bucket);
  if (category) params.set("category", category);
  if (scope) params.set("scope", scope);
  if (badge) params.set("badge", badge);
  if (badgeLimit != null) params.set("limit", String(badgeLimit));
  if (earners) params.set("earners", earners);
  if (earnersOffset != null && earnersOffset > 0) params.set("earners_offset", String(earnersOffset));
  if (earnersLimit != null) params.set("earners_limit", String(earnersLimit));
  const query = params.toString();
  return `/clubs/${clubSlug}/badge-codex${query ? `?${query}` : ""}${anchor ? `#${anchor}` : ""}`;
}

function badgeAnchor(badgeId: string): string {
  return `badge-${badgeId.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
}

function allBadges(buckets: BadgeCatalogBucket[]): PublicBadge[] {
  return buckets.flatMap((bucket) => bucket.sections.flatMap((section) => section.badges));
}

function sectionsForBadges(badges: PublicBadge[]): BadgeCodexSection[] {
  const grouped = new Map<string, PublicBadge[]>();
  for (const badge of badges) {
    const category = badge.category || "Other";
    grouped.set(category, [...(grouped.get(category) ?? []), badge]);
  }
  return Array.from(grouped.entries())
    .sort(([a], [b]) => CATEGORY_ORDER.indexOf(a) - CATEGORY_ORDER.indexOf(b) || a.localeCompare(b))
    .map(([name, items]) => ({
      name,
      badges: items.sort((a, b) => a.name.localeCompare(b.name))
    }));
}

function filterLinkStyle(active: boolean) {
  return {
    border: "1px solid #cbd5e1",
    borderRadius: "999px",
    padding: "0.38rem 0.68rem",
    background: active ? "#dbeafe" : "white",
    color: "#0f172a",
    textDecoration: "none",
    fontWeight: active ? 800 : 600
  };
}

function BadgeCard({
  badge,
  clubSlug,
  selected,
  bucket,
  category,
  scope,
  badgeLimit
}: {
  badge: PublicBadge;
  clubSlug: string;
  selected: boolean;
  bucket: string;
  category: string | null;
  scope: string | null;
  badgeLimit: number | "all";
}) {
  const anchor = badgeAnchor(badge.badge_id);
  return (
    <article
      id={anchor}
      data-badge-id={badge.badge_id}
      data-catalog-bucket={badge.catalog_bucket}
      style={{
        ...cardStyle,
        display: "flex",
        flexDirection: "column",
        gap: "0.65rem",
        borderColor: selected ? "#2563eb" : "#e2e8f0",
        boxShadow: selected ? "0 0 0 3px rgba(37,99,235,0.12)" : "none"
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "0.75rem" }}>
        <div>
          <div style={{ fontSize: "2rem", lineHeight: 1 }}>{iconForBadge(badge)}</div>
          <h3 style={{ margin: "0.35rem 0 0", lineHeight: 1.2 }}>{badge.name}</h3>
        </div>
        <span style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.15rem 0.45rem", fontSize: "0.75rem", color: "#475569" }}>
          {badgeAvailabilityLabel(badge)}
        </span>
      </div>
      <p style={{ margin: 0, color: "#334155", fontSize: "0.9rem" }}><strong>How to earn it:</strong> {badgeRequirements(badge)}</p>
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", fontSize: "0.82rem", color: "#475569" }}>
        <span style={{ border: "1px solid #e2e8f0", borderRadius: "999px", padding: "0.2rem 0.5rem" }}>Prestige {Number(badge.prestige || 0)}</span>
        <span style={{ border: "1px solid #e2e8f0", borderRadius: "999px", padding: "0.2rem 0.5rem" }}>{badge.earners_count ?? 0} {(badge.earners_count ?? 0) === 1 ? "earner" : "earners"}</span>
        {badge.rarity ? <span style={{ border: "1px solid #e2e8f0", borderRadius: "999px", padding: "0.2rem 0.5rem" }}>{humanizeLabel(badge.rarity)}</span> : null}
        {badge.badge_scope ? <span style={{ border: "1px solid #e2e8f0", borderRadius: "999px", padding: "0.2rem 0.5rem" }}>Counts toward: {badgeScopeLabel(badge.badge_scope)}</span> : null}
        <span style={{ border: "1px solid #e2e8f0", borderRadius: "999px", padding: "0.2rem 0.5rem" }}>Awarded: {badgeTimingLabel(badge.badge_award_timing)}</span>
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
        {Number(badge.earners_count || 0) > 0 ? (
          <p style={{ marginBottom: 0 }}>
            <Link href={pageHref({ clubSlug, bucket, category, scope, badge: badge.badge_id, badgeLimit, earners: badge.badge_id, earnersLimit: DEFAULT_EARNERS_LIMIT, anchor: `earners-${anchor}` })}>
              See all earners
            </Link>
          </p>
        ) : null}
      </div>
      <div style={{ marginTop: "auto", borderTop: "1px solid #f1f5f9", paddingTop: "0.55rem" }}>
        <Link href={pageHref({ clubSlug, bucket: badge.catalog_bucket, category: badge.category, scope: badge.badge_scope, badge: badge.badge_id, badgeLimit: "all", anchor })}>
          Share this badge
        </Link>
      </div>
    </article>
  );
}

function TrophyRoom({ entries, clubSlug }: { entries: BadgeTrophyRoomEntry[]; clubSlug: string }) {
  return (
    <section id="trophy-room" style={{ marginTop: "1.5rem" }}>
      <h2>All-time badge totals</h2>
      <p style={{ color: "#475569", maxWidth: "760px" }}>
        Players ranked by all-time prestige. Each recorded award adds to the total, including repeat awards.
      </p>
      {!entries.length ? <p style={{ color: "#64748b" }}>No badge earners are listed yet.</p> : null}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem" }}>
        {entries.map((entry) => (
          <article key={String(entry.player_id)} style={cardStyle} data-trophy-player={entry.player_id}>
            <h3 style={{ marginTop: 0 }}><Link href={`/clubs/${clubSlug}/players/${entry.player_id}`}>{entry.player_name}</Link></h3>
            <p style={{ margin: "0 0 0.35rem", color: "#475569" }}>{entry.prestige_total} prestige · {entry.unique_badge_count} badge type{entry.unique_badge_count === 1 ? "" : "s"}</p>
            <p style={{ margin: "0 0 0.35rem", color: "#64748b" }}>{entry.award_count} badge{entry.award_count === 1 ? "" : "s"} earned{dateLabel(entry.latest_earned_at) ? ` · latest ${dateLabel(entry.latest_earned_at)}` : ""}</p>
            <p style={{ marginBottom: 0 }}>{entry.latest_badges.map((badge) => badge.badge_name).join(" · ")}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function EarnersPanel({
  result,
  error,
  clubSlug,
  bucket,
  category,
  scope,
  badgeLimit
}: {
  result: BadgeEarnersResponse | null;
  error: string | null;
  clubSlug: string;
  bucket: string;
  category: string | null;
  scope: string | null;
  badgeLimit: number | "all";
}) {
  const badgeId = result?.badge_id ?? "selected";
  const anchor = `earners-${badgeAnchor(badgeId)}`;
  return (
    <section id={anchor} style={{ ...cardStyle, margin: "1.5rem 0", borderColor: error ? "#fecaca" : "#93c5fd" }} data-badge-earners-panel>
      <h2 style={{ marginTop: 0 }}>{result ? `${result.badge?.name || result.badge_id} earners` : "Badge earners"}</h2>
      {error ? <p style={{ color: "#b91c1c" }}>We couldn&apos;t load badge earners. Please try again.</p> : null}
      {result ? (
        <>
          <p style={{ color: "#475569" }}>
            Showing {result.earners.length ? result.offset + 1 : 0}–{result.offset + result.earners.length} of {result.total} earners.
          </p>
          {!result.earners.length ? <p>No earners are listed on this page.</p> : null}
          <ol start={result.offset + 1}>
            {result.earners.map((earner) => (
              <li key={`${earner.player_id}-${earner.earned_at ?? ""}`} style={{ marginBottom: "0.35rem" }}>
                <Link href={`/clubs/${clubSlug}/players/${earner.player_id}`}>{earner.player_name}</Link>
                {dateLabel(earner.earned_at) ? <span style={{ color: "#64748b" }}> · earned {dateLabel(earner.earned_at)}</span> : null}
              </li>
            ))}
          </ol>
          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            {result.offset > 0 ? (
              <Link href={pageHref({ clubSlug, bucket, category, scope, badge: result.badge_id, badgeLimit, earners: result.badge_id, earnersOffset: Math.max(0, result.offset - result.limit), earnersLimit: result.limit, anchor })}>
                Previous earners
              </Link>
            ) : null}
            {result.has_more && result.offset === 0 && result.limit < 100 ? (
              <Link data-load-more-earners href={pageHref({ clubSlug, bucket, category, scope, badge: result.badge_id, badgeLimit, earners: result.badge_id, earnersLimit: Math.min(100, result.limit + EARNERS_LOAD_MORE_STEP), anchor })}>
                Load more earners
              </Link>
            ) : null}
            {result.has_more && (result.offset > 0 || result.limit >= 100) ? (
              <Link href={pageHref({ clubSlug, bucket, category, scope, badge: result.badge_id, badgeLimit, earners: result.badge_id, earnersOffset: result.offset + result.earners.length, earnersLimit: DEFAULT_EARNERS_LIMIT, anchor })}>
                Next earners page
              </Link>
            ) : null}
          </div>
        </>
      ) : null}
    </section>
  );
}

export default async function BadgeCodexPage({ params, searchParams }: BadgeCodexPageProps) {
  const { clubSlug } = params;
  const requestedBucket = firstParam(searchParams, "bucket");
  const selectedCategory = firstParam(searchParams, "category");
  const selectedScope = firstParam(searchParams, "scope");
  const selectedBadge = firstParam(searchParams, "badge");
  const selectedEarners = firstParam(searchParams, "earners");
  const badgeLimit = parseBadgeLimit(firstParam(searchParams, "limit"));
  const earnersOffset = parseOffset(firstParam(searchParams, "earners_offset"));
  const earnersLimit = parsePositiveInt(firstParam(searchParams, "earners_limit"), DEFAULT_EARNERS_LIMIT, 100);

  const { data, error } = await getClubBadgeCodex(clubSlug);
  const buckets = data?.catalog_buckets ?? (data?.sections?.length ? [{ name: "Live Now", description: "Badges you can earn now.", badge_count: data.summary.badge_count, sections: data.sections }] : []);
  const badges = allBadges(buckets);
  const directBadge = selectedBadge ? badges.find((badge) => badge.badge_id === selectedBadge) ?? null : null;
  const bucket = requestedBucket || directBadge?.catalog_bucket || "Live Now";
  const bucketBadges = bucket === "all"
    ? badges
    : allBadges(buckets.filter((item) => item.name === bucket));
  const filteredBadges = bucketBadges.filter((badge) => {
    if (selectedCategory && badge.category !== selectedCategory) return false;
    if (selectedScope && badge.badge_scope !== selectedScope) return false;
    return true;
  });
  const visibleSections = sectionsForBadges(filteredBadges);
  const categories = data?.filters?.categories ?? Array.from(new Set(badges.map((badge) => badge.category || "Other"))).sort();
  const scopes = data?.filters?.scopes ?? Array.from(new Set(badges.map((badge) => badge.badge_scope).filter((value): value is string => Boolean(value)))).sort();
  const earnersResult = selectedEarners && data
    ? await getClubBadgeEarners(clubSlug, selectedEarners, { offset: earnersOffset, limit: earnersLimit })
    : { data: null, error: null };

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Badges &amp; trophies</p>
      <h1 style={{ marginTop: 0 }}>{data?.club.name ?? clubSlug} badges &amp; trophies</h1>
      <p style={{ color: "#334155", maxWidth: "800px" }}>
        Celebrate showing up, improving, playing with different partners, and earning major trophies. Each badge below explains exactly how to earn it.
      </p>

      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Badges are unavailable right now. Please try again shortly.</p> : null}
      {!error && data && !badges.length ? <p>No badges are available yet.</p> : null}

      {data ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Badges</strong><br />{data.summary.badge_count}</article>
            <article style={cardStyle}><strong>Earned badge types</strong><br />{data.summary.earned_badge_count}</article>
            <article style={cardStyle}><strong>Badge earners</strong><br />{data.summary.unique_earner_count ?? "—"}</article>
          </div>

          <nav aria-label="When badges are awarded" style={{ marginBottom: "1rem" }}>
            <strong>When badges are awarded</strong>
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "0.5rem" }}>
              <Link href={pageHref({ clubSlug, bucket: "all", category: selectedCategory, scope: selectedScope, badgeLimit })} style={filterLinkStyle(bucket === "all")}>All badges</Link>
              {buckets.map((item) => (
                <Link key={item.name} href={pageHref({ clubSlug, bucket: item.name, category: selectedCategory, scope: selectedScope, badgeLimit })} style={filterLinkStyle(bucket === item.name)} data-badge-bucket={item.name}>
                  {bucketLabel(item.name)} ({item.badge_count})
                </Link>
              ))}
            </div>
            {bucket !== "all" ? <p style={{ color: "#64748b", marginBottom: 0 }}>{bucketDescription(bucket)}</p> : null}
          </nav>

          <nav aria-label="Badge category" style={{ marginBottom: "1rem" }}>
            <strong>Category</strong>
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "0.5rem" }}>
              <Link href={pageHref({ clubSlug, bucket, scope: selectedScope, badgeLimit })} style={filterLinkStyle(!selectedCategory)}>All categories</Link>
              {categories.map((category) => (
                <Link key={category} href={pageHref({ clubSlug, bucket: "all", category, scope: selectedScope, badgeLimit })} style={filterLinkStyle(selectedCategory === category)}>{category}</Link>
              ))}
            </div>
          </nav>

          <nav aria-label="What badge activity counts" style={{ marginBottom: "1rem" }}>
            <strong>Counts toward</strong>
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "0.5rem" }}>
              <Link href={pageHref({ clubSlug, bucket, category: selectedCategory, badgeLimit })} style={filterLinkStyle(!selectedScope)}>All play</Link>
              {scopes.map((scope) => (
                <Link key={scope} href={pageHref({ clubSlug, bucket, category: selectedCategory, scope, badgeLimit })} style={filterLinkStyle(selectedScope === scope)}>{badgeScopeLabel(scope)}</Link>
              ))}
            </div>
          </nav>

          <p style={{ color: "#475569" }}>Match achievements count eligible recorded club matches. Deleted, invalid, PopUp, and tournament matches are excluded. Participation milestones and 100 lifetime wins use club standings totals. Weeks run Monday–Sunday; months and calendar years use UTC.</p>
          <Link href={pageHref({ clubSlug, bucket: "all", category: "Trophies", badgeLimit: "all" })}>Explore league awards and tournament trophies</Link>
        </>
      ) : null}

      {selectedEarners ? (
        <EarnersPanel
          result={earnersResult.data}
          error={earnersResult.error}
          clubSlug={clubSlug}
          bucket={bucket}
          category={selectedCategory}
          scope={selectedScope}
          badgeLimit={badgeLimit}
        />
      ) : null}

      {data && !filteredBadges.length ? (
        <p style={{ ...cardStyle, color: "#64748b" }} data-badge-empty-state>No badges match these filters.</p>
      ) : null}

      {visibleSections.map((section) => {
        const shownBadges = badgeLimit === "all" ? section.badges : section.badges.slice(0, badgeLimit);
        const canLoadMore = badgeLimit !== "all" && section.badges.length > shownBadges.length;
        const nextLimit = badgeLimit === "all" ? "all" : badgeLimit + BADGE_LOAD_MORE_STEP;
        return (
          <section key={section.name} style={{ marginTop: "1.5rem" }} data-badge-category={section.name}>
            <h2>{section.name}</h2>
            <p style={{ color: "#64748b" }}>{section.badges.length} badge{section.badges.length === 1 ? "" : "s"} in this category.</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
              {shownBadges.map((badge) => (
                <BadgeCard
                  key={badge.badge_id}
                  badge={badge}
                  clubSlug={clubSlug}
                  selected={badge.badge_id === selectedBadge}
                  bucket={bucket}
                  category={selectedCategory}
                  scope={selectedScope}
                  badgeLimit={badgeLimit}
                />
              ))}
            </div>
            {canLoadMore ? (
              <p style={{ marginTop: "1rem" }}>
                <Link data-load-more-badges href={pageHref({ clubSlug, bucket, category: selectedCategory, scope: selectedScope, badge: selectedBadge, badgeLimit: nextLimit })}>Load more badges</Link>
                <span style={{ color: "#64748b" }}> · showing {shownBadges.length} of {section.badges.length}</span>
              </p>
            ) : null}
          </section>
        );
      })}
      {data ? <TrophyRoom entries={data.trophy_room ?? []} clubSlug={clubSlug} /> : null}
    </section>
  );
}
