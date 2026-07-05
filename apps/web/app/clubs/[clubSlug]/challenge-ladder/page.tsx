import Link from "next/link";
import { getClubChallengeLadder } from "@/lib/challengeLadderApi";
import type { PublicLadderChallenge, PublicLadderPlayer, PublicLadderTier } from "@/lib/challengeLadderApi";

type ChallengeLadderPageProps = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

type SectionKey = "ladder" | "challenges" | "rules";

type FlatPlayer = PublicLadderPlayer & { tier_id: string; tier_label: string };

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const sectionLabels: Record<SectionKey, string> = {
  ladder: "Ladder",
  challenges: "Active challenges",
  rules: "Quick rules"
};

const statusLegend = [
  { label: "Ready to Defend", short: "Ready", publicMeaning: "Can initiate and receive challenges when otherwise eligible." },
  { label: "Locked", short: "Locked", publicMeaning: "Already involved in an active challenge; cannot start another." },
  { label: "Cooldown", short: "Cooldown", publicMeaning: "Can receive a challenge, but cannot initiate during the cooldown window." },
  { label: "Protected", short: "Protected", publicMeaning: "Can initiate, but cannot be challenged during the protection window." },
  { label: "Vacation", short: "Vacation", publicMeaning: "Temporarily unavailable for ladder activity." },
  { label: "Pass Hold", short: "Pass Hold", publicMeaning: "Monthly pass was used; challenge timing is temporarily restricted." },
  { label: "Reinstate Required", short: "Reinstate", publicMeaning: "Requires staff-managed reinstatement before normal ladder activity." }
];

const rulebook = [
  "The Challenge Ladder is an in-season challenge-anytime ranking system.",
  "You move up by challenging and defeating players ranked above you within your tier.",
  "Challenges are official only after staff records them in the Pro Shop Challenge Ledger.",
  "The defender must accept within the configured acceptance window; no response can become a forfeit.",
  "Once accepted, the challenge should be completed inside the configured play window.",
  "A ranked player may only be involved in one active challenge at a time.",
  "Monthly passes, vacation, reinstatement, disputes, and enforcement remain staff-managed.",
  "Swing partners never move on the ladder; only the ranked challenger/defender positions are affected."
];

function firstParam(searchParams: ChallengeLadderPageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function normalizeSection(value: string | null): SectionKey {
  if (value === "challenges" || value === "rules") return value;
  return "ladder";
}

function juprLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(3);
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10);
  return date.toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

function playerHref(clubSlug: string, playerId?: string | number | null): string {
  return playerId == null ? `/clubs/${clubSlug}/players` : `/clubs/${clubSlug}/players/${playerId}`;
}

function pageHref({
  clubSlug,
  section,
  tier,
  player,
  challenge,
  status,
  q,
  anchor
}: {
  clubSlug: string;
  section?: SectionKey | null;
  tier?: string | null;
  player?: string | number | null;
  challenge?: string | number | null;
  status?: string | null;
  q?: string | null;
  anchor?: string | null;
}): string {
  const params = new URLSearchParams();
  if (section && section !== "ladder") params.set("section", section);
  if (tier) params.set("tier", tier);
  if (player) params.set("player", String(player));
  if (challenge) params.set("challenge", String(challenge));
  if (status) params.set("status", status);
  if (q) params.set("q", q);
  const query = params.toString();
  return `/clubs/${clubSlug}/challenge-ladder${query ? `?${query}` : ""}${anchor ? `#${anchor}` : ""}`;
}

function playerAnchor(playerId: string | number): string {
  return `ladder-player-${encodeURIComponent(String(playerId))}`;
}

function challengeAnchor(challengeId?: string | number | null): string {
  return `ladder-challenge-${encodeURIComponent(String(challengeId ?? "unknown"))}`;
}

function flattenPlayers(tiers: PublicLadderTier[]): FlatPlayer[] {
  return tiers.flatMap((tier) => tier.players.map((player) => ({ ...player, tier_id: tier.tier_id, tier_label: tier.label })));
}

function statusText(player: PublicLadderPlayer): string {
  return String(player.status || player.status_short || "").toLowerCase();
}

function canInitiateChallenge(player: PublicLadderPlayer): boolean {
  const text = statusText(player);
  return text.includes("ready") || text.includes("protected");
}

function canReceiveChallenge(player: PublicLadderPlayer): boolean {
  const text = statusText(player);
  return text.includes("ready") || text.includes("cooldown");
}

function eligibleOpponents(player: FlatPlayer, allPlayers: FlatPlayer[], challengeRange: number): FlatPlayer[] {
  if (!canInitiateChallenge(player) || player.rank == null) return [];
  const rank = Number(player.rank);
  return allPlayers
    .filter((candidate) => candidate.tier_id === player.tier_id)
    .filter((candidate) => String(candidate.player_id) !== String(player.player_id))
    .filter((candidate) => candidate.rank != null && Number(candidate.rank) < rank && rank - Number(candidate.rank) <= challengeRange)
    .filter(canReceiveChallenge)
    .sort((a, b) => Number(a.rank ?? 9999) - Number(b.rank ?? 9999));
}

function PlayerRow({ player, clubSlug, selected }: { player: PublicLadderPlayer; clubSlug: string; selected: boolean }) {
  return (
    <tr id={playerAnchor(player.player_id)} style={{ background: selected ? "#eff6ff" : undefined }}>
      <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0", whiteSpace: "nowrap" }}>{player.rank ?? "—"}</td>
      <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>
        <Link href={pageHref({ clubSlug, player: player.player_id, anchor: playerAnchor(player.player_id) })}>{player.player_name}</Link>
        <span style={{ color: "#64748b" }}> · </span>
        <Link href={playerHref(clubSlug, player.player_id)}>profile</Link>
      </td>
      <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0", whiteSpace: "nowrap" }}>{juprLabel(player.rating_jupr)}</td>
      <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>
        <span style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.12rem 0.45rem", fontSize: "0.78rem" }}>{player.status_short || player.status}</span>
        {player.detail ? <div style={{ color: "#64748b", fontSize: "0.78rem", marginTop: "0.2rem" }}>{player.detail}</div> : null}
      </td>
    </tr>
  );
}

function TierCard({ tier, clubSlug, selectedPlayerId, q }: { tier: PublicLadderTier; clubSlug: string; selectedPlayerId: string | null; q: string | null }) {
  const query = String(q ?? "").trim().toLowerCase();
  const players = query
    ? tier.players.filter((player) => player.player_name.toLowerCase().includes(query) || String(player.status || "").toLowerCase().includes(query))
    : tier.players;

  return (
    <article id={`tier-${tier.tier_id}`} style={cardStyle}>
      <h2 style={{ marginTop: 0, marginBottom: "0.2rem" }}>
        <Link href={pageHref({ clubSlug, tier: tier.tier_id, anchor: `tier-${tier.tier_id}`, q })}>{tier.label}</Link>
      </h2>
      {tier.range ? <p style={{ marginTop: 0, color: "#64748b" }}>{tier.range}</p> : null}
      {!players.length ? <p style={{ color: "#64748b" }}>No matching active ladder players in this tier.</p> : null}
      {players.length ? (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "460px" }}>
            <thead>
              <tr>
                {[
                  "Rank",
                  "Player",
                  "JUPR",
                  "Status"
                ].map((heading) => (
                  <th key={heading} style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid #cbd5e1", color: "#475569", fontSize: "0.78rem" }}>{heading}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {players.map((player) => <PlayerRow key={String(player.player_id)} player={player} clubSlug={clubSlug} selected={String(player.player_id) === String(selectedPlayerId)} />)}
            </tbody>
          </table>
        </div>
      ) : null}
    </article>
  );
}

function ChallengeCard({ challenge, clubSlug, selected }: { challenge: PublicLadderChallenge; clubSlug: string; selected: boolean }) {
  return (
    <article id={challengeAnchor(challenge.id)} style={{ ...cardStyle, marginBottom: "0.75rem", borderColor: selected ? "#2563eb" : "#e2e8f0", boxShadow: selected ? "0 0 0 3px rgba(37,99,235,0.12)" : "none" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap" }}>
        <strong><Link href={pageHref({ clubSlug, section: "challenges", challenge: challenge.id, anchor: challengeAnchor(challenge.id) })}>Challenge #{challenge.id ?? "—"}</Link></strong>
        <span style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.12rem 0.45rem", fontSize: "0.78rem" }}>{challenge.status}</span>
      </div>
      <p style={{ margin: "0.5rem 0" }}>
        <Link href={playerHref(clubSlug, challenge.challenger.player_id)}>{challenge.challenger.player_name}</Link>
        <span style={{ color: "#64748b" }}> vs </span>
        <Link href={playerHref(clubSlug, challenge.defender.player_id)}>{challenge.defender.player_name}</Link>
      </p>
      {challenge.winner ? <p style={{ margin: "0.35rem 0", color: "#475569" }}>Winner: {challenge.winner.player_name}</p> : null}
      <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.4rem", margin: 0, color: "#475569", fontSize: "0.86rem" }}>
        <div><dt style={{ fontWeight: 700 }}>Created</dt><dd style={{ margin: 0 }}>{dateLabel(challenge.created_at)}</dd></div>
        <div><dt style={{ fontWeight: 700 }}>Accept by</dt><dd style={{ margin: 0 }}>{dateLabel(challenge.accept_by)}</dd></div>
        <div><dt style={{ fontWeight: 700 }}>Play by</dt><dd style={{ margin: 0 }}>{dateLabel(challenge.play_by)}</dd></div>
      </dl>
    </article>
  );
}

function PlayerChallengePanel({ player, opponents, clubSlug }: { player: FlatPlayer; opponents: FlatPlayer[]; clubSlug: string }) {
  return (
    <article style={{ ...cardStyle, borderColor: "#2563eb", marginBottom: "1rem" }}>
      <h2 style={{ marginTop: 0 }}>{player.player_name} challenge window</h2>
      <p style={{ color: "#475569" }}>
        Rank {player.rank ?? "—"} in {player.tier_label}; current status is <strong>{player.status_short || player.status}</strong>.
      </p>
      {!canInitiateChallenge(player) ? (
        <p style={{ color: "#b45309" }}>This public view shows the player as not currently able to initiate a challenge. Staff still owns final eligibility decisions.</p>
      ) : null}
      {canInitiateChallenge(player) && !opponents.length ? <p style={{ color: "#64748b" }}>No eligible public opponents are visible in range right now.</p> : null}
      {opponents.length ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
          {opponents.map((opponent) => (
            <div key={String(opponent.player_id)} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
              <strong><Link href={pageHref({ clubSlug, player: opponent.player_id, anchor: playerAnchor(opponent.player_id) })}>{opponent.player_name}</Link></strong>
              <p style={{ margin: "0.25rem 0 0", color: "#475569" }}>Rank {opponent.rank} · {opponent.status_short || opponent.status}</p>
            </div>
          ))}
        </div>
      ) : null}
      <p style={{ marginBottom: 0, color: "#64748b", fontSize: "0.9rem" }}>Public eligibility is informational only; official challenges must still be recorded by staff.</p>
    </article>
  );
}

export default async function ChallengeLadderPage({ params, searchParams }: ChallengeLadderPageProps) {
  const { clubSlug } = params;
  const section = normalizeSection(firstParam(searchParams, "section"));
  const selectedTier = firstParam(searchParams, "tier");
  const selectedPlayerId = firstParam(searchParams, "player");
  const selectedChallengeId = firstParam(searchParams, "challenge");
  const selectedStatus = firstParam(searchParams, "status");
  const q = firstParam(searchParams, "q");
  const { data, error } = await getClubChallengeLadder(clubSlug);
  const hasChallenges = Boolean(data?.challenge_sections?.some((challengeSection) => challengeSection.challenges.length > 0));
  const allPlayers = data ? flattenPlayers(data.tiers) : [];
  const selectedPlayer = selectedPlayerId ? allPlayers.find((player) => String(player.player_id) === String(selectedPlayerId)) ?? null : null;
  const selectedOpponents = selectedPlayer && data ? eligibleOpponents(selectedPlayer, allPlayers, data.settings.challenge_range) : [];
  const visibleTiers = data?.tiers.filter((tier) => !selectedTier || tier.tier_id === selectedTier) ?? [];

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Challenge Ladder
      </p>
      <h1 style={{ marginTop: 0 }}>{data?.club.name ?? clubSlug} challenge ladder</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Public ladder standings, player status, active challenge buckets, and quick rules. Challenge creation, notices, score entry, forfeits, passes, and rank changes remain staff-managed.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Challenge Ladder is temporarily unavailable. {error}</p> : null}
      {!error && data && data.summary.active_player_count === 0 ? <p>No active ladder roster is available yet.</p> : null}

      {data ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Active ladder players</strong><br />{data.summary.active_player_count}</article>
            <article style={cardStyle}><strong>Populated tiers</strong><br />{data.summary.populated_tier_count} / {data.summary.tier_count}</article>
            <article style={cardStyle}><strong>Active challenges</strong><br />{data.summary.active_challenge_count}</article>
            <article style={cardStyle}><strong>Challenge range</strong><br />Up to {data.settings.challenge_range} ranks</article>
          </div>

          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
            {(Object.keys(sectionLabels) as SectionKey[]).map((item) => {
              const active = item === section;
              return (
                <Link key={item} href={pageHref({ clubSlug, section: item, tier: selectedTier, player: selectedPlayerId, challenge: selectedChallengeId, status: selectedStatus, q })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                  {sectionLabels[item]}
                </Link>
              );
            })}
          </div>

          {section === "ladder" ? (
            <>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
                <Link href={pageHref({ clubSlug, q })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: !selectedTier ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: !selectedTier ? 800 : 600 }}>All tiers</Link>
                {data.tiers.map((tier) => {
                  const active = tier.tier_id === selectedTier;
                  return (
                    <Link key={tier.tier_id} href={pageHref({ clubSlug, tier: tier.tier_id, q, anchor: `tier-${tier.tier_id}` })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: active ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                      {tier.label}
                    </Link>
                  );
                })}
              </div>
              {q ? <p style={{ color: "#475569" }}>Filtering ladder by: <strong>{q}</strong></p> : null}
              {selectedPlayer ? <PlayerChallengePanel player={selectedPlayer} opponents={selectedOpponents} clubSlug={clubSlug} /> : null}
              <h2>Ladder standings</h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1rem" }}>
                {visibleTiers.map((tier) => <TierCard key={tier.tier_id} tier={tier} clubSlug={clubSlug} selectedPlayerId={selectedPlayerId} q={q} />)}
              </div>
            </>
          ) : null}

          {section === "challenges" ? (
            <>
              <h2>Active challenges</h2>
              {!hasChallenges ? <p style={{ color: "#64748b" }}>No public challenge activity yet.</p> : null}
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
                <Link href={pageHref({ clubSlug, section: "challenges", challenge: selectedChallengeId })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: !selectedStatus ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: !selectedStatus ? 800 : 600 }}>All buckets</Link>
                {data.challenge_sections.map((challengeSection) => {
                  const active = challengeSection.name === selectedStatus;
                  return (
                    <Link key={challengeSection.name} href={pageHref({ clubSlug, section: "challenges", status: challengeSection.name, challenge: selectedChallengeId })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: active ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                      {challengeSection.name} ({challengeSection.challenges.length})
                    </Link>
                  );
                })}
              </div>
              {data.challenge_sections.map((challengeSection) => {
                if (selectedStatus && challengeSection.name !== selectedStatus) return null;
                if (!challengeSection.challenges.length) return null;
                return (
                  <section key={challengeSection.name} style={{ marginBottom: "1rem" }}>
                    <h3>{challengeSection.name}</h3>
                    {challengeSection.challenges.map((challenge) => <ChallengeCard key={`${challengeSection.name}-${String(challenge.id)}`} challenge={challenge} clubSlug={clubSlug} selected={String(challenge.id) === String(selectedChallengeId)} />)}
                  </section>
                );
              })}
            </>
          ) : null}

          {section === "rules" ? (
            <>
              <h2>Quick rules</h2>
              <article style={cardStyle}>
                <ol style={{ margin: 0, paddingLeft: "1.25rem" }}>
                  {[...data.quick_rules, ...rulebook].map((rule, index) => <li key={`${index}-${rule}`} style={{ marginBottom: "0.35rem" }}>{rule}</li>)}
                </ol>
              </article>

              <h2>Status legend</h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem" }}>
                {statusLegend.map((item) => (
                  <article key={item.label} style={cardStyle}>
                    <h3 style={{ marginTop: 0 }}>{item.short}</h3>
                    <p style={{ margin: "0 0 0.35rem", fontWeight: 700 }}>{item.label}</p>
                    <p style={{ color: "#475569", marginBottom: 0 }}>{item.publicMeaning}</p>
                  </article>
                ))}
              </div>
            </>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
