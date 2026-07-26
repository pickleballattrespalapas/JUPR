import Link from "next/link";
import ChallengeLadderResultDetails from "@/components/ChallengeLadderResultDetails";
import { getClubChallengeLadder } from "@/lib/challengeLadderApi";
import type { PublicLadderChallenge, PublicLadderChallengeSide, PublicLadderPlayer, PublicLadderTier } from "@/lib/challengeLadderApi";

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
  challenges: "Challenges & results",
  rules: "Quick rules"
};

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

function ChallengeParticipant({
  label,
  player,
  clubSlug
}: {
  label: string;
  player: PublicLadderChallengeSide;
  clubSlug: string;
}) {
  const facts = [
    player.rank_at_create == null ? null : `At challenge: #${player.rank_at_create}`,
    player.current_rank == null ? null : `Current: #${player.current_rank}`,
    player.current_rating_jupr == null ? null : `Current JUPR: ${juprLabel(player.current_rating_jupr)}`
  ].filter((value): value is string => Boolean(value));

  return (
    <div style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.7rem" }}>
      <div style={{ color: "#64748b", fontSize: "0.75rem", fontWeight: 800, letterSpacing: "0.04em", textTransform: "uppercase" }}>{label}</div>
      <strong><Link href={playerHref(clubSlug, player.player_id)}>{player.player_name}</Link></strong>
      {facts.length ? <div style={{ color: "#475569", fontSize: "0.82rem", marginTop: "0.2rem" }}>{facts.join(" · ")}</div> : null}
    </div>
  );
}

function ChallengeCard({ challenge, clubSlug, selected }: { challenge: PublicLadderChallenge; clubSlug: string; selected: boolean }) {
  const isCompleted = challenge.bucket === "Recently Completed" || challenge.status === "COMPLETED" || challenge.status === "FORFEITED";
  const anchor = challengeAnchor(challenge.id);
  const headingId = `${anchor}-heading`;
  return (
    <article id={anchor} aria-labelledby={headingId} style={{ ...cardStyle, marginBottom: "0.75rem", borderColor: selected ? "#2563eb" : "#e2e8f0", boxShadow: selected ? "0 0 0 3px rgba(37,99,235,0.12)" : "none" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap" }}>
        <h4 id={headingId} style={{ margin: 0, fontSize: "1rem" }}><Link href={pageHref({ clubSlug, section: "challenges", challenge: challenge.id, anchor })}>Challenge #{challenge.id ?? "—"}</Link></h4>
        <span style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.12rem 0.45rem", fontSize: "0.78rem" }}>{challenge.status}</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.65rem", margin: "0.75rem 0" }}>
        <ChallengeParticipant label="Challenger" player={challenge.challenger} clubSlug={clubSlug} />
        <ChallengeParticipant label="Defender" player={challenge.defender} clubSlug={clubSlug} />
      </div>
      {challenge.winner ? (
        <p style={{ margin: "0.35rem 0", color: "#475569" }}>
          Winner: <Link href={playerHref(clubSlug, challenge.winner.player_id)}>{challenge.winner.player_name}</Link>
        </p>
      ) : null}
      <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.4rem", margin: 0, color: "#475569", fontSize: "0.86rem" }}>
        <div><dt style={{ fontWeight: 700 }}>Created</dt><dd style={{ margin: 0 }}>{dateLabel(challenge.created_at)}</dd></div>
        {challenge.accept_by ? <div><dt style={{ fontWeight: 700 }}>Accept by</dt><dd style={{ margin: 0 }}>{dateLabel(challenge.accept_by)}</dd></div> : null}
        {challenge.play_by ? <div><dt style={{ fontWeight: 700 }}>Play by</dt><dd style={{ margin: 0 }}>{dateLabel(challenge.play_by)}</dd></div> : null}
        {isCompleted ? <div><dt style={{ fontWeight: 700 }}>Completed</dt><dd style={{ margin: 0 }}>{dateLabel(challenge.completed_at)}</dd></div> : null}
      </dl>
      {isCompleted ? (
        <>
          <p style={{ margin: "0.65rem 0 0", color: "#64748b", fontSize: "0.82rem" }}>
            Current positions and ratings are live ladder values; they do not by themselves attribute a rank or rating change to this challenge.
          </p>
          {challenge.result_details ? <ChallengeLadderResultDetails challenge={challenge} details={challenge.result_details} clubSlug={clubSlug} /> : (
            <p data-result-details="unavailable" style={{ margin: "0.4rem 0 0", color: "#64748b", fontSize: "0.82rem" }}>
              {challenge.status === "FORFEITED"
                ? "Resolved by forfeit; played-match scores, swing partners, rating changes, and match links are not expected."
                : "Detailed scores, swing partners, rating changes, and match links are unavailable for this result. Older or imported records may not have linked match details."}
            </p>
          )}
        </>
      ) : null}
    </article>
  );
}

function PlayerChallengePanel({ player, clubSlug }: { player: FlatPlayer; clubSlug: string }) {
  const eligibility = player.eligibility;
  if (!eligibility) {
    return (
      <article style={{ ...cardStyle, borderColor: "#f59e0b", marginBottom: "1rem" }}>
        <h2 style={{ marginTop: 0 }}>{player.player_name} challenge window</h2>
        <p style={{ color: "#92400e", marginBottom: 0 }}>Eligibility hints are temporarily unavailable. Staff remains the authority for official challenges.</p>
      </article>
    );
  }
  const opponents = eligibility.eligible_opponents;
  return (
    <article style={{ ...cardStyle, borderColor: "#2563eb", marginBottom: "1rem" }} data-python-eligibility={eligibility.authority}>
      <h2 style={{ marginTop: 0 }}>{player.player_name} challenge window</h2>
      <p style={{ color: "#475569" }}>
        Rank {player.rank ?? "—"} in {player.tier_label}; current status is <strong>{player.status_short || player.status}</strong>.
      </p>
      {!eligibility.can_initiate ? (
        <p style={{ color: "#b45309" }}>{eligibility.hint} Staff still owns final official eligibility decisions.</p>
      ) : null}
      {eligibility.can_initiate && !opponents.length ? <p style={{ color: "#64748b" }}>{eligibility.hint}</p> : null}
      {opponents.length ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
          {opponents.map((opponent) => (
            <div key={String(opponent.player_id)} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
              <strong><Link href={pageHref({ clubSlug, player: opponent.player_id, anchor: playerAnchor(opponent.player_id) })}>{opponent.player_name}</Link></strong>
              <p style={{ margin: "0.25rem 0 0", color: "#475569" }}>Rank {opponent.rank} · {opponent.status_short || opponent.status} · {opponent.rank_gap} rank{opponent.rank_gap === 1 ? "" : "s"} up</p>
            </div>
          ))}
        </div>
      ) : null}
      <p style={{ marginBottom: 0, color: "#64748b", fontSize: "0.9rem" }}>Opponent hints are computed by the Python ladder policy from public status, tier, rank, and the configured range. Official challenges must still be recorded by staff.</p>
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
  const visibleTiers = data?.tiers.filter((tier) => !selectedTier || tier.tier_id === selectedTier) ?? [];

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Challenge Ladder
      </p>
      <h1 style={{ marginTop: 0 }}>{data?.club.name ?? clubSlug} challenge ladder</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Public ladder standings, player status, challenge activity, recent results, and quick rules. Challenge creation, notices, score entry, forfeits, passes, and rank changes remain staff-managed.
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
            <article style={cardStyle}><strong>Eligible public pairings</strong><br />{data.summary.eligible_pair_count ?? 0}</article>
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
              {selectedPlayer ? <PlayerChallengePanel player={selectedPlayer} clubSlug={clubSlug} /> : null}
              <h2>Ladder standings</h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1rem" }}>
                {visibleTiers.map((tier) => <TierCard key={tier.tier_id} tier={tier} clubSlug={clubSlug} selectedPlayerId={selectedPlayerId} q={q} />)}
              </div>
            </>
          ) : null}

          {section === "challenges" ? (
            <>
              <h2>Challenges & results</h2>
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
              <p style={{ color: "#475569" }}>The complete public rulebook below is generated by the same Python service that computes ladder status and eligible-opponent hints.</p>
              <div style={{ display: "grid", gap: "1rem" }} data-rulebook-authority={data.eligibility_authority || "unavailable"}>
                {(data.rulebook ?? []).map((section) => (
                  <article key={section.title} style={cardStyle}>
                    <h3 style={{ marginTop: 0 }}>{section.title}</h3>
                    <ol style={{ margin: 0, paddingLeft: "1.25rem" }}>
                      {section.rules.map((rule) => (
                        <li key={`${section.title}-${rule.title}`} style={{ marginBottom: "0.65rem" }}>
                          <strong>{rule.title}:</strong> {rule.body}
                        </li>
                      ))}
                    </ol>
                  </article>
                ))}
              </div>

              <h2>Status legend</h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem" }}>
                {(data.status_legend ?? []).map((item) => (
                  <article key={item.status} style={cardStyle} data-ladder-status={item.status}>
                    <h3 style={{ marginTop: 0 }}>{item.short}</h3>
                    <p style={{ margin: "0 0 0.35rem", fontWeight: 700 }}>{item.status}</p>
                    <p style={{ color: "#475569", marginBottom: "0.35rem" }}>{item.meaning}</p>
                    <p style={{ color: "#64748b", marginBottom: 0, fontSize: "0.85rem" }}>Initiate: {item.can_initiate ? "yes" : "no"} · Receive: {item.can_receive ? "yes" : "no"}</p>
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
