import Link from "next/link";
import { getClubChallengeLadder } from "@/lib/challengeLadderApi";
import type { PublicLadderChallenge, PublicLadderPlayer, PublicLadderTier } from "@/lib/challengeLadderApi";

type ChallengeLadderPageProps = {
  params: { clubSlug: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

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

function PlayerRow({ player, clubSlug }: { player: PublicLadderPlayer; clubSlug: string }) {
  return (
    <tr>
      <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0", whiteSpace: "nowrap" }}>{player.rank ?? "—"}</td>
      <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>
        <Link href={playerHref(clubSlug, player.player_id)}>{player.player_name}</Link>
      </td>
      <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0", whiteSpace: "nowrap" }}>{juprLabel(player.rating_jupr)}</td>
      <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>
        <span style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.12rem 0.45rem", fontSize: "0.78rem" }}>{player.status_short || player.status}</span>
        {player.detail ? <div style={{ color: "#64748b", fontSize: "0.78rem", marginTop: "0.2rem" }}>{player.detail}</div> : null}
      </td>
    </tr>
  );
}

function TierCard({ tier, clubSlug }: { tier: PublicLadderTier; clubSlug: string }) {
  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0, marginBottom: "0.2rem" }}>{tier.label}</h2>
      {tier.range ? <p style={{ marginTop: 0, color: "#64748b" }}>{tier.range}</p> : null}
      {!tier.players.length ? <p style={{ color: "#64748b" }}>No active ladder players in this tier.</p> : null}
      {tier.players.length ? (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "420px" }}>
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
              {tier.players.map((player) => <PlayerRow key={String(player.player_id)} player={player} clubSlug={clubSlug} />)}
            </tbody>
          </table>
        </div>
      ) : null}
    </article>
  );
}

function ChallengeCard({ challenge, clubSlug }: { challenge: PublicLadderChallenge; clubSlug: string }) {
  return (
    <article style={{ ...cardStyle, marginBottom: "0.75rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap" }}>
        <strong>Challenge #{challenge.id ?? "—"}</strong>
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

export default async function ChallengeLadderPage({ params }: ChallengeLadderPageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClubChallengeLadder(clubSlug);
  const hasChallenges = Boolean(data?.challenge_sections?.some((section) => section.challenges.length > 0));

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

          <h2>Ladder standings</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1rem" }}>
            {data.tiers.map((tier) => <TierCard key={tier.tier_id} tier={tier} clubSlug={clubSlug} />)}
          </div>

          <h2>Active challenges</h2>
          {!hasChallenges ? <p style={{ color: "#64748b" }}>No public challenge activity yet.</p> : null}
          {data.challenge_sections.map((section) => (
            section.challenges.length ? (
              <section key={section.name} style={{ marginBottom: "1rem" }}>
                <h3>{section.name}</h3>
                {section.challenges.map((challenge) => <ChallengeCard key={`${section.name}-${String(challenge.id)}`} challenge={challenge} clubSlug={clubSlug} />)}
              </section>
            ) : null
          ))}

          <h2>Quick rules</h2>
          <article style={cardStyle}>
            <ol style={{ margin: 0, paddingLeft: "1.25rem" }}>
              {data.quick_rules.map((rule) => <li key={rule} style={{ marginBottom: "0.35rem" }}>{rule}</li>)}
            </ol>
          </article>
        </>
      ) : null}
    </section>
  );
}
