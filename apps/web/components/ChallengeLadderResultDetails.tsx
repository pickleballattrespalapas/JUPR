import Link from "next/link";
import type {
  PublicLadderChallenge,
  PublicLadderResultDetails,
  PublicLadderResultPlayer
} from "@/lib/challengeLadderApi";

function playerHref(clubSlug: string, playerId?: string | number | null): string {
  const playersPath = `/clubs/${encodeURIComponent(clubSlug)}/players`;
  return playerId == null
    ? playersPath
    : `${playersPath}/${encodeURIComponent(String(playerId))}`;
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toISOString().replace("T", " ").replace(".000Z", " UTC");
}

function juprLabel(value?: number | null): string {
  return value == null || Number.isNaN(Number(value))
    ? "—"
    : Number(value).toFixed(3);
}

function signedJupr(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) {
    return "change unavailable";
  }
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(3)}`;
}

function ResultPlayerLink({
  player,
  clubSlug
}: {
  player: PublicLadderResultPlayer;
  clubSlug: string;
}) {
  return (
    <Link href={playerHref(clubSlug, player.player_id)}>
      {player.player_name}
    </Link>
  );
}

export default function ChallengeLadderResultDetails({
  challenge,
  details,
  clubSlug
}: {
  challenge: PublicLadderChallenge;
  details: PublicLadderResultDetails;
  clubSlug: string;
}) {
  const rank = details.rank_change;
  const challenger = rank?.challenger ?? challenge.challenger;
  const defender = rank?.defender ?? challenge.defender;

  return (
    <section
      data-result-details="available"
      data-result-completeness={details.completeness}
      aria-label="Played challenge result"
      style={{
        marginTop: "0.75rem",
        borderTop: "1px solid #e2e8f0",
        paddingTop: "0.75rem"
      }}
    >
      <h5 style={{ margin: "0 0 0.45rem", fontSize: "1rem" }}>
        Played result
      </h5>
      {details.notice ? (
        <p style={{ margin: "0 0 0.65rem", color: "#92400e" }}>
          {details.notice}
        </p>
      ) : null}
      {rank ? (
        <p style={{ margin: "0 0 0.65rem", color: "#334155" }}>
          Position change:{" "}
          <ResultPlayerLink player={rank.challenger} clubSlug={clubSlug} /> #
          {rank.challenger.before} → #{rank.challenger.after};{" "}
          <ResultPlayerLink player={rank.defender} clubSlug={clubSlug} /> #
          {rank.defender.before} → #{rank.defender.after}
          {rank.swapped ? "." : " (defender held position)."}
        </p>
      ) : null}
      {(details.warnings ?? []).map((warning) => (
        <p
          key={warning}
          role="note"
          style={{
            margin: "0 0 0.5rem",
            color: "#92400e",
            fontSize: "0.82rem"
          }}
        >
          {warning}
        </p>
      ))}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: "0.65rem"
        }}
      >
        {details.matches.map((match) => (
          <article
            key={`${match.slot}-${match.match_id}`}
            style={{
              border: "1px solid #cbd5e1",
              borderRadius: "10px",
              padding: "0.7rem",
              background: "#f8fafc"
            }}
          >
            <h6 style={{ margin: "0 0 0.35rem", fontSize: "0.95rem" }}>
              {match.match_id != null ? (
                <Link href={`/clubs/${clubSlug}/matches/${match.match_id}`}>
                  Match {match.slot.toUpperCase()}:{" "}
                  {match.score_challenger_team}–{match.score_defender_team}
                </Link>
              ) : (
                `Recorded match ${match.slot.toUpperCase()}: ${match.score_challenger_team}–${match.score_defender_team}`
              )}
            </h6>
            <p
              style={{
                margin: "0.2rem 0",
                color: "#475569",
                fontSize: "0.86rem"
              }}
            >
              Challenger team:{" "}
              <ResultPlayerLink player={challenger} clubSlug={clubSlug} /> +{" "}
              {match.challenger_partner ? (
                <ResultPlayerLink
                  player={match.challenger_partner}
                  clubSlug={clubSlug}
                />
              ) : (
                "partner unavailable"
              )}
            </p>
            <p
              style={{
                margin: "0.2rem 0",
                color: "#475569",
                fontSize: "0.86rem"
              }}
            >
              Defender team:{" "}
              <ResultPlayerLink player={defender} clubSlug={clubSlug} /> +{" "}
              {match.defender_partner ? (
                <ResultPlayerLink
                  player={match.defender_partner}
                  clubSlug={clubSlug}
                />
              ) : (
                "partner unavailable"
              )}
            </p>
            {match.date ? (
              <p
                style={{
                  margin: "0.2rem 0",
                  color: "#64748b",
                  fontSize: "0.8rem"
                }}
              >
                {dateLabel(match.date)}
              </p>
            ) : null}
            {match.games?.length ? (
              <p
                style={{
                  margin: "0.35rem 0",
                  color: "#475569",
                  fontSize: "0.82rem"
                }}
              >
                Games:{" "}
                {match.games
                  .map((game) => `${game.challenger}–${game.defender}`)
                  .join(", ")}
              </p>
            ) : null}
            <details style={{ marginTop: "0.45rem" }}>
              <summary style={{ cursor: "pointer", fontWeight: 700 }}>
                JUPR before → after
              </summary>
              {match.rating_changes.length ? (
                <ul
                  style={{
                    margin: "0.4rem 0 0",
                    paddingLeft: "1.2rem",
                    color: "#475569",
                    fontSize: "0.82rem"
                  }}
                >
                  {match.rating_changes.map((rating) => (
                    <li key={String(rating.player_id)}>
                      <ResultPlayerLink player={rating} clubSlug={clubSlug} />:{" "}
                      {rating.before_jupr == null ||
                      rating.after_jupr == null
                        ? "unrated / unavailable"
                        : `${juprLabel(rating.before_jupr)} → ${juprLabel(
                            rating.after_jupr
                          )} (${signedJupr(rating.delta_jupr)})`}
                    </li>
                  ))}
                </ul>
              ) : (
                <p style={{ color: "#64748b", fontSize: "0.82rem" }}>
                  Ratings before and after this match aren&apos;t available.
                </p>
              )}
            </details>
          </article>
        ))}
      </div>
    </section>
  );
}
