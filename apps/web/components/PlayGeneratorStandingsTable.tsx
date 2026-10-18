type StandingsSort = "wins" | "points" | "differential" | "moneyball_expectations";

export type PlayGeneratorStanding = {
  rank: number;
  moneyballEligible?: boolean;
  expectedWins?: number | null;
  expectedDifferential?: number | null;
  moneyballScore?: number | null;
  moneyballFinal?: boolean;
  roundRobinVictoryRanking?: boolean;
  winnerNeedsAdmin?: boolean;
  winnerDecidedByAdmin?: boolean;
  participantId: string;
  name: string;
  matches: number;
  wins: number;
  losses: number;
  pointsFor: number;
  pointsAgainst: number;
  differential: number;
};

type Props = {
  rows: PlayGeneratorStanding[];
  sortMode: StandingsSort;
  title?: string;
};

export function standingsSortLabel(mode: StandingsSort): string {
  if (mode === "moneyball_expectations") return "Moneyball vs Expectations";
  if (mode === "points") return "Total points";
  if (mode === "differential") return "Point differential";
  return "Total wins";
}

function tieBreakText(mode: StandingsSort): string {
  if (mode === "points") return "Ties are broken by wins, then point difference, then starting order.";
  if (mode === "differential") return "Ties are broken by wins, then total points, then starting order.";
  return "Ties are broken by point difference, then total points, then starting order.";
}

const primaryCell = {
  background: "#ecfdf5",
  fontWeight: 850
};

export default function PlayGeneratorStandingsTable({ rows, sortMode, title = "Standings" }: Props) {
  if (sortMode === "moneyball_expectations") return <article style={{ border: "1px solid #e2e8f0", borderRadius: 14, padding: "1rem", background: "white" }}>
    <h2>{title}</h2>
    <p>Highest Moneyball score wins. Your point performance is compared with expectations from frozen starting ratings. Wins are shown for context, with no separate win bonus.</p>
    <p>{rows.every(row => row.moneyballFinal) ? "All scheduled rounds are scored. Tied winners split the pool." : "Provisional standings — the winner is decided after the full rotation. Expected figures cover saved games only."}</p>
    <div style={{ overflowX: "auto" }}><table style={{ width: "100%", minWidth: 720, borderCollapse: "collapse" }}>
      <thead><tr>{["Place", "Player", "Wins / Games", "Expected wins", "Point diff.", "Expected point diff.", "Moneyball / 100"].map(label => <th key={label} style={{ padding: ".65rem", textAlign: "left" }}>{label}</th>)}</tr></thead>
      <tbody>{rows.map(row => <tr key={row.participantId} style={{ borderTop: "1px solid #e2e8f0" }}>
        <td style={{ padding: ".65rem" }}>{row.moneyballEligible === false ? "DQ" : row.moneyballScore == null ? "—" : row.rank}</td><td>{row.name}{row.moneyballEligible === false ? <small> · Ineligible after substitution</small> : null}</td>
        <td>{row.wins} / {row.matches}</td><td>{row.expectedWins?.toFixed(2) ?? "—"}</td>
        <td>{row.differential > 0 ? "+" : ""}{row.differential}</td>
        <td>{(row.expectedDifferential || 0) > 0 ? "+" : ""}{row.expectedDifferential?.toFixed(2) ?? "—"}</td>
        <td style={{ fontWeight: 850 }}>{row.moneyballScore?.toFixed(2) ?? "—"}</td>
      </tr>)}</tbody>
    </table></div>
  </article>;
  const finalRoundRobin = rows.some(row => row.roundRobinVictoryRanking);
  if (finalRoundRobin) sortMode = "wins";
  return (
    <article
      style={{
        border: "1px solid #e2e8f0",
        borderRadius: "14px",
        padding: "1rem",
        minWidth: 0,
        background: "white"
      }}
    >
      <div style={{ marginBottom: "0.85rem" }}>
        <h2 style={{ margin: "0 0 0.3rem" }}>{title}</h2>
        <p style={{ margin: 0, color: "#475569" }}>
          Ranked by <strong>{standingsSortLabel(sortMode)}</strong>. {finalRoundRobin ? "Ties are decided by point difference, then total points, then an admin decision." : tieBreakText(sortMode)}
          {" "}Skipped and unplayed rounds do not affect the table.
        </p>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", minWidth: 680, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #cbd5e1" }}>
              <th align="center" style={{ padding: "0.55rem" }}>Place</th>
              <th align="left" style={{ padding: "0.55rem" }}>Player</th>
              <th align="center" style={{ padding: "0.55rem" }}>Games</th>
              <th align="center" style={{ padding: "0.55rem", ...(sortMode === "wins" ? primaryCell : {}) }}>Wins</th>
              <th align="center" style={{ padding: "0.55rem" }}>Losses</th>
              <th align="center" style={{ padding: "0.55rem", ...(sortMode === "points" ? primaryCell : {}) }}>Points</th>
              <th align="center" style={{ padding: "0.55rem" }}>Points against</th>
              <th align="center" style={{ padding: "0.55rem", ...(sortMode === "differential" ? primaryCell : {}) }}>Point difference</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.participantId} style={{ borderBottom: "1px solid #e2e8f0" }}>
                <td align="center" style={{ padding: "0.65rem", fontWeight: 850 }}>{row.rank}</td>
                <td style={{ padding: "0.65rem", fontWeight: 750 }}>{row.name}{row.winnerNeedsAdmin ? <small> · Admin decision pending</small> : null}{row.winnerDecidedByAdmin ? <small> · Winner confirmed by admin</small> : null}</td>
                <td align="center" style={{ padding: "0.65rem" }}>{row.matches}</td>
                <td align="center" style={{ padding: "0.65rem", ...(sortMode === "wins" ? primaryCell : {}) }}>{row.wins}</td>
                <td align="center" style={{ padding: "0.65rem" }}>{row.losses}</td>
                <td align="center" style={{ padding: "0.65rem", ...(sortMode === "points" ? primaryCell : {}) }}>{row.pointsFor}</td>
                <td align="center" style={{ padding: "0.65rem" }}>{row.pointsAgainst}</td>
                <td align="center" style={{ padding: "0.65rem", ...(sortMode === "differential" ? primaryCell : {}) }}>
                  {row.differential > 0 ? "+" : ""}{row.differential}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}
