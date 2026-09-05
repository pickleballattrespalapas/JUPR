type StandingsSort = "wins" | "points" | "differential";

export type PlayGeneratorStanding = {
  rank: number;
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
};

export function standingsSortLabel(mode: StandingsSort): string {
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

export default function PlayGeneratorStandingsTable({ rows, sortMode }: Props) {
  return (
    <article
      style={{
        border: "1px solid #e2e8f0",
        borderRadius: "14px",
        padding: "1rem",
        background: "white"
      }}
    >
      <div style={{ marginBottom: "0.85rem" }}>
        <h2 style={{ margin: "0 0 0.3rem" }}>Standings</h2>
        <p style={{ margin: 0, color: "#475569" }}>
          Ranked by <strong>{standingsSortLabel(sortMode)}</strong>. {tieBreakText(sortMode)}
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
                <td style={{ padding: "0.65rem", fontWeight: 750 }}>{row.name}</td>
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
