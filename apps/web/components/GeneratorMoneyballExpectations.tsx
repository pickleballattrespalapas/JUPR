export type MoneyballExpectation = {
  participantId: string;
  name: string;
  scheduledGames: number;
  expectedWins: number;
  expectedLosses: number;
  expectedDifferential: number;
};

function signed(value: number): string {
  const rounded = value.toFixed(2);
  return Number(rounded) === 0 ? "0.00" : value > 0 ? `+${rounded}` : rounded;
}

export default function GeneratorMoneyballExpectations({ rows }: { rows?: MoneyballExpectation[] }) {
  return (
    <section aria-label="Moneyball player expectations" style={{ border: "1px solid #86efac", borderRadius: 12, padding: "1rem", marginBottom: "1rem", background: "#f0fdf4", minWidth: 0 }}>
      <h3 style={{ marginTop: 0 }}>Your expectations for the full event</h3>
      {rows?.length ? <>
        <p>Based on this starting lineup, its ratings and every scheduled matchup. Byes do not count as games. Expectations lock when you start.</p>
        <div role="region" aria-label="Full-event expectations table" tabIndex={0} style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", minWidth: 520, borderCollapse: "collapse", fontVariantNumeric: "tabular-nums" }}>
            <thead>
              <tr>{["Player", "Games", "Expected wins", "Expected losses", "Expected point diff."].map((label, index) => (
                <th key={label} scope="col" style={{ padding: ".65rem", textAlign: index === 0 ? "left" : "right" }}>{label}</th>
              ))}</tr>
            </thead>
            <tbody>{rows.map(row => (
              <tr key={row.participantId} style={{ borderTop: "1px solid #bbf7d0" }}>
                <th scope="row" style={{ padding: ".65rem", textAlign: "left" }}>{row.name}</th>
                <td style={{ padding: ".65rem", textAlign: "right" }}>{row.scheduledGames}</td>
                <td style={{ padding: ".65rem", textAlign: "right", fontWeight: 750 }}>{row.expectedWins.toFixed(2)}</td>
                <td style={{ padding: ".65rem", textAlign: "right" }}>{row.expectedLosses.toFixed(2)}</td>
                <td style={{ padding: ".65rem", textAlign: "right" }}>{signed(row.expectedDifferential)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <p>These are averages, so decimals are normal. For example, 2.6 expected wins means 3 wins would beat your win expectation.</p>
        <p style={{ marginBottom: 0 }}>Moneyball ranks your point performance against expectations. Wins are a guide, with no separate win bonus. Point difference is points scored minus points conceded; a higher result is better, even if it is still negative. Beating an expectation does not guarantee the prize.</p>
      </> : <p style={{ marginBottom: 0 }}>Generate a new preview to see each player’s full-event expectations.</p>}
    </section>
  );
}
