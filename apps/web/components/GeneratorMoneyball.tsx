export type MoneyballSettlement = {
  entryFeeMxn: number;
  prizePoolMxn: number;
  completedGames: number;
  totalGames: number;
  final: boolean;
  requiredRounds?: number;
  rotationSummary?: string;
  noEligibleWinner?: boolean;
  winners: Array<{ participantId: string; name: string; payoutMxn: number }>;
};

export default function GeneratorMoneyball({ value }: { value?: MoneyballSettlement | null }) {
  if (!value) return null;
  return <article style={{ border: "1px solid #86efac", borderRadius: 14, padding: "1rem", background: "#ecfdf5" }}>
    <h2 style={{ marginTop: 0 }}>Moneyball vs Expectations</h2>
    <p>{value.entryFeeMxn} pesos each · {value.prizePoolMxn}-peso pool · {value.completedGames}/{value.totalGames} games saved</p>
    {value.rotationSummary ? <p>{value.rotationSummary}</p> : null}
    {value.noEligibleWinner ? <p>No eligible winner remains. Return the pool to the starting players.</p> : value.final ? <p style={{ fontWeight: 800 }}>{value.winners.map(w => `${w.name}: ${w.payoutMxn.toFixed(2)} pesos`).join(" · ")}</p>
      : <p>Play all {value.requiredRounds || "scheduled"} rounds to decide the winner. Starting ratings and scheduled slots stay fixed. Substitutions disqualify both players. Highest Moneyball score wins; tied winners split the pool.</p>}
  </article>;
}
