import type { WeeklyHighlightCard } from "@/lib/leagueWeeklyHighlights";

export default function WeeklyHighlightCards({ cards }: { cards: WeeklyHighlightCard[] }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 240px), 1fr))", gap: "0.75rem" }}>
      {cards.map((card) => (
        <article key={card.id} style={{ border: "1px solid #bfdbfe", borderRadius: 14, padding: "1rem", background: "#f8fbff", minWidth: 0, overflowWrap: "anywhere" }}>
          <h3 style={{ margin: "0 0 0.75rem", color: "#1d4ed8", fontSize: "0.85rem" }}>{card.title}</h3>
          <p style={{ margin: "0 0 0.25rem", fontSize: "1.15rem", fontWeight: 800 }}>{card.name}</p>
          <p style={{ margin: "0 0 0.65rem", fontSize: "1.4rem", fontWeight: 800, color: "#0f172a" }}>{card.metric}</p>
          <p style={{ margin: 0, color: "#475569", lineHeight: 1.5, fontSize: "0.9rem" }}>{card.detail}</p>
        </article>
      ))}
    </div>
  );
}
