import Link from "next/link";

export default function LeagueManagerNav() {
  const links = [
    ["/admin/league-manager", "Overview"],
    ["/admin/league-manager/settings", "Settings"],
    ["/admin/league-manager/roster", "Roster"],
    ["/admin/league-manager/teams", "Team leagues"],
    ["/admin/league-manager/live", "Live rounds"],
    ["/admin/league-manager/awards", "Awards"]
  ];
  return (
    <nav aria-label="League Manager sections" style={{ display: "flex", gap: "0.55rem", flexWrap: "wrap", margin: "0 0 1rem" }}>
      {links.map(([href, label]) => (
        <Link key={href} href={href} style={{ padding: "0.45rem 0.75rem", border: "1px solid #cbd5e1", borderRadius: "999px", background: "white", textDecoration: "none", color: "#0f172a", fontWeight: 700 }}>
          {label}
        </Link>
      ))}
    </nav>
  );
}
