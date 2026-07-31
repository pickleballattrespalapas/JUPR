"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  ["/admin/league-manager", "Home"],
  ["/admin/league-manager/results", "Results"],
  ["/admin/league-manager/settings", "Settings"],
  ["/admin/league-manager/roster", "Roster"],
  ["/admin/league-manager/teams", "Team leagues"],
  ["/admin/league-manager/live", "Live rounds"],
  ["/admin/league-manager/awards", "Awards"]
] as const;

export default function LeagueManagerNav() {
  const pathname = usePathname() || "";
  return (
    <nav aria-label="League Manager sections" style={{ display: "flex", gap: "0.55rem", flexWrap: "wrap", margin: "0 0 1rem" }}>
      {links.map(([href, label]) => {
        const active = href === "/admin/league-manager"
          ? pathname === href
          : pathname === href || pathname.startsWith(`${href}/`);
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            style={{
              padding: "0.45rem 0.75rem",
              border: `1px solid ${active ? "#2563eb" : "#cbd5e1"}`,
              borderRadius: "999px",
              background: active ? "#dbeafe" : "white",
              textDecoration: "none",
              color: active ? "#1d4ed8" : "#0f172a",
              fontWeight: 700
            }}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
