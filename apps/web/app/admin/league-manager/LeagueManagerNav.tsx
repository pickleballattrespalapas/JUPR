"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

type Props = {
  leagueName?: string | null;
  leagueType?: string | null;
  managerOnly?: boolean;
};

type NavLink = {
  href: string;
  label: string;
};

const navStyle = {
  display: "flex",
  gap: "0.55rem",
  flexWrap: "wrap" as const,
  margin: "0 0 1rem"
};

function leagueHref(path: string, leagueName: string, leagueType?: string | null): string {
  const params = new URLSearchParams({ league: leagueName });
  if (leagueType) params.set("mode", leagueType);
  return `${path}?${params.toString()}`;
}

export default function LeagueManagerNav({ leagueName, leagueType, managerOnly = false }: Props) {
  const pathname = usePathname() || "";
  const hasLeague = Boolean(leagueName && !managerOnly);
  const links: NavLink[] = [
    { href: "/admin/league-manager", label: "League Manager Home" }
  ];

  if (hasLeague && leagueName) {
    links.push(
      { href: leagueHref("/admin/league-manager/league", leagueName, leagueType), label: "League Home" },
      { href: leagueHref("/admin/league-manager/results", leagueName, leagueType), label: "Results" },
      { href: leagueHref("/admin/league-manager/settings", leagueName, leagueType), label: "Settings" },
      { href: leagueHref("/admin/league-manager/roster", leagueName, leagueType), label: "Roster" },
      { href: leagueHref("/admin/league-manager/live", leagueName, leagueType), label: "Live rounds" },
      { href: leagueHref("/admin/league-manager/awards", leagueName, leagueType), label: "Awards" },
      { href: leagueHref("/admin/league-manager/print", leagueName, leagueType), label: "League night printout" }
    );
    if (String(leagueType || "Individual") === "Team") {
      links.splice(5, 0, {
        href: leagueHref("/admin/league-manager/teams", leagueName, leagueType),
        label: "Team league"
      });
    }
  }

  return (
    <nav aria-label="League Manager navigation" style={navStyle}>
      {links.map(({ href, label }) => {
        const hrefPath = href.split("?")[0];
        const active = hrefPath === "/admin/league-manager"
          ? pathname === hrefPath
          : pathname === hrefPath || pathname.startsWith(`${hrefPath}/`);
        return (
          <Link
            key={`${href}-${label}`}
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
