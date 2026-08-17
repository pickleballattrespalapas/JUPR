"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  isTeamLeagueType,
  leagueRouteHref,
  normalizeLeagueType,
  type LeagueRouteContext
} from "@/lib/leagueRouteContext";

type Props = {
  leagueId?: string | null;
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

function leagueHref(path: string, context: LeagueRouteContext): string {
  return leagueRouteHref(path, context);
}

export default function LeagueManagerNav({
  leagueId: providedLeagueId,
  leagueName: providedLeagueName,
  leagueType: providedLeagueType,
  managerOnly = false
}: Props) {
  const pathname = usePathname() || "";
  const leagueId = String(providedLeagueId || providedLeagueName || "").trim();
  const leagueName = String(providedLeagueName || leagueId).trim();
  const leagueType = normalizeLeagueType(providedLeagueType);
  const context = { leagueId, leagueName, leagueType };
  const hasLeague = Boolean(leagueId && leagueName && !managerOnly);
  const links: NavLink[] = [
    { href: "/admin/league-manager", label: "League Manager Home" }
  ];

  if (hasLeague && leagueName) {
    links.push(
      { href: leagueHref("/admin/league-manager/league", context), label: "League Home" },
      { href: leagueHref("/admin/league-manager/results", context), label: "Results" },
      { href: leagueHref("/admin/league-manager/settings", context), label: "Settings" },
      { href: leagueHref("/admin/league-manager/roster", context), label: "Roster" },
      { href: leagueHref("/admin/league-manager/live", context), label: "Live rounds" },
      { href: leagueHref("/admin/league-manager/awards", context), label: "Awards" },
      { href: leagueHref("/admin/league-manager/print", context), label: "League night printout" }
    );
    if (isTeamLeagueType(leagueType)) {
      links.splice(5, 0, {
        href: leagueHref("/admin/league-manager/teams", context),
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
