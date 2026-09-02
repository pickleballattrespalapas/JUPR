"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import {
  readTournamentRouteContext,
  tournamentRouteHref,
  type TournamentRouteContext
} from "@/lib/tournamentRouteContext";
import styles from "./TournamentAdminNav.module.css";

type NavigationItem = {
  href: string;
  label: string;
  match: (pathname: string) => boolean;
};

const exact = (href: string) => (pathname: string) => pathname === href;

function selectedItems(context: TournamentRouteContext): NavigationItem[] {
  return [
    {
      href: tournamentRouteHref("/admin/tournaments/tournament", context),
      label: "Tournament Home",
      match: exact("/admin/tournaments/tournament")
    },
    {
      href: tournamentRouteHref("/admin/tournaments/setup", context),
      label: "Tournament Builder",
      match: (pathname) =>
        pathname === "/admin/tournaments/setup" ||
        pathname.startsWith("/admin/tournaments/setup/") ||
        pathname === "/admin/tournament-setup" ||
        pathname === "/admin/tournaments/team-competition"
    },
    {
      href: tournamentRouteHref("/admin/tournaments/registration", context),
      label: "Registration",
      match: (pathname) =>
        pathname === "/admin/tournaments/registration" ||
        pathname.startsWith("/admin/tournaments/registration/") ||
        pathname === "/admin/tournaments/editor" ||
        pathname === "/admin/tournaments/registrations" ||
        pathname === "/admin/tournaments/bulk" ||
        pathname === "/admin/tournaments/commerce"
    },
    {
      href: tournamentRouteHref("/admin/tournaments/live-operations", context),
      label: "Live Operations",
      match: (pathname) =>
        pathname === "/admin/tournaments/live-operations" ||
        pathname.startsWith("/admin/tournaments/live-operations/") ||
        pathname === "/admin/tournament-live" ||
        pathname === "/admin/tournaments/status" ||
        pathname === "/admin/tournaments/ops" ||
        pathname === "/admin/tournaments/ops/draws" ||
        pathname === "/admin/tournaments/ops/import"
    },
    {
      href: tournamentRouteHref("/admin/tournaments/publish", context),
      label: "Publish",
      match: (pathname) =>
        pathname === "/admin/tournaments/publish" ||
        pathname.startsWith("/admin/tournaments/publish/") ||
        pathname === "/admin/tournaments/ops/results" ||
        pathname === "/admin/tournaments/ops/publish"
    }
  ];
}

function NavigationLinks({
  items,
  pathname
}: {
  items: NavigationItem[];
  pathname: string;
}) {
  return (
    <ul className={styles.list}>
      {items.map((item) => {
        const active = item.match(pathname);
        return (
          <li key={`${item.href}-${item.label}`}>
            <Link
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={`${styles.link} ${active ? styles.active : ""}`}
            >
              {item.label}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}

export default function TournamentAdminNav() {
  const pathname = usePathname() || "";
  const searchParams = useSearchParams();
  const context = readTournamentRouteContext(searchParams);
  const hasTournament = Boolean(context.tournamentId);
  const managerItems: NavigationItem[] = [
    {
      href: "/admin/tournaments",
      label: "Tournament Manager Home",
      match: exact("/admin/tournaments")
    }
  ];
  const tournamentItems = hasTournament
    ? selectedItems(context)
    : [];

  return (
    <div className={styles.shell} data-testid="tournament-admin-navigation">
      <nav aria-label="Tournament administration" className={styles.nav}>
        <NavigationLinks
          items={[...managerItems, ...tournamentItems]}
          pathname={pathname}
        />
      </nav>
    </div>
  );
}
