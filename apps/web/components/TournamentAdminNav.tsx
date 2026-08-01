"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import styles from "./TournamentAdminNav.module.css";

type NavigationItem = {
  href: string;
  label: string;
  match: (pathname: string) => boolean;
};

const exact = (href: string) => (pathname: string) => pathname === href;

function selectedHref(
  path: string,
  tournamentId: string,
  tournamentName: string
): string {
  const params = new URLSearchParams({ tournament: tournamentId });
  if (tournamentName) params.set("name", tournamentName);
  return `${path}?${params.toString()}`;
}

function selectedItems(
  tournamentId: string,
  tournamentName: string
): NavigationItem[] {
  return [
    {
      href: selectedHref(
        "/admin/tournaments/tournament",
        tournamentId,
        tournamentName
      ),
      label: "Tournament Home",
      match: exact("/admin/tournaments/tournament")
    },
    {
      href: selectedHref(
        "/admin/tournaments/setup",
        tournamentId,
        tournamentName
      ),
      label: "Setup",
      match: (pathname) =>
        pathname === "/admin/tournaments/setup" ||
        pathname === "/admin/tournament-setup" ||
        pathname === "/admin/tournaments/team-competition"
    },
    {
      href: selectedHref(
        "/admin/tournaments/registration",
        tournamentId,
        tournamentName
      ),
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
      href: selectedHref(
        "/admin/tournaments/live-operations",
        tournamentId,
        tournamentName
      ),
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
      href: selectedHref(
        "/admin/tournaments/publish",
        tournamentId,
        tournamentName
      ),
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
  const tournamentId = String(searchParams.get("tournament") || "").trim();
  const tournamentName = String(searchParams.get("name") || "").trim();
  const hasTournament = Boolean(tournamentId);
  const managerItems: NavigationItem[] = [
    {
      href: "/admin/tournaments",
      label: "Tournament Manager Home",
      match: exact("/admin/tournaments")
    }
  ];
  const tournamentItems = hasTournament
    ? selectedItems(tournamentId, tournamentName)
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
