"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import styles from "./TournamentAdminNav.module.css";

type NavigationItem = {
  href: string;
  label: string;
  match: (pathname: string) => boolean;
  danger?: boolean;
};

const exact = (href: string) => (pathname: string) => pathname === href;

function selectedHref(path: string, tournamentId: string, tournamentName: string): string {
  const params = new URLSearchParams({ tournament: tournamentId });
  if (tournamentName) params.set("name", tournamentName);
  return `${path}?${params.toString()}`;
}

function selectedItems(tournamentId: string, tournamentName: string): NavigationItem[] {
  return [
    { href: selectedHref("/admin/tournaments/tournament", tournamentId, tournamentName), label: "Tournament Home", match: exact("/admin/tournaments/tournament") },
    { href: selectedHref("/admin/tournament-setup", tournamentId, tournamentName), label: "Setup", match: exact("/admin/tournament-setup") },
    { href: selectedHref("/admin/tournaments/editor", tournamentId, tournamentName), label: "Registrations", match: exact("/admin/tournaments/editor") },
    { href: selectedHref("/admin/tournaments/registrations", tournamentId, tournamentName), label: "Reports", match: exact("/admin/tournaments/registrations") },
    { href: selectedHref("/admin/tournaments/bulk", tournamentId, tournamentName), label: "Bulk actions", match: exact("/admin/tournaments/bulk") },
    { href: selectedHref("/admin/tournaments/commerce", tournamentId, tournamentName), label: "Extras & fulfillment", match: exact("/admin/tournaments/commerce") },
    { href: selectedHref("/admin/tournaments/team-competition", tournamentId, tournamentName), label: "Ratings & team play", match: exact("/admin/tournaments/team-competition") },
    {
      href: selectedHref("/admin/tournaments/ops", tournamentId, tournamentName),
      label: "Operations",
      match: (pathname) => pathname === "/admin/tournaments/ops" || pathname.startsWith("/admin/tournaments/ops/")
    },
    { href: selectedHref("/admin/tournaments/ops/results", tournamentId, tournamentName), label: "Results", match: exact("/admin/tournaments/ops/results") },
    { href: selectedHref("/admin/tournament-live", tournamentId, tournamentName), label: "Live runner", match: exact("/admin/tournament-live") },
    { href: selectedHref("/admin/tournaments/ops/publish", tournamentId, tournamentName), label: "Official publish", match: exact("/admin/tournaments/ops/publish") },
    { href: selectedHref("/admin/tournaments/status", tournamentId, tournamentName), label: "Status & recovery", match: exact("/admin/tournaments/status") }
  ];
}

function operationsItems(tournamentId: string, tournamentName: string): NavigationItem[] {
  return [
    { href: selectedHref("/admin/tournaments/ops", tournamentId, tournamentName), label: "Operations home", match: exact("/admin/tournaments/ops") },
    { href: selectedHref("/admin/tournaments/ops/draws", tournamentId, tournamentName), label: "Draws & scoring", match: exact("/admin/tournaments/ops/draws") },
    { href: selectedHref("/admin/tournaments/ops/import", tournamentId, tournamentName), label: "Team imports", match: exact("/admin/tournaments/ops/import") },
    { href: selectedHref("/admin/tournaments/ops/results", tournamentId, tournamentName), label: "Results CSV", match: exact("/admin/tournaments/ops/results") },
    { href: selectedHref("/admin/tournaments/ops/publish", tournamentId, tournamentName), label: "Official publish", match: exact("/admin/tournaments/ops/publish") }
  ];
}

function NavigationLinks({ items, pathname }: { items: NavigationItem[]; pathname: string }) {
  return (
    <ul className={styles.list}>
      {items.map((item) => {
        const active = item.match(pathname);
        const classNames = [styles.link, active ? styles.active : "", item.danger ? styles.danger : ""].filter(Boolean).join(" ");
        return (
          <li key={`${item.href}-${item.label}`}>
            <Link href={item.href} aria-current={active ? "page" : undefined} className={classNames}>
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
    { href: "/admin/tournaments", label: "Tournament Manager Home", match: exact("/admin/tournaments") }
  ];
  const tournamentItems = hasTournament ? selectedItems(tournamentId, tournamentName) : [];
  const operationItems = hasTournament ? operationsItems(tournamentId, tournamentName) : [];

  return (
    <div className={styles.shell} data-testid="tournament-admin-navigation">
      <nav aria-label="Tournament administration" className={styles.nav}>
        <NavigationLinks items={[...managerItems, ...tournamentItems]} pathname={pathname} />
      </nav>
      {hasTournament && pathname.startsWith("/admin/tournaments/ops") ? (
        <nav aria-label="Tournament operations workflows" className={`${styles.nav} ${styles.subnav}`}>
          <NavigationLinks items={operationItems} pathname={pathname} />
        </nav>
      ) : null}
    </div>
  );
}
