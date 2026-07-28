"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import styles from "./TournamentAdminNav.module.css";

type NavigationItem = {
  href: string;
  label: string;
  match: (pathname: string) => boolean;
  danger?: boolean;
};

const exact = (href: string) => (pathname: string) => pathname === href;

const tournamentItems: NavigationItem[] = [
  { href: "/admin/tournament-setup", label: "Setup", match: exact("/admin/tournament-setup") },
  { href: "/admin/tournaments", label: "Registration editor", match: exact("/admin/tournaments") },
  { href: "/admin/tournaments/registrations", label: "Registration reports", match: exact("/admin/tournaments/registrations") },
  { href: "/admin/tournaments/bulk", label: "Bulk actions", match: exact("/admin/tournaments/bulk") },
  { href: "/admin/tournaments/commerce", label: "Extras & fulfillment", match: exact("/admin/tournaments/commerce") },
  { href: "/admin/tournaments/team-competition", label: "Ratings & team play", match: exact("/admin/tournaments/team-competition") },
  { href: "/admin/tournaments/status", label: "Status", match: exact("/admin/tournaments/status") },
  {
    href: "/admin/tournaments/ops",
    label: "Operations",
    match: (pathname) => pathname === "/admin/tournaments/ops" || pathname.startsWith("/admin/tournaments/ops/")
  },
  { href: "/admin/tournament-live", label: "Live runner", match: exact("/admin/tournament-live") },
  { href: "/admin/tournaments/delete-draft", label: "Delete draft", match: exact("/admin/tournaments/delete-draft"), danger: true }
];

const operationItems: NavigationItem[] = [
  { href: "/admin/tournaments/ops", label: "Overview", match: exact("/admin/tournaments/ops") },
  { href: "/admin/tournaments/ops/draws", label: "Draws and scoring", match: exact("/admin/tournaments/ops/draws") },
  { href: "/admin/tournaments/ops/import", label: "Team imports", match: exact("/admin/tournaments/ops/import") },
  { href: "/admin/tournaments/ops/results", label: "Results CSV", match: exact("/admin/tournaments/ops/results") },
  { href: "/admin/tournaments/ops/publish", label: "Official publish", match: exact("/admin/tournaments/ops/publish") }
];

function NavigationLinks({ items, pathname }: { items: NavigationItem[]; pathname: string }) {
  return (
    <ul className={styles.list}>
      {items.map((item) => {
        const active = item.match(pathname);
        const classNames = [
          styles.link,
          active ? styles.active : "",
          item.danger ? styles.danger : ""
        ].filter(Boolean).join(" ");

        return (
          <li key={item.href}>
            <Link
              href={item.href}
              aria-current={
                active ? (item.href === pathname ? "page" : "location") : undefined
              }
              className={classNames}
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

  return (
    <div className={styles.shell} data-testid="tournament-admin-navigation">
      <nav aria-label="Tournament administration" className={styles.nav}>
        <div className={styles.headingRow}>
          <p className={styles.eyebrow}>Tournament workspace</p>
          <p className={styles.hint}>Setup, registrations, event operations, and live play</p>
        </div>
        <NavigationLinks items={tournamentItems} pathname={pathname} />
      </nav>

      <nav aria-label="Tournament operations workflows" className={`${styles.nav} ${styles.subnav}`}>
        <p className={styles.subnavLabel}>Operations workflows</p>
        <NavigationLinks items={operationItems} pathname={pathname} />
      </nav>
    </div>
  );
}
