"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import styles from "./PublicSiteHeader.module.css";

type Props = {
  productName: string;
  isStaging: boolean;
};

type NavigationItem = {
  label: string;
  href: string;
  active: (pathname: string) => boolean;
  staff?: boolean;
};

const clubBase = "/clubs/tres-palapas";

const navigationItems: NavigationItem[] = [
  {
    label: "Home",
    href: "/",
    active: (pathname) => pathname === "/"
  },
  {
    label: "Club",
    href: clubBase,
    active: (pathname) => pathname === clubBase
  },
  {
    label: "Leagues",
    href: `${clubBase}/leagues`,
    active: (pathname) =>
      pathname === `${clubBase}/leagues` ||
      pathname.startsWith(`${clubBase}/league-results`) ||
      pathname.startsWith(`${clubBase}/team-leagues`) ||
      pathname.startsWith(`${clubBase}/challenge-ladder`)
  },
  {
    label: "Tournaments",
    href: `${clubBase}/tournaments`,
    active: (pathname) =>
      pathname === `${clubBase}/tournaments` ||
      pathname.startsWith(`${clubBase}/tournament-`)
  },
  {
    label: "Play",
    href: `${clubBase}/play`,
    active: (pathname) =>
      pathname.startsWith(`${clubBase}/play`) ||
      pathname.startsWith(`${clubBase}/round-robin-generator`) ||
      pathname.startsWith(`${clubBase}/ladder-generator`) ||
      pathname.startsWith(`${clubBase}/live`)
  },
  {
    label: "Leaderboards",
    href: `${clubBase}/leaderboards`,
    active: (pathname) => pathname.startsWith(`${clubBase}/leaderboards`)
  },
  {
    label: "Match Explorer",
    href: `${clubBase}/match-explorer`,
    active: (pathname) => pathname.startsWith(`${clubBase}/match-explorer`)
  },
  {
    label: "Weekly Recap",
    href: `${clubBase}/weekly-recap`,
    active: (pathname) => pathname.startsWith(`${clubBase}/weekly-recap`)
  },
  {
    label: "Players",
    href: `${clubBase}/players`,
    active: (pathname) => pathname.startsWith(`${clubBase}/players`)
  },
  {
    label: "Staff sign in",
    href: "/admin/login",
    active: (pathname) => pathname === "/admin/login",
    staff: true
  }
];

function Brand({ productName, isStaging }: Props) {
  return (
    <div className={styles.brandGroup}>
      <Link href="/" className={styles.brand}>
        {productName}
      </Link>
      {isStaging ? <span className={styles.environment}>STAGING</span> : null}
    </div>
  );
}

export default function PublicSiteHeader({ productName, isStaging }: Props) {
  const pathname = usePathname() || "/";

  if (pathname === "/admin" || pathname.startsWith("/admin/")) {
    return (
      <header className={styles.compactHeader}>
        <Brand productName={productName} isStaging={isStaging} />
      </header>
    );
  }

  return (
    <header className={styles.header}>
      <div className={styles.brandRow}>
        <Brand productName={productName} isStaging={isStaging} />
      </div>
      <nav className={styles.nav} aria-label="Primary navigation">
        {navigationItems.map((item) => {
          const active = item.active(pathname);
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={`${styles.link} ${active ? styles.active : ""} ${item.staff ? styles.staff : ""}`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
