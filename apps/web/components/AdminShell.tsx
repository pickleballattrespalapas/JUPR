"use client";

import type { ReactNode } from "react";
import { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { signOutAdminSession } from "@/lib/adminAuthClient";
import { useAdminSession } from "@/lib/useAdminSession";
import styles from "./AdminShell.module.css";

type Props = {
  children: ReactNode;
};

type AdminLink = {
  label: string;
  href: string;
  active: (pathname: string) => boolean;
};

type AdminGroup = {
  label: string;
  links: AdminLink[];
};

const adminGroups: AdminGroup[] = [
  {
    label: "Workspace",
    links: [
      {
        label: "Admin Home",
        href: "/admin",
        active: (pathname) => pathname === "/admin"
      },
      {
        label: "Match Uploader",
        href: "/admin/match-uploader",
        active: (pathname) => pathname.startsWith("/admin/match-uploader")
      },
      {
        label: "Match Log",
        href: "/admin/match-log",
        active: (pathname) => pathname.startsWith("/admin/match-log")
      },
      {
        label: "Players",
        href: "/admin/players",
        active: (pathname) => pathname.startsWith("/admin/players")
      }
    ]
  },
  {
    label: "Competitions",
    links: [
      {
        label: "League Manager",
        href: "/admin/league-manager",
        active: (pathname) => pathname.startsWith("/admin/league-manager")
      },
      {
        label: "Tournament Manager",
        href: "/admin/tournaments",
        active: (pathname) =>
          pathname.startsWith("/admin/tournaments") ||
          pathname.startsWith("/admin/tournament-")
      },
      {
        label: "JUPR Live",
        href: "/admin/jupr-live",
        active: (pathname) => pathname.startsWith("/admin/jupr-live")
      },
      {
        label: "Challenge Ladder",
        href: "/admin/challenge-ladder",
        active: (pathname) => pathname.startsWith("/admin/challenge-ladder")
      },
      {
        label: "Moneyball",
        href: "/admin/moneyball",
        active: (pathname) => pathname.startsWith("/admin/moneyball")
      }
    ]
  },
  {
    label: "Communications",
    links: [
      {
        label: "Player Updates",
        href: "/admin/player-updates",
        active: (pathname) => pathname.startsWith("/admin/player-updates")
      },
      {
        label: "Weekly Recap",
        href: "/admin/weekly-recap",
        active: (pathname) => pathname.startsWith("/admin/weekly-recap")
      },
      {
        label: "Support Requests",
        href: "/admin/support-requests",
        active: (pathname) => pathname.startsWith("/admin/support-requests")
      }
    ]
  },
  {
    label: "System",
    links: [
      {
        label: "Admin Tools",
        href: "/admin/tools",
        active: (pathname) =>
          pathname.startsWith("/admin/tools") ||
          pathname.startsWith("/admin/badges") ||
          pathname.startsWith("/admin/match-canonical-audit") ||
          pathname.startsWith("/admin/top-players-printable")
      },
      {
        label: "Replay History",
        href: "/admin/replay-history",
        active: (pathname) => pathname.startsWith("/admin/replay-history")
      }
    ]
  }
];

function SidebarLink({ item, pathname }: { item: AdminLink; pathname: string }) {
  const active = item.active(pathname);
  return (
    <Link
      href={item.href}
      aria-current={active ? "page" : undefined}
      className={`${styles.link} ${active ? styles.active : ""}`}
    >
      {item.label}
    </Link>
  );
}

export default function AdminShell({ children }: Props) {
  const pathname = usePathname() || "/admin";
  const router = useRouter();
  const { session, accessToken } = useAdminSession();
  const [signingOut, setSigningOut] = useState(false);
  const authPage =
    pathname === "/admin/login" || pathname === "/admin/reset-password";

  if (authPage || !accessToken) return <>{children}</>;

  async function signOut() {
    if (signingOut) return;
    setSigningOut(true);
    try {
      await signOutAdminSession();
    } finally {
      router.replace("/admin/login");
      router.refresh();
    }
  }

  return (
    <div className={styles.shell}>
      <aside className={styles.sidebar} aria-label="Admin workspace navigation">
        <div className={styles.identity}>
          <p className={styles.eyebrow}>Admin workspace</p>
          <p className={styles.email}>
            {session?.user?.email || "Authorized staff account"}
          </p>
        </div>

        {adminGroups.map((group) => (
          <nav key={group.label} className={styles.group} aria-label={group.label}>
            <p className={styles.groupLabel}>{group.label}</p>
            {group.links.map((item) => (
              <SidebarLink key={item.href} item={item} pathname={pathname} />
            ))}
          </nav>
        ))}

        <div className={styles.actions}>
          <Link href="/admin/login" className={styles.sessionLink}>
            Manage session
          </Link>
          <button
            type="button"
            className={styles.signOut}
            disabled={signingOut}
            onClick={() => void signOut()}
          >
            {signingOut ? "Signing out…" : "Sign out"}
          </button>
        </div>
      </aside>
      <div className={styles.content}>{children}</div>
    </div>
  );
}
