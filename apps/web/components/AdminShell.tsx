"use client";

import type { ReactNode } from "react";
import { useEffect, useState } from "react";
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
  newTab?: boolean;
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
        label: "Round-Robin Generator",
        href: "/admin/round-robin-generator",
        active: (pathname) =>
          pathname.startsWith("/admin/round-robin-generator")
      },
      {
        label: "Ladder Generator",
        href: "/admin/ladder-generator",
        active: (pathname) => pathname.startsWith("/admin/ladder-generator")
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
  },
  {
    label: "Public site",
    links: [
      {
        label: "Public Home ↗",
        href: "/",
        active: () => false,
        newTab: true
      },
      {
        label: "Club Home ↗",
        href: "/clubs/tres-palapas",
        active: () => false,
        newTab: true
      },
      {
        label: "Leagues ↗",
        href: "/clubs/tres-palapas/leagues",
        active: () => false,
        newTab: true
      },
      {
        label: "Tournaments ↗",
        href: "/clubs/tres-palapas/tournaments",
        active: () => false,
        newTab: true
      },
      {
        label: "Leaderboards ↗",
        href: "/clubs/tres-palapas/leaderboards",
        active: () => false,
        newTab: true
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
      target={item.newTab ? "_blank" : undefined}
      rel={item.newTab ? "noreferrer" : undefined}
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
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>(
    {}
  );
  const authPage =
    pathname === "/admin/login" || pathname === "/admin/reset-password";

  useEffect(() => {
    const activeGroup = adminGroups.find((group) =>
      group.links.some((item) => item.active(pathname))
    );
    if (!activeGroup) return;
    setCollapsedGroups((current) => {
      if (!current[activeGroup.label]) return current;
      return { ...current, [activeGroup.label]: false };
    });
  }, [pathname]);

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

  function toggleGroup(label: string) {
    setCollapsedGroups((current) => ({
      ...current,
      [label]: !current[label]
    }));
  }

  return (
    <div
      className={`${styles.shell} ${
        sidebarCollapsed ? styles.shellCollapsed : ""
      }`}
    >
      <aside
        className={`${styles.sidebar} ${
          sidebarCollapsed ? styles.sidebarCollapsed : ""
        }`}
        aria-label="Admin workspace navigation"
      >
        <button
          type="button"
          className={styles.sidebarToggle}
          aria-expanded={!sidebarCollapsed}
          aria-label={sidebarCollapsed ? "Expand admin sidebar" : "Collapse admin sidebar"}
          title={sidebarCollapsed ? "Expand admin sidebar" : "Collapse admin sidebar"}
          onClick={() => setSidebarCollapsed((current) => !current)}
        >
          <span aria-hidden="true">{sidebarCollapsed ? "☰" : "‹"}</span>
          {!sidebarCollapsed ? <span>Collapse</span> : null}
        </button>

        {!sidebarCollapsed ? (
          <>
            <div className={styles.identity}>
              <p className={styles.eyebrow}>Admin workspace</p>
              <p className={styles.email}>
                {session?.user?.email || "Authorized staff account"}
              </p>
            </div>

            {adminGroups.map((group) => {
              const collapsed = Boolean(collapsedGroups[group.label]);
              const activeGroup = group.links.some((item) => item.active(pathname));
              const groupId = `admin-group-${group.label
                .toLowerCase()
                .replace(/[^a-z0-9]+/g, "-")}`;
              return (
                <section key={group.label} className={styles.groupSection}>
                  <button
                    type="button"
                    className={`${styles.groupToggle} ${
                      activeGroup ? styles.groupActive : ""
                    }`}
                    aria-expanded={!collapsed}
                    aria-controls={groupId}
                    onClick={() => toggleGroup(group.label)}
                  >
                    <span>{group.label}</span>
                    <span aria-hidden="true">{collapsed ? "+" : "−"}</span>
                  </button>
                  {!collapsed ? (
                    <nav
                      id={groupId}
                      className={styles.group}
                      aria-label={group.label}
                    >
                      {group.links.map((item) => (
                        <SidebarLink
                          key={item.href}
                          item={item}
                          pathname={pathname}
                        />
                      ))}
                    </nav>
                  ) : null}
                </section>
              );
            })}

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
          </>
        ) : (
          <p className={styles.collapsedLabel} aria-hidden="true">
            Admin
          </p>
        )}
      </aside>
      <div className={styles.content}>{children}</div>
    </div>
  );
}
