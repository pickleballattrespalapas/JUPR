"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import styles from "./TournamentAdminNav.module.css";

export type TournamentPhase = "setup" | "registration" | "live" | "publish";

type PhaseItem = {
  label: string;
  href: string;
  match: (pathname: string) => boolean;
};

function selectedHref(
  path: string,
  tournamentId: string,
  tournamentName: string
): string {
  const params = new URLSearchParams({ tournament: tournamentId });
  if (tournamentName) params.set("name", tournamentName);
  return `${path}?${params.toString()}`;
}

function phaseItems(
  phase: TournamentPhase,
  tournamentId: string,
  tournamentName: string
): PhaseItem[] {
  if (phase === "setup") {
    return [
      {
        label: "Setup overview",
        href: selectedHref("/admin/tournaments/setup", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournaments/setup"
      },
      {
        label: "Tournament basics",
        href: selectedHref("/admin/tournaments/tournament", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournaments/tournament"
      },
      {
        label: "Events & formats",
        href: selectedHref("/admin/tournaments/team-competition", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournaments/team-competition"
      },
      {
        label: "Registration rules",
        href: selectedHref("/admin/tournament-setup", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournament-setup"
      },
      {
        label: "Pricing & extras",
        href: selectedHref("/admin/tournaments/commerce", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournaments/commerce"
      },
      {
        label: "Schedule & courts",
        href: selectedHref("/admin/tournament-setup", tournamentId, tournamentName),
        match: () => false
      }
    ];
  }

  if (phase === "registration") {
    return [
      {
        label: "Registration overview",
        href: selectedHref("/admin/tournaments/registration", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournaments/registration"
      },
      {
        label: "Registrants",
        href: selectedHref("/admin/tournaments/registration/registrants", tournamentId, tournamentName),
        match: (pathname) => pathname.startsWith("/admin/tournaments/registration/registrants")
      },
      {
        label: "Partners & teams",
        href: selectedHref("/admin/tournaments/registration/partners", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournaments/registration/partners"
      },
      {
        label: "Payments & extras",
        href: selectedHref("/admin/tournaments/commerce", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournaments/commerce"
      },
      {
        label: "Communications & reports",
        href: selectedHref("/admin/tournaments/registrations", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournaments/registrations"
      }
    ];
  }

  if (phase === "live") {
    return [
      {
        label: "Live overview",
        href: selectedHref("/admin/tournaments/live-operations", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournaments/live-operations"
      },
      {
        label: "Preflight & check-in",
        href: selectedHref("/admin/tournaments/live-operations/check-in", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournaments/live-operations/check-in"
      },
      {
        label: "Draws & schedule",
        href: selectedHref("/admin/tournaments/ops/draws", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournaments/ops/draws"
      },
      {
        label: "Live scoring",
        href: selectedHref("/admin/tournament-live", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournament-live"
      },
      {
        label: "Corrections & recovery",
        href: selectedHref("/admin/tournaments/status", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournaments/status"
      },
      {
        label: "Podium draft",
        href: selectedHref("/admin/tournaments/ops", tournamentId, tournamentName),
        match: (pathname) => pathname === "/admin/tournaments/ops"
      }
    ];
  }

  return [
    {
      label: "Publish overview",
      href: selectedHref("/admin/tournaments/publish", tournamentId, tournamentName),
      match: (pathname) => pathname === "/admin/tournaments/publish"
    },
    {
      label: "Review results",
      href: selectedHref("/admin/tournaments/ops/results", tournamentId, tournamentName),
      match: (pathname) => pathname === "/admin/tournaments/ops/results"
    },
    {
      label: "Publish divisions",
      href: selectedHref("/admin/tournaments/ops/publish", tournamentId, tournamentName),
      match: (pathname) => pathname === "/admin/tournaments/ops/publish"
    },
    {
      label: "Tournament closeout",
      href: selectedHref("/admin/tournaments/publish/closeout", tournamentId, tournamentName),
      match: (pathname) => pathname === "/admin/tournaments/publish/closeout"
    }
  ];
}

export default function TournamentPhaseNav({ phase }: { phase: TournamentPhase }) {
  const pathname = usePathname() || "";
  const searchParams = useSearchParams();
  const tournamentId = String(searchParams.get("tournament") || "").trim();
  const tournamentName = String(searchParams.get("name") || "").trim();
  if (!tournamentId) return null;

  const items = phaseItems(phase, tournamentId, tournamentName);
  return (
    <nav
      aria-label={`${phase} tournament workflow`}
      className={`${styles.nav} ${styles.subnav}`}
    >
      <ul className={styles.list}>
        {items.map((item) => {
          const active = item.match(pathname);
          return (
            <li key={`${phase}-${item.label}`}>
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
    </nav>
  );
}
