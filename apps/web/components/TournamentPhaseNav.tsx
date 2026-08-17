"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import {
  readTournamentRouteContext,
  tournamentRouteHref,
  type TournamentRouteContext
} from "@/lib/tournamentRouteContext";
import styles from "./TournamentAdminNav.module.css";

export type TournamentPhase = "setup" | "registration" | "live" | "publish";

type PhaseItem = {
  label: string;
  href: string;
  match: (pathname: string) => boolean;
};

function phaseItems(
  phase: TournamentPhase,
  context: TournamentRouteContext
): PhaseItem[] {
  if (phase === "setup") {
    return [
      {
        label: "Tournament",
        href: tournamentRouteHref("/admin/tournaments/setup/basics", context),
        match: (pathname) =>
          pathname === "/admin/tournaments/setup/basics" ||
          pathname === "/admin/tournaments/setup/schedule" ||
          pathname === "/admin/tournaments/setup/registration-rules"
      },
      {
        label: "Competition",
        href: tournamentRouteHref("/admin/tournaments/setup/events", context),
        match: (pathname) =>
          pathname === "/admin/tournaments/setup/events" ||
          pathname === "/admin/tournaments/setup/divisions" ||
          pathname === "/admin/tournaments/team-competition"
      },
      {
        label: "Commerce",
        href: tournamentRouteHref("/admin/tournaments/setup/pricing", context),
        match: (pathname) => pathname === "/admin/tournaments/setup/pricing"
      },
      {
        label: "Review",
        href: tournamentRouteHref("/admin/tournaments/setup/review", context),
        match: (pathname) => pathname === "/admin/tournaments/setup/review"
      }
    ];
  }

  if (phase === "registration") {
    return [
      {
        label: "Registration overview",
        href: tournamentRouteHref("/admin/tournaments/registration", context),
        match: (pathname) => pathname === "/admin/tournaments/registration"
      },
      {
        label: "Registrants",
        href: tournamentRouteHref("/admin/tournaments/registration/registrants", context),
        match: (pathname) =>
          pathname.startsWith("/admin/tournaments/registration/registrants")
      },
      {
        label: "Partners & teams",
        href: tournamentRouteHref("/admin/tournaments/registration/partners", context),
        match: (pathname) =>
          pathname === "/admin/tournaments/registration/partners"
      },
      {
        label: "Payments & extras",
        href: tournamentRouteHref("/admin/tournaments/commerce", context),
        match: (pathname) => pathname === "/admin/tournaments/commerce"
      },
      {
        label: "Communications & reports",
        href: tournamentRouteHref("/admin/tournaments/registrations", context),
        match: (pathname) => pathname === "/admin/tournaments/registrations"
      }
    ];
  }

  if (phase === "live") {
    return [
      {
        label: "Day workspace",
        href: tournamentRouteHref("/admin/tournaments/live-operations", context),
        match: (pathname) =>
          pathname === "/admin/tournaments/live-operations" ||
          pathname === "/admin/tournaments/live-operations/draws" ||
          pathname === "/admin/tournaments/ops/draws" ||
          pathname === "/admin/tournament-live"
      },
      {
        label: "Preflight & check-in",
        href: tournamentRouteHref("/admin/tournaments/live-operations/check-in", context),
        match: (pathname) =>
          pathname === "/admin/tournaments/live-operations/check-in"
      },
      {
        label: "Corrections & recovery",
        href: tournamentRouteHref("/admin/tournaments/live-operations/corrections", context),
        match: (pathname) => pathname === "/admin/tournaments/live-operations/corrections" || pathname === "/admin/tournaments/status"
      },
      {
        label: "Podium & awards",
        href: tournamentRouteHref("/admin/tournaments/live-operations/podium", context),
        match: (pathname) => pathname === "/admin/tournaments/live-operations/podium" || pathname === "/admin/tournaments/ops"
      }
    ];
  }

  return [
    {
      label: "Publish overview",
      href: tournamentRouteHref("/admin/tournaments/publish", context),
      match: (pathname) => pathname === "/admin/tournaments/publish"
    },
    {
      label: "Review results",
      href: tournamentRouteHref("/admin/tournaments/ops/results", context),
      match: (pathname) => pathname === "/admin/tournaments/ops/results"
    },
    {
      label: "Import results",
      href: tournamentRouteHref("/admin/tournaments/publish/import-results", context),
      match: (pathname) => pathname === "/admin/tournaments/publish/import-results"
    },
    {
      label: "Publish divisions",
      href: tournamentRouteHref("/admin/tournaments/ops/publish", context),
      match: (pathname) => pathname === "/admin/tournaments/ops/publish"
    },
    {
      label: "Tournament closeout",
      href: tournamentRouteHref("/admin/tournaments/publish/closeout", context),
      match: (pathname) =>
        pathname === "/admin/tournaments/publish/closeout"
    }
  ];
}

export default function TournamentPhaseNav({ phase }: { phase: TournamentPhase }) {
  const pathname = usePathname() || "";
  const searchParams = useSearchParams();
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) return null;

  const items = phaseItems(phase, context);
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
