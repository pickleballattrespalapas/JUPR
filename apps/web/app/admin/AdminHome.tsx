"use client";

import Link from "next/link";
import type { AdminSession } from "@/lib/adminAuthClient";
import AdminNotificationCenter from "@/components/AdminNotificationCenter";
import { useAdminSession } from "@/lib/useAdminSession";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import styles from "./AdminHome.module.css";

const quickActions = [
  { label: "Start a round robin", openLabel: "Open round robin", program: "round_robin", description: "Build teams, play rounds, and record scores.", href: "/admin/round-robin-generator", permission: "manage_matches" },
  { label: "Start a ladder", openLabel: "Open ladder", program: "ladder", description: "Set up courts and move players between rounds.", href: "/admin/ladder-generator", permission: "manage_matches" },
  { label: "Manage tournaments", openLabel: "Manage tournaments", program: "tournaments", description: "Registration, draws, and live play.", href: "/admin/tournaments", permission: "manage_tournaments" },
  { label: "Manage leagues", openLabel: "Manage leagues", program: "leagues", description: "Players, sessions, and league results.", href: "/admin/league-manager", permission: "manage_matches" }
];

function ClubDashboard({ accessToken, session, clubId, clubSlug }: {
  accessToken: string; session: AdminSession; clubId: string; clubSlug: string;
}) {
  const assignments = session.capabilities?.assignments.filter(item => item.club_id === clubId) ?? [];
  const permissions = new Set(assignments.flatMap(item => item.permissions));
  const isAdmin = assignments.some(item => ["administrator", "club_owner", "super_admin"].includes(item.role));
  const can = (permission: string) => isAdmin || permissions.has(permission);
  const operatorScopes = assignments.filter(item => item.role === "operator").flatMap(item => item.scopes ?? []);
  const availableActions = quickActions.filter(action => isAdmin
    || assignments.some(item => item.role !== "operator" && item.permissions.includes(action.permission))
    || operatorScopes.some(scope => scope.kind === "club" || scope.program_type === action.program)
  ).map(action => {
    const canCreate = isAdmin
      || assignments.some(item => item.role !== "operator" && item.permissions.includes(action.permission))
      || operatorScopes.some(scope => scope.kind === "club" || (scope.kind === "program_type" && scope.program_type === action.program));
    return { ...action, label: canCreate ? action.label : action.openLabel,
      description: canCreate ? action.description : "Open your assigned programs and continue play." };
  });

  return (
    <section className={styles.home}>
      <header className={styles.header}>
        <div>
          <h1>Admin Home</h1>
          <p>Review pending work and get your club ready for play.</p>
        </div>
        <Link className={styles.publicLink} href={`/clubs/${encodeURIComponent(clubSlug)}`} target="_blank" rel="noreferrer">View club website ↗</Link>
      </header>

      <AdminNotificationCenter accessToken={accessToken} clubId={clubId} compact />

      {availableActions.length > 0 ? <section aria-labelledby="play-heading">
        <h2 id="play-heading">Run play</h2>
        <div className={styles.actions}>{availableActions.map(action => <Link key={action.href} href={action.href} className={styles.action}>
          <strong>{action.label}<span aria-hidden="true">↗</span></strong><span>{action.description}</span>
        </Link>)}</div>
      </section> : null}
      {isAdmin || can("manage_players") ? <section className={styles.desk} aria-labelledby="desk-heading">
        <h2 id="desk-heading">Club desk</h2>
        <div className={styles.deskLinks}>
          {can("manage_players") ? <Link href="/admin/players">Find or add a player</Link> : null}
          {isAdmin ? <><Link href="/admin/match-uploader">Enter match results</Link><Link href="/admin/match-log">Review match history</Link></> : null}
        </div>
      </section> : null}
    </section>
  );
}

export default function AdminHome() {
  const { session, accessToken, loading: sessionLoading, message } = useAdminSession();
  const { clubId, clubSlug } = useAdminWorkspace();
  if (sessionLoading || !accessToken || !session) {
    return <section><h1>{sessionLoading ? "Checking admin access…" : "Admin sign-in required"}</h1>
      {sessionLoading ? <p role="status">Checking your staff access…</p> : <><p><Link href="/admin/login">Sign in</Link> to open your club workspace.</p>{message ? <p role="alert">{message}</p> : null}</>}
    </section>;
  }
  // Remount on token or club changes so no prior club's counts can appear,
  // including the render before an effect has a chance to clear old data.
  return <ClubDashboard key={`${accessToken}\u0000${clubId}`} accessToken={accessToken} session={session} clubId={clubId} clubSlug={clubSlug} />;
}
