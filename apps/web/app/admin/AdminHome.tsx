"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { clearAdminSession, type AdminSession } from "@/lib/adminAuthClient";
import { getAdminDashboard, type AdminDashboard } from "@/lib/adminDashboardApi";
import { useAdminSession } from "@/lib/useAdminSession";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
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
  const [data, setData] = useState<AdminDashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const inFlight = useRef(false);
  const mounted = useRef(true);
  const request = useLatestRequestGuard(`${accessToken}\u0000${clubId}`);

  async function loadDashboard() {
    if (inFlight.current) return;
    inFlight.current = true;
    const generation = request.begin();
    setLoading(true);
    setError(null);
    try {
      const result = await getAdminDashboard(accessToken, clubId);
      if (!mounted.current || !request.isCurrent(generation)) return;
      if (result.status === 401 || result.status === 403) {
        setData(null);
        clearAdminSession();
        return;
      }
      if (!result.data) {
        setData(null);
        setError(result.error || "We couldn’t check pending work. Please try again.");
        return;
      }
      setData(result.data);
    } catch {
      if (mounted.current && request.isCurrent(generation)) {
        setData(null);
        setError("We couldn’t check pending work. Please try again.");
      }
    } finally {
      inFlight.current = false;
      if (mounted.current && request.isCurrent(generation)) setLoading(false);
    }
  }

  useAuthenticatedAutoLoad(accessToken, loadDashboard, clubId);
  const refreshRef = useRef(loadDashboard);
  refreshRef.current = loadDashboard;
  useEffect(() => {
    mounted.current = true;
    const refresh = () => {
      if (document.visibilityState === "visible") void refreshRef.current();
    };
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", refresh);
    const timer = window.setInterval(refresh, 60_000);
    return () => {
      mounted.current = false;
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", refresh);
      window.clearInterval(timer);
    };
  }, [request]);

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
  const pending = data?.queues.filter(queue => queue.status === "ready" && Number(queue.count) > 0) ?? [];
  const unavailable = data?.queues.filter(queue => queue.status === "unavailable") ?? [];
  const ready = data?.queues.filter(queue => queue.status === "ready") ?? [];
  const total = pending.reduce((sum, queue) => sum + Number(queue.count), 0);

  return (
    <section className={styles.home}>
      <header className={styles.header}>
        <div>
          <h1>Admin Home</h1>
          <p>Review pending work and get your club ready for play.</p>
        </div>
        <Link className={styles.publicLink} href={`/clubs/${encodeURIComponent(clubSlug)}`} target="_blank" rel="noreferrer">View club website ↗</Link>
      </header>

      <section className={`${styles.attention} ${total > 0 ? styles.hasPending : ""}`} aria-labelledby="attention-heading" aria-busy={loading}>
        <div className={styles.sectionHeader}>
          <div>
            <p className={styles.eyebrow}>Notifications</p>
            <h2 id="attention-heading">Needs attention</h2>
          </div>
          <button type="button" className={styles.refresh} disabled={loading} onClick={() => void loadDashboard()}>{loading ? "Checking…" : "Refresh"}</button>
        </div>
        <p role="status" aria-live="polite" className={styles.summary}>
          {loading && !data ? "Checking for pending work…" : error ? "Notifications unavailable" : total > 0
            ? `${total}${unavailable.length ? "+" : ""} ${total === 1 && !unavailable.length ? "item needs" : "items need"} attention`
            : unavailable.length ? "Some notifications couldn’t be checked" : data?.queues.length
              ? "No pending work in the checked queues" : "No review queues are available for your role"}
        </p>
        {error ? <div role="alert" className={styles.warning}><p>{error}</p><p>Your club tools are still available below and in the menu.</p></div> : null}
        {pending.length > 0 ? <ul className={styles.queueList}>
          {pending.map(queue => <li key={queue.key}>
            <Link href={queue.href} className={styles.queueLink}>
              <span className={styles.count}>{queue.count}</span>
              <span className={styles.queueText}><strong>{queue.label}</strong><span>{queue.description}</span></span>
              <span aria-hidden="true" className={styles.arrow}>→</span>
            </Link>
          </li>)}
        </ul> : null}
        {unavailable.length > 0 ? <div role="alert" className={styles.warning}>
          <p>We couldn’t check {unavailable.length === 1 ? "this queue" : "these queues"}. Open {unavailable.length === 1 ? "it" : "them"} to review pending work, or try Refresh.</p>
          <ul>{unavailable.map(queue => <li key={queue.key}><Link href={queue.href}>{queue.label}</Link></li>)}</ul>
        </div> : null}
        {data && ready.length > 0 ? <details className={styles.checked}>
          <summary>{ready.length} {ready.length === 1 ? "queue" : "queues"} checked · See all review queues</summary>
          <ul>{ready.map(queue => <li key={queue.key}><Link href={queue.href}>{queue.label}</Link><span>{queue.count} pending</span></li>)}</ul>
        </details> : null}
        {data ? <p className={styles.updated}>{loading ? "Updating notifications…" : `Checked at ${new Date(data.checked_at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`} · Refreshes automatically while this page is open.</p> : null}
      </section>

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
          {isAdmin ? <><Link href="/admin/weekly-recap">Prepare a weekly recap</Link><Link href="/admin/website">Update club website</Link><Link href="/admin/staff">Manage staff</Link></> : null}
          {isAdmin ? <Link href="/admin/interclub">Interclub seasons</Link> : null}
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
