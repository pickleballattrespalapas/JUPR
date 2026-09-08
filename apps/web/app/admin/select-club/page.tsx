"use client";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useAdminSession } from "@/lib/useAdminSession";
import { useAvailableWorkspaces } from "@/lib/useAvailableWorkspaces";
import { canChooseAdminWorkspace, selectAdminWorkspace, type AvailableWorkspace } from "@/lib/adminWorkspace";

export default function SelectClubPage() {
  const { session, accessToken, loading, message } = useAdminSession();
  const { workspaces, error, loaded, retry } = useAvailableWorkspaces(accessToken, session?.user?.id || session?.user?.email || accessToken);
  const [opening, setOpening] = useState("");
  const [selectionError, setSelectionError] = useState("");
  const canChoose = canChooseAdminWorkspace(workspaces);
  const open = useCallback((workspace: AvailableWorkspace) => {
    if (opening) return;
    setOpening(workspace.club_id); setSelectionError("");
    try { selectAdminWorkspace(workspace); }
    catch (error) { setSelectionError(error instanceof Error ? error.message : "Unable to open club."); setOpening(""); }
  }, [opening]);
  useEffect(() => {
    if (!loading && accessToken && loaded && !error && !selectionError && !canChoose && workspaces.length === 1) {
      open(workspaces[0]);
    }
  }, [loading, accessToken, loaded, error, selectionError, canChoose, workspaces, open]);
  if (loading) return <p role="status">Checking your access…</p>;
  if (!accessToken) return <section><h1>Club workspace</h1><p>{message || "Sign in to open your clubs."}</p><Link href="/admin/login?next=/admin/select-club">Sign in</Link></section>;
  return <section style={{ maxWidth: 780, margin: "0 auto", padding: 24 }}>
    <h1>{canChoose ? "Choose your club" : "Your club"}</h1>
    <p>{canChoose ? "Open a club to manage its players and programs." : "Your club is linked to your sign-in."}</p>
    {opening && <p role="status">Opening your club…</p>}
    {error || selectionError ? <p role="alert">{error || selectionError} {error && <button onClick={retry}>Try again</button>}</p> : null}
    {!loaded && <p role="status">Loading your clubs…</p>}
    {loaded && !error && !workspaces.length && <p>No club workspace is available. Ask a club administrator to check your assignment.</p>}
    <div style={{ display: "grid", gap: 16 }}>
      {workspaces.map(workspace => <article key={workspace.club_id} style={{ border: "1px solid #cbd5e1", borderRadius: 12, padding: 20 }}>
        <h2 style={{ marginTop: 0 }}>{workspace.club_name}</h2>
        <p>{workspace.roles.map(role => role.replaceAll("_", " ")).join(", ")}</p>
        <button disabled={Boolean(opening)} onClick={() => open(workspace)}>{opening === workspace.club_id ? "Opening…" : `Open ${workspace.club_name}`}</button>
      </article>)}
    </div>
    <p><Link href="/admin/platform">PCS administration</Link> · <Link href="/admin/login">Manage sign-in</Link></p>
  </section>;
}
