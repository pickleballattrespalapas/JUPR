"use client";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useAdminSession } from "@/lib/useAdminSession";
import { useAvailableWorkspaces } from "@/lib/useAvailableWorkspaces";
import { canChooseAdminWorkspace, selectAdminWorkspace, type AvailableWorkspace } from "@/lib/adminWorkspace";
import ClubWorkspaceCards from "./ClubWorkspaceCards";
import styles from "./selector.module.css";

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
  return <section className={styles.page}>
    <h1>{canChoose ? "Choose your club" : "Your club"}</h1>
    <p>{canChoose ? "Select a club below to open its players and programs." : "Your club is linked to your sign-in."}</p>
    {opening && <p role="status">Opening your club…</p>}
    {error || selectionError ? <p className={styles.error} role="alert">{error || selectionError} {error && <button type="button" className={styles.retry} onClick={retry}>Try again</button>}</p> : null}
    {!loaded && <p role="status">Loading your clubs…</p>}
    {loaded && !error && !workspaces.length && <p>No club workspace is available. Ask a club administrator to check your assignment.</p>}
    <ClubWorkspaceCards workspaces={workspaces} opening={opening} onOpen={open} />
    <p><Link href="/admin/login">Manage sign-in</Link></p>
  </section>;
}
