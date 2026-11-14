"use client";
import Link from "next/link";
import { use } from "react";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import { useAdminSession } from "@/lib/useAdminSession";
import ManagedHome from "./managed/ManagedHome";

export default function InterclubPage({ searchParams }: { searchParams: Promise<{ season?: string }> }) {
  const params = use(searchParams);
  const { session, accessToken, loading } = useAdminSession();
  const { clubId } = useAdminWorkspace();
  const allowed = session?.capabilities?.assignments.some(a => a.club_id === clubId && ["administrator", "club_owner", "super_admin"].includes(a.role));
  if (loading) return <p>Checking club access…</p>;
  if (!allowed) return <p>Sign in as the league organizer. <Link href="/admin/login">Sign in</Link></p>;
  return <ManagedHome key={`${clubId}:${session?.user?.id || session?.user?.email}`} clubId={clubId} accessToken={accessToken} initialId={params.season || ""} />;
}
