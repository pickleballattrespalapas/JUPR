"use client";

import Link from "next/link";
import AdminNotificationCenter from "@/components/AdminNotificationCenter";
import { useAdminSession } from "@/lib/useAdminSession";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";

export default function AdminNotificationsPage() {
  const { session, accessToken, loading, message } = useAdminSession();
  const { clubId } = useAdminWorkspace();
  if (loading) return <section><h1>Notifications</h1><p role="status">Checking your staff access…</p></section>;
  if (!accessToken || !session) return <section><h1>Admin sign-in required</h1><p><Link href="/admin/login">Sign in</Link> to view your notifications.</p>{message ? <p role="alert">{message}</p> : null}</section>;
  return <div style={{ maxWidth: 1100, minWidth: 0 }}><h1>Notification center</h1><p>Follow club activity and keep track of work that matters to you.</p><AdminNotificationCenter accessToken={accessToken} clubId={clubId} /></div>;
}
