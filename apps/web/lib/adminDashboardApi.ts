import { getAdminApiBaseUrl } from "@/lib/adminAuthClient";

export type AdminDashboardQueue = {
  key: string;
  label: string;
  description: string;
  href: string;
  count: number | null;
  status: "ready" | "unavailable";
};

export type AdminDashboard = {
  club_id: string;
  checked_at: string;
  queues: AdminDashboardQueue[];
};

export async function getAdminDashboard(accessToken: string, clubId: string): Promise<{
  data: AdminDashboard | null; error: string | null; status: number | null;
}> {
  if (!accessToken) return { data: null, error: "Sign in to view your club’s pending work.", status: 401 };
  const api = getAdminApiBaseUrl();
  if (!api) return { data: null, error: "Notifications are unavailable. Please try again.", status: null };
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20_000);
  try {
    const response = await fetch(`${api.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/dashboard`, {
      cache: "no-store",
      headers: { accept: "application/json", Authorization: `Bearer ${accessToken}` },
      signal: controller.signal
    });
    if (!response.ok) return { data: null, error: "We couldn’t check pending work. Please try again.", status: response.status };
    const data = await response.json() as AdminDashboard;
    if (data.club_id !== clubId || !Array.isArray(data.queues)) throw new Error("Unexpected club response");
    return { data, error: null, status: response.status };
  } catch {
    return { data: null, error: "We couldn’t check pending work. Please try again.", status: null };
  } finally {
    clearTimeout(timeout);
  }
}
