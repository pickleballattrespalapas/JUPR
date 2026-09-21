import { getAdminApiBaseUrl } from "@/lib/adminAuthClient";

export type NotificationState = "new" | "flagged" | "cleared";
export type AdminNotificationCategory = {
  key: string;
  label: string;
  description: string;
  href: string;
  kind: "action" | "activity";
  enabled: boolean;
  status: "ready" | "unavailable";
  total_count: number | null;
};
export type AdminNotification = {
  key: string;
  category: string;
  kind: "action" | "activity";
  title: string;
  description: string;
  href: string;
  occurred_at: string | null;
  state: NotificationState;
};
export type AdminNotifications = {
  club_id: string;
  checked_at: string;
  history_days: number;
  categories: AdminNotificationCategory[];
  items: AdminNotification[];
  truncated: boolean;
};
export type NotificationResult = { data: AdminNotifications | null; error: string | null; status: number | null };

async function request(accessToken: string, clubId: string, suffix = "", body?: unknown): Promise<NotificationResult> {
  if (!accessToken) return { data: null, error: "Sign in to view notifications.", status: 401 };
  const api = getAdminApiBaseUrl();
  if (!api) return { data: null, error: "Notifications are unavailable. Please try again.", status: null };
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000);
  try {
    const response = await fetch(`${api.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/notifications${suffix}`, {
      method: body === undefined ? "GET" : "PUT",
      cache: "no-store",
      headers: { accept: "application/json", Authorization: `Bearer ${accessToken}`, ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      signal: controller.signal
    });
    if (!response.ok) return { data: null, error: response.status === 409
      ? "This notice has changed. Refresh notifications and try again."
      : body === undefined ? "We couldn’t check notifications. Please try again." : "Your change couldn’t be confirmed. Refresh to check its status, then try again.", status: response.status };
    const data = await response.json() as AdminNotifications;
    if (data.club_id !== clubId || !Array.isArray(data.categories) || !Array.isArray(data.items)) throw new Error("Unexpected club response");
    // Notification destinations are internal admin routes, including persisted flags.
    if ([...data.categories, ...data.items].some(item => typeof item.href !== "string" || !item.href.startsWith("/admin/") || /[\\\r\n]/.test(item.href))) throw new Error("Unexpected notification destination");
    return { data, error: null, status: response.status };
  } catch {
    return { data: null, error: body === undefined ? "We couldn’t check notifications. Please try again." : "Your change couldn’t be confirmed. Refresh to check its status, then try again.", status: null };
  } finally {
    clearTimeout(timeout);
  }
}

export const getAdminNotifications = (accessToken: string, clubId: string) => request(accessToken, clubId);
export const updateAdminNotificationPreferences = (accessToken: string, clubId: string, categories: Record<string, boolean>) => request(accessToken, clubId, "/preferences", { categories });
export const updateAdminNotificationState = (accessToken: string, clubId: string, key: string, state: NotificationState) => request(accessToken, clubId, `/items/${encodeURIComponent(key)}`, { state });
