import type { AdminNotification, AdminNotifications } from "./adminNotificationsApi";

export type NotificationNavigationGroup = {
  label: string;
  links: { href: string; active: (pathname: string) => boolean; newTab?: boolean }[];
};

export function notificationNavigation(data: AdminNotifications | null, groups: NotificationNavigationGroup[]) {
  const byLink: Record<string, AdminNotification[]> = {};
  const byGroup: Record<string, AdminNotification[]> = {};
  const enabled = new Set(data?.categories.filter(category => category.enabled).map(category => category.key));
  const items = (data?.items || []).filter(item => item.state !== "cleared" && enabled.has(item.category));
  const unique = [...new Map(items.map(item => [item.key, item])).values()];
  const destinations = groups.flatMap(group => group.links)
    .filter(link => !link.newTab && link.href !== "/admin/notifications")
    .sort((left, right) => right.href.length - left.href.length);

  for (const item of unique) {
    const pathname = item.href.split(/[?#]/, 1)[0];
    // Prefer the most specific destination; active() includes existing route
    // aliases such as /admin/tournament-* and /admin/badges.
    const destination = destinations.find(link => pathname === link.href ||
      (link.href !== "/admin" && pathname.startsWith(`${link.href}/`)) ||
      (!pathname.startsWith(link.href) && link.active(pathname)));
    if (destination) (byLink[destination.href] ||= []).push(item);
  }
  byLink["/admin/notifications"] = unique;
  for (const group of groups) {
    // Notifications is a global inbox, not a second source of Workspace work.
    const groupItems = group.links.filter(link => link.href !== "/admin/notifications")
      .flatMap(link => byLink[link.href] || []);
    byGroup[group.label] = [...new Map(groupItems.map(item => [item.key, item])).values()];
  }
  return { byLink, byGroup };
}

export function notificationSummary(items: AdminNotification[]): string {
  const titles = items.slice(0, 4).map(item => item.title);
  if (items.length > titles.length) titles.push(`${items.length - titles.length} more`);
  return `${items.length} active notification${items.length === 1 ? "" : "s"}: ${titles.join("; ")}`;
}
