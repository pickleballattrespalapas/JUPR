import type { AdminNotifications } from "./adminNotificationsApi";

type Subscriber = {
  accessToken: string;
  clubId: string;
  listener: (data: AdminNotifications) => void;
};

// Keep scoped responses inside this module, never in DOM events or storage.
// No response cache survives a sidebar unmount or account/workspace change.
const subscribers = new Set<Subscriber>();

export function subscribeAdminNotifications(
  accessToken: string,
  clubId: string,
  listener: (data: AdminNotifications) => void
): () => void {
  const subscriber = { accessToken, clubId, listener };
  subscribers.add(subscriber);
  return () => { subscribers.delete(subscriber); };
}

export function publishAdminNotifications(accessToken: string, clubId: string, data: AdminNotifications): void {
  if (data.club_id !== clubId) return;
  for (const subscriber of subscribers) {
    if (subscriber.accessToken === accessToken && subscriber.clubId === clubId) subscriber.listener(data);
  }
}
