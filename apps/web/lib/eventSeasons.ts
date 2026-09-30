export type EventKind = "interclub" | "league" | "tournament";
export type EventSeason = {
  source_id: string; name: string; label: string; position: number;
  start_date: string | null; end_date: string | null; status: string;
  complete: boolean; public: boolean; selected?: boolean;
  league_type?: string | null;
  results_href: string | null; admin_href?: string; fingerprint?: string;
  honors: { id: string; title: string; recipient: string; placement: number; record?: string }[];
};
export type EventHistory = {
  kind: EventKind; series_id: string | null; series_name: string; seasons: EventSeason[];
};
export type AdminEventHistory = EventHistory & {
  current: EventSeason & { fingerprint: string; admin_href: string };
  current_label: string; can_start: boolean; reason: string;
  past_candidates: { source_id: string; name: string }[];
};
