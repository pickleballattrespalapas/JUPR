export type WeeklyHighlightCard = {
  id: string;
  category: string;
  title: string;
  name: string;
  metric: string;
  detail: string;
  player_ids: number[];
  value: number;
  sample: number;
};

export type WeeklyHighlightSelection = {
  week_num: number | null;
  mode: "automatic" | "curated";
  cards: WeeklyHighlightCard[];
};

export type WeeklyHighlightsEditorData = WeeklyHighlightSelection & {
  revision: number;
  updated_at?: string | null;
  selected_ids: string[];
  automatic_cards: WeeklyHighlightCard[];
  max_highlights: number;
  available: boolean;
  warnings: string[];
  unavailable_ids: string[];
  categories: Array<{
    id: string;
    title: string;
    qualification: string;
    candidates: WeeklyHighlightCard[];
  }>;
};
