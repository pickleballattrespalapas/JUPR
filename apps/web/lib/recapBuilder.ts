import type { AdminWeeklyRecapCandidate } from "./adminWeeklyRecapApi";
import type { WeeklyRecapJson } from "./weeklyRecapApi";

export const RECAP_SECTIONS = {
  opening: "From the courts",
  numbers: "The week in numbers",
  events: "Event results",
  spotlight: "Player highlights",
  races: "Season awards race",
  achievements: "New achievements",
  welcome: "Welcome to the courts",
  notes: "Around the club",
  custom: "Club stories",
  upcoming: "Coming up",
} as const;
export type SectionKey = keyof typeof RECAP_SECTIONS;
export type RecapItem = {
  id: string;
  title: string;
  text: string;
  lines: string[];
  kind?: string;
  date?: string;
  href?: string;
  url?: string;
  image_url?: string;
  alt?: string;
  link_label?: string;
  layout?: "plain" | "photo" | "callout" | "promotion";
  image_fit?: "contain" | "cover";
  caption?: string;
  credit?: string;
  links?: Array<{ label: string; href: string }>;
};
export type RecapBlock = { key: SectionKey; title: string; items: RecapItem[] };
export type RecapCatalog = Record<SectionKey, RecapItem[]>;
export type SectionSetting = {
  enabled: boolean;
  email_enabled: boolean;
  title: string;
  detail: "brief" | "full";
  item_ids?: string[];
  email_item_ids?: string[];
};
export type BuilderConfig = {
  version: 1;
  order: SectionKey[];
  email_order?: SectionKey[];
  sections: Record<SectionKey, SectionSetting>;
  custom_blocks: RecapItem[];
  step: number;
};
export type RecapTemplate = {
  id: string;
  name: string;
  builder: BuilderConfig;
};
export type RecapPreferences = {
  revision: number;
  settings: { templates: RecapTemplate[]; default_template_id: string };
  leagues: string[];
};
export type Editorial = {
  headline: string;
  opening: string;
  event_notes: string;
  upcoming: Array<{ text: string; url: string }>;
};
export type SpotlightEdit = {
  include: boolean;
  order: string;
  description: string;
  players: string[];
};
export type RecapEdits = {
  editorial: Editorial;
  spotlight_overrides: Record<
    string,
    { include: boolean; order: number; description: string; players: string[] }
  >;
  looking_ahead: string[];
  builder: BuilderConfig;
  scope: { league_names: string[] };
};
const emailLimits: Partial<Record<SectionKey, number>> = {
  events: 3,
  spotlight: 2,
  races: 2,
  achievements: 3,
};
export function normalizeBuilder(
  value?: Partial<BuilderConfig> | null,
): BuilderConfig {
  const keys = Object.keys(RECAP_SECTIONS) as SectionKey[];
  const order = [
    ...new Set([
      ...(value?.order || []).filter((k) => keys.includes(k)),
      ...keys,
    ]),
  ];
  return {
    version: 1,
    order,
    ...(Array.isArray(value?.email_order)
      ? {
          email_order: [
            ...new Set([
              ...value.email_order.filter((k) => keys.includes(k)),
              ...order,
            ]),
          ],
        }
      : {}),
    sections: Object.fromEntries(
      keys.map((key) => [
        key,
        {
          enabled: true,
          email_enabled: true,
          title: RECAP_SECTIONS[key],
          detail: "full",
          ...(value?.sections?.[key] || {}),
        },
      ]),
    ) as Record<SectionKey, SectionSetting>,
    custom_blocks: value?.custom_blocks || [],
    step: Math.min(4, Math.max(0, value?.step || 0)),
  };
}
export function layoutTemplate(config: BuilderConfig): BuilderConfig {
  return {
    ...config,
    step: 0,
    custom_blocks: [],
    sections: Object.fromEntries(
      config.order.map((key) => {
        const { item_ids, email_item_ids, ...section } = config.sections[key];
        return [
          key,
          ["numbers", "spotlight"].includes(key)
            ? config.sections[key]
            : section,
        ];
      }),
    ) as Record<SectionKey, SectionSetting>,
  };
}
export function projectEditions(
  groups: RecapCatalog,
  builder: BuilderConfig,
): { edition: RecapBlock[]; email_edition: RecapBlock[] } {
  const result = {
    edition: [] as RecapBlock[],
    email_edition: [] as RecapBlock[],
  };
  for (const channel of ["edition", "email_edition"] as const)
    for (const key of channel === "email_edition"
      ? builder.email_order || builder.order
      : builder.order) {
      const setting = builder.sections[key];
      if (
        !setting.enabled ||
        (channel === "email_edition" && !setting.email_enabled)
      )
        continue;
      const available = new Map(
        (groups[key] || []).map((item) => [item.id, item]),
      );
      let ids = setting.item_ids || [...available.keys()];
      if (channel === "email_edition") {
        const allowed = new Set(ids);
        ids = (
          setting.email_item_ids || ids.slice(0, emailLimits[key] || 150)
        ).filter((id) => allowed.has(id));
      }
      const items = ids.flatMap((id) => {
        const item = available.get(id);
        if (!item) return [];
        const lines =
          setting.detail === "brief" && key !== "spotlight"
            ? item.lines.slice(0, 1)
            : item.lines;
        return item.title || item.text || lines.length || item.image_url
          ? [{ ...item, lines }]
          : [];
      });
      if (items.length)
        result[channel].push({
          key,
          title: setting.title || RECAP_SECTIONS[key],
          items,
        });
    }
  return result;
}
export function previewRecap(
  generated: Partial<WeeklyRecapJson>,
  edits: RecapEdits,
  candidates: Record<string, AdminWeeklyRecapCandidate[]>,
): Partial<WeeklyRecapJson> {
  const groups = structuredClone(
    generated.builder_catalog || {},
  ) as RecapCatalog;
  for (const key of Object.keys(RECAP_SECTIONS) as SectionKey[])
    groups[key] ||= [];
  groups.opening = [
    {
      id: groups.opening[0]?.id || "opening",
      title: "",
      text: edits.editorial.opening,
      lines: [],
    },
  ];
  groups.notes = [
    {
      id: groups.notes[0]?.id || "notes",
      title: "",
      text: edits.editorial.event_notes,
      lines: [],
    },
  ];
  groups.spotlight = Object.entries(edits.spotlight_overrides)
    .sort((a, b) => a[1].order - b[1].order)
    .flatMap(([key, selection]) => {
      if (!selection.include) return [];
      const selected = [...new Set(selection.players.filter(Boolean))].flatMap(
        (id) => (candidates[key] || []).filter((c) => c.candidate_id === id),
      );
      if (!selected.length) return [];
      return [
        {
          id:
            groups.spotlight.find((item) => item.kind === key)?.id ||
            `spotlight-${key}`,
          kind: key,
          title: selected[0].label,
          text: selection.description,
          lines: selected.map((c) => c.display),
        },
      ];
    });
  groups.custom = edits.builder.custom_blocks;
  groups.upcoming = [
    ...edits.editorial.upcoming.map((v, i) => ({
      id: `link-${i}`,
      title: "",
      text: v.text,
      url: v.url,
      link_label: v.text,
      lines: [],
    })),
    ...edits.looking_ahead
      .filter(Boolean)
      .map((v, i) => ({ id: `ahead-${i}`, title: "", text: v, lines: [] })),
  ];
  return {
    ...generated,
    builder_catalog: groups,
    report: {
      ...generated.report,
      ...edits.editorial,
      ...projectEditions(groups, edits.builder),
    },
  };
}
