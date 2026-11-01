import type { AdminWeeklyRecapCandidate, AdminWeeklyRecapRow } from "./adminWeeklyRecapApi";
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
  credit_label?: string;
  results_label?: string;
  links?: Array<{ label: string; href: string }>;
  court_players?: Array<{ id: string; name: string }>;
  selection_aliases?: string[];
};

export type CourtRecognition = {
  key: string;
  name: string;
  status: "winner" | "tie" | "leader";
  ids: string[];
  label: string;
};

/** Read saved court evidence too, so existing drafts need no regeneration. */
export function ladderRecognitions(items: RecapItem[]): CourtRecognition[] {
  const rows = new Map<string, CourtRecognition>();
  for (const item of items) {
    if (item.kind !== "Ladder court") continue;
    const status = item.text.startsWith("Winner:")
      ? "winner"
      : item.text.startsWith("Tied for first:")
        ? "tie"
        : "leader";
    const players = item.court_players?.length
      ? item.court_players
      : item.lines.flatMap((line) => {
          const match = line.match(/^(.*) — \d+–\d+ · point difference [+-]\d+$/);
          return match ? [{ id: "", name: match[1] }] : [];
        });
    // The summary is a safe fallback for old single-winner snapshots.
    if (!players.length && status === "winner")
      players.push({ id: "", name: item.text.slice("Winner:".length).trim() });
    for (const player of players) {
      if (!player.name) continue;
      const key = `${status}:${player.id || player.name}`;
      const row = rows.get(key) || {
        key, name: player.name, status, ids: [], label: "",
      };
      if (!row.ids.includes(item.id)) row.ids.push(item.id);
      row.label = `${row.name}${row.ids.length > 1 ? ` ×${row.ids.length}` : ""}${
        status === "tie" ? " (tied for first)" : status === "leader" ? " (incomplete court leader)" : ""
      }`;
      rows.set(key, row);
    }
  }
  return [...rows.values()];
}

export function compactLadderCourts(items: RecapItem[]): RecapItem[] {
  const courts = items.filter((item) => item.kind === "Ladder court");
  if (!courts.length) return items;
  const rows = ladderRecognitions(courts);
  // Unrecognized legacy evidence stays visible without displaying score lines.
  const recognized = new Set(rows.flatMap((row) => row.ids));
  const lines = [
    ...rows.map((row) => row.label),
    ...courts.filter((item) => !recognized.has(item.id)).map((item) => item.text || item.title),
  ];
  const summary: RecapItem = {
    id: "ladder-court-winners", title: "Ladder court winners", text: "", lines,
  };
  return items.flatMap((item) =>
    item.kind !== "Ladder court" ? [item] : item.id === courts[0].id ? [summary] : [],
  );
}
export const RECAP_TEXT_LIMITS = {
  title: 240,
  text: 3000,
  kind: 80,
  caption: 600,
  credit: 160,
  credit_label: 80,
  results_label: 100,
  link_label: 100,
  alt: 240,
} as const;
export type RecapTextField = keyof typeof RECAP_TEXT_LIMITS;
export type RecapTextOverride = Partial<Record<RecapTextField, string>> & {
  lines?: Record<string, string>;
  link_labels?: Record<string, string>;
};
export type RecapDisplayText = Partial<
  Record<"date" | "scope" | "footer_title" | "footer_link", string>
>;
const cleanText = (value: string, limit: number) =>
  value.replace(/[<>]/g, "").trim().slice(0, limit);
// Authored notes lack a source ID shared by the client and server; bind their
// edits to the original text/link so deleting or moving a note cannot swap edits.
// Generated evidence always retains its stable source ID.
export function textOverrideKey(section: SectionKey, item: RecapItem) {
  return section === "upcoming"
    ? `upcoming:${JSON.stringify([cleanText(item.text, 240), (item.url || "").trim()])}`
    : item.id;
}
export function applyTextOverride(
  item: RecapItem,
  patch?: RecapTextOverride,
): RecapItem {
  if (!patch) return item;
  const fields = Object.fromEntries(
    Object.entries(RECAP_TEXT_LIMITS).flatMap(([field, limit]) => {
      const value = patch[field as RecapTextField];
      return typeof value === "string"
        ? [[field, cleanText(value, limit)]]
        : [];
    }),
  );
  return {
    ...item,
    ...fields,
    lines: item.lines.map((line) =>
      typeof patch.lines?.[line] === "string"
        ? cleanText(patch.lines[line], 600)
        : line,
    ),
    ...(item.links
      ? {
          links: item.links.map((link) => ({
            ...link,
            label:
              typeof patch.link_labels?.[link.href] === "string"
                ? cleanText(patch.link_labels[link.href], 80)
                : link.label,
          })),
        }
      : {}),
  };
}
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
  email_order_mode?: "website" | "custom";
  sections: Record<SectionKey, SectionSetting>;
  custom_blocks: RecapItem[];
  step: number;
  text_overrides?: Record<string, RecapTextOverride>;
  display_text?: RecapDisplayText;
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
    email_order_mode: value?.email_order_mode === "custom" ? "custom" : "website",
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
    ...(value?.text_overrides ? { text_overrides: value.text_overrides } : {}),
    ...(value?.display_text ? { display_text: value.display_text } : {}),
    step: Math.min(4, Math.max(0, value?.step || 0)),
  };
}
export function layoutTemplate(config: BuilderConfig): BuilderConfig {
  const { text_overrides, display_text, ...layout } = config;
  return {
    ...layout,
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
export function resolveBuilderSelections(
  builder: BuilderConfig,
  groups: Partial<RecapCatalog>,
): BuilderConfig {
  const resolved = structuredClone(builder);
  for (const key of Object.keys(RECAP_SECTIONS) as SectionKey[]) {
    const items = groups[key] || [];
    const current = new Set(items.map((item) => item.id));
    const aliases = new Map<string, string[]>();
    for (const item of items)
      for (const alias of item.selection_aliases || [])
        aliases.set(alias, [...(aliases.get(alias) || []), item.id]);
    const settings = resolved.sections[key];
    for (const field of ["item_ids", "email_item_ids"] as const)
      if (settings[field])
        settings[field] = [...new Set(settings[field].flatMap((id) =>
          current.has(id) ? [id] : aliases.get(id) || [id],
        ))];
    const overrides = resolved.text_overrides;
    for (const [alias, targets] of aliases)
      if (targets.length === 1 && !current.has(alias) && overrides?.[alias])
        overrides[targets[0]] ??= structuredClone(overrides[alias]);
  }
  return resolved;
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object")
    return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(
      ([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`,
    ).join(",")}}`;
  return JSON.stringify(value) ?? "null";
}

/** Only retry when the saved editorial work is exactly the version we loaded. */
export function sourceOnlyRecapChange(before: AdminWeeklyRecapRow, after: AdminWeeklyRecapRow) {
  return Boolean(before.id && before.id === after.id && before.club_id === after.club_id
    && before.week_start === after.week_start && before.week_end === after.week_end
    && before.status === "draft" && after.status === "draft"
    && after.row_version > before.row_version
    && canonicalJson(before.edits_json || {}) === canonicalJson(after.edits_json || {})
    && canonicalJson(before.published_json ?? null) === canonicalJson(after.published_json ?? null));
}

export type RecapOrderAction =
  | { type: "section"; section: SectionKey; delta: -1 | 1 }
  | { type: "item"; section: SectionKey; itemId: string; delta: -1 | 1 }
  | { type: "group_events" };

/** Start a separate email order from what the administrator currently sees. */
export function setEmailOrderMode(builder: BuilderConfig, groups: RecapCatalog, mode: "website" | "custom"): BuilderConfig {
  const next = resolveBuilderSelections(builder, groups);
  if (mode === "custom" && next.email_order_mode !== "custom") {
    next.email_order = [...next.order];
    for (const key of Object.keys(RECAP_SECTIONS) as SectionKey[]) {
      const section = next.sections[key];
      const website = section.item_ids || (groups[key] || []).map(item => item.id);
      const selected = new Set(section.email_item_ids || website);
      section.email_item_ids = website.filter(id => selected.has(id));
    }
  }
  next.email_order_mode = mode;
  return next;
}

/** Move what the administrator sees, including the combined ladder winners. */
export function reorderRecapPreview(
  builder: BuilderConfig,
  groups: RecapCatalog,
  channel: "edition" | "email_edition",
  action: RecapOrderAction,
): BuilderConfig {
  const next = channel === "email_edition"
    ? setEmailOrderMode(builder, groups, "custom")
    : resolveBuilderSelections(builder, groups);
  const blocks = projectEditions(groups, next)[channel];
  if (action.type === "section") {
    const index = blocks.findIndex((block) => block.key === action.section);
    const neighbor = blocks[index + action.delta];
    if (index < 0 || !neighbor) return builder;
    const order = [...(channel === "email_edition" ? next.email_order || next.order : next.order)];
    const from = order.indexOf(action.section), to = order.indexOf(neighbor.key);
    [order[from], order[to]] = [order[to], order[from]];
    if (channel === "email_edition") next.email_order = order;
    else next.order = order;
    return next;
  }
  const section = action.type === "group_events" ? "events" : action.section;
  const block = blocks.find((entry) => entry.key === section);
  if (!block || section === "upcoming") return builder;
  const ordered = [...block.items];
  if (action.type === "item") {
    const index = ordered.findIndex((item) => item.id === action.itemId);
    const to = index + action.delta;
    if (index < 0 || to < 0 || to >= ordered.length) return builder;
    [ordered[index], ordered[to]] = [ordered[to], ordered[index]];
  } else {
    // Keep the first group's position and the order within each group.
    const byKind = new Map<string, RecapItem[]>();
    for (const item of ordered) {
      const kind = item.id === "ladder-court-winners" ? "Ladder court" : item.kind || "Other";
      byKind.set(kind, [...(byKind.get(kind) || []), item]);
    }
    ordered.splice(0, ordered.length, ...[...byKind.values()].flat());
  }
  const setting = next.sections[section];
  const website = setting.item_ids || (groups[section] || []).map((item) => item.id);
  const email = (setting.email_item_ids || website)
    .filter((id) => website.includes(id));
  const selected = channel === "email_edition" ? email : website;
  const courts = new Set((groups[section] || []).filter((item) => item.kind === "Ladder court").map((item) => item.id));
  const visible = ordered.flatMap((item) => item.id === "ladder-court-winners"
    ? selected.filter((id) => courts.has(id)) : [item.id]);
  const order = [...new Set([...visible, ...selected.filter((id) => !visible.includes(id))])];
  if (channel === "email_edition") setting.email_item_ids = order;
  else {
    setting.email_item_ids ??= email;
    setting.item_ids = order;
  }
  return next;
}

export function projectEditions(
  groups: RecapCatalog,
  builder: BuilderConfig,
): { edition: RecapBlock[]; email_edition: RecapBlock[] } {
  builder = resolveBuilderSelections(builder, groups);
  const result = {
    edition: [] as RecapBlock[],
    email_edition: [] as RecapBlock[],
  };
  for (const channel of ["edition", "email_edition"] as const)
    for (const key of channel === "email_edition" && builder.email_order_mode === "custom"
      ? builder.email_order || builder.order
      : builder.order) {
      const setting = builder.sections[key];
      if (
        !setting.enabled ||
        (channel === "email_edition" && !setting.email_enabled)
      )
        continue;
      const available = new Map(
        (groups[key] || []).map((item) => [
          item.id,
          item.kind === "Ladder court" ? item : applyTextOverride(
            item,
            builder.text_overrides?.[textOverrideKey(key, item)],
          ),
        ]),
      );
      let ids = setting.item_ids || [...available.keys()];
      if (channel === "email_edition") {
        const allowed = new Set(ids);
        const selected = setting.email_item_ids || ids;
        ids = builder.email_order_mode === "custom"
          ? selected.filter(id => allowed.has(id))
          : ids.filter(id => selected.includes(id));
      }
      const selectedItems = ids.flatMap((id) => {
        const item = available.get(id);
        if (!item) return [];
        const lines = (
          setting.detail === "brief" && key !== "spotlight" && item.kind !== "Ladder court"
            ? item.lines.slice(0, 1)
            : item.lines
        ).filter(Boolean);
        return item.title || item.text || lines.length || item.image_url
          ? [{ ...item, lines }]
          : [];
      });
      const items = key === "events"
        ? compactLadderCourts(selectedItems).map((item) =>
            item.id === "ladder-court-winners"
              ? applyTextOverride(item, builder.text_overrides?.[item.id])
              : item,
          )
        : selectedItems;
      if (items.length)
        result[channel].push({
          key,
          title: setting.title ?? RECAP_SECTIONS[key],
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
    ...edits.editorial.upcoming
      .slice(0, 10)
      .filter((v) => v.text.trim())
      .map((v, i) => ({
        id: `link-${i}`,
        title: "",
        text: v.text,
        url: v.url,
        link_label: v.text,
        lines: [],
      })),
    ...edits.looking_ahead
      .filter((v) => v.trim())
      .map((v, i) => ({ id: `ahead-${i}`, title: "", text: v, lines: [] })),
  ];
  return {
    ...generated,
    builder_catalog: groups,
    report: {
      ...generated.report,
      ...edits.editorial,
      ...(edits.builder.display_text
        ? {
            display_text: Object.fromEntries(
              Object.entries(edits.builder.display_text).map(([key, value]) => [
                key,
                cleanText(value, 160),
              ]),
            ),
          }
        : {}),
      ...projectEditions(groups, edits.builder),
    },
  };
}
