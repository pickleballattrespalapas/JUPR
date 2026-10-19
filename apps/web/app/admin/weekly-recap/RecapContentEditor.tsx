"use client";
import type {
  BuilderConfig,
  RecapCatalog,
  RecapItem,
  SectionKey,
  SectionSetting,
} from "@/lib/recapBuilder";
import { RECAP_SECTIONS } from "@/lib/recapBuilder";
import styles from "./recap-builder.module.css";

const itemSections = new Set([
  "numbers",
  "events",
  "races",
  "achievements",
  "welcome",
]);
export function RecapContentEditor({
  builder,
  catalog,
  onChange,
  disabled,
}: {
  builder: BuilderConfig;
  catalog: RecapCatalog;
  onChange: (value: BuilderConfig) => void;
  disabled: boolean;
}) {
  function section(key: SectionKey, patch: Partial<SectionSetting>) {
    onChange({
      ...builder,
      sections: {
        ...builder.sections,
        [key]: { ...builder.sections[key], ...patch },
      },
    });
  }
  function move(key: SectionKey, delta: number) {
    const order = [...builder.order];
    const index = order.indexOf(key);
    [order[index], order[index + delta]] = [order[index + delta], order[index]];
    onChange({ ...builder, order });
  }
  function select(
    key: SectionKey,
    id: string,
    checked: boolean,
    email: boolean,
  ) {
    const config = builder.sections[key];
    const all = (catalog[key] || []).map((item) => item.id);
    const included = config.item_ids || all;
    const prop = email ? "email_item_ids" : "item_ids";
    const defaults = email
      ? included.slice(
          0,
          ({ events: 3, races: 2, achievements: 3 } as Record<string, number>)[
            key
          ] || 150,
        )
      : all;
    const current = config[prop] || defaults;
    section(key, {
      [prop]: checked
        ? [...current.filter((v) => v !== id), id]
        : current.filter((v) => v !== id),
    });
  }
  return (
    <div className={styles.sections}>
      <p>
        Choose what readers see. Game totals still cover the scope selected in
        Setup. Empty sections stay hidden.
      </p>
      {builder.order.map((key, index) => {
        const config = builder.sections[key];
        const items = catalog[key] || [];
        const selected = config.item_ids || items.map((item) => item.id);
        const emailIds =
          config.email_item_ids ||
          selected.slice(
            0,
            (
              { events: 3, races: 2, achievements: 3 } as Record<string, number>
            )[key] || 150,
          );
        const sorted = [...items].sort((a, b) => {
          const ai = selected.indexOf(a.id),
            bi = selected.indexOf(b.id);
          return (ai < 0 ? 999 : ai) - (bi < 0 ? 999 : bi);
        });
        return (
          <section className={styles.section} key={key}>
            <div className={styles.row}>
              <label className={styles.check}>
                <input
                  type="checkbox"
                  checked={config.enabled}
                  disabled={disabled}
                  onChange={(e) => section(key, { enabled: e.target.checked })}
                />
                <strong>{RECAP_SECTIONS[key]}</strong>
              </label>
              <div className={styles.actions}>
                <button
                  type="button"
                  disabled={disabled || index === 0}
                  onClick={() => move(key, -1)}
                  aria-label={`Move ${RECAP_SECTIONS[key]} up`}
                >
                  ↑
                </button>
                <button
                  type="button"
                  disabled={disabled || index === builder.order.length - 1}
                  onClick={() => move(key, 1)}
                  aria-label={`Move ${RECAP_SECTIONS[key]} down`}
                >
                  ↓
                </button>
              </div>
            </div>
            {config.enabled ? (
              <>
                <div className={styles.fields}>
                  <label>
                    Section heading
                    <input
                      value={config.title}
                      maxLength={100}
                      disabled={disabled}
                      onChange={(e) => section(key, { title: e.target.value })}
                    />
                  </label>
                  <label>
                    Detail
                    <select
                      value={config.detail}
                      disabled={disabled}
                      onChange={(e) =>
                        section(key, {
                          detail: e.target.value as "brief" | "full",
                        })
                      }
                    >
                      <option value="full">Full details</option>
                      <option value="brief">Brief highlights</option>
                    </select>
                  </label>
                </div>
                <label className={styles.check}>
                  <input
                    type="checkbox"
                    checked={config.email_enabled}
                    disabled={disabled}
                    onChange={(e) =>
                      section(key, { email_enabled: e.target.checked })
                    }
                  />
                  Include section in email
                </label>
                {itemSections.has(key) ? (
                  <details>
                    <summary>
                      {
                        selected.filter((id) => items.some((i) => i.id === id))
                          .length
                      }{" "}
                      of {items.length} items selected
                    </summary>
                    {sorted.map((item) => {
                      const position = selected.indexOf(item.id);
                      return (
                        <div className={styles.item} key={item.id}>
                          <label className={styles.check}>
                            <input
                              type="checkbox"
                              checked={selected.includes(item.id)}
                              disabled={disabled}
                              onChange={(e) =>
                                select(key, item.id, e.target.checked, false)
                              }
                            />
                            {item.title || item.text}
                            {item.kind ? ` · ${item.kind}` : ""}
                          </label>
                          <div className={styles.actions}>
                            <label className={styles.check}>
                              <input
                                type="checkbox"
                                aria-label={`Feature ${item.title || item.text} in email`}
                                checked={
                                  emailIds.includes(item.id) &&
                                  selected.includes(item.id)
                                }
                                disabled={
                                  disabled ||
                                  !config.email_enabled ||
                                  !selected.includes(item.id)
                                }
                                onChange={(e) =>
                                  select(key, item.id, e.target.checked, true)
                                }
                              />
                              Email
                            </label>
                            <button
                              type="button"
                              aria-label={`Move ${item.title || item.text} earlier`}
                              disabled={disabled || position < 1}
                              onClick={() => {
                                const ids = [...selected];
                                [ids[position - 1], ids[position]] = [
                                  ids[position],
                                  ids[position - 1],
                                ];
                                section(key, {
                                  item_ids: ids,
                                  ...(config.email_item_ids
                                    ? {
                                        email_item_ids: ids.filter((id) =>
                                          config.email_item_ids!.includes(id),
                                        ),
                                      }
                                    : {}),
                                });
                              }}
                            >
                              ↑
                            </button>
                          </div>
                        </div>
                      );
                    })}
                    {!items.length ? (
                      <p>No recorded items in this date range.</p>
                    ) : null}
                  </details>
                ) : (
                  <p className={styles.muted}>
                    {key === "spotlight"
                      ? "Choose players and email highlights in Personalize."
                      : "Edit this content in Personalize."}
                  </p>
                )}
              </>
            ) : null}
          </section>
        );
      })}
    </div>
  );
}

export function RecapStoryEditor({
  builder,
  onChange,
  disabled,
}: {
  builder: BuilderConfig;
  onChange: (value: BuilderConfig) => void;
  disabled: boolean;
}) {
  function update(id: string, patch: Partial<RecapItem>) {
    onChange({
      ...builder,
      custom_blocks: builder.custom_blocks.map((item) =>
        item.id === id ? { ...item, ...patch } : item,
      ),
    });
  }
  function add(kind: string) {
    onChange({
      ...builder,
      custom_blocks: [
        ...builder.custom_blocks,
        {
          id: crypto.randomUUID(),
          kind,
          title: kind,
          text: "",
          lines: [],
          image_url: "",
          alt: "",
          url: "",
          link_label: "",
        },
      ],
    });
  }
  return (
    <section>
      <h3>Club stories</h3>
      <div className={styles.actions}>
        {[
          "Photo of the week",
          "Coach’s tip",
          "Community recognition",
          "Sponsor thank-you",
          "Club announcement",
        ].map((kind) => (
          <button
            type="button"
            key={kind}
            disabled={disabled || builder.custom_blocks.length >= 12}
            onClick={() => add(kind)}
          >
            + {kind}
          </button>
        ))}
      </div>
      {builder.custom_blocks.map((item, index) => (
        <fieldset className={styles.section} key={item.id} disabled={disabled}>
          <legend>{item.kind}</legend>
          <label>
            Heading
            <input
              value={item.title}
              maxLength={240}
              onChange={(e) => update(item.id, { title: e.target.value })}
            />
          </label>
          <label>
            Story
            <textarea
              aria-label="Story"
              value={item.text}
              maxLength={3000}
              onChange={(e) => update(item.id, { text: e.target.value })}
            />
          </label>
          <div className={styles.fields}>
            <label>
              Photo URL
              <input
                type="url"
                placeholder="https://…"
                value={item.image_url || ""}
                onChange={(e) => update(item.id, { image_url: e.target.value })}
              />
            </label>
            <label>
              Photo description
              <input
                value={item.alt || ""}
                maxLength={240}
                onChange={(e) => update(item.id, { alt: e.target.value })}
              />
            </label>
          </div>
          <div className={styles.fields}>
            <label>
              Link URL
              <input
                type="url"
                placeholder="https://…"
                value={item.url || ""}
                onChange={(e) => update(item.id, { url: e.target.value })}
              />
            </label>
            <label>
              Link label
              <input
                value={item.link_label || ""}
                maxLength={100}
                onChange={(e) =>
                  update(item.id, { link_label: e.target.value })
                }
              />
            </label>
          </div>
          <div className={styles.actions}>
            <button
              type="button"
              disabled={index === 0}
              onClick={() => {
                const blocks = [...builder.custom_blocks];
                [blocks[index - 1], blocks[index]] = [
                  blocks[index],
                  blocks[index - 1],
                ];
                onChange({ ...builder, custom_blocks: blocks });
              }}
            >
              Move up
            </button>
            <button
              type="button"
              onClick={() =>
                onChange({
                  ...builder,
                  custom_blocks: builder.custom_blocks.filter(
                    (v) => v.id !== item.id,
                  ),
                })
              }
            >
              Remove story
            </button>
          </div>
        </fieldset>
      ))}
    </section>
  );
}
