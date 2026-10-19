"use client";
import type {
  BuilderConfig,
  RecapCatalog,
  RecapItem,
  SectionKey,
  SectionSetting,
} from "@/lib/recapBuilder";
import { RECAP_SECTIONS } from "@/lib/recapBuilder";
import RecapEdition from "@/components/RecapEdition";
import styles from "./recap-builder.module.css";

const itemSections = new Set([
  "numbers",
  "events",
  "races",
  "achievements",
  "welcome",
  "custom",
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
      <RecapEmailOrder
        builder={builder}
        catalog={catalog}
        onChange={onChange}
        disabled={disabled}
      />
    </div>
  );
}

function move<T>(values: T[], index: number, delta: number): T[] {
  const result = [...values];
  if (index + delta < 0 || index + delta >= result.length) return result;
  [result[index], result[index + delta]] = [
    result[index + delta],
    result[index],
  ];
  return result;
}

function RecapEmailOrder({
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
  const order = builder.email_order || builder.order;
  function selected(key: SectionKey) {
    const section = builder.sections[key];
    const available = (catalog[key] || []).filter(
      (i) => i.title || i.text || i.lines?.length || i.image_url,
    );
    const included = section.item_ids || available.map((i) => i.id);
    const ids =
      section.email_item_ids ||
      included.slice(
        0,
        (
          { events: 3, spotlight: 2, races: 2, achievements: 3 } as Record<
            string,
            number
          >
        )[key] || 150,
      );
    return ids.flatMap((id) =>
      included.includes(id) ? available.filter((i) => i.id === id) : [],
    );
  }
  const visible = order.filter(
    (key) =>
      builder.sections[key].enabled &&
      builder.sections[key].email_enabled &&
      selected(key).length,
  );
  return (
    <details className={styles.section} data-testid="recap-email-order">
      <summary>
        <strong>Arrange your email</strong>
      </summary>
      <p>
        Choose email items above, then arrange them here. Website ordering stays
        separate once you customize this order.
      </p>
      {!visible.length ? <p>No email content selected yet.</p> : null}
      {visible.map((key, index) => (
        <section key={key} className={styles.section}>
          <div className={styles.row}>
            <strong>{builder.sections[key].title}</strong>
            <div className={styles.actions}>
              {[-1, 1].map((delta) => (
                <button
                  key={delta}
                  type="button"
                  disabled={
                    disabled ||
                    (delta < 0 ? index === 0 : index === visible.length - 1)
                  }
                  aria-label={`Move ${RECAP_SECTIONS[key]} ${delta < 0 ? "up" : "down"} in email`}
                  onClick={() =>
                    onChange({
                      ...builder,
                      email_order: [
                        ...move(visible, index, delta),
                        ...order.filter((k) => !visible.includes(k)),
                      ],
                    })
                  }
                >
                  {delta < 0 ? "↑" : "↓"}
                </button>
              ))}
            </div>
          </div>
          {selected(key).map((item, position, items) => (
            <div key={item.id} className={styles.item}>
              <span>{item.title || item.text}</span>
              <div className={styles.actions}>
                {[-1, 1].map((delta) => (
                  <button
                    key={delta}
                    type="button"
                    disabled={
                      disabled ||
                      (delta < 0
                        ? position === 0
                        : position === items.length - 1)
                    }
                    aria-label={`Move ${item.title || item.text} ${delta < 0 ? "earlier" : "later"} in email`}
                    onClick={() =>
                      onChange({
                        ...builder,
                        sections: {
                          ...builder.sections,
                          [key]: {
                            ...builder.sections[key],
                            email_item_ids: move(
                              items.map((i) => i.id),
                              position,
                              delta,
                            ),
                          },
                        },
                      })
                    }
                  >
                    {delta < 0 ? "↑" : "↓"}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </section>
      ))}
      <button
        type="button"
        disabled={disabled}
        onClick={() =>
          onChange({
            ...builder,
            email_order: undefined,
            sections: Object.fromEntries(
              builder.order.map((key) => {
                const config = builder.sections[key];
                return [
                  key,
                  {
                    ...config,
                    ...(config.email_item_ids
                      ? {
                          email_item_ids: (
                            config.item_ids ||
                            (catalog[key] || []).map((i) => i.id)
                          ).filter((id) => config.email_item_ids!.includes(id)),
                        }
                      : {}),
                  },
                ];
              }),
            ) as BuilderConfig["sections"],
          })
        }
      >
        Use website order for email
      </button>
    </details>
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
    const id = crypto.randomUUID();
    const config = builder.sections.custom;
    onChange({
      ...builder,
      sections: {
        ...builder.sections,
        custom: {
          ...config,
          ...(config.item_ids ? { item_ids: [...config.item_ids, id] } : {}),
          ...(config.email_item_ids
            ? { email_item_ids: [...config.email_item_ids, id] }
            : {}),
        },
      },
      custom_blocks: [
        ...builder.custom_blocks,
        {
          id,
          kind,
          title: kind,
          text: "",
          lines: [],
          image_url: "",
          alt: "",
          url: "",
          link_label: kind === "Featured event" ? "Register now" : "Learn more",
          layout:
            kind === "Photo of the week"
              ? "photo"
              : ["Featured event", "Sponsor thank-you"].includes(kind)
                ? "promotion"
                : "callout",
          image_fit: "contain",
        },
      ],
    });
  }
  function visibility(id: string, checked: boolean, email: boolean) {
    const config = builder.sections.custom;
    const all = builder.custom_blocks.map((item) => item.id);
    const prop = email ? "email_item_ids" : "item_ids";
    const current = config[prop] || (email ? config.item_ids || all : all);
    onChange({
      ...builder,
      sections: {
        ...builder.sections,
        custom: {
          ...config,
          [prop]: checked
            ? [...current.filter((v) => v !== id), id]
            : current.filter((v) => v !== id),
        },
      },
    });
  }
  function reorder(index: number, delta: number) {
    const blocks = move(builder.custom_blocks, index, delta);
    const config = builder.sections.custom;
    onChange({
      ...builder,
      custom_blocks: blocks,
      sections: {
        ...builder.sections,
        custom: {
          ...config,
          ...(config.item_ids
            ? {
                item_ids: blocks
                  .filter((item) => config.item_ids!.includes(item.id))
                  .map((item) => item.id),
              }
            : {}),
        },
      },
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
          "Featured event",
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
      {!builder.sections.custom.enabled ? (
        <p>
          Club stories are hidden. Enable the Club stories section in Content to
          show these cards.
        </p>
      ) : null}
      {builder.custom_blocks.map((item, index) => (
        <fieldset className={styles.section} key={item.id} disabled={disabled}>
          <legend>{item.kind}</legend>
          <div className={styles.actions}>
            <label className={styles.check}>
              <input
                type="checkbox"
                checked={
                  !builder.sections.custom.item_ids ||
                  builder.sections.custom.item_ids.includes(item.id)
                }
                onChange={(e) => visibility(item.id, e.target.checked, false)}
              />
              Include this story in recap
            </label>
            <label className={styles.check}>
              <input
                type="checkbox"
                disabled={Boolean(
                  builder.sections.custom.item_ids &&
                  !builder.sections.custom.item_ids.includes(item.id),
                )}
                checked={
                  (!builder.sections.custom.item_ids ||
                    builder.sections.custom.item_ids.includes(item.id)) &&
                  (!builder.sections.custom.email_item_ids ||
                    builder.sections.custom.email_item_ids.includes(item.id))
                }
                onChange={(e) => visibility(item.id, e.target.checked, true)}
              />
              Include this story in email
            </label>
          </div>
          <label>
            Card style
            <select
              aria-label="Card style"
              value={item.layout || "plain"}
              onChange={(e) =>
                update(item.id, {
                  layout: e.target.value as RecapItem["layout"],
                })
              }
            >
              <option value="plain">Simple story</option>
              <option value="photo">Photo feature</option>
              <option value="callout">Highlighted note</option>
              <option value="promotion">Promotion with button</option>
            </select>
          </label>
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
              Photo presentation
              <select
                aria-label="Photo presentation"
                value={item.image_fit || "contain"}
                onChange={(e) =>
                  update(item.id, {
                    image_fit: e.target.value as RecapItem["image_fit"],
                  })
                }
              >
                <option value="contain">Show the whole image</option>
                <option value="cover">Crop to a wide banner</option>
              </select>
            </label>
            <label>
              Photo credit
              <input
                value={item.credit || ""}
                maxLength={160}
                onChange={(e) => update(item.id, { credit: e.target.value })}
              />
            </label>
          </div>
          <label>
            Photo caption
            <input
              value={item.caption || ""}
              maxLength={600}
              onChange={(e) => update(item.id, { caption: e.target.value })}
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
              onClick={() => reorder(index, -1)}
            >
              Move up
            </button>
            <button
              type="button"
              disabled={index === builder.custom_blocks.length - 1}
              onClick={() => reorder(index, 1)}
            >
              Move down
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
          <details>
            <summary>Preview this card</summary>
            <RecapEdition
              clubSlug=""
              blocks={[{ key: "custom", title: "", items: [item] }]}
            />
          </details>
        </fieldset>
      ))}
    </section>
  );
}
