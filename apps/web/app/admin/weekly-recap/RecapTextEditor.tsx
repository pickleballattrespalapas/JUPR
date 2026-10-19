"use client";

import { useId } from "react";
import {
  projectEditions,
  RECAP_SECTIONS,
  RECAP_TEXT_LIMITS,
  textOverrideKey,
  type RecapCatalog,
  type RecapDisplayText,
  type RecapEdits,
  type RecapTextField,
  type RecapTextOverride,
} from "@/lib/recapBuilder";
import styles from "./recap-builder.module.css";

function TextField({
  label,
  value,
  changed,
  limit,
  multiline,
  onChange,
  onReset,
}: {
  label: string;
  value: string;
  changed: boolean;
  limit: number;
  multiline?: boolean;
  onChange: (value: string) => void;
  onReset: () => void;
}) {
  const id = useId();
  const props = {
    id,
    value,
    maxLength: limit,
    onChange: (
      event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
    ) => onChange(event.target.value),
  };
  return (
    <div className={styles.textField}>
      <div className={styles.textFieldHeading}>
        <div className={styles.textFieldLabel}>
          <label htmlFor={id}>{label}</label>
          {changed ? (
            <span className={styles.editedBadge} aria-hidden="true">
              Edited
            </span>
          ) : null}
        </div>
        <button
          type="button"
          disabled={!changed}
          aria-label={`Reset ${label}`}
          onClick={onReset}
        >
          Reset
        </button>
      </div>
      {multiline ? (
        <textarea
          {...props}
          rows={Math.min(6, Math.max(2, value.split("\n").length))}
        />
      ) : (
        <input {...props} />
      )}
    </div>
  );
}

export default function RecapTextEditor({
  catalog,
  edits,
  channel,
  disabled,
  weekStart,
  weekEnd,
  scopeLabel,
  generatedHeadline,
  onChange,
}: {
  catalog: RecapCatalog;
  edits: RecapEdits;
  channel: "edition" | "email_edition";
  disabled: boolean;
  weekStart: string;
  weekEnd: string;
  scopeLabel?: string;
  generatedHeadline: string;
  onChange: (edits: RecapEdits) => void;
}) {
  const builder = edits.builder;
  // Render the selected source cards so clearing text never removes its editor.
  const blocks = projectEditions(catalog, {
    ...builder,
    text_overrides: undefined,
  })[channel];
  function updateItem(
    key: string,
    field: RecapTextField | "lines" | "link_labels",
    value: string | undefined,
    source?: string,
  ) {
    const overrides = { ...builder.text_overrides };
    const patch: RecapTextOverride = { ...overrides[key] };
    if (field === "lines" || field === "link_labels") {
      const values = { ...patch[field] };
      if (value === undefined) delete values[String(source)];
      else values[String(source)] = value;
      if (Object.keys(values).length) patch[field] = values;
      else delete patch[field];
    } else if (value === undefined) delete patch[field];
    else patch[field] = value;
    if (Object.keys(patch).length) overrides[key] = patch;
    else delete overrides[key];
    onChange({ ...edits, builder: { ...builder, text_overrides: overrides } });
  }
  function displayField(
    field: keyof RecapDisplayText,
    label: string,
    original: string,
  ) {
    const current = builder.display_text?.[field];
    const update = (value?: string) => {
      const display = { ...builder.display_text };
      if (value === undefined) delete display[field];
      else display[field] = value;
      onChange({ ...edits, builder: { ...builder, display_text: display } });
    };
    return (
      <TextField
        label={label}
        value={current ?? original}
        changed={current !== undefined}
        limit={160}
        onChange={update}
        onReset={() => update()}
      />
    );
  }
  return (
    <fieldset
      className={styles.textEditor}
      disabled={disabled}
      data-testid="recap-text-editor"
    >
      <legend>Edit recap text</legend>
      <p className={styles.muted}>
        Changes save with this draft and appear wherever the item is included:
        website, email, and exports. Reset restores the current source wording.
        Recorded scores and player records stay intact.
      </p>
      <section className={styles.section} aria-label="Recap heading text">
        <TextField
          label="Recap headline"
          value={edits.editorial.headline}
          changed={edits.editorial.headline !== generatedHeadline}
          limit={160}
          onChange={(headline) =>
            onChange({ ...edits, editorial: { ...edits.editorial, headline } })
          }
          onReset={() =>
            onChange({
              ...edits,
              editorial: { ...edits.editorial, headline: generatedHeadline },
            })
          }
        />
        {displayField("date", "Date wording", `${weekStart} – ${weekEnd}`)}
        {displayField(
          "scope",
          "Club or league wording",
          scopeLabel && scopeLabel !== "Whole club"
            ? `League: ${scopeLabel}`
            : "",
        )}
        <p className={styles.muted}>
          The date and league wording change the presentation. Choose the games
          to include in Setup and Content.
        </p>
      </section>
      {blocks.map((block) => (
        <section
          className={styles.section}
          key={block.key}
          data-edit-section={block.key}
        >
          <h3>{RECAP_SECTIONS[block.key]}</h3>
          <TextField
            label="Section heading"
            value={builder.sections[block.key].title}
            changed={
              builder.sections[block.key].title !== RECAP_SECTIONS[block.key]
            }
            limit={100}
            onChange={(title) =>
              onChange({
                ...edits,
                builder: {
                  ...builder,
                  sections: {
                    ...builder.sections,
                    [block.key]: { ...builder.sections[block.key], title },
                  },
                },
              })
            }
            onReset={() =>
              onChange({
                ...edits,
                builder: {
                  ...builder,
                  sections: {
                    ...builder.sections,
                    [block.key]: {
                      ...builder.sections[block.key],
                      title: RECAP_SECTIONS[block.key],
                    },
                  },
                },
              })
            }
          />
          {block.items.map((item, index) => {
            const key = textOverrideKey(block.key, item);
            const patch = builder.text_overrides?.[key] || {};
            const field = (
              name: RecapTextField,
              label: string,
              fallback = "",
              multiline = false,
            ) => (
              <TextField
                key={name}
                label={label}
                value={patch[name] ?? item[name] ?? fallback}
                changed={patch[name] !== undefined}
                limit={RECAP_TEXT_LIMITS[name]}
                multiline={multiline}
                onChange={(value) => updateItem(key, name, value)}
                onReset={() => updateItem(key, name, undefined)}
              />
            );
            return (
              <fieldset
                className={styles.textCard}
                key={item.id}
                data-edit-item={item.id}
              >
                <legend>{item.title || `Item ${index + 1}`}</legend>
                {["events", "custom"].includes(block.key) &&
                (item.kind || patch.kind !== undefined)
                  ? field("kind", "Type label")
                  : null}
                {item.title || patch.title !== undefined
                  ? field("title", "Title")
                  : null}
                {item.text || patch.text !== undefined
                  ? field(
                      "text",
                      block.key === "numbers" ? "Statistic value" : "Summary",
                      "",
                      block.key !== "numbers",
                    )
                  : null}
                {item.lines.map((line, lineIndex) => (
                  <TextField
                    key={`line-${lineIndex}`}
                    label={`Detail ${lineIndex + 1}`}
                    value={patch.lines?.[line] ?? line}
                    changed={patch.lines?.[line] !== undefined}
                    limit={600}
                    multiline
                    onChange={(value) => updateItem(key, "lines", value, line)}
                    onReset={() => updateItem(key, "lines", undefined, line)}
                  />
                ))}
                {item.image_url ? (
                  <>
                    <img
                      src={item.image_url}
                      alt={item.alt || item.title}
                      className={styles.editorPhoto}
                    />
                    {field("caption", "Photo caption")}
                    {field("credit_label", "Photo credit prefix", "Photo:")}
                    {field("credit", "Photo credit")}
                    {field("alt", "Photo description", item.title)}
                  </>
                ) : null}
                {item.href
                  ? field("results_label", "Results link label", "View results")
                  : null}
                {item.links?.map((link, linkIndex) => (
                  <TextField
                    key={`link-${linkIndex}`}
                    label={`Related link ${linkIndex + 1} label`}
                    value={patch.link_labels?.[link.href] ?? link.label}
                    changed={patch.link_labels?.[link.href] !== undefined}
                    limit={80}
                    onChange={(value) =>
                      updateItem(key, "link_labels", value, link.href)
                    }
                    onReset={() =>
                      updateItem(key, "link_labels", undefined, link.href)
                    }
                  />
                ))}
                {item.url
                  ? field("link_label", "Link label", "Learn more")
                  : null}
              </fieldset>
            );
          })}
        </section>
      ))}
      {channel === "email_edition" ? (
        <section className={styles.section} aria-label="Email footer text">
          <h3>Email footer</h3>
          {displayField(
            "footer_title",
            "Footer heading",
            "KEEP UP WITH THE CLUB",
          )}
          {displayField(
            "footer_link",
            "Full recap link label",
            "View the full weekly report",
          )}
          <p className={styles.muted}>
            The full recap link appears once this edition is published.
          </p>
        </section>
      ) : null}
    </fieldset>
  );
}
