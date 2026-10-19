"use client";

import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess } from "@/components/interaction";
import type { RecapPreferences } from "@/lib/recapBuilder";
import styles from "./recap-builder.module.css";

export default function RecapLayoutManager({
  preferences,
  disabled,
  onSave,
  onReload,
}: {
  preferences: RecapPreferences;
  disabled: boolean;
  onSave: (
    settings: RecapPreferences["settings"],
    revision: number,
    message: string,
  ) => Promise<void>;
  onReload: () => Promise<void>;
}) {
  const { settings, revision } = preferences;
  const [selectedId, setSelectedId] = useState(settings.templates[0]?.id || "");
  const selected = settings.templates.find((t) => t.id === selectedId);
  const [name, setName] = useState(selected?.name || "");
  const [error, setError] = useState("");
  async function perform(action: () => Promise<void>) {
    setError("");
    try {
      await action();
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "The layout could not be updated.",
      );
    }
  }
  async function rename() {
    const next = name.trim();
    if (!selected || !next) throw new Error("Give the layout a name.");
    if (
      settings.templates.some(
        (t) =>
          t.id !== selected.id &&
          t.name.toLocaleLowerCase() === next.toLocaleLowerCase(),
      )
    )
      throw new Error(
        "Choose a different name. A layout with that name already exists.",
      );
    await onSave(
      {
        ...settings,
        templates: settings.templates.map((t) =>
          t.id === selected.id ? { ...t, name: next } : t,
        ),
      },
      revision,
      "Layout renamed.",
    );
  }
  return (
    <details className={styles.section} data-testid="recap-layout-manager">
      <summary>
        <strong>Manage saved layouts</strong>
      </summary>
      <p>
        Layouts control the starting point for future recaps. Your current
        edition keeps its content and settings.
      </p>
      {error ? <p role="alert">{error}</p> : null}
      <label>
        Default for new recaps
        <select
          aria-label="Default for new recaps"
          disabled={disabled}
          value={settings.default_template_id}
          onChange={(e) => {
            const id = e.target.value;
            void perform(() =>
              onSave(
                { ...settings, default_template_id: id },
                revision,
                id
                  ? "Club default layout updated."
                  : "New recaps will use the standard layout.",
              ),
            );
          }}
        >
          <option value="">Standard recap</option>
          {settings.templates.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
      </label>
      {settings.templates.length ? (
        <>
          <label>
            Saved layout
            <select
              aria-label="Saved layout"
              disabled={disabled}
              value={selectedId}
              onChange={(e) => {
                setSelectedId(e.target.value);
                setName(
                  settings.templates.find((t) => t.id === e.target.value)
                    ?.name || "",
                );
                setError("");
              }}
            >
              {settings.templates.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Rename layout
            <input
              disabled={disabled}
              value={name}
              maxLength={80}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <div className={styles.actions}>
            <button
              type="button"
              disabled={disabled || !selected || name.trim() === selected.name}
              onClick={() => void perform(rename)}
            >
              Rename saved layout
            </button>
            {selected ? (
              <ConfirmAction
                triggerLabel="Delete saved layout"
                title={`Delete “${selected.name}”?`}
                description={`Remove this saved layout from the club?${selected.id === settings.default_template_id ? " New recaps will start with the standard layout." : ""} Existing recap editions keep their settings.`}
                confirmationText="DELETE LAYOUT"
                confirmLabel="Delete layout"
                tone="danger"
                disabled={disabled}
                onConfirm={async () => {
                  await onSave(
                    {
                      templates: settings.templates.filter(
                        (t) => t.id !== selected.id,
                      ),
                      default_template_id:
                        settings.default_template_id === selected.id
                          ? ""
                          : settings.default_template_id,
                    },
                    revision,
                    "Saved layout deleted.",
                  );
                  return actionSuccess(
                    "Layout deleted",
                    `“${selected.name}” was removed from saved layouts.`,
                  );
                }}
              />
            ) : null}
          </div>
        </>
      ) : (
        <p>No saved layouts yet. Save your choices in Publish &amp; share.</p>
      )}
      <button
        type="button"
        disabled={disabled}
        onClick={() => void perform(onReload)}
      >
        Reload saved layouts
      </button>
    </details>
  );
}
