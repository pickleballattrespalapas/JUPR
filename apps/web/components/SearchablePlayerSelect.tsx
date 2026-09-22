"use client";

import { Children, isValidElement, useEffect, useMemo, useState, type ReactNode, type SelectHTMLAttributes } from "react";
import PlayerSearchInput, { type PlayerSearchOption } from "@/components/PlayerSearchInput";

type Props = Omit<SelectHTMLAttributes<HTMLSelectElement>, "onChange"> & {
  onValueChange?: (value: string) => void;
  onValuesChange?: (values: string[]) => void;
};

function plainText(node: ReactNode): string {
  return Children.toArray(node).map(child => isValidElement<{ children?: ReactNode }>(child) ? plainText(child.props.children) : String(child)).join("");
}

function readOptions(children: ReactNode): PlayerSearchOption[] {
  return Children.toArray(children).flatMap(child => {
    if (!isValidElement<{ value?: string | number; children?: ReactNode; disabled?: boolean }>(child)) return [];
    if (child.type === "option") return [{ value: String(child.props.value ?? plainText(child.props.children)), label: plainText(child.props.children), disabled: child.props.disabled }];
    return readOptions(child.props.children);
  });
}

/** Keeps existing IDs, eligibility, form serialization and change handlers intact. */
export default function SearchablePlayerSelect({ children, value, defaultValue, onValueChange, onValuesChange, multiple, required, disabled,
  style, className, id, name, ...props }: Props) {
  const options = useMemo(() => readOptions(children), [children]);
  const [localValue, setLocalValue] = useState(defaultValue ?? (multiple ? [] : options[0]?.value ?? ""));
  const selection = value ?? localValue;
  const selectedIds = (Array.isArray(selection) ? selection : [selection]).map(String);
  const selectedLabel = String(selection) ? options.find(option => option.value === String(selection))?.label || "" : "";
  const [query, setQuery] = useState<string | null>(null);
  useEffect(() => { setQuery(null); }, [selection, selectedLabel, multiple]);

  function choose(option: PlayerSearchOption) {
    if (disabled || option.disabled) return;
    if (multiple) {
      const next = selectedIds.includes(option.value) ? selectedIds.filter(id => id !== option.value) : [...selectedIds, option.value];
      setLocalValue(next); onValuesChange?.(next); setQuery(null);
    } else {
      setLocalValue(option.value); onValueChange?.(option.value); setQuery(null);
    }
  }

  return <span style={{ display: "block", minWidth: 0 }}>
    <PlayerSearchInput id={id} aria-label={props["aria-label"]} aria-describedby={props["aria-describedby"]} aria-invalid={props["aria-invalid"]}
      className={className} disabled={disabled} style={style} value={query ?? (multiple ? "" : selectedLabel)} onTextChange={setQuery}
      onPick={choose} options={multiple ? options.filter(option => !selectedIds.includes(option.value)) : options}
      browse onDismiss={() => setQuery(null)}
      validationMessage={required && !selectedIds.some(Boolean) ? "Choose a player from the suggestions." : ""}
      required={required && !selectedIds.some(Boolean)} />
    {multiple && selectedIds.length ? <span style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 }}>
      {selectedIds.map(id => { const option = options.find(row => row.value === id); return option ? <button key={id} type="button" disabled={disabled}
        aria-label={`Remove ${option.label}`} onClick={() => choose(option)} style={{ border: "1px solid #cbd5e1", background: "#f1f5f9", borderRadius: 16, padding: ".4rem .65rem", font: "inherit" }}>{option.label} ×</button> : null; })}
    </span> : null}
    <select hidden aria-hidden="true" tabIndex={-1} name={name} disabled={disabled} multiple={multiple} value={selection}
      onChange={event => {
        if (multiple) { const ids = Array.from(event.currentTarget.selectedOptions).map(option => option.value); setLocalValue(ids); onValuesChange?.(ids); }
        else { const option = options.find(row => row.value === event.target.value); if (option) choose(option); }
      }}>{children}</select>
  </span>;
}
