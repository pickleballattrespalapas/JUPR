"use client";

import { useEffect, useId, useRef, useState, type InputHTMLAttributes } from "react";
import { matchesPlayerSearch } from "@/lib/playerSearch";

export type PlayerSearchOption = { value: string; label: string; disabled?: boolean };
type Props = Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "onChange" | "children"> & {
  value: string;
  onTextChange: (value: string) => void;
  onPick: (option: PlayerSearchOption) => void;
  options: PlayerSearchOption[];
  browse?: boolean;
  loading?: boolean;
  error?: string;
  onDismiss?: () => void;
  validationMessage?: string;
};

/** Suggestions remain local to the caller's eligible roster or scoped API results. */
export default function PlayerSearchInput({ value, onTextChange, onPick, options, browse = false, loading = false, error,
  onDismiss, validationMessage = "", onFocus, style, id, disabled, ...props }: Props) {
  const generatedId = useId();
  const inputId = id || `player-search-${generatedId}`;
  const listId = `${inputId}-suggestions`;
  const input = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const [browsing, setBrowsing] = useState(false);
  useEffect(() => { input.current?.setCustomValidity(validationMessage); }, [validationMessage]);
  const query = browsing ? "" : value.trim();
  const ready = query.length >= 2 || (browse && !query);
  const matches = ready ? options.filter(option => matchesPlayerSearch(option.label, query)) : [];
  const choices = matches.slice(0, 25);
  const expanded = open && !disabled;

  function pick(option: PlayerSearchOption) {
    if (option.disabled || disabled) return;
    setOpen(false); setActive(-1); setBrowsing(false);
    onPick(option);
  }

  return <span style={{ display: "block", minWidth: 0, position: "relative" }} onBlur={event => {
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
    setOpen(false); setActive(-1); setBrowsing(false); onDismiss?.();
  }}>
    <input {...props} id={inputId} ref={input} disabled={disabled} value={value} autoComplete="off"
      role="combobox" aria-autocomplete="list" aria-expanded={expanded} aria-controls={expanded ? listId : undefined}
      aria-activedescendant={expanded && active >= 0 && choices[active] ? `${listId}-${active}` : undefined}
      placeholder={props.placeholder || "Type a first or last name"}
      style={{ width: "100%", boxSizing: "border-box", padding: ".65rem", border: "1px solid #cbd5e1", borderRadius: 8, background: "white", color: "#0f172a", font: "inherit", ...style }}
      onFocus={event => { setOpen(true); setBrowsing(browse); setActive(-1); event.currentTarget?.select?.(); onFocus?.(event); }}
      onChange={event => { setOpen(true); setBrowsing(false); setActive(-1); onTextChange(event.target.value); }}
      onKeyDown={event => {
        props.onKeyDown?.(event);
        if (event.defaultPrevented) return;
        if (event.nativeEvent?.isComposing) return;
        if (event.key === "Escape") { event.preventDefault(); setOpen(false); setActive(-1); onDismiss?.(); }
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault(); setOpen(true);
          const direction = event.key === "ArrowDown" ? 1 : -1;
          let next = active < 0 && direction < 0 ? 0 : active;
          for (let count = 0; count < choices.length; count++) {
            next = (next + direction + choices.length) % choices.length;
            if (!choices[next].disabled) break;
          }
          setActive(next);
          if (typeof document !== "undefined") document.getElementById(`${listId}-${next}`)?.scrollIntoView({ block: "nearest" });
        }
        if (event.key === "Enter" && expanded && active >= 0 && choices[active]) { event.preventDefault(); pick(choices[active]); }
      }} />
    {expanded ? <span style={{ display: "block", marginTop: 3, maxHeight: 300, overflowY: "auto", border: "1px solid #cbd5e1", borderRadius: 8, background: "white", color: "#0f172a", boxShadow: "0 6px 18px #0f172a22" }}>
      <span id={listId} role="listbox" aria-label="Matching players" style={{ display: "block" }}>
        {!loading && !error && choices.map((option, index) => <button key={`${option.value}-${index}`} id={`${listId}-${index}`} type="button"
          role="option" data-player-value={option.value} aria-selected={active === index} disabled={option.disabled} tabIndex={-1}
          onMouseDown={event => event.preventDefault()} onClick={event => { event.preventDefault(); pick(option); }}
          style={{ display: "block", width: "100%", minHeight: 44, border: 0, borderBottom: "1px solid #f1f5f9", padding: ".7rem", textAlign: "left", font: "inherit", whiteSpace: "normal", color: option.disabled ? "#94a3b8" : "#0f172a", background: active === index ? "#dbeafe" : "white", cursor: "pointer" }}>
          {option.label}{choices.some(other => other.value !== option.value && other.label === option.label) ? ` · #${option.value}` : ""}
        </button>)}
      </span>
      <span role="status" style={{ display: "block", padding: ".5rem .7rem", color: "#64748b", fontSize: ".85rem" }}>
        {error || (loading ? "Finding players…" : !ready ? "Type at least 2 characters of a first or last name." : !choices.length ? "No matching players." : matches.length > choices.length ? "Keep typing to narrow these matches." : `${choices.length} matching ${choices.length === 1 ? "player" : "players"}`)}
      </span>
    </span> : null}
  </span>;
}
