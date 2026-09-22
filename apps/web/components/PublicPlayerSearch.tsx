"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import PlayerSearchInput, { type PlayerSearchOption } from "@/components/PlayerSearchInput";

type Props = {
  clubSlug: string;
  defaultValue?: string;
  id?: string;
  style?: CSSProperties;
  scope?: "players" | "leaderboards";
  filters?: Record<string, string>;
};

export default function PublicPlayerSearch({ clubSlug, defaultValue = "", id, style, scope = "players", filters = {} }: Props) {
  const [query, setQuery] = useState(defaultValue);
  const [result, setResult] = useState<{ key: string; options: PlayerSearchOption[]; error?: string } | null>(null);
  const root = useRef<HTMLSpanElement>(null);
  const params = new URLSearchParams(filters).toString();
  const key = `${clubSlug}/${scope}?${params}&q=${query.trim()}`;
  useEffect(() => { setQuery(defaultValue); }, [defaultValue, clubSlug, scope]);

  useEffect(() => {
    if (query.trim().length < 2) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const base = process.env.NEXT_PUBLIC_JUPR_API_BASE_URL;
        if (!base) throw new Error("Player suggestions are unavailable. You can still search by name.");
        const search = new URLSearchParams(params);
        search.set("q", query.trim().slice(0, 80)); search.set("limit", "20"); search.set("offset", "0");
        const response = await fetch(`${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubSlug)}/${scope}?${search}`, { signal: controller.signal, cache: "no-store" });
        if (!response.ok) throw new Error("Player suggestions are unavailable. You can still search by name.");
        const payload = await response.json();
        const rows = (scope === "players" ? payload.players : payload.leaderboard) || [];
        const options = rows.map((row: { id?: number; player_id?: number; name?: string; player_name?: string; rating_jupr?: number }) => ({
          value: String(row.id ?? row.player_id),
          label: `${row.name || row.player_name || "Player"}${row.rating_jupr != null ? ` · ${Number(row.rating_jupr).toFixed(2)}` : ""}`,
        }));
        if (!controller.signal.aborted) setResult({ key, options });
      } catch (error) {
        if (!controller.signal.aborted) setResult({ key, options: [], error: error instanceof Error ? error.message : "Player suggestions are unavailable." });
      }
    }, 250);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [clubSlug, scope, params, query, key]);

  const current = result?.key === key ? result : null;
  return <span ref={root} style={{ display: "block" }}><PlayerSearchInput id={id} name="q" type="search" aria-label="Find player" style={style}
    value={query} onTextChange={setQuery} options={current?.options || []} error={current?.error}
    loading={query.trim().length >= 2 && !current} onPick={option => {
      if (scope === "players") window.location.assign(`/clubs/${encodeURIComponent(clubSlug)}/players/${encodeURIComponent(option.value)}`);
      else {
        const form = root.current?.closest("form");
        if (!form) return;
        const url = new URL(form.action);
        const search = new URLSearchParams();
        new FormData(form).forEach((value, key) => { if (typeof value === "string") search.set(key, value); });
        search.delete("q"); search.delete("page"); search.set("player", option.value);
        window.location.assign(`${url.pathname}?${search}`);
      }
    }} /></span>;
}
