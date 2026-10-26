"use client";

import { useEffect, useState } from "react";
import WeeklyHighlightCards from "@/components/WeeklyHighlightCards";
import type { WeeklyHighlightsEditorData } from "@/lib/leagueWeeklyHighlights";

type Props = {
  apiBase: string;
  clubId: string;
  leagueName: string;
  week: number;
  accessToken: string;
  writesEnabled: boolean;
};
type WeekDraft = {
  saved: WeeklyHighlightsEditorData;
  mode: "automatic" | "curated";
  ids: string[];
};
const button = { padding: "0.5rem 0.75rem", border: "1px solid #cbd5e1", borderRadius: 8, background: "white", color: "#0f172a", cursor: "pointer" };

function draftFrom(saved: WeeklyHighlightsEditorData): WeekDraft {
  return { saved, mode: saved.mode, ids: saved.mode === "automatic" ? saved.automatic_cards.map((card) => card.id) : saved.selected_ids };
}

async function readResponse(response: Response) {
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(typeof payload?.detail === "string" ? payload.detail : "Unable to update weekly highlights. Please try again.");
  return payload;
}

export default function WeeklyHighlightsEditor({ apiBase, clubId, leagueName, week, accessToken, writesEnabled }: Props) {
  // Keep drafts when moving between weeks. The parent keys this editor by club,
  // league and signed-in session so protected drafts cannot cross those scopes.
  const [drafts, setDrafts] = useState<Record<number, WeekDraft>>({});
  const [messages, setMessages] = useState<Record<number, string>>({});
  const [busy, setBusy] = useState(false);
  const [reload, setReload] = useState(0);
  const [errors, setErrors] = useState<Record<number, string>>({});
  const draft = drafts[week];
  const endpoint = `${apiBase.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(leagueName)}/weeks/${week}/highlights`;

  useEffect(() => {
    if (draft) return;
    const controller = new AbortController();
    void fetch(endpoint, { cache: "no-store", headers: { Authorization: `Bearer ${accessToken}` }, signal: controller.signal })
      .then(readResponse)
      .then((saved: WeeklyHighlightsEditorData) => {
        if (controller.signal.aborted) return;
        if (saved.week_num !== week) throw new Error("The returned week did not match your selection.");
        setDrafts((current) => ({ ...current, [week]: draftFrom(saved) }));
        setErrors((current) => ({ ...current, [week]: "" }));
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setErrors((current) => ({ ...current, [week]: error instanceof Error ? error.message : "Unable to load weekly highlights." }));
      });
    return () => controller.abort();
  }, [accessToken, draft, endpoint, reload, week]);

  function reloadSaved() {
    setDrafts((current) => {
      const next = { ...current };
      delete next[week];
      return next;
    });
    setErrors((current) => ({ ...current, [week]: "" }));
    setMessages((current) => ({ ...current, [week]: "" }));
    setReload((value) => value + 1);
  }

  function edit(patch: Partial<Pick<WeekDraft, "mode" | "ids">>) {
    setDrafts((current) => ({ ...current, [week]: { ...current[week], ...patch } }));
    setMessages((current) => ({ ...current, [week]: "" }));
    setErrors((current) => ({ ...current, [week]: "" }));
  }

  async function save() {
    if (!draft) return;
    const targetWeek = week;
    const submittedIds = draft.mode === "automatic" ? [] : draft.ids;
    setBusy(true);
    setErrors((current) => ({ ...current, [targetWeek]: "" }));
    try {
      const saved = await fetch(endpoint, {
        method: "PUT", headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
        body: JSON.stringify({ mode: draft.mode, selected_ids: submittedIds, expected_revision: draft.saved.revision })
      }).then(readResponse);
      setDrafts((current) => ({ ...current, [targetWeek]: {
        ...draft, saved: { ...draft.saved, mode: saved.mode, selected_ids: saved.selected_ids, revision: saved.revision }
      } }));
      setMessages((current) => ({ ...current, [targetWeek]: `Week ${targetWeek} highlights saved. Public results may take up to a minute to refresh.` }));
    } catch (error) {
      setErrors((current) => ({ ...current, [targetWeek]: error instanceof Error ? error.message : "Unable to save weekly highlights." }));
    } finally {
      setBusy(false);
    }
  }

  if (!draft) return (
    <section aria-label="Weekly highlights editor">
      <h2>Choose weekly highlights</h2>
      {errors[week] ? <><p role="alert">{errors[week]}</p><button type="button" style={button} onClick={reloadSaved}>Retry</button></> : <p role="status">Finding Week {week} standouts…</p>}
    </section>
  );

  const candidates = draft.saved.categories.flatMap((category) => category.candidates);
  const lookup = new Map(candidates.map((card) => [card.id, card]));
  const preview = draft.mode === "automatic" ? draft.saved.automatic_cards : draft.ids.flatMap((id) => lookup.get(id) ? [lookup.get(id)!] : []);
  const unavailable = draft.ids.filter((id) => !lookup.has(id));
  const dirty = draft.mode !== draft.saved.mode || (draft.mode === "curated" && JSON.stringify(draft.ids) !== JSON.stringify(draft.saved.selected_ids));
  const disabled = busy || !writesEnabled || !draft.saved.available;

  function toggle(id: string) {
    if (!draft) return;
    edit({ ids: draft.ids.includes(id) ? draft.ids.filter((key) => key !== id) : [...draft.ids, id] });
  }

  function move(index: number, offset: number) {
    if (!draft) return;
    const ids = [...draft.ids];
    [ids[index], ids[index + offset]] = [ids[index + offset], ids[index]];
    edit({ ids });
  }

  return (
    <section aria-label="Weekly highlights editor" style={{ display: "grid", gap: "1rem" }}>
      <div>
        <h2 style={{ margin: "0 0 0.4rem" }}>Choose Week {week} highlights</h2>
        <p style={{ margin: 0, color: "#475569" }}>Review standout performances, then choose up to {draft.saved.max_highlights} highlights in the order you want them shown above the public rankings.</p>
      </div>
      <fieldset disabled={disabled} style={{ border: 0, padding: 0, margin: 0, display: "flex", gap: "1rem", flexWrap: "wrap" }}>
        <legend style={{ marginBottom: "0.5rem", fontWeight: 700 }}>Highlights for this week</legend>
        <label><input type="radio" name={`highlights-mode-${week}`} checked={draft.mode === "curated"} onChange={() => edit({ mode: "curated" })} /> Choose highlights</label>
        <label><input type="radio" name={`highlights-mode-${week}`} checked={draft.mode === "automatic"} onChange={() => edit({ mode: "automatic" })} /> Automatic suggestions</label>
      </fieldset>
      {draft.saved.warnings.map((warning) => <p key={warning} style={{ margin: 0, color: "#92400e" }}>{warning}</p>)}
      {draft.mode === "curated" ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 300px), 1fr))", gap: "0.75rem" }}>
            {draft.saved.categories.map((category) => (
              <details key={category.id} style={{ border: "1px solid #e2e8f0", borderRadius: 10, padding: "0.85rem", alignSelf: "start" }} open={category.candidates.some((card) => draft.ids.includes(card.id))}>
                <summary style={{ cursor: "pointer", fontWeight: 800 }}>{category.title} <span style={{ fontWeight: 400 }}>({category.candidates.length})</span></summary>
                <p style={{ color: "#64748b", fontSize: "0.85rem" }}>{category.qualification}</p>
                {!category.candidates.length ? <p>No qualifying performances this week.</p> : null}
                <div style={{ display: "grid", gap: "0.65rem", maxHeight: 340, overflowY: "auto" }}>
                  {category.candidates.map((card) => (
                    <label key={card.id} style={{ display: "flex", gap: "0.6rem", alignItems: "flex-start", padding: "0.5rem", borderRadius: 8, background: draft.ids.includes(card.id) ? "#eff6ff" : "#f8fafc" }}>
                      <input type="checkbox" checked={draft.ids.includes(card.id)} onChange={() => toggle(card.id)} disabled={disabled || (!draft.ids.includes(card.id) && draft.ids.length >= draft.saved.max_highlights)} style={{ marginTop: "0.3rem" }} />
                      <span><strong>{card.name} · {card.metric}</strong><br /><span style={{ color: "#475569", fontSize: "0.85rem" }}>{card.detail}</span></span>
                    </label>
                  ))}
                </div>
              </details>
            ))}
          </div>
          <div>
            <h3 style={{ margin: "0 0 0.5rem" }}>Selected highlights · {draft.ids.length}/{draft.saved.max_highlights}</h3>
            {unavailable.length ? <p role="alert" style={{ color: "#b91c1c" }}>Some saved selections no longer qualify after results changed. Remove them or reload the suggestions before saving.</p> : null}
            <ol style={{ paddingLeft: "1.5rem", margin: 0 }}>
              {draft.ids.map((id, index) => (
                <li key={id} style={{ marginBottom: "0.6rem" }}>
                  <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
                    <span style={{ flex: "1 1 220px" }}>{lookup.has(id) ? `${lookup.get(id)!.title}: ${lookup.get(id)!.name}` : "Previously selected highlight · no longer available"}</span>
                    <button type="button" style={button} disabled={disabled || index === 0} onClick={() => move(index, -1)} aria-label={`Move highlight ${index + 1} up`}>↑</button>
                    <button type="button" style={button} disabled={disabled || index === draft.ids.length - 1} onClick={() => move(index, 1)} aria-label={`Move highlight ${index + 1} down`}>↓</button>
                    <button type="button" style={button} disabled={disabled} onClick={() => toggle(id)} aria-label={`Remove highlight ${index + 1}`}>Remove</button>
                  </div>
                </li>
              ))}
            </ol>
            {!draft.ids.length ? <p>No highlights selected. Saving this choice hides the highlights section for Week {week}.</p> : null}
          </div>
        </>
      ) : <p style={{ margin: 0, color: "#475569" }}>Automatic suggestions pick different standouts as the results change. Choose highlights to make your own selection for this week.</p>}
      <div>
        <h3 style={{ marginTop: 0 }}>Public preview · Week {week}</h3>
        {preview.length ? <WeeklyHighlightCards cards={preview} /> : <p>No highlights will display for this week.</p>}
      </div>
      {!writesEnabled || !draft.saved.available ? <p role="status">Saving weekly highlights is currently unavailable.</p> : null}
      {errors[week] ? <p role="alert" style={{ color: "#b91c1c", margin: 0 }}>{errors[week]}</p> : null}
      {messages[week] ? <p role="status" style={{ color: "#166534", margin: 0 }}>{messages[week]}</p> : null}
      <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }}>
        <button type="button" onClick={() => void save()} disabled={disabled || !dirty || (draft.mode === "curated" && unavailable.length > 0)} style={{ ...button, background: "#1d4ed8", color: "white", fontWeight: 800 }}>{busy ? "Saving…" : `Save Week ${week} highlights`}</button>
        <button type="button" onClick={reloadSaved} disabled={busy} style={button}>{dirty ? "Discard edits and reload" : "Reload saved highlights"}</button>
        {dirty ? <span role="status">Unsaved changes for Week {week}</span> : null}
      </div>
    </section>
  );
}
