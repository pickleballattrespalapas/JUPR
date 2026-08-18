"use client";

import { useMemo, useState } from "react";
import { useAdminSession } from "@/lib/useAdminSession";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";

type Props = {
  apiBase: string | null;
  clubId: string;
  leagueName: string;
  leagueStatus: string;
};

type AwardCatalogRow = {
  key: string;
  label: string;
  recipient_type: "player" | "team";
  metric: string;
  minimum_metric?: string;
  default_enabled?: boolean;
};

type AwardDraft = {
  enabled: boolean;
  depth: "1" | "2" | "3";
  minimum: string;
};

type AwardsSetupResponse = {
  league?: { awards_config?: Record<string, unknown>; min_games?: number | null };
  award_catalog?: AwardCatalogRow[];
  awards_config_version?: number;
  writes_enabled?: boolean;
  wizard?: { status?: string };
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function draftsFrom(payload: AwardsSetupResponse): Record<string, AwardDraft> {
  const awardsConfig = asRecord(payload.league?.awards_config);
  const categories = asRecord(awardsConfig.categories);
  const defaultMinimum = Number(awardsConfig.default_min_games ?? payload.league?.min_games ?? 0);
  return Object.fromEntries((payload.award_catalog || []).map((category) => {
    const configured = asRecord(categories[category.key]);
    const depth = String(configured.depth || "1");
    return [category.key, {
      enabled: typeof configured.enabled === "boolean"
        ? configured.enabled
        : Boolean(category.default_enabled),
      depth: (["1", "2", "3"].includes(depth) ? depth : "1") as AwardDraft["depth"],
      minimum: String(configured.minimum ?? configured.min_games ?? defaultMinimum)
    }];
  }));
}

export default function LeagueAwardsSetupPanel({ apiBase, clubId, leagueName, leagueStatus }: Props) {
  const { accessToken } = useAdminSession();
  const [state, setState] = useState<AwardsSetupResponse | null>(null);
  const [drafts, setDrafts] = useState<Record<string, AwardDraft>>({});
  const [loadedDrafts, setLoadedDrafts] = useState<Record<string, AwardDraft>>({});
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const requestGuard = useLatestRequestGuard(`${accessToken}\u0000${leagueName}`, clearProtectedState);

  function clearProtectedState() {
    setState(null);
    setDrafts({});
    setLoadedDrafts({});
    setBusy(false);
    setMessage(null);
  }

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before loading award setup.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  function awardPath(suffix = ""): string {
    return `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(leagueName)}/awards${suffix}`;
  }

  function hydrate(payload: AwardsSetupResponse) {
    const next = draftsFrom(payload);
    setState(payload);
    setDrafts(next);
    setLoadedDrafts(next);
  }

  async function load() {
    const generation = requestGuard.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AwardsSetupResponse>(awardPath());
      if (!requestGuard.isCurrent(generation)) return;
      hydrate(payload);
    } catch (error) {
      if (requestGuard.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to load award setup.");
      }
    } finally {
      if (requestGuard.isCurrent(generation)) setBusy(false);
    }
  }

  async function save() {
    if (!state) return;
    const categories: Record<string, unknown> = {};
    for (const category of state.award_catalog || []) {
      const draft = drafts[category.key];
      if (!draft) continue;
      const minimum = Number(draft.minimum);
      if (!Number.isInteger(minimum) || minimum < 0 || minimum > 1000) {
        setMessage(`${category.label} minimum must be a whole number from 0 to 1000.`);
        return;
      }
      categories[category.key] = {
        enabled: draft.enabled,
        depth: Number(draft.depth),
        minimum
      };
    }
    const generation = requestGuard.begin();
    setBusy(true);
    setMessage(null);
    try {
      const currentConfig = asRecord(state.league?.awards_config);
      const payload = await requestJson<AwardsSetupResponse>(awardPath("/config"), {
        method: "PUT",
        body: JSON.stringify({
          awards_config: { ...currentConfig, categories },
          expected_config_version: Number(state.awards_config_version || 0),
          source: "next_selected_league_settings_awards_setup"
        })
      });
      if (!requestGuard.isCurrent(generation)) return;
      hydrate(payload);
      setMessage(`Saved ${Object.values(drafts).filter((draft) => draft.enabled).length} award category choice(s).`);
    } catch (error) {
      if (requestGuard.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to save award setup.");
      }
    } finally {
      if (requestGuard.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(accessToken ? `${accessToken}\u0000${leagueName}` : "", load);

  const configured = useMemo(() => {
    const categories = asRecord(asRecord(state?.league?.awards_config).categories);
    return (state?.award_catalog || []).flatMap((category) => {
      const value = asRecord(categories[category.key]);
      if (value.enabled !== true) return [];
      return [{
        ...category,
        depth: Number(value.depth || 1),
        minimum: Number(value.minimum ?? value.min_games ?? 0)
      }];
    });
  }, [state]);
  const isDraft = leagueStatus === "draft";
  const canEdit = isDraft && state?.writes_enabled === true && state?.wizard?.status === "not_started";
  const hasChanges = JSON.stringify(drafts) !== JSON.stringify(loadedDrafts);

  return (
    <article style={cardStyle} data-testid="league-awards-setup">
      <h2 style={{ marginTop: 0 }}>Awards setup</h2>
      <p style={{ color: "#475569" }}>
        Award categories, places, and minimum criteria are configured here before the league starts. The Awards tab tracks live progress and handles final review.
      </p>
      {busy && !state ? <p role="status">Loading award setup…</p> : null}
      {state && canEdit ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "0.75rem" }}>
            {(state.award_catalog || []).map((category) => {
              const draft = drafts[category.key];
              if (!draft) return null;
              return (
                <fieldset key={category.key} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: draft.enabled ? "#f0fdf4" : "#f8fafc" }}>
                  <legend style={{ fontWeight: 800 }}>{category.label}</legend>
                  <p style={{ color: "#64748b", marginTop: 0 }}>{category.recipient_type === "team" ? "Team" : "Player"} · {category.metric.replace(/_/g, " ")}</p>
                  <label><input type="checkbox" checked={draft.enabled} onChange={(event) => setDrafts((current) => ({ ...current, [category.key]: { ...draft, enabled: event.target.checked } }))} /> Enabled</label>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.6rem", marginTop: "0.6rem" }}>
                    <label><strong>Places</strong><br /><select value={draft.depth} disabled={!draft.enabled} onChange={(event) => setDrafts((current) => ({ ...current, [category.key]: { ...draft, depth: event.target.value as AwardDraft["depth"] } }))} style={inputStyle}><option value="1">Top 1</option><option value="2">Top 2</option><option value="3">Top 3</option></select></label>
                    <label><strong>Minimum {String(category.minimum_metric || "games").replace(/_/g, " ")}</strong><br /><input type="number" min={0} max={1000} value={draft.minimum} disabled={!draft.enabled} onChange={(event) => setDrafts((current) => ({ ...current, [category.key]: { ...draft, minimum: event.target.value } }))} style={inputStyle} /></label>
                  </div>
                </fieldset>
              );
            })}
          </div>
          <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <button type="button" onClick={() => void save()} disabled={busy || !hasChanges} style={buttonStyle}>{busy ? "Saving…" : "Save award setup"}</button>
            <button type="button" onClick={() => setDrafts(loadedDrafts)} disabled={busy || !hasChanges} style={{ ...buttonStyle, background: "white", color: "#0f172a" }}>Reset</button>
          </p>
        </>
      ) : state ? (
        <>
          <p><strong>Saved award configuration · read-only</strong></p>
          {configured.length ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
              {configured.map((category) => (
                <div key={category.key} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem", background: "#f8fafc" }}>
                  <strong>{category.label}</strong><br />Top {category.depth} · Minimum {category.minimum} {String(category.minimum_metric || "games").replace(/_/g, " ")}
                </div>
              ))}
            </div>
          ) : <p style={{ color: "#64748b" }}>No award categories were configured for this league.</p>}
          {isDraft && !canEdit ? <p style={{ color: "#92400e" }}>Award setup changes are currently unavailable.</p> : null}
        </>
      ) : null}
      {message ? <p role="status" style={{ color: /unable|error|must|unavailable/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </article>
  );
}
