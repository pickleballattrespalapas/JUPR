"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { DivisionRule, apiError } from "@/lib/interclubRegistration";

export default function OpenRegistration({ api, clubId, accessToken, seasonId, revision, divisions }: {
  api: string; clubId: string; accessToken: string; seasonId: string; revision: number; divisions: string[];
}) {
  const [status, setStatus] = useState("loading");
  const [rules, setRules] = useState<Record<string, DivisionRule>>(() => Object.fromEntries(divisions.map(d => [d, { min_rating: null, max_rating: null, women_required: null }])));
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [reload, setReload] = useState(0);
  const pending = useRef(false), token = useRef(accessToken), mutation = useRef<AbortController | null>(null);
  token.current = accessToken;
  const url = `${api}/admin/clubs/${encodeURIComponent(clubId)}/interclub/registrations/${seasonId}`;
  useEffect(() => {
    const controller = new AbortController(); setStatus("loading");
    fetch(url, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal })
      .then(response => { if (!controller.signal.aborted) setStatus(response.ok ? "open" : response.status === 404 ? "unopened" : "error"); })
      .catch(() => { if (!controller.signal.aborted) setStatus("error"); });
    return () => { controller.abort(); mutation.current?.abort(); };
  }, [url, reload]);
  function ruleEdit(division: string, field: keyof DivisionRule, value: string) {
    setRules(old => ({ ...old, [division]: { ...old[division], [field]: value === "" ? null : Number(value) } }));
  }
  async function open() {
    if (pending.current || status !== "unopened") return;
    pending.current = true; setBusy(true); setMessage("");
    const controller = new AbortController(); mutation.current = controller;
    try {
      const response = await fetch(`${url}/open`, { method: "POST", signal: controller.signal,
        headers: { Authorization: `Bearer ${token.current}`, "Content-Type": "application/json" },
        body: JSON.stringify({ expected_revision: revision, rules }) });
      const data = await response.json();
      if (controller.signal.aborted) return;
      if (!response.ok) { if ([409, 503].includes(response.status)) setStatus("error"); throw new Error(apiError(data, "Unable to open invitations.")); }
      setStatus("open"); setMessage("Club invitations are open. Each club can accept from its interclub page.");
    } catch (error) { if (!controller.signal.aborted) setMessage(error instanceof Error ? error.message : "Unable to open invitations."); }
    finally { pending.current = false; if (!controller.signal.aborted) setBusy(false); }
  }
  return <section style={{ marginTop: 32, padding: 20, border: "1px solid #cbd5e1", borderRadius: 12 }}>
    <h2>Club invitations</h2>
    {status === "loading" && <p>Checking invitations…</p>}
    {status === "open" && <p><Link href={`/admin/interclub/registrations?season=${seasonId}`}>Open club responses and meet rosters</Link>. Later changes to the planning draft do not change this open registration.</p>}
    {status === "error" && <p>Reload to check whether invitations are already open. <button disabled={busy} onClick={() => setReload(n => n + 1)}>Check again</button></p>}
    {status === "unopened" && <form onSubmit={e => { e.preventDefault(); void open(); }}>
      <p>Set the eligibility rules before inviting the proposed clubs. These eligibility rules apply throughout the season. Clubs choose their available players separately for each meet. Clubs receive the invitation in their workspace; no email is sent.</p>
      <fieldset disabled={busy} style={{ display: "grid", gap: 16 }}><legend>Registration rules</legend>
        <p>Ratings are captured from the represented club when each player first enters the season. Blank limits mean no rating restriction. Set roster deadlines for individual meets after opening invitations.</p>
        {divisions.map(division => <fieldset key={division}><legend>{division}</legend>
          <label>Minimum starting rating <input aria-label={`${division} minimum rating`} type="number" min={1} max={7} step="0.001" value={rules[division].min_rating ?? ""} onChange={e => ruleEdit(division, "min_rating", e.target.value)} /></label>{" "}
          <label>Maximum starting rating <input aria-label={`${division} maximum rating`} type="number" min={1} max={7} step="0.001" value={rules[division].max_rating ?? ""} onChange={e => ruleEdit(division, "max_rating", e.target.value)} /></label>{" "}
          <label>Team composition <select aria-label={`${division} team composition`} value={rules[division].women_required ?? ""} onChange={e => ruleEdit(division, "women_required", e.target.value)}>
            <option value="">Any four players</option>{[0, 1, 2, 3, 4].map(n => <option key={n} value={n}>{n} women, {4 - n} men</option>)}
          </select></label>
        </fieldset>)}
        <button type="submit">{busy ? "Opening…" : "Open invitations with these rules"}</button>
      </fieldset>
    </form>}
    {message && <p role="status">{message}</p>}
  </section>;
}
