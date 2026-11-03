"use client";

import { useEffect, useRef, useState } from "react";
import type { Engagement, EngagementMode, PollFollowup, RecapPoll, WeeklyRecapJson } from "@/lib/weeklyRecapApi";
import { shortRecapEmail } from "@/lib/weeklyReportEmail";
import styles from "./recap-builder.module.css";

type Stats = {
  counts: null | { readers: number; email_readers: number; email_participants: number; votes: number; first: number; second: number; nominations: number };
  pending: { id: string; nominee: string; story: string }[];
  published_poll: RecapPoll | null;
  previous: Pick<PollFollowup, "week" | "question" | "options" | "counts"> | null;
};
type Props = {
  week: string; recap: Partial<WeeklyRecapJson>; config?: Engagement;
  polls: RecapPoll[]; defaultMode: EngagementMode; defaultShoutouts: boolean; disabled: boolean; canManage: boolean;
  request: <T>(path: string, method?: string, body?: unknown) => Promise<T>;
  onChange: (v: Engagement) => void; onDefault: (v: EngagementMode, shoutouts: boolean) => Promise<void>;
};
const modes: EngagementMode[] = ["automatic", "review", "off"];
const label = { automatic: "Automatic", review: "Review first", off: "Off" };

export default function RecapEngagementEditor({ week, recap, config, polls, defaultMode, defaultShoutouts, disabled, canManage, request, onChange, onDefault }: Props) {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState("");
  const [permission, setPermission] = useState<string[]>([]);
  const [editing, setEditing] = useState(false);
  const [pendingReview, setPendingReview] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [draftPoll, setDraftPoll] = useState<RecapPoll | null>(null);
  const req = useRef(request);
  useEffect(() => { req.current = request; }, [request]);
  const path = "/recaps/" + encodeURIComponent(week) + "/engagement";
  useEffect(() => {
    let active = true;
    req.current<Stats>(path).then(result => { if (active) setStats(result); })
      .catch(() => { if (active) setError("Could not load participation. Try Refresh totals."); });
    return () => { active = false; };
  }, [path, refresh]);
  const generated = shortRecapEmail(recap);
  const update = (patch: Partial<Engagement>) => { if (config) onChange({ ...config, ...patch }); };
  const locked = Boolean(stats?.counts?.votes);
  async function review(id: string, decision: "featured" | "dismissed") {
    setPendingReview(true);
    try {
      await request(path + "/nominations/" + id, "POST", { decision, permission: permission.includes(id) });
      setError(""); setRefresh(v => v + 1);
    } catch (e) { setError(e instanceof Error ? e.message : "Could not review this nomination."); }
    finally { setPendingReview(false); }
  }
  return <section className={styles.section} aria-label="Recap engagement">
    <h2>Engagement</h2>
    <p>Send one short story with a link to this edition. Members take part on the recap webpage.</p>
    <label>Club default for new editions{" "}
      <select value={defaultMode} disabled={!canManage} onChange={async e => {
        const mode = e.target.value as EngagementMode;
        try { await onDefault(mode, defaultShoutouts); if (config && !disabled) update({ mode }); }
        catch { /* The parent displays the preference save error. */ }
      }}>{modes.map(m => <option key={m} value={m}>{label[m]}</option>)}</select>
    </label>
    <p><label><input type="checkbox" checked={defaultShoutouts} disabled={!canManage}
      onChange={async e => { const shoutouts = e.target.checked;
        try { await onDefault(defaultMode, shoutouts); if (config && !disabled) update({ shoutouts }); }
        catch { /* The parent displays the preference save error. */ }
      }} /> Accept shout-outs alongside polls in new editions</label></p>
    {!config ? <p><button disabled={disabled || !polls.length} onClick={() => onChange({
      mode: defaultMode, featured: "poll", approved: false, shoutouts: defaultShoutouts, poll: polls[0], subject: "", body: "",
    })}>Set up engagement for this edition</button></p> : <>
      <fieldset disabled={disabled}>
        <legend>This issue</legend>
        <label>Activity mode <select value={config.mode} onChange={e => update({ mode: e.target.value as EngagementMode })}>
          {modes.map(m => <option value={m} key={m}>{label[m]}</option>)}
        </select></label>{" "}
        <label>Featured activity <select value={config.featured} onChange={e => update({ featured: e.target.value as Engagement["featured"] })}>
          <option value="poll">Weekly vote</option><option value="shoutout">Member shout-out</option><option value="none">Skip activity this week</option>
        </select></label>
        {config.mode === "review" && config.featured !== "none" && <p><label><input type="checkbox" checked={config.approved}
          onChange={e => update({ approved: e.target.checked })} /> Include the reviewed activity in this edition</label></p>}
        {config.featured === "poll" && config.mode !== "off" && <>
          <h3>{config.poll.question}</h3><p>{config.poll.options.join(" or ")}</p>
          <button type="button" disabled={locked || polls.length < 2} onClick={() => {
            setDraftPoll(null); update({ poll: polls[(polls.findIndex(p => p.id === config.poll.id) + 1) % polls.length], approved: false });
          }}>Try another</button>{" "}
          <button type="button" disabled={locked} onClick={() => setDraftPoll(structuredClone(config.poll))}>Edit question and tips</button>
          {locked && <p>The published question is locked because members have voted.</p>}
          {draftPoll && <div>
            <label>Question<input maxLength={200} value={draftPoll.question} onChange={e => setDraftPoll({ ...draftPoll, question: e.target.value })} /></label>
            {draftPoll.options.map((_, i) => <div key={i}>{(["options","tips"] as const).map(key => <label key={key}>{key === "options" ? "Answer" : "Follow-up drill"} {i + 1}
              <input maxLength={key === "options" ? 80 : 450} value={draftPoll[key][i]} onChange={e => {
                const next = structuredClone(draftPoll); next[key][i] = e.target.value; setDraftPoll(next);
              }} /></label>)}</div>)}
            <button disabled={locked || !draftPoll.question.trim() || ![...draftPoll.options,...draftPoll.tips].every(t => t.trim())
              || draftPoll.options[0].trim().toLowerCase() === draftPoll.options[1].trim().toLowerCase()}
              onClick={() => { update({ poll: draftPoll, approved: false }); setDraftPoll(null); }}>Use this question</button>{" "}
            <button onClick={() => setDraftPoll(null)}>Cancel question edits</button>
          </div>}
          <p><label><input type="checkbox" checked={config.shoutouts} onChange={e => update({ shoutouts: e.target.checked })} /> Also accept member shout-outs on the webpage</label></p>
        </>}
      </fieldset>
      <h3>Member email</h3><p><strong>{generated.subject}</strong></p>{generated.paragraphs.map((v, i) => <p key={i}>{v}</p>)}
      <button disabled={disabled} onClick={() => setEditing(!editing)}>{editing ? "Done editing email" : "Edit email"}</button>{" "}
      {(config.subject || config.body) && <button disabled={disabled} onClick={() => update({ subject: "", body: "" })}>Use automatic email again</button>}
      {editing && <fieldset disabled={disabled}><legend>Edition-specific wording</legend>
        <label>Subject<input maxLength={140} value={config.subject || generated.subject} onChange={e => update({ subject: e.target.value })} /></label>
        <label>One or two paragraphs<textarea rows={5} maxLength={1600} value={config.body || generated.paragraphs.join("\n\n")}
          onChange={e => update({ body: e.target.value })} /></label>
      </fieldset>}
    </>}
    <h3>Participation</h3>
    {stats?.counts ? <>
      <p>{stats.counts.email_readers} readers arrived from the email. {stats.counts.email_participants} of them voted or submitted a shout-out.</p>
      <p>{stats.counts.votes} votes across all sources · {stats.counts.nominations} shout-outs · {stats.counts.readers} readers</p>
      {stats.published_poll && stats.counts.votes > 0 && <p>{stats.published_poll.options[0]} ({stats.counts.first}) · {stats.published_poll.options[1]} ({stats.counts.second})</p>}
    </> : <p>Participation totals appear after publication.</p>}
    {stats?.previous && <div><h4>Previous edition · {stats.previous.week}</h4>
      <p>{stats.previous.question} {stats.previous.options.map((v, i) => v + " (" + stats.previous!.counts[i] + ")").join(" · ")}</p>
    </div>}
    <button onClick={() => { setError(""); setRefresh(v => v + 1); }}>Refresh totals</button>
    {stats?.pending.map(item => <article key={item.id}><h4>{item.nominee}</h4><p>{item.story}</p>
      <label><input type="checkbox" disabled={!canManage} checked={permission.includes(item.id)} onChange={e => setPermission(prev =>
        e.target.checked ? [...prev, item.id] : prev.filter(id => id !== item.id))} /> Permission to feature this story confirmed</label>{" "}
      <button disabled={!canManage || pendingReview || !permission.includes(item.id)} onClick={() => void review(item.id, "featured")}>Feature on recap</button>{" "}
      <button disabled={!canManage || pendingReview} onClick={() => void review(item.id, "dismissed")}>Dismiss</button>
    </article>)}
    {error && <p role="alert">{error}</p>}
  </section>;
}
