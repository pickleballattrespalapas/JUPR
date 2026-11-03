"use client";

import { useEffect, useRef, useState } from "react";
import type { PollFollowup } from "@/lib/weeklyRecapApi";
import styles from "./recap-participation.module.css";

type Activity = {
  featured: "poll" | "shoutout" | "none"; open: boolean; shoutouts: boolean;
  poll: { question: string; options: string[] } | null;
  stories: { nominee: string; story: string }[]; followup: PollFollowup | null;
};
type Receipt = { choice: number | null; nominated: boolean };

export function Followup({ result }: { result: PollFollowup }) {
  const total = result.counts.reduce((sum, v) => sum + v, 0);
  return <section className={styles.card}><p className={styles.kicker}>Last week’s vote</p>
    <h3>{result.winner === null ? "A tie in the club vote" : "You picked " + result.options[result.winner] + "."}</h3>
    <p>{result.question}</p><p>{result.options.map((v, i) => v + " · " + result.counts[i]).join(" / ")} ({total} votes)</p>
    {result.tip && <p>{result.tip}</p>}
    {result.clinic && <a href={result.clinic.url}>{result.clinic.title}</a>}
  </section>;
}

export default function RecapParticipation({ club, week, source }: { club: string; week: string; source: string }) {
  const endpoint = "/api/clubs/" + encodeURIComponent(club) + "/weekly-recaps/" + encodeURIComponent(week) + "/engagement";
  const browser = useRef("");
  const [data, setData] = useState<Activity | null>(null);
  const [receipt, setReceipt] = useState<Receipt>({ choice: null, nominated: false });
  const [answer, setAnswer] = useState("");
  const [name, setName] = useState(""), [story, setStory] = useState("");
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const key = "pcs:recap-reader:" + club + ":" + week;
    try {
      browser.current = localStorage.getItem(key) || browser.current || crypto.randomUUID();
      localStorage.setItem(key, browser.current);
    } catch { browser.current ||= crypto.randomUUID(); }
    async function load() {
      const res = await fetch(endpoint, { cache: "no-store", signal: controller.signal });
      if (!res.ok) throw Error("Could not load this week’s activity.");
      const activity: Activity = await res.json();
      if (!active) return;
      setData(activity);
      const visit = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({ visitor: browser.current, source, action: "visit" }) });
      if (visit.ok) {
        const state: Receipt = await visit.json();
        if (active) setReceipt(previous => ({ choice: previous.choice ?? state.choice, nominated: previous.nominated || state.nominated }));
      }
    }
    load().catch(() => { if (active) setError("Could not load this week’s activity."); });
    return () => { active = false; controller.abort(); };
  }, [club, week, endpoint, source, reload]);
  async function submit(action: "vote" | "shoutout") {
    if (busy) return;
    setBusy(true); setError("");
    try {
      const response = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ visitor: browser.current, action, source, question: data?.poll?.question, answer, name, story }) });
      const result = await response.json();
      if (!response.ok) throw Error(typeof result.detail === "string" ? result.detail : "Could not save your response.");
      setReceipt(result);
    } catch (e) { setError(e instanceof Error ? e.message : "Could not save your response."); }
    finally { setBusy(false); }
  }
  return <div className={styles.activities} id="recap-activities">
    {data?.poll && <section className={styles.card}><p className={styles.kicker}>Weekly vote</p>
      <h3>{data.poll.question}</h3>
      {receipt.choice !== null ? <p role="status">Your vote for <strong>{data.poll.options[receipt.choice]}</strong> is recorded.</p>
        : data.open ? <form onSubmit={e => { e.preventDefault(); void submit("vote"); }}>
          <fieldset><legend>Choose one answer</legend>{data.poll.options.map(option => <label key={option}>
            <input type="radio" name="recap-poll" value={option} checked={answer === option} onChange={() => setAnswer(option)} /> {option}
          </label>)}</fieldset><button disabled={busy || !answer}>Cast my vote</button>
        </form> : <p>Voting for this edition has closed.</p>}
    </section>}
    {data?.shoutouts && <section className={styles.card}><p className={styles.kicker}>Member shout-outs</p>
      <h3>Who made you feel welcome on court?</h3>
      {receipt.nominated ? <p role="status">Your shout-out is with the club team for review.</p>
        : data.open ? <form onSubmit={e => { e.preventDefault(); void submit("shoutout"); }}>
          <label>Member’s name<input value={name} maxLength={80} required onChange={e => setName(e.target.value)} /></label>
          <label>What did they do?<textarea value={story} maxLength={400} required rows={2} onChange={e => setStory(e.target.value)} /></label>
          <p>The club reviews stories and checks permission before featuring them.</p>
          <button disabled={busy || !name.trim() || !story.trim()}>Give a shout-out</button>
        </form> : <p>Nominations for this edition have closed.</p>}
      {data.stories.map((s, i) => <article key={i}><strong>{s.nominee}</strong><p>{s.story}</p></article>)}
    </section>}
    {data?.followup && <Followup result={data.followup} />}
    {error && <p role="alert">{error} <button type="button" onClick={() => { setError(""); setReload(v => v + 1); }}>Refresh activity</button></p>}
  </div>;
}
