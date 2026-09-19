"use client";

import { useEffect, useRef, useState } from "react";
import { PlayerSignupError, playerSignupRequest, seasonDates, SeasonSignupDetails, signupErrorMessage, signupMeetTime } from "@/lib/interclubPlayerSignup";
import styles from "../../player-signup.module.css";

type Registration = { status: "registered" | "already_registered"; message: string; manage_url?: string };

export default function SeasonSignup({ shareId }: { shareId: string }) {
  const [data, setData] = useState<SeasonSignupDetails | null>(null);
  const [loading, setLoading] = useState(true), [reload, setReload] = useState(0);
  const [error, setError] = useState("");
  const [name, setName] = useState(""), [email, setEmail] = useState("");
  const [divisions, setDivisions] = useState<string[]>([]), [notes, setNotes] = useState("");
  const [consent, setConsent] = useState(false), [busy, setBusy] = useState(false);
  const [refreshRequired, setRefreshRequired] = useState(false);
  const [registration, setRegistration] = useState<Registration | null>(null), [copied, setCopied] = useState(false);
  const pending = useRef(false), mutation = useRef<AbortController | null>(null);
  const retry = useRef<{ signature: string; requestId: string } | null>(null);
  const success = useRef<HTMLHeadingElement>(null);
  const path = `/public/interclub-signups/${encodeURIComponent(shareId)}`;

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError("");
    playerSignupRequest<SeasonSignupDetails>(path, controller.signal).then(value => {
      if (!controller.signal.aborted) { setData(value); setRefreshRequired(false); }
    }).catch(err => { if (!controller.signal.aborted) { setData(null); setError(signupErrorMessage(err)); } })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [path, reload]);
  useEffect(() => () => mutation.current?.abort(), []);
  useEffect(() => { if (registration) success.current?.focus(); }, [registration]);

  async function register() {
    if (pending.current || refreshRequired || !data?.signup.open || !consent) return;
    const body = { name: name.trim(), email: email.trim(), divisions, notes: notes.trim(), email_consent: true };
    if (!body.name || !body.email) { setError("Enter your name and email."); return; }
    const signature = JSON.stringify(body);
    if (retry.current?.signature !== signature) retry.current = { signature, requestId: crypto.randomUUID() };
    const controller = new AbortController(); mutation.current = controller; pending.current = true;
    setBusy(true); setError("");
    try {
      const result = await playerSignupRequest<Registration>(path, controller.signal, { ...body, request_id: retry.current.requestId });
      if (!controller.signal.aborted) setRegistration(result);
    } catch (err) { if (!controller.signal.aborted) { setError(signupErrorMessage(err)); if (err instanceof PlayerSignupError && [404, 409].includes(err.status)) setRefreshRequired(true); } }
    finally { pending.current = false; if (!controller.signal.aborted) setBusy(false); }
  }

  return <div className={styles.page}><section className={styles.card}>
    <p className={styles.eyebrow}>{data?.club.name || "Interclub pickleball"}</p>
    <h1>{registration ? registration.status === "registered" ? "You’re in the player pool" : "You’re already signed up" : "Join your club’s player pool"}</h1>
    {loading ? <p role="status">Loading season signup…</p> : data && <>
      <h2>{data.season.name}</h2><p className={styles.muted}>{seasonDates(data.season)}</p>
      {registration ? <div className={styles.success}>
        <h2 ref={success} tabIndex={-1}>{registration.status === "registered" ? "Your interest is registered" : "Your earlier signup is still saved"}</h2>
        <p>Your club will invite players to upcoming meets. You can say yes, maybe, or no for each one. Your administrator confirms the final teams.</p>
        {registration.status === "registered" && registration.manage_url ? <>
          <p><strong>Keep your personal link.</strong> Use it to update your details or leave the player pool. This link is just for you.</p>
          <div className={styles.actions}><a className={`${styles.buttonLink} ${styles.primary}`} href={registration.manage_url} rel="noreferrer">Manage my season signup</a>
            <button type="button" onClick={async () => { try { await navigator.clipboard.writeText(registration.manage_url!); setCopied(true); } catch { setCopied(false); setError("Copy the personal link from the box below."); } }}>{copied ? "Link copied" : "Copy my personal link"}</button></div>
          <label className={styles.form}>Your personal link<input readOnly value={registration.manage_url} aria-label="Your personal link" onFocus={event => event.target.select()} /></label>
        </> : <p>Use the personal link from your original signup to make changes. If you no longer have it, ask your club administrator for help.</p>}
      </div> : !data.signup.open ? <div className={styles.notice}><h2>Signup is currently closed</h2><p>Contact your club administrator if you’d like to take part.</p></div> : <>
        <p className={styles.intro}>Interested in playing for {data.club.name}? Sign up once for the season. Your club will invite you to individual meets when it’s time to plan the teams.</p>
        <div className={styles.notice}><strong>Travel plans can change.</strong><p>Joining this pool does not commit you to every meet or reserve a team place. You’ll choose your availability separately for each meet. No player account is needed.</p></div>
        {data.meets.length > 0 && <details><summary>Planned meets for your club ({data.meets.length})</summary><ul>{data.meets.map(meet => <li key={meet.id}>{signupMeetTime(meet.starts_at, data.season.timezone)}{meet.host_club_name ? ` · ${meet.host_club_name}` : ""}</li>)}</ul><p className={styles.hint}>You’ll respond separately when your club invites you to a meet.</p></details>}
        <form className={styles.form} onSubmit={event => { event.preventDefault(); void register(); }}>
          <label>Your name<input required maxLength={120} autoComplete="name" value={name} disabled={busy} onChange={event => setName(event.target.value)} /></label>
          <label>Email<input required type="email" maxLength={254} autoComplete="email" value={email} disabled={busy} onChange={event => setEmail(event.target.value)} /><span className={styles.hint}>Your club will send meet invitations to this address.</span></label>
          <fieldset disabled={busy}><legend>Divisions you’re interested in</legend><div className={styles.choices}>{data.season.divisions.map(division => <label className={styles.choice} key={division}><input type="checkbox" checked={divisions.includes(division)} onChange={event => setDivisions(current => event.target.checked ? [...current, division] : current.filter(item => item !== division))} /><span>{division}</span></label>)}</div><p className={styles.hint}>Select any that suit you. Your club checks ratings and eligibility before choosing teams.</p></fieldset>
          <label>Availability notes <span className={styles.hint}>(optional)</span><textarea maxLength={1000} value={notes} disabled={busy} placeholder="For example: in Baja November through March, away for two weeks in January." onChange={event => setNotes(event.target.value)} /></label>
          <label className={styles.choice}><input type="checkbox" required checked={consent} disabled={busy} onChange={event => setConsent(event.target.checked)} /><span>My club can email me invitations for this interclub season. I’ll choose my availability for each meet.</span></label>
          <p className={styles.hint}>Your contact information and notes are available to your club’s administrators.</p>
          <button className={styles.primary} disabled={busy || refreshRequired || !consent}>{busy ? "Saving your signup…" : "Join the season player pool"}</button>
        </form>
      </>}
    </>}
    {error && <div className={styles.error} role="alert"><p>{error}</p>{!loading && (!data || refreshRequired) && <button type="button" onClick={() => setReload(value => value + 1)}>{refreshRequired ? "Reload signup" : "Try again"}</button>}</div>}
  </section></div>;
}
