"use client";

import { useEffect, useRef, useState } from "react";
import { adminSessionIsFresh, loadAdminSession } from "@/lib/adminAuthClient";
import { registrationNameKey } from "@/lib/tournamentRegistrationProfile";
import { formatSignupRating, PlayerSignupError, playerSignupRequest, seasonDates, SeasonSignupDetails, signupErrorMessage, signupMeetTime, SignupPlayer, SignupPlayerMatches, sortSignupDivisions } from "@/lib/interclubPlayerSignup";
import styles from "../../player-signup.module.css";
import { registrationCanAccept, registrationWindowDates, registrationWindowMessage } from "@/lib/interclubRegistrationWindow";
import { useRegistrationWindow } from "@/lib/useRegistrationWindow";

type Registration = { status: "registered" | "already_registered"; message: string; manage_url?: string };

export default function SeasonSignup({ shareId }: { shareId: string }) {
  const [data, setData] = useState<SeasonSignupDetails | null>(null);
  const [loading, setLoading] = useState(true), [reload, setReload] = useState(0);
  const [error, setError] = useState("");
  const [name, setName] = useState(""), [email, setEmail] = useState("");
  const [session, setSession] = useState({ accessToken: "", email: "" });
  const [profile, setProfile] = useState<SignupPlayer | null>(null);
  const [candidates, setCandidates] = useState<SignupPlayer[] | null>(null);
  const [withoutProfile, setWithoutProfile] = useState(false);
  const [lookupPending, setLookupPending] = useState(false), [lookupError, setLookupError] = useState("");
  const [lookupRetry, setLookupRetry] = useState(0);
  const [divisions, setDivisions] = useState<string[]>([]), [notes, setNotes] = useState("");
  const [consent, setConsent] = useState(false), [busy, setBusy] = useState(false);
  const [refreshRequired, setRefreshRequired] = useState(false);
  const [registration, setRegistration] = useState<Registration | null>(null), [copied, setCopied] = useState(false);
  const pending = useRef(false), mutation = useRef<AbortController | null>(null);
  const retry = useRef<{ signature: string; requestId: string } | null>(null);
  const success = useRef<HTMLHeadingElement>(null);
  const nameRef = useRef(""), identityVersion = useRef(0);
  const previousShare = useRef(shareId);
  const path = `/public/interclub-signups/${encodeURIComponent(shareId)}`;
  const { canRegister, now } = useRegistrationWindow(data?.season.registration, () => setReload(value => value + 1));
  const registrationDates = data ? registrationWindowDates(data.season.registration, data.season.timezone) : null;
  const needsProfileChoice = !profile && !withoutProfile && Boolean(candidates?.length || lookupError);

  useEffect(() => {
    // Signup remains public. Only use an existing fresh session for account prefill.
    try {
      const stored = loadAdminSession();
      if (stored && adminSessionIsFresh(stored)) setSession({ accessToken: stored.access_token, email: stored.user?.email || "" });
    } catch { /* A browser that blocks storage can still join without an account. */ }
  }, []);

  useEffect(() => {
    if (previousShare.current === shareId) return;
    previousShare.current = shareId;
    identityVersion.current += 1; nameRef.current = "";
    mutation.current?.abort(); pending.current = false; retry.current = null;
    setName(""); setEmail(""); setProfile(null); setCandidates(null); setWithoutProfile(false);
    setLookupPending(false); setLookupError(""); setDivisions([]); setNotes(""); setConsent(false);
    setBusy(false); setRegistration(null); setCopied(false);
  }, [shareId]);

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

  useEffect(() => {
    if (!session.accessToken || !canRegister) return;
    const controller = new AbortController();
    const version = identityVersion.current;
    playerSignupRequest<SignupPlayerMatches>(`${path}/players`, controller.signal, undefined, session.accessToken).then(result => {
      if (controller.signal.aborted || version !== identityVersion.current || nameRef.current.trim() || pending.current) return;
      if (result.linked_player) {
        nameRef.current = result.linked_player.name;
        setName(result.linked_player.name); setProfile(result.linked_player);
        setEmail(current => current || session.email);
      }
    }).catch(() => { /* Name search is available if account prefill is unavailable. */ });
    return () => controller.abort();
  }, [path, session, canRegister]);

  useEffect(() => {
    if (!canRegister || registration || profile || withoutProfile || name.trim().length < 2) {
      setLookupPending(false);
      return;
    }
    const controller = new AbortController();
    const version = identityVersion.current;
    setLookupPending(true); setLookupError("");
    const timer = setTimeout(() => {
      playerSignupRequest<SignupPlayerMatches>(`${path}/players?q=${encodeURIComponent(name.trim())}`, controller.signal, undefined, session.accessToken).then(result => {
        if (controller.signal.aborted || version !== identityVersion.current) return;
        const rows = result.players;
        setCandidates(rows);
        const exact = rows.filter(player => registrationNameKey(player.name) === registrationNameKey(name));
        if (exact.length === 1) setProfile(exact[0]);
      }).catch(err => {
        if (!controller.signal.aborted && version === identityVersion.current) setLookupError(signupErrorMessage(err));
      }).finally(() => { if (!controller.signal.aborted && version === identityVersion.current) setLookupPending(false); });
    }, 250);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [path, name, session.accessToken, canRegister, registration, profile, withoutProfile, lookupRetry]);

  function changeName(value: string) {
    identityVersion.current += 1;
    nameRef.current = value;
    setName(value); setProfile(null); setCandidates(null); setWithoutProfile(false); setLookupError("");
    setLookupPending(value.trim().length >= 2);
  }

  function chooseProfile(player: SignupPlayer | null) {
    identityVersion.current += 1;
    setLookupPending(false); setLookupError(""); setProfile(player); setWithoutProfile(!player);
    if (player) { nameRef.current = player.name; setName(player.name); }
  }

  async function register() {
    if (pending.current || refreshRequired || !consent) return;
    if (!registrationCanAccept(data?.season.registration)) { setError(registrationWindowMessage(data?.season.registration)); setRefreshRequired(true); return; }
    if (lookupPending) { setError("Wait a moment while we check for your club profile."); return; }
    if (needsProfileChoice) { setError("Choose your profile, or continue without one."); return; }
    const body = {
      name: name.trim(), email: email.trim(), divisions, notes: notes.trim(), email_consent: true,
      ...(profile && registrationNameKey(name) === registrationNameKey(profile.name) ? { player_id: profile.id } : withoutProfile ? { player_id: null } : {}),
    };
    if (!body.name || !body.email) { setError("Enter your name and email."); return; }
    const signature = JSON.stringify(body);
    if (retry.current?.signature !== signature) retry.current = { signature, requestId: crypto.randomUUID() };
    const controller = new AbortController(); mutation.current = controller; pending.current = true;
    setBusy(true); setError("");
    try {
      const result = await playerSignupRequest<Registration>(path, controller.signal, { ...body, request_id: retry.current.requestId }, session.accessToken);
      if (!controller.signal.aborted) setRegistration(result);
    } catch (err) { if (!controller.signal.aborted) { setError(signupErrorMessage(err)); if (err instanceof PlayerSignupError && [404, 409].includes(err.status)) setRefreshRequired(true); } }
    finally { pending.current = false; if (!controller.signal.aborted) setBusy(false); }
  }

  return <div className={styles.page}><section className={styles.card}>
    <p className={styles.eyebrow}>{data?.club.name || "Interclub pickleball"}</p>
    <h1>{registration ? registration.status === "registered" ? "You’re in the player pool" : "You’re already signed up" : "Join your club’s player pool"}</h1>
    {loading ? <p role="status">Loading season signup…</p> : data && <>
      <h2>{data.season.name}</h2><p className={styles.muted}>{seasonDates(data.season)}</p>
      {registrationDates && <p className={styles.hint}><strong>Season registration:</strong> {registrationDates}</p>}
      {registration ? <div className={styles.success}>
        <h2 ref={success} tabIndex={-1}>{registration.status === "registered" ? "Your interest is registered" : "Your earlier signup is still saved"}</h2>
        <p>Your club will invite players to upcoming meets. You can say yes, maybe, or no for each one. Your administrator confirms the final teams.</p>
        {registration.status === "registered" && registration.manage_url ? <>
          <p><strong>Keep your personal link.</strong> Use it to update your details or leave the player pool. This link is just for you.</p>
          <div className={styles.actions}><a className={`${styles.buttonLink} ${styles.primary}`} href={registration.manage_url} rel="noreferrer">Manage my season signup</a>
            <button type="button" onClick={async () => { try { await navigator.clipboard.writeText(registration.manage_url!); setCopied(true); } catch { setCopied(false); setError("Copy the personal link from the box below."); } }}>{copied ? "Link copied" : "Copy my personal link"}</button></div>
          <label className={styles.form}>Your personal link<input readOnly value={registration.manage_url} aria-label="Your personal link" onFocus={event => event.target.select()} /></label>
        </> : <p>Use the personal link from your original signup to make changes. If you no longer have it, ask your club administrator for help.</p>}
      </div> : !canRegister ? <div className={styles.notice}><h2>{registrationWindowMessage(data.season.registration, now)}</h2><p>The commissioner sets registration dates for every club in this league. {data.season.registration?.status === "closed" ? "Contact your club administrator about your existing signup." : "Return when registration opens to join your club’s player pool."}</p></div> : <>
        <p className={styles.intro}>Interested in playing for {data.club.name}? Sign up once for the season. Your club will invite you to individual meets when it’s time to plan the teams.</p>
        <div className={styles.notice}><strong>Travel plans can change.</strong><p>Joining this pool does not commit you to every meet or reserve a team place. You’ll choose your availability separately for each meet. No player account is needed.</p></div>
        {data.meets.length > 0 && <details><summary>Planned meets for your club ({data.meets.length})</summary><ul>{data.meets.map(meet => <li key={meet.id}>{signupMeetTime(meet.starts_at, data.season.timezone)}{meet.host_club_name ? ` · ${meet.host_club_name}` : ""}</li>)}</ul><p className={styles.hint}>You’ll respond separately when your club invites you to a meet.</p></details>}
        <form className={styles.form} onSubmit={event => { event.preventDefault(); void register(); }}>
          <div className={styles.profileSearch}>
            <label>Your name<input required maxLength={120} autoComplete="name" value={name} disabled={busy} onChange={event => changeName(event.target.value)} aria-describedby="profile-search-help" /></label>
            <p id="profile-search-help" className={styles.hint}>We’ll look for your existing {data.club.name} player profile and rating.</p>
            {lookupPending && <p className={styles.hint} role="status">Looking for your club profile…</p>}
            {lookupError && <div className={styles.profileNotice} role="alert"><p>We couldn’t check player profiles. You can try again or let your club link your signup later.</p><div className={styles.actions}><button type="button" disabled={busy} onClick={() => setLookupRetry(value => value + 1)}>Try profile search again</button><button type="button" disabled={busy} onClick={() => chooseProfile(null)}>Continue without a profile</button></div></div>}
            {profile && <div className={styles.profileNotice}>
              <p><strong>{profile.name}</strong> · Existing club profile</p>
              <label>Club rating<input aria-label="Club rating" readOnly value={formatSignupRating(profile.rating)} /></label>
              {profile.league_rating != null && <label>League rating<input aria-label="League rating" readOnly value={formatSignupRating(profile.league_rating)} /></label>}
              <p className={styles.hint}>Your saved ratings are used to check division eligibility.{profile.eligible_divisions.length > 0 ? ` Eligible divisions: ${sortSignupDivisions(profile.eligible_divisions).join(", ")}.` : " Your club can confirm your eligible divisions."}</p>
              <button type="button" disabled={busy} onClick={() => chooseProfile(null)}>This isn’t my profile</button>
            </div>}
            {!lookupPending && Boolean(candidates?.length) && !profile && !withoutProfile && <fieldset disabled={busy} className={styles.profileMatches}>
              <legend>Is one of these your club profile?</legend>
              <p className={styles.hint}>Choose your profile so your club can use your existing rating.</p>
              {candidates!.map(player => <label key={player.id} className={styles.choice}><input type="radio" name="club_profile" value={player.id} checked={false} onChange={() => chooseProfile(player)} /><span><strong>{player.name}</strong><small>Rating: {formatSignupRating(player.rating)}{player.gender ? ` · ${player.gender}` : ""}</small></span></label>)}
              <label className={styles.choice}><input type="radio" name="club_profile" value="none" checked={false} onChange={() => chooseProfile(null)} /><span>None of these is me</span></label>
            </fieldset>}
            {!lookupPending && !lookupError && !profile && (withoutProfile || candidates?.length === 0) && <p className={styles.hint}>You can join without an existing profile. Your club can help link your player record and confirm your rating.</p>}
          </div>
          <label>Email<input required type="email" maxLength={254} autoComplete="email" value={email} disabled={busy} onChange={event => setEmail(event.target.value)} /><span className={styles.hint}>Your club will send meet invitations to this address.</span></label>
          <fieldset disabled={busy}><legend>Divisions you’re interested in</legend><div className={styles.choices}>{sortSignupDivisions(data.season.divisions).map(division => <label className={styles.choice} key={division}><input type="checkbox" checked={divisions.includes(division)} onChange={event => setDivisions(current => event.target.checked ? [...current, division] : current.filter(item => item !== division))} /><span>{division}</span></label>)}</div><p className={styles.hint}>Select any that suit you. Your club checks ratings and eligibility before choosing teams.</p></fieldset>
          <label>Availability notes <span className={styles.hint}>(optional)</span><textarea maxLength={1000} value={notes} disabled={busy} placeholder="For example: in Baja November through March, away for two weeks in January." onChange={event => setNotes(event.target.value)} /></label>
          <label className={styles.choice}><input type="checkbox" required checked={consent} disabled={busy} onChange={event => setConsent(event.target.checked)} /><span>My club can email me invitations for this interclub season. I’ll choose my availability for each meet.</span></label>
          <p className={styles.hint}>Your contact information and notes are available to your club’s administrators.</p>
          <button className={styles.primary} disabled={busy || refreshRequired || !consent || lookupPending || needsProfileChoice}>{busy ? "Saving your signup…" : "Join the season player pool"}</button>
        </form>
      </>}
    </>}
    {error && <div className={styles.error} role="alert"><p>{error}</p>{!loading && (!data || refreshRequired) && <button type="button" onClick={() => setReload(value => value + 1)}>{refreshRequired ? "Reload signup" : "Try again"}</button>}</div>}
  </section></div>;
}
