"use client";
import { useEffect, useRef, useState } from "react";
import MeetSignupGender from "@/components/MeetSignupGender";
import MeetSignupQueues from "@/components/MeetSignupQueues";
import { MeetSignupBoard, PrivateMeetSignup, SignupGender, signupGender, signupGenderLabels, divisionPriorityHint, meetSignupOpen, signupPlacement } from "@/lib/interclubMeetSignup";
import { SignupPlayer, playerSignupRequest, signupErrorMessage, signupMeetTime } from "@/lib/interclubPlayerSignup";
import styles from "../player-signup.module.css";

const base = "/public/interclub-meet-signups";
export default function MeetSignup({ shareId }: { shareId?: string }) {
  const [data, setData] = useState<MeetSignupBoard | null>(null), [own, setOwn] = useState<PrivateMeetSignup["entry"] | null>(null);
  const [loading, setLoading] = useState(true), [busy, setBusy] = useState(false), [reload, setReload] = useState(0);
  const [error, setError] = useState(""), [notice, setNotice] = useState(""), [copied, setCopied] = useState(false);
  const [name, setName] = useState(""), [email, setEmail] = useState(""), [division, setDivision] = useState("");
  const [profile, setProfile] = useState<SignupPlayer | null>(null), [matches, setMatches] = useState<SignupPlayer[]>([]);
  const [searching, setSearching] = useState(false), [searchError, setSearchError] = useState(""), [confirmedSelf, setConfirmedSelf] = useState(false);
  const [gender, setGender] = useState<SignupGender | "">("");
  const [now, setNow] = useState(Date.now());
  const token = useRef(""), pending = useRef(false), mutation = useRef<AbortController | null>(null);
  const request = useRef<{ signature: string; id: string } | null>(null), resultHeading = useRef<HTMLHeadingElement>(null);
  const open = Boolean(data && meetSignupOpen(data, now));
  useEffect(() => {
    if (!shareId && !token.current) {
      token.current = new URLSearchParams(window.location.hash.slice(1)).get("token") || "";
      if (window.location.hash) window.history.replaceState(null, "", window.location.pathname + window.location.search);
    }
    const controller = new AbortController(); setLoading(true); setError("");
    if (!shareId && !token.current) { setError("Open the private link you saved after signing up, or ask your club for it."); setLoading(false); return; }
    playerSignupRequest<MeetSignupBoard | PrivateMeetSignup>(token.current ? `${base}/review` : `${base}/${encodeURIComponent(shareId!)}`, controller.signal, token.current ? { token: token.current } : undefined)
      .then(value => { if (!controller.signal.aborted) { setData(value); if ("entry" in value) setOwn(value.entry); setDivision(current => value.season.divisions.includes(current) ? current : value.season.divisions[0] || ""); } })
      .catch(cause => { if (!controller.signal.aborted) setError(signupErrorMessage(cause)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [shareId, reload]);
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer); }, []);
  useEffect(() => () => mutation.current?.abort(), []);
  useEffect(() => { if (own) resultHeading.current?.focus(); }, [own]);
  useEffect(() => {
    setMatches([]); setSearchError("");
    if (!shareId || !open || profile || name.trim().length < 2 || own) { setSearching(false); return; }
    const controller = new AbortController(); setSearching(true);
    const timer = setTimeout(() => {
      playerSignupRequest<{ players: SignupPlayer[] }>(`${base}/${encodeURIComponent(shareId)}/players?q=${encodeURIComponent(name.trim())}`, controller.signal)
        .then(value => { if (!controller.signal.aborted) setMatches(value.players); })
        .catch(cause => { if (!controller.signal.aborted) setSearchError(signupErrorMessage(cause)); })
        .finally(() => { if (!controller.signal.aborted) setSearching(false); });
    }, 250);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [shareId, name, profile, open, own]);

  async function submit(withdraw = false) {
    if (pending.current || !data || !meetSignupOpen(data)) return;
    pending.current = true; setBusy(true); setError(""); setNotice("");
    const controller = new AbortController(); mutation.current = controller;
    try {
      let body: object;
      if (withdraw && own) body = { token: token.current, expected_revision: own.revision };
      else {
        if (!profile || !confirmedSelf || !gender) return;
        const details = { player_id: profile.id, name: profile.name, email, division, gender, confirm_self: true };
        const signature = JSON.stringify(details);
        if (request.current?.signature !== signature) request.current = { signature, id: crypto.randomUUID() };
        body = { ...details, request_id: request.current.id };
      }
      const value = await playerSignupRequest<PrivateMeetSignup | { duplicate: true; message: string }>(withdraw ? `${base}/withdraw` : `${base}/${encodeURIComponent(shareId!)}`, controller.signal, body);
      if (controller.signal.aborted) return;
      if ("duplicate" in value) { setNotice(value.message); return; }
      setData(value); setOwn(value.entry);
      token.current = value.entry.manage_url?.split("#token=")[1] || token.current;
    } catch (cause) { if (!controller.signal.aborted) setError(signupErrorMessage(cause)); }
    finally { pending.current = false; if (!controller.signal.aborted) setBusy(false); }
  }

  return <main className={styles.page}><section className={styles.card}>
    <p className={styles.eyebrow}>Interclub meet signup</p><h1>{data ? `Play for ${data.club.name}` : "Sign up to play"}</h1>
    {loading && <p role="status">Loading meet signup…</p>}{error && <p role="alert" className={styles.error}>{error}</p>}
    {notice && <p role="status" className={styles.notice}>{notice}</p>}
    {data && <>
      <p className={styles.intro}>{data.season.name}</p>
      <dl className={styles.details}><dt>Meet</dt><dd>{signupMeetTime(data.meet.starts_at, data.season.timezone)}</dd>
        <dt>Host</dt><dd>{data.meet.host_club_name}</dd><dt>Signup closes</dt><dd>{signupMeetTime(data.signup.deadline, data.season.timezone)}</dd></dl>
      <p>Each division has two women’s spots and two men’s spots for {data.club.name}. Players in that rating band get spots in registration order. Later signups join the substitute pool; players playing up wait behind them.</p>
      {!open && <p className={styles.notice}>{data.signup.schedule_changed ? "The meet schedule changed. Contact your club to reconfirm your place." : "Signup is closed. Contact your club for lineup changes."}</p>}
      {own ? <section className={own.placement === "confirmed" ? styles.success : styles.notice} aria-label="Your meet signup">
        <h2 ref={resultHeading} tabIndex={-1}>{signupPlacement(own)}</h2><p><strong>{own.name}</strong> · {own.division} division</p><p>{own.reason}</p>{own.declared_gender && <p>Gender: {signupGenderLabels[own.declared_gender]}</p>}
        <p>Save your private link to check your place or withdraw. Your club can also provide this link. Substitutes should check it for promotions.</p>
        {own.manage_url && <div className={styles.actions}><button type="button" onClick={async () => { try { await navigator.clipboard.writeText(own.manage_url!); setCopied(true); } catch { setError("Copy the private link below."); } }}>{copied ? "Private link copied" : "Copy my private link"}</button>
          <a className={styles.privateLink} href={own.manage_url}>My private signup link</a></div>}
        {open && own.status === "active" && <button disabled={busy} onClick={() => void submit(true)}>Withdraw from this meet</button>}
        {own.status === "withdrawn" && data.signup.url && <a className={styles.buttonLink} href={data.signup.url}>Register again</a>}
      </section> : open && <form className={styles.form} onSubmit={event => { event.preventDefault(); void submit(); }}>
        <label>Find your name in the approved season pool<input value={name} disabled={busy} autoComplete="name" onChange={event => { setName(event.target.value); setProfile(null); setGender(""); setConfirmedSelf(false); }} /></label>
        {searching && <p role="status">Finding players…</p>}{searchError && <p role="alert">{searchError}</p>}
        {!profile && matches.length > 0 && <fieldset><legend>Choose your player profile</legend>{matches.map(player => <label className={styles.choice} key={player.id}>
          <input type="radio" name="profile" disabled={busy} onChange={() => { setProfile(player); setGender(signupGender(player.gender)); setName(player.name); if (!player.eligible_divisions.includes(division)) setDivision(player.eligible_divisions[0] || ""); }} />
          <span><strong>{player.name}</strong><small>League rating {player.league_rating == null ? "not available" : Number(player.league_rating).toFixed(2)} · {player.gender === "female" ? "Woman" : player.gender === "male" ? "Man" : "Gender needed"}</small></span>
        </label>)}</fieldset>}
        {!profile && name.trim().length >= 2 && !searching && !matches.length && !searchError && <p>Can’t find your name? Ask your club to add or approve you in the season player pool first.</p>}
        {profile && <div className={styles.profileNotice}><strong>{profile.name}</strong><p>League rating: {profile.league_rating == null ? "Unavailable" : Number(profile.league_rating).toFixed(2)}</p></div>}
        {profile && <MeetSignupGender value={gender} onChange={setGender} disabled={busy} />}
        <label>Division<select value={division} disabled={busy} onChange={event => setDivision(event.target.value)} required>{data.season.divisions.map(value => <option key={value} value={value} disabled={Boolean(profile && !profile.eligible_divisions.includes(value))}>{value}</option>)}</select></label>
        {profile && <p className={styles.notice}>{divisionPriorityHint(division, profile.league_rating ?? null)}</p>}
        <label>Email (optional, visible only to your club)<input type="email" value={email} disabled={busy} autoComplete="email" onChange={event => setEmail(event.target.value)} /></label>
        <label className={styles.choice}><input type="checkbox" checked={confirmedSelf} disabled={busy} onChange={event => setConfirmedSelf(event.target.checked)} /><span>This is my profile and I want to play at this meet. My name, league rating and signup position will appear on this meet’s signup page.</span></label>
        <button className={styles.primary} disabled={busy || !profile || !gender || !confirmedSelf || !profile.eligible_divisions.includes(division)} type="submit">{busy ? "Registering…" : "Sign me up for this meet"}</button>
      </form>}
      <h2 style={{ marginTop: "2rem" }}>Spots and substitute pool</h2>
      <p className={styles.hint}>A full team is created when both women’s and both men’s spots are filled. Before signup closes, an eligible substitute moves up automatically when a spot opens. Playing up requires an admin to approve a vacancy and keeps rating-band players first.</p>
      <MeetSignupQueues board={data} />
    </>}
    <div className={styles.actions}><button disabled={busy || loading} onClick={() => setReload(value => value + 1)}>Refresh signup</button></div>
  </section></main>;
}
