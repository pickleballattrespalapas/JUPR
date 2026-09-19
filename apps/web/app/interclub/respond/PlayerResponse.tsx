"use client";

import { useEffect, useRef, useState } from "react";
import { PlayerResponseDetails, PlayerSignupError, playerSignupRequest, seasonDates, signupErrorMessage, signupMeetTime } from "@/lib/interclubPlayerSignup";
import styles from "../player-signup.module.css";

type MeetStatus = "available" | "maybe" | "unavailable";

function usableReview(value: PlayerResponseDetails): boolean {
  return (value.kind === "season" || value.kind === "meet") && !!value.member?.id && Number.isInteger(value.member.revision)
    && (value.kind !== "meet" || (!!value.meet?.id && Number.isInteger(value.availability?.revision)));
}

export default function PlayerResponse() {
  const token = useRef<string | null>(null), pending = useRef(false);
  const mutation = useRef<AbortController | null>(null), confirmation = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<PlayerResponseDetails | null>(null);
  const [loading, setLoading] = useState(true), [reload, setReload] = useState(0), [busy, setBusy] = useState(false);
  const [error, setError] = useState(""), [blocked, setBlocked] = useState(false), [message, setMessage] = useState("");
  const [name, setName] = useState(""), [email, setEmail] = useState("");
  const [divisions, setDivisions] = useState<string[]>([]), [notes, setNotes] = useState("");
  const [answer, setAnswer] = useState<MeetStatus | "">(""), [confirmWithdrawal, setConfirmWithdrawal] = useState(false);

  function show(value: PlayerResponseDetails) {
    setData(value); setName(value.member.name); setEmail(value.member.email);
    setDivisions(value.member.divisions); setNotes(value.member.notes || "");
    setAnswer(value.availability?.status === "invited" ? "" : value.availability?.status || "");
    setConfirmWithdrawal(false);
  }

  useEffect(() => {
    // A personal capability stays out of request URLs, referrers and persistent
    // browser storage. Preserve it in this mounted component for safe retries.
    if (token.current === null) {
      token.current = new URLSearchParams(window.location.hash.slice(1)).get("token") || "";
      if (window.location.hash) window.history.replaceState(window.history.state, "", `${window.location.pathname}${window.location.search}`);
    }
    if (!token.current) {
      setError("Open your full personal link from the invitation or your original signup. No sign-in is needed.");
      setLoading(false); return;
    }
    const controller = new AbortController();
    setLoading(true); setError(""); setMessage(""); setBlocked(true);
    playerSignupRequest<PlayerResponseDetails>("/public/interclub-player-response/review", controller.signal, { token: token.current })
      .then(value => {
        if (!usableReview(value)) throw new PlayerSignupError("We could not read your invitation. Ask your club administrator for a new link.", 503);
        if (!controller.signal.aborted) { show(value); setBlocked(false); }
      }).catch(err => { if (!controller.signal.aborted) { setData(null); setError(signupErrorMessage(err)); } })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [reload]);
  useEffect(() => () => mutation.current?.abort(), []);
  useEffect(() => { if (message) confirmation.current?.focus(); }, [message]);

  async function save(body: Record<string, unknown>, success: string) {
    const withdrawalAllowed = data?.kind === "season" && data.can_withdraw && body.action === "update_season" && body.status === "withdrawn";
    if (!data || !token.current || pending.current || blocked || (!data.can_respond && !withdrawalAllowed)) return;
    const previous = data;
    const controller = new AbortController(); mutation.current = controller; pending.current = true;
    setBusy(true); setError(""); setMessage("");
    try {
      const result = await playerSignupRequest<PlayerResponseDetails>("/public/interclub-player-response/respond", controller.signal, { token: token.current, ...body });
      if (!usableReview(result) || result.kind !== previous.kind || result.member.id !== previous.member.id || result.club.id !== previous.club.id || result.season.id !== previous.season.id || result.meet?.id !== previous.meet?.id) {
        throw new PlayerSignupError("We could not confirm the saved details. Reload before making another change.", 409);
      }
      if (!controller.signal.aborted) { show(result); setMessage(success); }
    } catch (err) {
      if (!controller.signal.aborted) { setError(signupErrorMessage(err)); if (!(err instanceof PlayerSignupError) || [404, 409, 503].includes(err.status)) setBlocked(true); }
    } finally { pending.current = false; if (!controller.signal.aborted) setBusy(false); }
  }

  function saveSeason(status: "active" | "withdrawn") {
    if (!data || data.kind !== "season") return;
    const fields = status === "withdrawn" ? data.member : { name: name.trim(), email: email.trim(), divisions, notes: notes.trim() };
    void save({ action: "update_season", expected_revision: data.member.revision, name: fields.name, email: fields.email, divisions: fields.divisions, notes: fields.notes, status },
      status === "withdrawn" ? "You’ve left the player pool. Your club will no longer send you new meet invitations for this season."
      : data.member.status === "withdrawn" ? "You’re back in the season player pool. Your club can invite you to upcoming meets." : "Your season signup is updated.");
  }

  const disabled = busy || blocked || loading || !data?.can_respond;
  const withdrawalDisabled = busy || blocked || loading || !data?.can_withdraw;
  return <div className={styles.page}><section className={styles.card}>
    <p className={styles.eyebrow}>{data?.club.name || "Your interclub invitation"}</p>
    <h1>{data?.kind === "season" ? "Your season signup" : data?.kind === "meet" ? "Can you play in this meet?" : "Your interclub availability"}</h1>
    {loading ? <p role="status">Loading your invitation…</p> : data && <>
      <h2>{data.season.name}</h2>
      <p className={styles.muted}>{data.kind === "season" ? seasonDates(data.season) : `Hi ${data.member.name}. Let your club know whether you’re available.`}</p>
      {message && <div className={styles.success} role="status" tabIndex={-1} ref={confirmation}><strong>{message}</strong></div>}
      {data.kind === "meet" && data.meet && data.availability ? <>
        <dl className={styles.details}>
          <dt>Meet</dt><dd>{signupMeetTime(data.meet.starts_at, data.season.timezone)}</dd>
          {data.meet.host_club_name && <><dt>Host club</dt><dd>{data.meet.host_club_name}</dd></>}
          <dt>Time zone</dt><dd>{data.season.timezone.replace(/_/g, " ")}</dd>
          {data.availability.deadline && <><dt>Reply by</dt><dd>{signupMeetTime(data.availability.deadline, data.season.timezone)}</dd></>}
          <dt>Your response</dt><dd>{data.availability.status === "invited" ? "Not answered yet" : data.availability.status === "available" ? "Available" : data.availability.status === "maybe" ? "Maybe" : "Not available"}</dd>
        </dl>
        <p className={styles.intro}>This is for this meet only. Your administrator will choose the final teams and confirm who is playing.</p>
        {!data.can_respond ? <div className={styles.notice}><h2>Responses are closed</h2><p>{data.member.status === "withdrawn" ? "You’ve left the season player pool. Contact your club administrator if your plans have changed." : "Your saved response is shown above. Contact your club administrator if your availability changes."}</p></div> : <form className={styles.form} onSubmit={event => {
          event.preventDefault();
          if (answer) void save({ action: "respond_meet", expected_revision: data.availability!.revision, status: answer }, answer === "available" ? "Your club knows you’re available. Your administrator will confirm the final teams." : answer === "maybe" ? "Your club knows you’re a maybe. You can update your response before the deadline." : "Your club knows you’re not available for this meet. You’re still in the season player pool.");
        }}>
          <fieldset disabled={disabled}><legend>Your availability for this meet</legend><div className={styles.responseChoices}>{([
            ["available", "Yes, I’m available", "I’d like to be considered for a team."],
            ["maybe", "Maybe", "I’m interested, but my plans aren’t confirmed."],
            ["unavailable", "No, I can’t play this meet", "Keep me in the pool for future meets."],
          ] as const).map(([value, label, hint]) => <label key={value} className={styles.choice}><input type="radio" name="availability" value={value} checked={answer === value} onChange={() => setAnswer(value)} /><span><strong>{label}</strong><small>{hint}</small></span></label>)}</div></fieldset>
          <button className={styles.primary} disabled={disabled || !answer}>{busy ? "Saving your response…" : "Save my response"}</button>
          <p className={styles.hint}>You can use your personal invitation link again to update this response before the deadline.</p>
        </form>}
      </> : data.kind === "season" && <>
        <div className={styles.notice}><strong>{data.member.status === "active" ? "You’re in your club’s player pool" : "You’ve left this season’s player pool"}</strong><p>{data.member.status === "active" ? "You’ll decide separately for each meet. Joining the pool does not commit you to every date or reserve a team place." : "You won’t receive new meet invitations for this season. Your previous meet results and rosters stay on record."}</p></div>
        {!data.can_respond && <p className={styles.notice}>Season signup changes are currently closed. {data.can_withdraw ? "You can still leave the player pool below. Contact your club administrator for other changes." : "Contact your club administrator if you need a change."}</p>}
        <form className={styles.form} onSubmit={event => { event.preventDefault(); saveSeason("active"); }}>
          <label>Your name<input required maxLength={120} autoComplete="name" value={name} disabled={disabled} onChange={event => setName(event.target.value)} /></label>
          <label>Email<input required type="email" maxLength={254} autoComplete="email" value={email} disabled={disabled} onChange={event => setEmail(event.target.value)} /></label>
          <fieldset disabled={disabled}><legend>Divisions you’re interested in</legend><div className={styles.choices}>{data.season.divisions.map(division => <label className={styles.choice} key={division}><input type="checkbox" checked={divisions.includes(division)} onChange={event => setDivisions(current => event.target.checked ? [...current, division] : current.filter(item => item !== division))} /><span>{division}</span></label>)}</div></fieldset>
          <label>Availability notes <span className={styles.hint}>(optional)</span><textarea maxLength={1000} value={notes} disabled={disabled} onChange={event => setNotes(event.target.value)} /></label>
          {data.can_respond && <button className={styles.primary} disabled={disabled}>{busy ? "Saving your signup…" : data.member.status === "withdrawn" ? "Rejoin the season player pool" : "Save my changes"}</button>}
        </form>
        {data.can_withdraw && data.member.status === "active" && <><hr className={styles.divider} />{confirmWithdrawal ? <div className={styles.notice}><h2>Leave the player pool?</h2><p>This stops new meet invitations for this season. It won’t remove you from teams already confirmed; contact your club administrator about those.</p><div className={styles.actions}><button type="button" disabled={withdrawalDisabled} onClick={() => saveSeason("withdrawn")}>{busy ? "Leaving…" : "Yes, leave the player pool"}</button><button type="button" disabled={withdrawalDisabled} onClick={() => setConfirmWithdrawal(false)}>Stay in the pool</button></div></div>
          : <button type="button" disabled={withdrawalDisabled} onClick={() => setConfirmWithdrawal(true)}>Leave the season player pool</button>}</>}
      </>}
    </>}
    {error && <div className={styles.error} role="alert"><p>{error}</p>{token.current && !loading && <button type="button" disabled={busy} onClick={() => setReload(value => value + 1)}>Reload latest details</button>}</div>}
  </section></div>;
}
