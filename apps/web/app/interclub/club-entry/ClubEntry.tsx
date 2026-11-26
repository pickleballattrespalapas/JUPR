"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import type { ClubEntryContactMethod, ClubEntryContext, ClubEntryOffer, ClubEntryParticipation, ClubEntryRegistration } from "@/lib/interclubClubEntry";
import styles from "../player-signup.module.css";
import entryStyles from "./club-entry.module.css";
import { copy, type EntryLanguage } from "./copy";

type EntryError = "expired" | "unavailable" | "conflict" | "invalid" | "limited" | "uncertain" | "missing"
  | "invalidTimes" | "duplicateDate" | "phoneError" | "noDivisions";

async function requestEntry(action: "read" | "save", payload: object, signal: AbortSignal): Promise<ClubEntryContext> {
  const response = await fetch("/api/interclub-club-entry/" + action, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    cache: "no-store", credentials: "omit", referrerPolicy: "no-referrer", signal,
  });
  if (!response.ok) throw Object.assign(new Error(), { status: response.status });
  const data: ClubEntryContext = await response.json();
  if (!data || !Number.isInteger(data.revision) || typeof data.club_name !== "string" || !Array.isArray(data.divisions)
      || (action === "save" && !data.registration)) throw new Error();
  return data;
}

function requestError(cause: unknown, fallback: EntryError): EntryError {
  const status = (cause as {status?: number} | null)?.status;
  if (status === 404 || status === 401 || status === 403 || status === 410) return "expired";
  if (status === 409) return "conflict";
  if (status === 400 || status === 422) return "invalid";
  if (status === 429) return "limited";
  return fallback;
}

export default function ClubEntry() {
  const [language, setLanguage] = useState<EntryLanguage>("en");
  const token = useRef("");
  const [load, setLoad] = useState(0);
  const [context, setContext] = useState<ClubEntryContext | null>(null);
  const [error, setError] = useState<EntryError | null>(null);
  const t = copy[language];

  useEffect(() => {
    const param = new URLSearchParams(window.location.search).get("lang");
    setLanguage(param === "es" || (param !== "en" && navigator.language.toLowerCase().startsWith("es")) ? "es" : "en");
    function takeFragment() {
      if (!window.location.hash) return;
      token.current = new URLSearchParams(window.location.hash.slice(1)).get("token") || "";
      window.history.replaceState(window.history.state, "", window.location.pathname + window.location.search);
      setContext(null);
      setError(null);
      setLoad(n => n + 1);
    }
    takeFragment();
    window.addEventListener("hashchange", takeFragment);
    return () => window.removeEventListener("hashchange", takeFragment);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setContext(null);
    setError(null);
    if (!token.current) {
      setError("missing");
      return;
    }
    requestEntry("read", { token: token.current }, controller.signal).then(result => {
      if (!controller.signal.aborted) setContext(result);
    }).catch(reason => {
      if (!controller.signal.aborted) setError(requestError(reason, "unavailable"));
    });
    return () => controller.abort();
  }, [load]);

  return <div className={styles.page} lang={language}><section className={styles.card}>
    <div className={entryStyles.language} aria-label="Language / Idioma" role="group">
      <button type="button" lang="en" aria-pressed={language === "en"} onClick={() => setLanguage("en")}>English</button>
      <button type="button" lang="es" aria-pressed={language === "es"} onClick={() => setLanguage("es")}>Español</button>
    </div>
    <p className={styles.eyebrow}>{context?.season_name || t.privateEntry}</p>
    <h1>{t.pageTitle}{context ? " · " + context.club_name : ""}</h1>
    {error && <div className={styles.error} role="alert"><p>{t[error]}</p>
      {error !== "missing" && error !== "expired" && <button type="button" onClick={() => setLoad(n => n + 1)}>{t.retry}</button>}
    </div>}
    {!error && !context && <p role="status">{t.loading}</p>}
    {context && <EntryForm key={load} initial={context} language={language} token={token.current} onReload={() => setLoad(n => n + 1)} />}
  </section></div>;
}

function EntryForm({ initial, language, token, onReload }: {
  initial: ClubEntryContext; language: EntryLanguage; token: string; onReload: () => void;
}) {
  const [context, setContext] = useState(initial);
  const saved = context.registration;
  const [editing, setEditing] = useState(!saved);
  const [captain, setCaptain] = useState(saved?.captain ?? "");
  const [method, setMethod] = useState<ClubEntryContactMethod>(saved?.contact_method ?? "whatsapp");
  const [contact, setContact] = useState(saved?.contact ?? "");
  const [divisions, setDivisions] = useState<string[]>(saved?.divisions ?? []);
  const [participation, setParticipation] = useState<ClubEntryParticipation | "">(saved?.participation ?? "");
  const [travelNotes, setTravelNotes] = useState(saved?.travel_notes ?? "");
  const [offers, setOffers] = useState<ClubEntryOffer[]>(saved?.host_offers ?? []);
  const [error, setError] = useState<EntryError | null>(null);
  const [busy, setBusy] = useState(false);
  const [inviteState, setInviteState] = useState<"copied" | "copyInstead" | null>(null);
  const mutation = useRef<AbortController | null>(null);
  const statusRef = useRef<HTMLDivElement>(null);
  const t = copy[language];
  const hosts = participation === "host" || participation === "host_travel" || participation === "undecided";
  const blocked = error === "conflict" || error === "uncertain" || error === "expired";
  const invitation = t.invitation(context.club_name, context.season_name);

  useEffect(() => () => mutation.current?.abort(), []);
  useEffect(() => { if (!editing) statusRef.current?.focus(); }, [editing]);

  function editOffer(index: number, patch: Partial<ClubEntryOffer>) {
    setOffers(current => current.map((offer, i) => i === index ? { ...offer, ...patch } : offer));
  }

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (mutation.current || blocked || !participation) return;
    if (!divisions.length) { setError("noDivisions"); return; }
    if (method !== "email" && (!/^[\d\s()+.\-]+$/.test(contact) || !/^\d{7,15}$/.test(contact.replace(/\D/g, "")))) {
      setError("phoneError"); return;
    }
    const hostOffers = hosts ? offers : [];
    if (new Set(hostOffers.map(offer => offer.date)).size !== hostOffers.length) { setError("duplicateDate"); return; }
    if (hostOffers.some(o => o.start_time && o.end_time && o.end_time <= o.start_time)) { setError("invalidTimes"); return; }
    const registration: ClubEntryRegistration = {
      captain: captain.trim(), contact_method: method, contact: contact.trim(), divisions, participation,
      travel_notes: travelNotes.trim(), host_offers: hostOffers,
    };
    const controller = new AbortController();
    mutation.current = controller;
    setBusy(true);
    setError(null);
    try {
      const result = await requestEntry("save", { token, expected_revision: context.revision, registration }, controller.signal);
      if (!controller.signal.aborted) {
        setContext(result);
        setEditing(false);
        setInviteState(null);
      }
    } catch (cause) {
      if (!controller.signal.aborted) setError(requestError(cause, "uncertain"));
    } finally {
      mutation.current = null;
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  async function copyInvitation() {
    try {
      await navigator.clipboard.writeText(invitation);
      setInviteState("copied");
    } catch {
      setInviteState("copyInstead");
    }
  }

  function showDate(iso: string) {
    return new Intl.DateTimeFormat(language, { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" })
      .format(new Date(iso + "T12:00:00Z"));
  }

  if (!editing && saved) return <>
    <div ref={statusRef} className={styles.success} role="status" tabIndex={-1}>
      <h2>{t.saved}</h2><p>{t.savedNote}</p>
    </div>
    <dl className={styles.details}>
      <dt>{t.contactHeading}</dt><dd>{saved.captain} · {t[saved.contact_method]} · {saved.contact}</dd>
      <dt>{t.teams}</dt><dd>{saved.divisions.join(", ")}</dd>
      <dt>{t.participation}</dt><dd>{t[saved.participation]}</dd>
      {saved.host_offers.map((offer, index) => <div key={offer.date} style={{ display: "contents" }}><dt>{t.offerNumber} {index + 1}</dt>
        <dd>{showDate(offer.date)}{offer.start_time && " · " + offer.start_time}{offer.end_time && " – " + offer.end_time}
          {offer.courts && " · " + t.courts + ": " + offer.courts}</dd></div>)}
      {saved.travel_notes && <><dt>{t.travelNotes}</dt><dd>{saved.travel_notes}</dd></>}
    </dl>
    <button type="button" onClick={() => setEditing(true)}>{t.edit}</button>
    <hr className={styles.divider} /><h2>{t.inviteTitle}</h2><p>{invitation}</p>
    <button type="button" onClick={copyInvitation}>{t.copyInvite}</button>
    {inviteState && <p role="status">{t[inviteState]}</p>}
    {inviteState === "copyInstead" && <textarea aria-label={t.copyInvite} readOnly value={invitation} onFocus={e => e.target.select()} />}
  </>;

  return <>
    <p className={styles.intro}>{t.intro}</p>
    <details className={styles.notice}><summary>{t.title}</summary>
      <p>{t.commitment}</p>
      {context.start_date && context.end_date && <p>{t.season} · {showDate(context.start_date)} – {showDate(context.end_date)}</p>}
    </details>
    <form className={styles.form} onSubmit={save} autoComplete="off">
      <label>{t.name}<input required maxLength={120} value={captain} disabled={busy || blocked} onChange={e => setCaptain(e.target.value)} /></label>
      <label>{t.method}<select value={method} disabled={busy || blocked} onChange={e => {
        const next = e.target.value as ClubEntryContactMethod;
        if ((method === "email") !== (next === "email")) setContact("");
        setMethod(next);
      }}>{(["whatsapp", "email", "phone"] as const).map(value => <option key={value} value={value}>{t[value]}</option>)}</select></label>
      <label>{method === "email" ? t.emailField : t.phoneField}<input type={method === "email" ? "email" : "tel"} required
        pattern={method === "email" ? "[^@\\s]+@[^@\\s]+\\.[^@\\s]+" : undefined} maxLength={254} value={contact} disabled={busy || blocked} onChange={e => setContact(e.target.value)} /></label>
      <fieldset disabled={busy || blocked}><legend>{t.teams}</legend><p className={styles.hint}>{t.teamHint}</p>
        <div className={styles.choices}>{context.divisions.map((division, index) => <label className={styles.choice} key={division}>
          <input type="checkbox" checked={divisions.includes(division)} required={index === 0 && !divisions.length}
            onChange={e => setDivisions(current => e.target.checked ? [...current, division] : current.filter(item => item !== division))} />
          {division}</label>)}</div>
      </fieldset>
      <label>{t.participation}<select required value={participation} disabled={busy || blocked}
        onChange={e => {
          const next = e.target.value as ClubEntryParticipation | "";
          setParticipation(next);
          if (next === "travel") setOffers([]);
        }}><option value="">{t.choose}</option>{(["host_travel", "host", "travel", "undecided"] as const).map(value => <option key={value} value={value}>{t[value]}</option>)}</select></label>
      {hosts && <fieldset disabled={busy || blocked}><legend>{t.hosting}</legend><p className={styles.hint}>{t.hostingHint}</p>
        {offers.map((offer, index) => <fieldset className={entryStyles.offer} key={index}><legend>{t.offerNumber} {index + 1}</legend>
          <label>{t.date}<input type="date" required value={offer.date} min={context.start_date || undefined} max={context.end_date || undefined}
            onChange={e => editOffer(index, { date: e.target.value })} /></label>
          <label>{t.starts} ({t.optional})<input type="time" value={offer.start_time || ""} onChange={e => editOffer(index, { start_time: e.target.value || null })} /></label>
          <label>{t.ends} ({t.optional})<input type="time" value={offer.end_time || ""} onChange={e => editOffer(index, { end_time: e.target.value || null })} /></label>
          <label>{t.courts} ({t.optional})<input type="number" min={1} max={100} step={1} value={offer.courts ?? ""} onChange={e => editOffer(index, { courts: e.target.value === "" ? null : Number(e.target.value) })} /></label>
          <button type="button" onClick={() => setOffers(current => current.filter((_, i) => i !== index))}>{t.remove}</button>
        </fieldset>)}
        {offers.length < 2 && <button type="button" onClick={() => setOffers(current => [...current, { date: "", start_time: null, end_time: null, courts: null }])}>{offers.length ? t.addSecond : t.addDate}</button>}
      </fieldset>}
      <label>{t.travelNotes} ({t.optional})<span className={styles.hint}>{t.travelHint}</span>
        <textarea maxLength={1000} value={travelNotes} disabled={busy || blocked} onChange={e => setTravelNotes(e.target.value)} /></label>
      {error && <div className={styles.error} role="alert"><p>{t[error]}</p>
        {(error === "conflict" || error === "uncertain") && <button type="button" onClick={onReload}>{t.reload}</button>}</div>}
      <button type="submit" className={styles.primary} disabled={busy || blocked}>{busy ? t.saving : saved ? t.update : t.save}</button>
    </form>
  </>;
}
