"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  authorizeAndSaveAdminSession, consumeStaffInvitationSession, getAdminApiBaseUrl,
  getAdminAuthConfig, refreshAdminSession, signInWithPassword, type AdminSession,
} from "@/lib/adminAuthClient";
import { selectAdminWorkspace, type AvailableWorkspace } from "@/lib/adminWorkspace";
import {
  clearClubCreationDraft, clubDetailsError, readClubCreationDraft,
  saveClubCreationDraft, suggestClubSlug, type ClubCreationStep,
} from "@/lib/clubCreationDraft";
import styles from "@/components/ClubWebsite.module.css";
import layout from "./createClub.module.css";

const steps = ["Club details", "Administrator", "Review and create"];

export default function CreateClubPage() {
  const [step, setStep] = useState<ClubCreationStep>(1);
  const [session, setSession] = useState<AdminSession | null>(null);
  const [ready, setReady] = useState(false);
  const [draftLoaded, setDraftLoaded] = useState(false);
  const [savedLocally, setSavedLocally] = useState(true);
  const [mode, setMode] = useState<"signin" | "signup">("signup");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [name, setName] = useState("");
  const [slug, setSlug] = useState("");
  const [slugEdited, setSlugEdited] = useState(false);
  const [emailEnabled, setEmailEnabled] = useState<boolean | null>(null);
  const [optionsLoaded, setOptionsLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [verificationSent, setVerificationSent] = useState(false);
  const [created, setCreated] = useState<AvailableWorkspace | null>(null);
  const lock = useRef(false);
  const initialized = useRef<Promise<AdminSession | null> | null>(null);
  const mounted = useRef(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const previousStep = useRef(step);
  const api = getAdminApiBaseUrl();

  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    const draft = readClubCreationDraft();
    if (draft) {
      setName(draft.name);
      setSlug(draft.slug);
      setSlugEdited(draft.slugEdited);
    }
    setDraftLoaded(true);
    initialized.current ||= consumeStaffInvitationSession();
    initialized.current.then((restored) => {
      if (!mounted.current) return;
      setSession(restored);
      if (draft && !clubDetailsError(draft.name, draft.slug) && draft.step > 1) {
        setStep(restored ? 3 : 2);
      }
    }).catch(() => {
      if (mounted.current && draft && draft.step > 1 && !clubDetailsError(draft.name, draft.slug)) {
        setStep(2);
        setMode("signin");
        setError("Sign in again to finish creating your club. Your club details are still here.");
      }
    }).finally(() => { if (mounted.current) setReady(true); });
    void fetch(`${api}/public/club-signup-options`, { cache: "no-store", signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("Options unavailable");
        const data = await response.json();
        if (!controller.signal.aborted) {
          setEmailEnabled(data.email_enabled === true);
          if (data.email_enabled !== true) setMode("signin");
        }
      }).catch(() => {}).finally(() => {
        if (!controller.signal.aborted) setOptionsLoaded(true);
      });
    return () => { mounted.current = false; controller.abort(); };
  }, [api]);

  useEffect(() => {
    if (draftLoaded && !created) setSavedLocally(saveClubCreationDraft({ name, slug, slugEdited, step }));
  }, [draftLoaded, name, slug, slugEdited, step, created]);

  useEffect(() => {
    if (previousStep.current !== step) heading.current?.focus();
    previousStep.current = step;
  }, [step]);

  function goTo(next: ClubCreationStep) {
    setError("");
    setStep(next);
  }

  async function perform(action: () => Promise<void>) {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError("");
    try { await action(); }
    catch (e) { if (mounted.current) setError(e instanceof Error ? e.message : "Please try again."); }
    finally { lock.current = false; if (mounted.current) setBusy(false); }
  }

  function continueDetails() {
    const invalid = clubDetailsError(name, slug);
    if (invalid) { setError(invalid); return; }
    setName(name.trim());
    goTo(session ? 3 : 2);
  }

  async function authenticate() {
    if (mode === "signin") {
      const authenticated = await signInWithPassword(email, password);
      if (!mounted.current) return;
      setSession(authenticated);
      setPassword("");
      setConfirmation("");
      goTo(3);
      return;
    }
    if (!emailEnabled) throw new Error("Use an existing test account to continue in this environment.");
    if (password.length < 8 || password !== confirmation) {
      throw new Error("Use at least 8 characters and enter the same password twice.");
    }
    const config = getAdminAuthConfig();
    if (!config) throw new Error("Account setup is unavailable.");
    const response = await fetch(
      `${config.supabaseUrl}/auth/v1/signup?redirect_to=${encodeURIComponent(window.location.origin + "/create-club")}`,
      {
        method: "POST",
        headers: { apikey: config.supabaseAnonKey, "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim(), password }),
      },
    );
    const data = await response.json();
    if (!response.ok) throw new Error("Account setup could not complete. Use your existing account if you already registered, or try again later.");
    if (!mounted.current) return;
    setPassword("");
    setConfirmation("");
    if (data.access_token) {
      setSession({
        access_token: data.access_token, refresh_token: data.refresh_token,
        expires_at: Date.now() + (Number(data.expires_in) || 3600) * 1000, user: data.user,
      });
      goTo(3);
    } else setVerificationSent(true);
  }

  async function create() {
    if (clubDetailsError(name, slug)) { goTo(1); setError(clubDetailsError(name, slug)!); return; }
    let current: AdminSession | null;
    try { current = session ? await refreshAdminSession(session) : null; }
    catch { current = null; }
    if (!current) {
      setSession(null);
      setMode("signin");
      goTo(2);
      throw new Error("Sign in again to finish creating your club. Your club details are still here.");
    }
    setSession(current);
    let workspace = created;
    if (!workspace) {
      const response = await fetch(`${api}/clubs/create`, {
        method: "POST",
        headers: { Authorization: `Bearer ${current.access_token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim(), slug }),
      });
      const data = await response.json();
      if (!mounted.current) return;
      if (!response.ok) {
        if (response.status === 409) goTo(1);
        if (response.status === 401) { setSession(null); setMode("signin"); goTo(2); }
        throw new Error(typeof data.detail === "string" ? data.detail : "Check the club name and address.");
      }
      workspace = data as AvailableWorkspace;
      setCreated(workspace);
      clearClubCreationDraft();
    }
    await authorizeAndSaveAdminSession(current, workspace.club_id, { preserveOnUnavailable: true });
    if (mounted.current) selectAdminWorkspace(workspace);
  }

  const clubSummary = <div className={layout.summary}>
    <strong>{name}</strong><p className={layout.hint}>/clubs/{slug}</p>
  </div>;

  return <section className={layout.page}>
    <p className={styles.eyebrow}>A home for your pickleball club</p>
    <h1>Create a club</h1>
    <p className={layout.intro}>Start with your club’s name and web address. Then connect its administrator account and create your club.</p>
    <ol className={layout.steps} aria-label="Club creation progress">
      {steps.map((label, index) => <li key={label} aria-current={step === index + 1 ? "step" : undefined}>
        <span className={layout.number} aria-hidden="true">{index + 1}</span>{label}
      </li>)}
    </ol>
    <section className={`${styles.card} ${styles.form} ${layout.panel}`} aria-labelledby="creation-step">
      <div>
        <p className={styles.eyebrow}>Step {step} of 3</p>
        <h2 id="creation-step" ref={heading} tabIndex={-1}>{steps[step - 1]}</h2>
      </div>

      {step === 1 && <form className={styles.form} onSubmit={(e) => { e.preventDefault(); continueDetails(); }}>
        <p className={layout.intro}>What is your club called? You can add your location, logo and visitor information after creating it.</p>
        <fieldset className={`${styles.form} ${layout.fields}`} disabled={busy || !!created}>
          <label htmlFor="club-name">Club name</label>
          <input id="club-name" name="club-name" required maxLength={120} autoComplete="organization"
            value={name} placeholder="e.g. Baja Pickleball Club" onChange={(e) => {
              setName(e.target.value);
              if (!slugEdited) setSlug(suggestClubSlug(e.target.value));
            }} />
          <label htmlFor="club-address">Club web address</label>
          <div className={layout.address}>
            <span aria-hidden="true">/clubs/</span>
            <input id="club-address" name="club-address" required minLength={3} maxLength={60}
              pattern="[a-z0-9]+(-[a-z0-9]+)*" autoCapitalize="none" spellCheck={false}
              aria-describedby="club-address-help" value={slug} placeholder="baja-pickleball-club"
              onChange={(e) => { setSlugEdited(true); setSlug(e.target.value.toLowerCase()); }} />
          </div>
          <small id="club-address-help" className={layout.hint}>Suggested from your club name. You can change it using lowercase letters, numbers and hyphens. Availability is checked when you create the club.</small>
        </fieldset>
        <p className={styles.notice}>You control when your website goes live. Your new club starts unpublished.</p>
        {error && <p className={styles.error} role="alert">{error}</p>}
        <div className={layout.footer}>
          <span className={layout.hint}>{ready ? "No account needed for this step." : "Preparing the next step…"}</span>
          <button className={styles.primary} disabled={!ready || busy}>Continue →</button>
        </div>
      </form>}

      {step === 2 && <>
        {clubSummary}
        <p className={layout.intro}>Connect the account that will manage {name}. Your club details are ready.</p>
        {session ? <>
          <p>Administrator: <strong>{session.user?.email || "your verified account"}</strong></p>
          <button className={styles.primary} disabled={busy} onClick={() => goTo(3)}>Review your club →</button>
        </> : <>
          {emailEnabled === false && <p className={styles.notice}>For staging testing, use your existing test administrator account. New account verification emails are turned off here.</p>}
          {optionsLoaded && emailEnabled === null && <p className={styles.notice}>New account setup could not be checked. You can continue with an existing account.</p>}
          <div className={styles.actions}>
            <button type="button" aria-pressed={mode === "signup"} disabled={busy || !emailEnabled}
              className={mode === "signup" ? styles.primary : styles.button} onClick={() => { setMode("signup"); setError(""); setVerificationSent(false); }}>
              New administrator
            </button>
            <button type="button" aria-pressed={mode === "signin"} disabled={busy}
              className={mode === "signin" ? styles.primary : styles.button} onClick={() => { setMode("signin"); setError(""); setVerificationSent(false); }}>
              Use existing account
            </button>
          </div>
          {verificationSent ? <div className={styles.notice} role="status">
            <strong>Check your email</strong>
            <p>Open the verification link sent to {email}. It returns you here to review and create {name}.</p>
            <p>Your club has not been created yet. {savedLocally ? "Your club details are saved in this browser for 24 hours." : "Keep this tab open to retain your club details."}</p>
          </div> : <form className={styles.form} onSubmit={(e) => { e.preventDefault(); void perform(authenticate); }}>
            <fieldset className={`${styles.form} ${layout.fields}`} disabled={busy || (mode === "signup" && !emailEnabled)}>
              <label>Administrator email
                <input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
              </label>
              <label>{mode === "signup" ? "Choose a password" : "Password"}
                <input type="password" autoComplete={mode === "signup" ? "new-password" : "current-password"}
                  required minLength={mode === "signup" ? 8 : undefined} value={password} onChange={(e) => setPassword(e.target.value)} />
              </label>
              {mode === "signup" && <label>Confirm password
                <input type="password" autoComplete="new-password" required value={confirmation} onChange={(e) => setConfirmation(e.target.value)} />
              </label>}
              <button className={styles.primary}>{busy ? "Please wait…" : mode === "signup" ? "Create administrator account →" : "Continue with this account →"}</button>
            </fieldset>
            {mode === "signin" && <Link href="/admin/reset-password">Forgot password?</Link>}
          </form>}
        </>}
        {error && <p className={styles.error} role="alert">{error}</p>}
        <div className={layout.footer}><button className={styles.button} disabled={busy || !!created} onClick={() => goTo(1)}>← Club details</button></div>
      </>}

      {step === 3 && <>
        <p className={layout.intro}>Check these details, then create your club. You’ll open its admin workspace next.</p>
        <div className={layout.summary}><dl>
          <dt>Club name</dt><dd>{name}</dd>
          <dt>Club web address</dt><dd>/clubs/{slug}</dd>
          <dt>Administrator</dt><dd>{session?.user?.email || "Your verified account"}</dd>
          <dt>Website visibility</dt><dd>Unpublished — publish when you’re ready</dd>
        </dl></div>
        <p>Next, add players, organize games and personalize your club website. Only administrators need an account.</p>
        {created && <p className={styles.notice} role="status">Your club was created. Open it below, or <Link href="/admin/select-club">choose it from your clubs</Link>.</p>}
        {error && <p className={styles.error} role="alert">{error}</p>}
        <div className={layout.footer}>
          <button className={styles.button} disabled={busy || !!created} onClick={() => goTo(1)}>Edit club details</button>
          <button className={styles.primary} disabled={busy} onClick={() => void perform(create)}>
            {busy ? "Please wait…" : created ? "Open your new club →" : "Create club →"}
          </button>
        </div>
      </>}
    </section>
    {!savedLocally && <p className={layout.hint} role="status">This browser cannot save your progress. Keep this tab open while setting up your club.</p>}
    <p><Link href="/clubs">Find an existing club</Link></p>
  </section>;
}
