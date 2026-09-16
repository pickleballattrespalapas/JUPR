"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  authorizeAndSaveAdminSession,
  consumeStaffInvitationSession,
  getAdminApiBaseUrl,
  getAdminAuthConfig,
  refreshAdminSession,
  signInWithPassword,
  type AdminSession,
} from "@/lib/adminAuthClient";
import {
  selectAdminWorkspace,
  type AvailableWorkspace,
} from "@/lib/adminWorkspace";
import styles from "@/components/ClubWebsite.module.css";
export default function CreateClubPage() {
  const [session, setSession] = useState<AdminSession | null>(null),
    [ready, setReady] = useState(false),
    [mode, setMode] = useState("signin");
  const [email, setEmail] = useState(""),
    [password, setPassword] = useState(""),
    [confirmation, setConfirmation] = useState("");
  const [name, setName] = useState(""),
    [slug, setSlug] = useState(""),
    [emailEnabled, setEmailEnabled] = useState(false),
    [busy, setBusy] = useState(false),
    [message, setMessage] = useState("");
  const [created, setCreated] = useState<AvailableWorkspace | null>(null);
  const lock = useRef(false),
    initialized = useRef<Promise<AdminSession | null> | null>(null),
    mounted = useRef(false);
  const api = getAdminApiBaseUrl();
  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    initialized.current ||= consumeStaffInvitationSession();
    initialized.current
      .then((s) => {
        if (mounted.current) setSession(s);
      })
      .catch(() => {})
      .finally(() => {
        if (mounted.current) setReady(true);
      });
    void fetch(`${api}/public/club-signup-options`, {
      cache: "no-store",
      signal: controller.signal,
    })
      .then((r) => r.json())
      .then((d) => {
        if (!controller.signal.aborted)
          setEmailEnabled(d.email_enabled === true);
      })
      .catch(() => {});
    return () => {
      mounted.current = false;
      controller.abort();
    };
  }, [api]);
  async function perform(action: () => Promise<void>) {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setMessage("");
    try {
      await action();
    } catch (e) {
      if (mounted.current)
        setMessage(e instanceof Error ? e.message : "Please try again.");
    } finally {
      lock.current = false;
      if (mounted.current) setBusy(false);
    }
  }
  async function authenticate() {
    if (mode === "signin") {
      const s = await signInWithPassword(email, password);
      if (mounted.current) {
        setSession(s);
        setPassword("");
      }
      return;
    }
    if (!emailEnabled)
      throw new Error("Use an existing test login in this environment.");
    if (password.length < 8 || password !== confirmation)
      throw new Error(
        "Use at least 8 characters and enter the same password twice.",
      );
    const config = getAdminAuthConfig();
    if (!config) throw new Error("Account setup is unavailable.");
    const response = await fetch(
      `${config.supabaseUrl}/auth/v1/signup?redirect_to=${encodeURIComponent(window.location.origin + "/create-club")}`,
      {
        method: "POST",
        headers: {
          apikey: config.supabaseAnonKey,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email: email.trim(), password }),
      },
    );
    const data = await response.json();
    if (!response.ok)
      throw new Error(
        "Account setup could not complete. Try signing in if you already have an account, or try again later.",
      );
    setPassword("");
    setConfirmation("");
    if (data.access_token)
      setSession({
        access_token: data.access_token,
        refresh_token: data.refresh_token,
        expires_at: Math.floor(Date.now() / 1000) + (data.expires_in || 3600),
        user: data.user,
      });
    else
      setMessage(
        "Check your email to confirm your account. The verification link returns you here to create your club. Already registered? Use Sign in.",
      );
  }
  async function create() {
    if (!session) return;
    const current = await refreshAdminSession(session);
    if (!current) {
      setSession(null);
      throw new Error("Sign in again to create your club.");
    }
    let workspace = created;
    if (!workspace) {
      const response = await fetch(`${api}/clubs/create`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${current.access_token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name, slug }),
      });
      const data = await response.json();
      if (!response.ok)
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Check the club name and address.",
        );
      workspace = data as AvailableWorkspace;
      setCreated(workspace);
    }
    await authorizeAndSaveAdminSession(current, workspace.club_id);
    selectAdminWorkspace(workspace);
  }
  return (
    <section className={styles.page} style={{ maxWidth: 720 }}>
      <p className={styles.eyebrow}>Start your club’s next chapter</p>
      <h1>Create a club</h1>
      <p>
        Use your administrator account to create a club workspace. Add your
        players, build your website, and publish when you’re ready.
      </p>
      {!ready ? (
        <p role="status">Checking your sign-in…</p>
      ) : !session ? (
        <section className={`${styles.card} ${styles.form}`}>
          <div className={styles.actions}>
            <button
              className={mode === "signin" ? styles.primary : styles.button}
              onClick={() => {
                setMode("signin");
                setMessage("");
              }}
            >
              Sign in
            </button>
            <button
              className={mode === "signup" ? styles.primary : styles.button}
              onClick={() => {
                setMode("signup");
                setMessage("");
              }}
            >
              New administrator account
            </button>
          </div>
          {!emailEnabled && (
            <p className={styles.notice}>
              Staging uses existing test logins. Sign in with your test account
              to create a club; no verification email is needed for an already
              verified account.
            </p>
          )}
          <form
            className={styles.form}
            onSubmit={(e) => {
              e.preventDefault();
              void perform(authenticate);
            }}
          >
            <fieldset
              className={styles.form}
              disabled={busy || (mode === "signup" && !emailEnabled)}
              style={{ border: 0, padding: 0 }}
            >
              <label>
                Email
                <input
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </label>
              <label>
                {mode === "signup" ? "Choose a password" : "Password"}
                <input
                  type="password"
                  autoComplete={
                    mode === "signup" ? "new-password" : "current-password"
                  }
                  required
                  minLength={8}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </label>
              {mode === "signup" && (
                <label>
                  Confirm password
                  <input
                    type="password"
                    autoComplete="new-password"
                    required
                    value={confirmation}
                    onChange={(e) => setConfirmation(e.target.value)}
                  />
                </label>
              )}
              <button className={styles.primary}>
                {busy
                  ? "Please wait…"
                  : mode === "signin"
                    ? "Sign in to continue"
                    : "Create administrator account"}
              </button>
            </fieldset>
          </form>
          <Link href="/admin/reset-password">Forgot password?</Link>
        </section>
      ) : (
        <form
          className={`${styles.card} ${styles.form}`}
          onSubmit={(e) => {
            e.preventDefault();
            void perform(create);
          }}
        >
          <p>
            Signed in as{" "}
            <strong>
              {session.user?.email || "your administrator account"}
            </strong>
            .
          </p>
          <fieldset
            className={styles.form}
            disabled={busy || !!created}
            style={{ border: 0, padding: 0 }}
          >
            <label>
              Club name
              <input
                required
                maxLength={120}
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Your club’s name"
              />
            </label>
            <label>
              Club web address
              <input
                required
                minLength={3}
                maxLength={60}
                pattern="[a-z0-9]+(-[a-z0-9]+)*"
                value={slug}
                onChange={(e) => setSlug(e.target.value.toLowerCase())}
                placeholder="your-club"
              />
            </label>
            <small>
              Your website will be at /clubs/{slug || "your-club"}. Use
              lowercase letters, numbers and hyphens.
            </small>
          </fieldset>
          <p>
            Your club starts with an unpublished website. You become its
            administrator. No player accounts are created.
          </p>
          <button className={styles.primary} disabled={busy}>
            {busy
              ? "Please wait…"
              : created
                ? "Open your new club"
                : "Create club workspace"}
          </button>
          {created && (
            <p>
              Your club was created. Use the button to continue, or{" "}
              <Link href="/admin/select-club">choose it from your clubs</Link>.
            </p>
          )}
        </form>
      )}
      {message && (
        <p className={styles.notice} role="status">
          {message}
        </p>
      )}
      <p>
        <Link href="/clubs">Find an existing club</Link>
      </p>
    </section>
  );
}
