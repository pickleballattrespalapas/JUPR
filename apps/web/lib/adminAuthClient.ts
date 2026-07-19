export type AdminAuthConfig = {
  supabaseUrl: string;
  supabaseAnonKey: string;
};

export type AdminCapabilityAssignment = {
  club_id: string;
  role: string;
  permissions: string[];
};

export type AdminCapabilities = {
  authorized: true;
  user: { email?: string | null };
  requested_club_id?: string | null;
  assignments: AdminCapabilityAssignment[];
};

export type AdminSession = {
  access_token: string;
  refresh_token?: string | null;
  expires_at?: number | null;
  token_type?: string | null;
  recovery?: boolean;
  capabilities?: AdminCapabilities | null;
  user?: {
    id?: string | null;
    email?: string | null;
  } | null;
};

export const ADMIN_PASSWORD_MIN_LENGTH = 8;

const STORAGE_KEY = "jupr_admin_session_v1";
const RECOVERY_SESSION_KEY = "jupr_admin_recovery_session_v1";
const RECOVERY_PKCE_KEY = "jupr_admin_recovery_pkce_v1";
const RECOVERY_PKCE_MAX_AGE_MS = 60 * 60 * 1000;

function nowMs(): number {
  return Date.now();
}

function canUseBrowserStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function cleanBaseUrl(value?: string): string {
  return String(value || "").trim().replace(/\/$/, "");
}

function genericSignInError(): Error {
  return new Error("Sign-in failed. Check your email and password and try again.");
}

function genericRecoveryRequestError(): Error {
  return new Error("Unable to request a recovery email right now. Wait a moment and try again.");
}

function genericRecoveryLinkError(): Error {
  return new Error("This password recovery link is invalid or expired. Request a new email.");
}

function authResponseError(payload: unknown): string {
  if (!payload || typeof payload !== "object") return "";
  const record = payload as Record<string, unknown>;
  return String(record.error_description || record.msg || record.message || record.error || "").toLowerCase();
}

function expiryMs(payload: Record<string, unknown>): number | null {
  const explicit = Number(payload.expires_at || 0);
  if (explicit > 0) return explicit > 10_000_000_000 ? explicit : explicit * 1000;
  const expiresIn = Number(payload.expires_in || 0);
  return expiresIn > 0 ? nowMs() + expiresIn * 1000 : null;
}

function normalizeSession(
  payload: Record<string, unknown>,
  options: { recovery?: boolean; fallbackRefreshToken?: string | null } = {}
): AdminSession {
  const user = (payload.user && typeof payload.user === "object" ? payload.user : {}) as Record<string, unknown>;
  return {
    access_token: String(payload.access_token || ""),
    refresh_token: payload.refresh_token
      ? String(payload.refresh_token)
      : options.fallbackRefreshToken || null,
    expires_at: expiryMs(payload),
    token_type: payload.token_type ? String(payload.token_type) : "bearer",
    recovery: Boolean(options.recovery),
    user: {
      id: user.id ? String(user.id) : null,
      email: user.email ? String(user.email) : null
    }
  };
}

function safeSameOriginRedirect(redirectTo: string | undefined, expectedPath: string): string | undefined {
  if (!redirectTo || typeof window === "undefined") return undefined;
  try {
    const candidate = new URL(redirectTo, window.location.origin);
    if (candidate.origin !== window.location.origin || candidate.pathname !== expectedPath) return undefined;
    candidate.hash = "";
    return candidate.toString();
  } catch {
    return undefined;
  }
}

function bytesToBase64Url(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return window.btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function randomPkceVerifier(): string {
  const bytes = new Uint8Array(48);
  window.crypto.getRandomValues(bytes);
  return bytesToBase64Url(bytes);
}

async function pkceChallenge(verifier: string): Promise<string> {
  const digest = await window.crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
  return bytesToBase64Url(new Uint8Array(digest));
}

function saveRecoveryVerifier(verifier: string): void {
  if (!canUseBrowserStorage()) return;
  window.localStorage.setItem(RECOVERY_PKCE_KEY, JSON.stringify({ verifier, created_at: nowMs() }));
}

function consumeRecoveryVerifier(): string | null {
  if (!canUseBrowserStorage()) return null;
  const raw = window.localStorage.getItem(RECOVERY_PKCE_KEY);
  window.localStorage.removeItem(RECOVERY_PKCE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as { verifier?: unknown; created_at?: unknown };
    const verifier = String(parsed.verifier || "");
    const createdAt = Number(parsed.created_at || 0);
    if (!verifier || !createdAt || nowMs() - createdAt > RECOVERY_PKCE_MAX_AGE_MS) return null;
    return verifier;
  } catch {
    return null;
  }
}

function cleanAuthCallbackUrl(): void {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  for (const key of [
    "code",
    "token_hash",
    "type",
    "error",
    "error_code",
    "error_description",
    "access_token",
    "refresh_token",
    "expires_in",
    "token_type"
  ]) {
    url.searchParams.delete(key);
  }
  url.hash = "";
  window.history.replaceState({}, document.title, `${url.pathname}${url.search}`);
}

export function getAdminAuthConfig(): AdminAuthConfig | null {
  const supabaseUrl = cleanBaseUrl(process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_JUPR_SUPABASE_URL);
  const supabaseAnonKey = String(process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_JUPR_SUPABASE_ANON_KEY || "").trim();
  if (!supabaseUrl || !supabaseAnonKey) return null;
  return { supabaseUrl, supabaseAnonKey };
}

export function getAdminApiBaseUrl(): string | null {
  return cleanBaseUrl(process.env.NEXT_PUBLIC_JUPR_API_BASE_URL) || null;
}

export function getDefaultAdminClubId(): string {
  return String(process.env.NEXT_PUBLIC_JUPR_ADMIN_CLUB_ID || "tres_palapas").trim() || "tres_palapas";
}

export function safeAdminNextPath(value: string | null | undefined, fallback = "/admin"): string {
  const requested = String(value || "").trim();
  if (!requested || !requested.startsWith("/") || requested.startsWith("//") || requested.includes("\\") || /[\u0000-\u001f]/.test(requested)) {
    return fallback;
  }
  try {
    const parsed = new URL(requested, "https://jupr.invalid");
    const allowed = parsed.pathname === "/admin" || parsed.pathname.startsWith("/admin/") || /^\/clubs\/[^/]+\/admin(?:\/|$)/.test(parsed.pathname);
    const authLoop = parsed.pathname === "/admin/login" || parsed.pathname === "/admin/reset-password";
    if (parsed.origin !== "https://jupr.invalid" || !allowed || authLoop) return fallback;
    return `${parsed.pathname}${parsed.search}${parsed.hash}`;
  } catch {
    return fallback;
  }
}

export function loadAdminSession(): AdminSession | null {
  if (!canUseBrowserStorage()) return null;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    const session = JSON.parse(raw) as AdminSession;
    if (!session?.access_token || session.recovery) throw new Error("invalid session");
    return session;
  } catch {
    window.localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

export function saveAdminSession(session: AdminSession): void {
  if (!canUseBrowserStorage() || !session.access_token || session.recovery) return;
  const serialized = JSON.stringify(session);
  if (window.localStorage.getItem(STORAGE_KEY) === serialized) return;
  window.localStorage.setItem(STORAGE_KEY, serialized);
  window.dispatchEvent(new CustomEvent("jupr-admin-session-change"));
}

export function clearAdminSession(): void {
  if (!canUseBrowserStorage()) return;
  window.localStorage.removeItem(STORAGE_KEY);
  window.dispatchEvent(new CustomEvent("jupr-admin-session-change"));
}

export function loadRecoverySession(): AdminSession | null {
  if (typeof window === "undefined" || typeof window.sessionStorage === "undefined") return null;
  const raw = window.sessionStorage.getItem(RECOVERY_SESSION_KEY);
  if (!raw) return null;
  try {
    const session = JSON.parse(raw) as AdminSession;
    if (!session?.access_token || !session.recovery) throw new Error("invalid recovery session");
    return session;
  } catch {
    window.sessionStorage.removeItem(RECOVERY_SESSION_KEY);
    return null;
  }
}

export function saveRecoverySession(session: AdminSession): void {
  if (typeof window === "undefined" || typeof window.sessionStorage === "undefined") return;
  if (!session.access_token || !session.recovery) return;
  window.sessionStorage.setItem(RECOVERY_SESSION_KEY, JSON.stringify(session));
}

export function clearRecoveryArtifacts(): void {
  if (canUseBrowserStorage()) window.localStorage.removeItem(RECOVERY_PKCE_KEY);
  if (typeof window !== "undefined" && typeof window.sessionStorage !== "undefined") {
    window.sessionStorage.removeItem(RECOVERY_SESSION_KEY);
  }
  cleanAuthCallbackUrl();
}

export function adminSessionIsFresh(session: AdminSession | null, graceSeconds = 120): boolean {
  if (!session?.access_token) return false;
  if (!session.expires_at) return true;
  return Number(session.expires_at) - nowMs() > graceSeconds * 1000;
}

export async function signInWithPassword(email: string, password: string): Promise<AdminSession> {
  const config = getAdminAuthConfig();
  if (!config) throw new Error("Supabase public auth configuration is missing.");
  let response: Response;
  try {
    response = await fetch(`${config.supabaseUrl}/auth/v1/token?grant_type=password`, {
      method: "POST",
      headers: {
        apikey: config.supabaseAnonKey,
        Authorization: `Bearer ${config.supabaseAnonKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ email: email.trim(), password })
    });
  } catch {
    throw genericSignInError();
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw genericSignInError();
  const session = normalizeSession((payload || {}) as Record<string, unknown>);
  if (!session.access_token) throw genericSignInError();
  return session;
}

export async function sendMagicLink(email: string, redirectTo?: string): Promise<void> {
  const config = getAdminAuthConfig();
  if (!config) throw new Error("Supabase public auth configuration is missing.");
  const safeRedirect = safeSameOriginRedirect(redirectTo, "/admin/login");
  try {
    const response = await fetch(`${config.supabaseUrl}/auth/v1/otp`, {
      method: "POST",
      headers: {
        apikey: config.supabaseAnonKey,
        Authorization: `Bearer ${config.supabaseAnonKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ email: email.trim(), create_user: false, redirect_to: safeRedirect })
    });
    if (!response.ok) throw new Error("request failed");
  } catch {
    throw new Error("Unable to request a sign-in link right now. Wait a moment and try again.");
  }
}

export async function sendPasswordResetEmail(email: string, redirectTo?: string): Promise<void> {
  const config = getAdminAuthConfig();
  if (!config) throw new Error("Supabase public auth configuration is missing.");
  if (typeof window === "undefined" || !window.crypto?.subtle) throw genericRecoveryRequestError();
  const safeRedirect = safeSameOriginRedirect(redirectTo, "/admin/reset-password");
  if (!safeRedirect) throw genericRecoveryRequestError();

  try {
    const verifier = randomPkceVerifier();
    const challenge = await pkceChallenge(verifier);
    saveRecoveryVerifier(verifier);
    const response = await fetch(`${config.supabaseUrl}/auth/v1/recover`, {
      method: "POST",
      headers: {
        apikey: config.supabaseAnonKey,
        Authorization: `Bearer ${config.supabaseAnonKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        email: email.trim(),
        redirect_to: safeRedirect,
        code_challenge: challenge,
        code_challenge_method: "s256"
      })
    });
    if (!response.ok) throw new Error("request failed");
  } catch {
    clearRecoveryArtifacts();
    throw genericRecoveryRequestError();
  }
}

export async function authorizeAdminSession(
  session: AdminSession,
  requestedClubId = getDefaultAdminClubId()
): Promise<AdminSession> {
  const apiBase = getAdminApiBaseUrl();
  if (!apiBase) throw new Error("JUPR admin API configuration is missing.");
  const params = requestedClubId ? `?club_id=${encodeURIComponent(requestedClubId)}` : "";
  let response: Response;
  try {
    response = await fetch(`${apiBase}/admin/auth/capabilities${params}`, {
      method: "GET",
      headers: { Authorization: `Bearer ${session.access_token}` },
      cache: "no-store"
    });
  } catch {
    throw new Error("Admin access could not be verified. Try again.");
  }
  if (response.status === 401) throw new Error("Your session is invalid or expired. Sign in again.");
  if (response.status === 403) throw new Error("This account is not authorized for the requested JUPR admin workspace.");
  if (!response.ok) throw new Error("Admin access could not be verified. Try again.");
  const capabilities = (await response.json().catch(() => null)) as AdminCapabilities | null;
  if (!capabilities?.authorized || !capabilities.assignments?.length) {
    throw new Error("This account is not authorized for the requested JUPR admin workspace.");
  }
  return {
    ...session,
    capabilities,
    user: {
      ...session.user,
      email: capabilities.user?.email || session.user?.email || null
    }
  };
}

export async function authorizeAndSaveAdminSession(
  session: AdminSession,
  requestedClubId = getDefaultAdminClubId(),
  options: { preserveOnUnavailable?: boolean } = {}
): Promise<AdminSession> {
  try {
    const authorized = await authorizeAdminSession(session, requestedClubId);
    saveAdminSession(authorized);
    return authorized;
  } catch (error) {
    const message = error instanceof Error ? error.message : "";
    const denied = message.includes("invalid or expired") || message.includes("not authorized");
    if (!options.preserveOnUnavailable || denied) {
      clearAdminSession();
      await revokeAdminSession(session);
    }
    throw error;
  }
}

export async function refreshAdminSession(session = loadAdminSession()): Promise<AdminSession | null> {
  if (!session?.access_token) return null;
  if (adminSessionIsFresh(session)) return session;
  if (!session.refresh_token) {
    clearAdminSession();
    throw new Error("Your session is invalid or expired. Sign in again.");
  }
  const config = getAdminAuthConfig();
  if (!config) throw new Error("Supabase public auth configuration is missing.");
  let response: Response;
  try {
    response = await fetch(`${config.supabaseUrl}/auth/v1/token?grant_type=refresh_token`, {
      method: "POST",
      headers: {
        apikey: config.supabaseAnonKey,
        Authorization: `Bearer ${config.supabaseAnonKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ refresh_token: session.refresh_token })
    });
  } catch {
    throw new Error("Your session could not be refreshed. Sign in again.");
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    clearAdminSession();
    throw new Error("Your session is invalid or expired. Sign in again.");
  }
  const refreshed = normalizeSession((payload || {}) as Record<string, unknown>, {
    fallbackRefreshToken: session.refresh_token
  });
  if (!refreshed.access_token) {
    clearAdminSession();
    throw new Error("Your session is invalid or expired. Sign in again.");
  }
  return refreshed;
}

export async function restoreAuthorizedAdminSession(
  requestedClubId = getDefaultAdminClubId()
): Promise<AdminSession | null> {
  const current = loadAdminSession();
  if (!current) return null;
  const refreshed = await refreshAdminSession(current);
  if (!refreshed) return null;
  return authorizeAndSaveAdminSession(refreshed, requestedClubId, { preserveOnUnavailable: true });
}

async function revokeAdminSession(session: AdminSession | null): Promise<void> {
  const config = getAdminAuthConfig();
  if (!session?.access_token || !config) return;
  await fetch(`${config.supabaseUrl}/auth/v1/logout?scope=local`, {
    method: "POST",
    headers: {
      apikey: config.supabaseAnonKey,
      Authorization: `Bearer ${session.access_token}`
    }
  }).catch(() => undefined);
}

export async function signOutAdminSession(): Promise<void> {
  const session = loadAdminSession();
  clearAdminSession();
  await revokeAdminSession(session);
}

export function consumeHashSession(options: { requireRecovery?: boolean } = {}): AdminSession | null {
  if (typeof window === "undefined") return null;
  const hash = window.location.hash.replace(/^#/, "");
  if (!hash) return null;
  const params = new URLSearchParams(hash);
  const accessToken = params.get("access_token");
  if (!accessToken) return null;
  const flowType = String(params.get("type") || "").toLowerCase();
  const recovery = flowType === "recovery";
  if (options.requireRecovery && !recovery) {
    cleanAuthCallbackUrl();
    return null;
  }
  if (!options.requireRecovery && recovery) return null;
  const session = normalizeSession(
    {
      access_token: accessToken,
      refresh_token: params.get("refresh_token"),
      expires_in: params.get("expires_in"),
      token_type: params.get("token_type")
    },
    { recovery }
  );
  cleanAuthCallbackUrl();
  return session;
}

async function exchangeRecoveryCode(code: string): Promise<AdminSession> {
  const config = getAdminAuthConfig();
  const verifier = consumeRecoveryVerifier();
  if (!config || !verifier) throw genericRecoveryLinkError();
  try {
    const response = await fetch(`${config.supabaseUrl}/auth/v1/token?grant_type=pkce`, {
      method: "POST",
      headers: {
        apikey: config.supabaseAnonKey,
        Authorization: `Bearer ${config.supabaseAnonKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ auth_code: code, code_verifier: verifier })
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw genericRecoveryLinkError();
    const session = normalizeSession((payload || {}) as Record<string, unknown>, { recovery: true });
    if (!session.access_token) throw genericRecoveryLinkError();
    return session;
  } catch {
    throw genericRecoveryLinkError();
  }
}

async function verifyRecoveryTokenHash(tokenHash: string): Promise<AdminSession> {
  const config = getAdminAuthConfig();
  if (!config) throw genericRecoveryLinkError();
  try {
    const response = await fetch(`${config.supabaseUrl}/auth/v1/verify`, {
      method: "POST",
      headers: {
        apikey: config.supabaseAnonKey,
        Authorization: `Bearer ${config.supabaseAnonKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ token_hash: tokenHash, type: "recovery" })
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw genericRecoveryLinkError();
    const session = normalizeSession((payload || {}) as Record<string, unknown>, { recovery: true });
    if (!session.access_token) throw genericRecoveryLinkError();
    return session;
  } catch {
    throw genericRecoveryLinkError();
  }
}

export async function consumeRecoverySession(): Promise<AdminSession | null> {
  if (typeof window === "undefined") return null;
  const params = new URLSearchParams(window.location.search);
  if (params.get("error") || params.get("error_code")) {
    clearRecoveryArtifacts();
    throw genericRecoveryLinkError();
  }

  let session: AdminSession | null = null;
  const code = String(params.get("code") || "").trim();
  const tokenHash = String(params.get("token_hash") || "").trim();
  try {
    if (code) session = await exchangeRecoveryCode(code);
    else if (tokenHash) session = await verifyRecoveryTokenHash(tokenHash);
    else session = consumeHashSession({ requireRecovery: true });
  } finally {
    if (code || tokenHash) cleanAuthCallbackUrl();
  }
  return session;
}

export async function updateAdminPassword(password: string, session = loadRecoverySession()): Promise<void> {
  const config = getAdminAuthConfig();
  if (!config) throw new Error("Supabase public auth configuration is missing.");
  if (!session?.access_token || !session.recovery) throw genericRecoveryLinkError();
  let response: Response;
  try {
    response = await fetch(`${config.supabaseUrl}/auth/v1/user`, {
      method: "PUT",
      headers: {
        apikey: config.supabaseAnonKey,
        Authorization: `Bearer ${session.access_token}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ password })
    });
  } catch {
    throw new Error("Unable to update the password right now. Try again.");
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = authResponseError(payload);
    if (detail.includes("different from the old") || detail.includes("same password")) {
      throw new Error("Choose a password that is different from the current password.");
    }
    if (detail.includes("weak") || detail.includes("password should contain") || detail.includes("password must") || detail.includes("at least") || detail.includes("character")) {
      throw new Error("That password does not meet the Supabase password policy. Use a longer, unique password with the required character types.");
    }
    throw genericRecoveryLinkError();
  }
}

export async function finishPasswordRecovery(session: AdminSession | null): Promise<void> {
  await revokeAdminSession(session);
  clearAdminSession();
  clearRecoveryArtifacts();
}
