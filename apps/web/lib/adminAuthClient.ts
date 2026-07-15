export type AdminAuthConfig = {
  supabaseUrl: string;
  supabaseAnonKey: string;
};

export type AdminSession = {
  access_token: string;
  refresh_token?: string | null;
  expires_at?: number | null;
  token_type?: string | null;
  user?: {
    id?: string | null;
    email?: string | null;
  } | null;
};

const STORAGE_KEY = "jupr_admin_session_v1";

function nowMs(): number {
  return Date.now();
}

function canUseBrowserStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function cleanBaseUrl(value?: string): string {
  return String(value || "").trim().replace(/\/$/, "");
}

function parseAuthError(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") return fallback;
  const record = payload as Record<string, unknown>;
  return String(record.error_description || record.msg || record.message || record.error || fallback);
}

function normalizeSession(payload: Record<string, unknown>): AdminSession {
  const expiresIn = Number(payload.expires_in || 0);
  const user = (payload.user && typeof payload.user === "object" ? payload.user : {}) as Record<string, unknown>;
  return {
    access_token: String(payload.access_token || ""),
    refresh_token: payload.refresh_token ? String(payload.refresh_token) : null,
    expires_at: expiresIn > 0 ? nowMs() + expiresIn * 1000 : null,
    token_type: payload.token_type ? String(payload.token_type) : "bearer",
    user: {
      id: user.id ? String(user.id) : null,
      email: user.email ? String(user.email) : null
    }
  };
}

export function getAdminAuthConfig(): AdminAuthConfig | null {
  const supabaseUrl = cleanBaseUrl(process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_JUPR_SUPABASE_URL);
  const supabaseAnonKey = String(process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_JUPR_SUPABASE_ANON_KEY || "").trim();
  if (!supabaseUrl || !supabaseAnonKey) return null;
  return { supabaseUrl, supabaseAnonKey };
}

export function loadAdminSession(): AdminSession | null {
  if (!canUseBrowserStorage()) return null;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    const session = JSON.parse(raw) as AdminSession;
    if (!session?.access_token) return null;
    return session;
  } catch {
    window.localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

export function saveAdminSession(session: AdminSession): void {
  if (!canUseBrowserStorage()) return;
  if (!session.access_token) return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  window.dispatchEvent(new CustomEvent("jupr-admin-session-change"));
}

export function clearAdminSession(): void {
  if (!canUseBrowserStorage()) return;
  window.localStorage.removeItem(STORAGE_KEY);
  window.dispatchEvent(new CustomEvent("jupr-admin-session-change"));
}

export function adminSessionIsFresh(session: AdminSession | null, graceSeconds = 120): boolean {
  if (!session?.access_token) return false;
  if (!session.expires_at) return true;
  return Number(session.expires_at) - nowMs() > graceSeconds * 1000;
}

export async function signInWithPassword(email: string, password: string): Promise<AdminSession> {
  const config = getAdminAuthConfig();
  if (!config) throw new Error("Supabase public auth configuration is missing.");
  const response = await fetch(`${config.supabaseUrl}/auth/v1/token?grant_type=password`, {
    method: "POST",
    headers: {
      apikey: config.supabaseAnonKey,
      Authorization: `Bearer ${config.supabaseAnonKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ email: email.trim(), password })
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(parseAuthError(payload, `Supabase sign-in failed (${response.status}).`));
  const session = normalizeSession((payload || {}) as Record<string, unknown>);
  if (!session.access_token) throw new Error("Supabase sign-in did not return an access token.");
  saveAdminSession(session);
  return session;
}

export async function sendMagicLink(email: string, redirectTo?: string): Promise<void> {
  const config = getAdminAuthConfig();
  if (!config) throw new Error("Supabase public auth configuration is missing.");
  const response = await fetch(`${config.supabaseUrl}/auth/v1/otp`, {
    method: "POST",
    headers: {
      apikey: config.supabaseAnonKey,
      Authorization: `Bearer ${config.supabaseAnonKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      email: email.trim(),
      create_user: false,
      redirect_to: redirectTo
    })
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(parseAuthError(payload, `Supabase magic-link request failed (${response.status}).`));
}

export async function sendPasswordResetEmail(email: string, redirectTo?: string): Promise<void> {
  const config = getAdminAuthConfig();
  if (!config) throw new Error("Supabase public auth configuration is missing.");
  const response = await fetch(`${config.supabaseUrl}/auth/v1/recover`, {
    method: "POST",
    headers: {
      apikey: config.supabaseAnonKey,
      Authorization: `Bearer ${config.supabaseAnonKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      email: email.trim(),
      redirect_to: redirectTo
    })
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(parseAuthError(payload, `Supabase password reset request failed (${response.status}).`));
}

export async function updateAdminPassword(password: string, session = loadAdminSession()): Promise<void> {
  const config = getAdminAuthConfig();
  if (!config) throw new Error("Supabase public auth configuration is missing.");
  if (!session?.access_token) throw new Error("Open a valid password reset link or sign in before setting a new password.");
  const response = await fetch(`${config.supabaseUrl}/auth/v1/user`, {
    method: "PUT",
    headers: {
      apikey: config.supabaseAnonKey,
      Authorization: `Bearer ${session.access_token}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ password })
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(parseAuthError(payload, `Supabase password update failed (${response.status}).`));
}

export async function refreshAdminSession(session = loadAdminSession()): Promise<AdminSession | null> {
  if (!session?.refresh_token) return session;
  if (adminSessionIsFresh(session)) return session;
  const config = getAdminAuthConfig();
  if (!config) return session;
  const response = await fetch(`${config.supabaseUrl}/auth/v1/token?grant_type=refresh_token`, {
    method: "POST",
    headers: {
      apikey: config.supabaseAnonKey,
      Authorization: `Bearer ${config.supabaseAnonKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ refresh_token: session.refresh_token })
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    clearAdminSession();
    throw new Error(parseAuthError(payload, `Supabase session refresh failed (${response.status}).`));
  }
  const refreshed = normalizeSession((payload || {}) as Record<string, unknown>);
  if (!refreshed.access_token) throw new Error("Supabase session refresh did not return an access token.");
  saveAdminSession(refreshed);
  return refreshed;
}

export async function signOutAdminSession(): Promise<void> {
  const session = loadAdminSession();
  const config = getAdminAuthConfig();
  clearAdminSession();
  if (!session?.access_token || !config) return;
  await fetch(`${config.supabaseUrl}/auth/v1/logout`, {
    method: "POST",
    headers: {
      apikey: config.supabaseAnonKey,
      Authorization: `Bearer ${session.access_token}`
    }
  }).catch(() => undefined);
}

export function consumeHashSession(): AdminSession | null {
  if (typeof window === "undefined") return null;
  const hash = window.location.hash.replace(/^#/, "");
  if (!hash) return null;
  const params = new URLSearchParams(hash);
  const accessToken = params.get("access_token");
  if (!accessToken) return null;
  const expiresIn = Number(params.get("expires_in") || 0);
  const session: AdminSession = {
    access_token: accessToken,
    refresh_token: params.get("refresh_token"),
    expires_at: expiresIn > 0 ? nowMs() + expiresIn * 1000 : null,
    token_type: params.get("token_type") || "bearer",
    user: null
  };
  saveAdminSession(session);
  const cleanUrl = `${window.location.pathname}${window.location.search}`;
  window.history.replaceState({}, document.title, cleanUrl);
  return session;
}
