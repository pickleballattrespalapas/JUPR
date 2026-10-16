"use client";
import { useState } from "react";
import styles from "./playerPool.module.css";
export function ShareLink({ url, label, openLabel }: { url: string; label: string; openLabel?: string }) {
  const [copied, setCopied] = useState(false), [error, setError] = useState("");
  return <div className={styles.share}><label>{label}<input readOnly value={url} onFocus={event => event.currentTarget.select()} /></label>
    <button type="button" onClick={async () => { try { await navigator.clipboard.writeText(url); setCopied(true); setError(""); } catch { setError("Select the link above and copy it."); } }}>{copied ? "Link copied" : "Copy link"}</button>{openLabel && <a className={styles.buttonLink} href={url} target="_blank" rel="noopener noreferrer">{openLabel}</a>}{error && <span role="status">{error}</span>}</div>;
}
export function RequestStatus({ error, blocked, loading }: { error: string; blocked: boolean; loading: boolean }) {
  return <>{loading && <p role="status">Loading…</p>}{error && <p role="alert" className={styles.error}>{error}{blocked && " Reload this section to check the latest information before continuing."}</p>}</>;
}
