"use client";

import { useEffect, useRef, useState } from "react";
import { FormDialog, actionSuccess, InteractionActionError } from "@/components/interaction";
import { ConfirmAction } from "@/components/ConfirmAction";
import TournamentSponsorDisplay from "@/components/TournamentSponsorDisplay";
import { normalizeSponsorWebsite, sponsorPlacement, sponsorTierLabels, sponsorTiers, type SponsorDraft, type SponsorTier } from "@/lib/tournamentSponsors";

type Props = {
  sponsors: SponsorDraft[];
  tournamentName: string;
  disabled: boolean;
  onSave: (sponsors: SponsorDraft[]) => Promise<boolean>;
  onUpload: (base64: string) => Promise<{ logo_path: string; logo_url: string }>;
};
const input = { width: "100%", minWidth: 0, boxSizing: "border-box" as const, padding: ".6rem", border: "1px solid #cbd5e1", borderRadius: 8, font: "inherit" };
const button = { padding: ".5rem .75rem", background: "white", border: "1px solid #cbd5e1", borderRadius: 8, cursor: "pointer" };
const newSponsor = (): SponsorDraft => ({ id: crypto.randomUUID(), name: "", tier: "supporting", level: "", notes: "", website: "", logo_path: "", logo_url: "", is_visible: true });

export default function TournamentSponsorEditor({ sponsors, tournamentName, disabled, onSave, onUpload }: Props) {
  const [draft, setDraft] = useState<SponsorDraft | null>(null);
  const [baseline, setBaseline] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [localUrl, setLocalUrl] = useState("");
  const [preview, setPreview] = useState(false);
  const [phone, setPhone] = useState(false);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const nameRef = useRef<HTMLInputElement>(null);
  const uploadRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (!file) { setLocalUrl(""); return; }
    const url = URL.createObjectURL(file); setLocalUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  function edit(value: SponsorDraft) { setDraft({ ...value }); setBaseline(JSON.stringify(value)); setFile(null); setError(""); }
  async function save(rows: SponsorDraft[]) {
    if (!await onSave(rows)) throw new InteractionActionError("The sponsor save could not be confirmed. Check the current tournament draft before retrying. Your edits are still here.");
    setMessage("Sponsor draft saved. Publish from Review when you’re ready.");
  }
  async function submit() {
    if (!draft || !draft.name.trim()) throw new InteractionActionError("Enter the sponsor name.", { kind: "validation" });
    if (error) throw new InteractionActionError(error, { kind: "validation" });
    let website: string;
    try { website = normalizeSponsorWebsite(draft.website); }
    catch { throw new InteractionActionError("Enter an HTTP or HTTPS sponsor website, such as https://example.com.", { kind: "validation" }); }
    const next = { ...draft, name: draft.name.trim(), website };
    setWorking(true);
    try {
      if (file) {
        const encoded = await new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(",")[1]); reader.onerror = () => reject(new InteractionActionError("The logo could not be read. Choose the image again.", { kind: "validation" })); reader.readAsDataURL(file); });
        Object.assign(next, await onUpload(encoded));
        // Reuse a completed upload if the subsequent draft save needs retrying.
        setDraft(next); setFile(null);
      }
      await save(sponsors.some(s => s.id === next.id) ? sponsors.map(s => s.id === next.id ? next : s) : [...sponsors, next]);
      return actionSuccess("Sponsor saved", "Saved to the tournament draft.");
    } finally { setWorking(false); }
  }
  async function move(id: string, direction: number) {
    const sponsor = sponsors.find(s => s.id === id)!;
    const peers = sponsors.filter(s => s.tier === sponsor.tier);
    const other = peers[peers.indexOf(sponsor) + direction];
    if (!other) return;
    const rows = [...sponsors], a = rows.indexOf(sponsor), b = rows.indexOf(other);
    [rows[a], rows[b]] = [rows[b], rows[a]];
    setWorking(true); setError("");
    try { await save(rows); } catch (exc) { setError(exc instanceof Error ? exc.message : "Unable to reorder sponsors."); } finally { setWorking(false); }
  }
  const busy = disabled || working;
  const visible = sponsors.filter(s => s.is_visible);
  return <article style={{ border: "1px solid #e2e8f0", borderRadius: 14, padding: "1rem", background: "white" }}>
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: 12 }}><div><h3 style={{ margin: 0 }}>Sponsors</h3><p style={{ color: "#64748b", margin: ".25rem 0" }}>Names, logos, and links for your tournament pages.</p></div><div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}><button type="button" style={button} disabled={busy || sponsors.length >= 50} onClick={() => edit(newSponsor())}>Add sponsor</button><button type="button" style={button} onClick={() => setPreview(!preview)} aria-expanded={preview}>Preview placement</button></div></div>
    {sponsorTiers.map(tier => { const rows = sponsors.filter(s => s.tier === tier); return <section key={tier} style={{ marginTop: 20 }}><h4 style={{ margin: 0 }}>{sponsorTierLabels[tier]}</h4><p style={{ margin: "4px 0", color: "#64748b", fontSize: ".85rem" }}>{sponsorPlacement(tier)}</p>{rows.map((s, index) => <div key={s.id} style={{ padding: "12px 0", borderBottom: "1px solid #e2e8f0", display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center" }}><div style={{ flex: "1 1 160px", minWidth: 0, overflowWrap: "anywhere" }}><strong>{s.name}</strong><p style={{ color: "#64748b", fontSize: ".85rem", margin: "3px 0" }}>{s.is_visible ? "Visible" : "Hidden"} · {s.logo_path ? "Logo uploaded" : "Name only"}{s.website ? " · Website linked" : ""}{s.level ? ` · ${s.level}` : ""}</p></div><div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}><button type="button" style={button} disabled={busy} onClick={() => edit(s)}>Edit</button><button type="button" style={button} disabled={busy || index === 0} onClick={() => void move(s.id, -1)} aria-label={`Move ${s.name} up`}>↑</button><button type="button" style={button} disabled={busy || index === rows.length - 1} onClick={() => void move(s.id, 1)} aria-label={`Move ${s.name} down`}>↓</button><ConfirmAction triggerLabel="Remove" title={`Remove ${s.name}?`} description="The sponsor will be removed from the tournament draft. Publish the draft to update the public pages." confirmLabel="Remove sponsor" confirmationText="REMOVE SPONSOR" tone="danger" disabled={busy} onConfirm={async () => { await save(sponsors.filter(row => row.id !== s.id)); return actionSuccess("Sponsor removed", "The tournament draft was saved."); }} /></div></div>)}{!rows.length ? <p style={{ color: "#64748b", fontSize: ".85rem" }}>No sponsors in this tier.</p> : null}</section>; })}
    {message ? <div role="status" style={{ marginTop: 12 }}>{message} <button type="button" style={button} onClick={() => setMessage("")}>Done</button></div> : null}{error && !draft ? <p role="alert" style={{ color: "#b91c1c" }}>{error}</p> : null}
    {preview ? <section style={{ marginTop: 24, borderTop: "1px solid #e2e8f0", paddingTop: 16 }}><div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", justifyContent: "space-between", gap: 8 }}><strong>Draft preview</strong><button type="button" style={button} aria-pressed={phone} onClick={() => setPhone(!phone)}>{phone ? "Desktop view" : "Phone view"}</button></div><div style={{ maxWidth: phone ? 360 : "100%", margin: "16px auto 0", padding: 12 }}><h2 style={{ margin: 0 }}>{tournamentName}</h2><TournamentSponsorDisplay sponsors={visible} placement="header" /><p style={{ padding: "24px 0", color: "#64748b" }}>Tournament page content</p><TournamentSponsorDisplay sponsors={visible} placement="footer" /></div></section> : null}
    <FormDialog open={!!draft} mode={draft && sponsors.some(s => s.id === draft.id) ? "edit" : "create"} title={draft && sponsors.some(s => s.id === draft.id) ? "Edit sponsor" : "Add sponsor"} dirty={!!file || (!!draft && JSON.stringify(draft) !== baseline)} submitLabel="Save sponsor" submitDisabled={working || !!error} workingLabel="Saving sponsor…" initialFocusRef={nameRef} onSubmit={submit} onAcknowledge={() => { setDraft(null); setFile(null); }} onCancel={() => { setDraft(null); setFile(null); setError(""); }}>
      {draft ? <div style={{ display: "grid", gap: 16 }}><label>Sponsor name<input ref={nameRef} style={input} maxLength={120} value={draft.name} disabled={working} onChange={e => setDraft({ ...draft, name: e.target.value })} required /></label><label>Tier<select style={input} value={draft.tier} disabled={working} onChange={e => setDraft({ ...draft, tier: e.target.value as SponsorTier, level: "" })}>{sponsorTiers.map(t => <option key={t} value={t}>{sponsorTierLabels[t]}</option>)}</select><span style={{ color: "#64748b", fontSize: ".85rem" }}>{sponsorPlacement(draft.tier)}</span></label><label>Public tier label (optional)<input style={input} maxLength={80} value={draft.level} placeholder={sponsorTierLabels[draft.tier]} disabled={working} onChange={e => setDraft({ ...draft, level: e.target.value })} /></label><label>Logo<input ref={uploadRef} type="file" style={input} accept="image/png,image/jpeg,image/webp" disabled={working} onChange={e => { const next = e.target.files?.[0] || null; if (next && (!['image/png','image/jpeg','image/webp'].includes(next.type) || next.size > 5 * 1024 * 1024)) { setError("Choose a PNG, JPG, or WebP logo under 5 MB."); return; } setError(""); setFile(next); }} /><span style={{ color: "#64748b", fontSize: ".85rem" }}>PNG, JPG, or WebP · up to 5 MB</span></label><TournamentSponsorDisplay sponsors={[{ ...draft, logo_url: localUrl || draft.logo_url }]} placement={draft.tier === "presenting" ? "header" : "footer"} />{draft.logo_path || file ? <button type="button" style={button} disabled={working} onClick={() => { setFile(null); setDraft({ ...draft, logo_path: "", logo_url: "" }); setError(""); if (uploadRef.current) uploadRef.current.value = ""; }}>Remove logo</button> : null}<label>Website (optional)<input style={input} maxLength={2048} value={draft.website} inputMode="url" placeholder="https://example.com" disabled={working} onChange={e => setDraft({ ...draft, website: e.target.value })} /></label><label><input type="checkbox" checked={draft.is_visible} disabled={working} onChange={e => setDraft({ ...draft, is_visible: e.target.checked })} /> Show sponsor</label><label>Internal notes<textarea style={input} maxLength={2000} value={draft.notes} disabled={working} onChange={e => setDraft({ ...draft, notes: e.target.value })} /><span style={{ color: "#64748b", fontSize: ".85rem" }}>Only visible to tournament staff.</span></label>{error ? <p role="alert" style={{ color: "#b91c1c" }}>{error}</p> : null}</div> : null}
    </FormDialog>
  </article>;
}
