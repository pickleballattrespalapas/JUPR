"use client";
import { useEffect, useRef, useState } from "react";
import type { WeeklyRecapJson } from "@/lib/weeklyRecapApi";
import { buildWeeklyReportEmail } from "@/lib/weeklyReportEmail";

function download(content: Blob, filename: string) {
  const url = URL.createObjectURL(content);
  const a = document.createElement("a"); a.href = url; a.download = filename; a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export default function WeeklyReportExport({ recap, clubSlug, published }: { recap: Partial<WeeklyRecapJson>; clubSlug: string; published: boolean }) {
  const [compact, setCompact] = useState(true);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const frame = useRef<HTMLIFrameElement>(null);
  const [origin, setOrigin] = useState("");
  useEffect(() => setOrigin(window.location.origin), []);
  const options = { origin, clubSlug, published, compact };
  const fragment = buildWeeklyReportEmail(recap, options);
  const html = buildWeeklyReportEmail(recap, { ...options, document: true });
  const week = recap.start_date || recap.week_start || "report";
  async function copyHtml() {
    try { await navigator.clipboard.writeText(fragment); setMessage("HTML copied. Paste it into the newsletter’s HTML/source editor, or give the downloaded file to Tres 2027."); }
    catch { setMessage("Clipboard unavailable. Use Download HTML or copy the source below."); }
  }
  async function png() {
    setBusy(true);
    const exportFrame = document.createElement("iframe");
    exportFrame.setAttribute("sandbox", "allow-same-origin");
    exportFrame.style.cssText = "position:fixed;left:-10000px;top:0;width:636px;height:1px;border:0;";
    exportFrame.title = "Preparing weekly report image";
    try {
      const loaded = new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => reject(new Error("Preview timed out")), 10000);
        exportFrame.onload = () => { window.clearTimeout(timeout); resolve(); };
      });
      exportFrame.srcdoc = html;
      document.body.appendChild(exportFrame);
      await loaded;
      const target = exportFrame.contentDocument?.body;
      if (!target) throw new Error("Preview unavailable");
      const { toBlob } = await import("html-to-image");
      const blob = await toBlob(target, { pixelRatio: 2, backgroundColor: "#f4f0e8", width: 636, height: target.scrollHeight, skipFonts: true });
      if (!blob) throw new Error("No image was created.");
      download(blob, `weekly-report-${week}.png`); setMessage("PNG downloaded. Use HTML when you want clickable links in the email.");
    } catch { setMessage("The image could not be created. The HTML download and browser print preview remain available."); }
    finally { exportFrame.remove(); setBusy(false); }
  }
  return <section data-testid="weekly-report-export" style={{ border: "1px solid #e6e8e1", borderRadius: 14, padding: "1rem", background: "#f8f3e9" }}>
    <h2 style={{ marginTop: 0, color: "#123e37" }}>Add to your member email</h2>
    <p>A newsletter section in the Tres Palapas colors and 620-pixel layout. Exports use the saved report shown below.</p>
    <label>Content size <select value={compact ? "compact" : "full"} onChange={e => setCompact(e.target.value === "compact")}><option value="compact">Compact newsletter section</option><option value="full">Full weekly report</option></select></label>
    <p style={{ fontSize: 14 }}>The compact section features three events, two player highlights, and two award races. Published reports include a link to the complete report.</p>
    <div style={{ display: "flex", flexWrap: "wrap", gap: ".6rem" }}>
      <button type="button" onClick={copyHtml}>Copy email HTML</button>
      <button type="button" onClick={() => download(new Blob([fragment], { type: "text/html;charset=utf-8" }), `weekly-report-${week}-email-section.html`)}>Download email section</button>
      <button type="button" onClick={() => download(new Blob([html], { type: "text/html;charset=utf-8" }), `weekly-report-${week}.html`)}>Download HTML preview</button>
      <button type="button" onClick={png} disabled={busy}>{busy ? "Creating PNG…" : "Download PNG"}</button>
    </div>
    {!published ? <p style={{ color: "#92400e" }}>Draft exports are labeled for review. Publish when ready to enable the public report link.</p> : null}
    {message ? <p role="status">{message}</p> : null}
    <details><summary style={{ marginTop: "1rem" }}>Preview email section and HTML source</summary>
      <iframe ref={frame} title="Weekly report email preview" sandbox="allow-same-origin" srcDoc={html} style={{ width: "100%", maxWidth: 636, height: 700, border: "1px solid #e6e8e1", marginTop: "1rem" }} />
      <label style={{ display: "block", marginTop: "1rem" }}>HTML source<textarea readOnly value={fragment} style={{ width: "100%", minHeight: 140 }} /></label>
    </details>
  </section>;
}
