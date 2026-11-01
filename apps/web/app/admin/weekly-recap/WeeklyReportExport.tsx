"use client";
import { useEffect, useRef, useState } from "react";
import type { WeeklyRecapJson } from "@/lib/weeklyRecapApi";
import { buildWeeklyReportEmail } from "@/lib/weeklyReportEmail";

function download(content: Blob, filename: string) {
  const url = URL.createObjectURL(content);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export default function WeeklyReportExport({
  recap,
  clubSlug,
  published,
}: {
  recap: Partial<WeeklyRecapJson>;
  clubSlug: string;
  published: boolean;
}) {
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
    try {
      await navigator.clipboard.writeText(fragment);
      setMessage(
        "HTML copied. Paste it into the newsletter’s HTML/source editor, or give the downloaded file to Tres 2027.",
      );
    } catch {
      setMessage(
        "Clipboard unavailable. Use Download HTML or copy the source below.",
      );
    }
  }
  async function exportImage(format: "png" | "pdf") {
    setBusy(true);
    setMessage("");
    try {
      const { exportWeeklyRecap } = await import("@/lib/weeklyRecapExport");
      await exportWeeklyRecap(html, format, `weekly-report-${week}`);
      setMessage(
        format === "pdf"
          ? "PDF downloaded with your photos, styling, and clickable links."
          : "PNG downloaded. Use HTML or PDF for clickable links.",
      );
    } catch (e) {
      setMessage(
        e instanceof Error
          ? e.message
          : "The export could not be created. Use the HTML download or Print-friendly view.",
      );
    } finally {
      setBusy(false);
    }
  }
  return (
    <section
      data-testid="weekly-report-export"
      style={{
        border: "1px solid #e6e8e1",
        borderRadius: 14,
        padding: "1rem",
        background: "#f8f3e9",
      }}
    >
      <h2 style={{ marginTop: 0, color: "#123e37" }}>
        Add to your member email
      </h2>
      <p>
        A newsletter section in the Tres Palapas colors and 620-pixel layout.
        Exports use the content selected in your current preview.
      </p>
      <label>
        Export content{" "}
        <select
          value={compact ? "compact" : "full"}
          onChange={(e) => setCompact(e.target.value === "compact")}
        >
          <option value="compact">Member email selections</option>
          <option value="full">Website edition</option>
        </select>
      </label>
      <p style={{ fontSize: 14 }}>
        The member email keeps your edited text and card styles. Published reports
        include a link to the complete report.
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: ".6rem" }}>
        <button type="button" onClick={copyHtml}>
          Copy email HTML
        </button>
        <button
          type="button"
          onClick={() =>
            download(
              new Blob([fragment], { type: "text/html;charset=utf-8" }),
              `weekly-report-${week}-email-section.html`,
            )
          }
        >
          Download email section
        </button>
        <button
          type="button"
          onClick={() =>
            download(
              new Blob([html], { type: "text/html;charset=utf-8" }),
              `weekly-report-${week}.html`,
            )
          }
        >
          Download HTML preview
        </button>
        <button
          type="button"
          onClick={() => void exportImage("png")}
          disabled={busy}
        >
          Download PNG
        </button>
        <button
          type="button"
          onClick={() => void exportImage("pdf")}
          disabled={busy}
        >
          Download styled PDF
        </button>
        {busy ? <span role="status">Creating download…</span> : null}
      </div>
      {!published ? (
        <p style={{ color: "#92400e" }}>
          Draft exports are labeled for review. Publish when ready to enable the
          public report link.
        </p>
      ) : null}
      {message ? <p role="status">{message}</p> : null}
      <details>
        <summary style={{ marginTop: "1rem" }}>
          Preview email section and HTML source
        </summary>
        <iframe
          ref={frame}
          title="Weekly report email preview"
          sandbox="allow-same-origin"
          srcDoc={html}
          style={{
            width: "100%",
            maxWidth: 636,
            height: 700,
            border: "1px solid #e6e8e1",
            marginTop: "1rem",
          }}
        />
        <label style={{ display: "block", marginTop: "1rem" }}>
          HTML source
          <textarea
            readOnly
            value={fragment}
            style={{ width: "100%", minHeight: 140 }}
          />
        </label>
      </details>
    </section>
  );
}
