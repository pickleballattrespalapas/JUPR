"use client";
import { useState } from "react";
import type { WeeklyRecapJson } from "@/lib/weeklyRecapApi";
import { buildWeeklyReportEmail } from "@/lib/weeklyReportEmail";

export default function WeeklyRecapPdfDownload({
  recap,
  clubSlug,
}: {
  recap: WeeklyRecapJson;
  clubSlug: string;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function download() {
    setBusy(true);
    setError("");
    try {
      const { exportWeeklyRecap } = await import("@/lib/weeklyRecapExport");
      const html = buildWeeklyReportEmail(recap, {
        origin: window.location.origin,
        clubSlug,
        published: true,
        compact: false,
        document: true,
      });
      await exportWeeklyRecap(
        html,
        "pdf",
        `weekly-recap-${recap.week_start || recap.start_date || "report"}`,
      );
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "The PDF could not be created. Use Print-friendly view to save a PDF.",
      );
    } finally {
      setBusy(false);
    }
  }
  return (
    <span>
      <button
        type="button"
        data-testid="weekly-recap-styled-pdf"
        disabled={busy}
        onClick={download}
      >
        {busy ? "Creating PDF…" : "Download PDF"}
      </button>
      {error ? (
        <span role="alert" style={{ display: "block", maxWidth: 420 }}>
          {error}
        </span>
      ) : null}
    </span>
  );
}
