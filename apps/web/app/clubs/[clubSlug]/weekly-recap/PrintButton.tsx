"use client";

export default function PrintButton({ label = "Print / save PDF" }: { label?: string }) {
  return (
    <button
      data-testid="weekly-recap-print-button"
      type="button"
      onClick={() => window.print()}
      style={{
        border: "1px solid #cbd5e1",
        borderRadius: "999px",
        padding: "0.55rem 0.85rem",
        background: "white",
        color: "#0f172a",
        fontWeight: 800,
        cursor: "pointer"
      }}
    >
      {label}
    </button>
  );
}
