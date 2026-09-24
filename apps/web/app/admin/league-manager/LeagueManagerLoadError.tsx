"use client";

import { useRouter } from "next/navigation";
import { useTransition } from "react";

export default function LeagueManagerLoadError({ error, service = "League Manager" }: { error: string; service?: string }) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();

  return (
    <div role="alert" style={{ border: "1px solid #fecaca", borderRadius: "12px", padding: "1rem", background: "#fff7f7" }}>
      <p style={{ color: "#b91c1c", marginTop: 0 }}><strong>{service} couldn’t load.</strong> {error}</p>
      <button
        type="button"
        disabled={pending}
        onClick={() => startTransition(() => router.refresh())}
        style={{ padding: "0.6rem 0.9rem", borderRadius: "8px", border: "1px solid #0f172a", background: "white", color: "#0f172a", font: "inherit", fontWeight: 700, cursor: pending ? "wait" : "pointer" }}
      >
        {pending ? "Trying again…" : "Retry loading"}
      </button>
    </div>
  );
}
