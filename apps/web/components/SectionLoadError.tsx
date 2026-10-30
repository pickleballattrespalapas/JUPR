"use client";

import { useTransition } from "react";
import { useRouter } from "next/navigation";

export default function SectionLoadError({ title, message }: { title: string; message: string }) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  return (
    <div role="alert" style={{ padding: "1rem", border: "1px solid #cbd5e1", borderRadius: "0.5rem" }}>
      <p style={{ marginTop: 0, fontWeight: 700 }}>{title}</p>
      <p>{message}</p>
      <button type="button" disabled={pending} onClick={() => startTransition(() => router.refresh())}>
        {pending ? "Retrying…" : "Try again"}
      </button>
    </div>
  );
}
