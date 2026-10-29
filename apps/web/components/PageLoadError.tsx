"use client";

import { useTransition } from "react";

export default function PageLoadError({ retry }: { error: unknown; retry: () => void }) {
  const [pending, startTransition] = useTransition();
  return (
    <section role="alert" style={{ padding: "1.25rem" }}>
      <h1>This page couldn’t load</h1>
      <p>The page is temporarily unavailable. Try again to reload it.</p>
      <button type="button" disabled={pending} onClick={() => startTransition(() => retry())}>
        {pending ? "Retrying…" : "Try again"}
      </button>
    </section>
  );
}
