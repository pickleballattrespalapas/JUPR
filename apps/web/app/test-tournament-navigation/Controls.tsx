"use client";

import { useState, useSyncExternalStore } from "react";
import Link from "@/components/NavigationLink";

const subscribe = () => () => {};
const clientReady = () => true;
const serverReady = () => false;

export default function Controls() {
  const [fail, setFail] = useState(false);
  const ready = useSyncExternalStore(subscribe, clientReady, serverReady);
  return <>
    <p>{ready ? "Test controls ready" : "Preparing test controls…"}</p>
    <nav aria-label="Navigation test" style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
      <Link href="/test-tournament-navigation/slow?tournament=fixture&draw=one&day=two" prefetch={false}>Slow page</Link>
      <Link href="/test-tournament-navigation/fast?tournament=fixture&draw=one&day=two" prefetch={false}>Fast page</Link>
      <Link href="/test-tournament-navigation/failure?tournament=fixture&draw=one&day=two" prefetch={false}>Failure page</Link>
    </nav>
    <label><input type="checkbox" checked={fail} onChange={event => {
      const next = event.target.checked;
      document.cookie = `pcs_navigation_test_fail=${next ? "1" : "0"}; Path=/test-tournament-navigation; SameSite=Lax`;
      setFail(next);
    }} />Simulate a failed request</label>
  </>;
}
