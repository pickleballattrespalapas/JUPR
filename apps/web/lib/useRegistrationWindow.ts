"use client";
import { useEffect, useRef, useState } from "react";
import { registrationCanAccept, registrationMeetPlanning, registrationPhase, type SeasonRegistrationWindow } from "./interclubRegistrationWindow";

/** Recheck at the published opening/closing boundary and after returning to the page. */
export function useRegistrationWindow(registration: SeasonRegistrationWindow | null | undefined, onBoundary?: (reason: "boundary" | "focus") => void) {
  const [now, setNow] = useState(Date.now);
  const callback = useRef(onBoundary); callback.current = onBoundary;
  const latest = useRef(registration); latest.current = registration;
  const opened = now >= Date.parse(registration?.opens_at || ""), closed = now >= Date.parse(registration?.closes_at || "");
  useEffect(() => {
    const current = Date.now(); setNow(current);
    const boundary = [registration?.opens_at, registration?.closes_at].map(value => Date.parse(value || "")).filter(value => Number.isFinite(value) && value > current).sort((a, b) => a - b)[0];
    let timer: ReturnType<typeof setTimeout> | undefined;
    const schedule = () => {
      if (boundary == null) return;
      timer = setTimeout(() => {
        const time = Date.now();
        if (time < boundary) { schedule(); return; }
        setNow(time); callback.current?.("boundary");
      }, Math.min(Math.max(0, boundary - Date.now()) + 25, 2_147_483_647));
    };
    schedule();
    const refresh = () => {
      const time = Date.now(); setNow(time);
      const currentRegistration = latest.current;
      callback.current?.(currentRegistration && registrationPhase(currentRegistration, time) !== currentRegistration.status ? "boundary" : "focus");
    };
    if (typeof window !== "undefined") window.addEventListener?.("focus", refresh);
    return () => { if (timer != null) clearTimeout(timer); if (typeof window !== "undefined") window.removeEventListener?.("focus", refresh); };
  }, [registration?.opens_at, registration?.closes_at, registration?.revision, registration?.status, opened, closed]);
  const current = Date.now();
  return { now: current, phase: registrationPhase(registration, current), canRegister: registrationCanAccept(registration, current), meetPlanningOpen: registrationMeetPlanning(registration, current) };
}
