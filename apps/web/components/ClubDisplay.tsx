"use client";
import { createContext, useContext, type ReactNode } from "react";
import { visible, type DisplayKey, type DisplaySettings } from "@/lib/clubSite";
export const ClubDisplayContext = createContext<DisplaySettings>({});
export function ClubDisplayProvider({
  display,
  children,
}: {
  display: DisplaySettings;
  children: ReactNode;
}) {
  return (
    <ClubDisplayContext.Provider value={display}>
      {children}
    </ClubDisplayContext.Provider>
  );
}
export function Display({
  field,
  children,
}: {
  field?: DisplayKey;
  children: ReactNode;
}) {
  const display = useContext(ClubDisplayContext);
  return !field || visible(display, field) ? <>{children}</> : null;
}
