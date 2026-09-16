"use client";
import { useEffect } from "react";
import { LAST_CLUB_COOKIE, validClubSlug } from "@/lib/clubSite";
export default function RememberPublicClub({ slug }: { slug: string }) {
  useEffect(() => {
    if (!validClubSlug(slug)) return;
    document.cookie = `${LAST_CLUB_COOKIE}=${encodeURIComponent(slug)}; Path=/; SameSite=Lax; Max-Age=31536000${location.protocol === "https:" ? "; Secure" : ""}`;
  }, [slug]);
  return null;
}
