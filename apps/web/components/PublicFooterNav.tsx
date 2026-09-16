"use client";
import Link from "next/link";

export default function PublicFooterNav() {
  // Club navigation lives inside ClubLayout, where published visibility applies.
  // This global footer is outside that context and only advertises PCS pages.
  return <nav style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }} aria-label="Footer navigation">
    <Link href="/?welcome=1">Powered by PCS</Link>
    <Link href="/clubs">Find a club</Link>
    <Link href="/admin/login">Staff sign in</Link>
    <Link href="/site-map">Site map</Link>
    <Link href="/how-ratings-work">How ratings work</Link>
    <Link href="/faq">FAQ</Link>
    <Link href="/privacy">Privacy</Link>
    <Link href="/terms">Terms</Link>
    <Link href="/support">Contact</Link>
    <Link href="/data-corrections">Data corrections</Link>
  </nav>;
}
