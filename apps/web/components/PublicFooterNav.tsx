"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function PublicFooterNav() {
  const pathname = usePathname() || "/";
  const clubBase = pathname.match(/^\/clubs\/[^/]+/)?.[0];
  const admin = pathname === "/admin" || pathname.startsWith("/admin/");
  return <nav style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }} aria-label="Footer navigation">
    {!admin && clubBase && <>
      <Link href={`${clubBase}/leagues`}>Leagues</Link>
      <Link href={`${clubBase}/tournaments`}>Tournaments</Link>
      <Link href={`${clubBase}/badge-codex`}>Badges &amp; Trophies</Link>
      <Link href={`${clubBase}/matches`}>Matches</Link>
    </>}
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
