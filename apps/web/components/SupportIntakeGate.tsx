import Link from "next/link";
import type { ReactNode } from "react";
import { getClubDirectory } from "@/lib/clubSiteServer";
import { SUPPORT_EMAIL } from "@/lib/publicSupportContent";
import { getPublicSupportIntakeStatus } from "@/lib/supportIntakeApi";

export type SupportSearchParams = Record<string, string | string[] | undefined>;
export const supportParam = (params: SupportSearchParams | undefined, key: string) =>
  typeof params?.[key] === "string" ? (params[key] as string).trim() : "";

function EmailFallback() {
  return <p>Contact your club organizer or <a href={`mailto:${SUPPORT_EMAIL}`}>email PCS support</a>. Include your club and the details of your request.</p>;
}

export default async function SupportIntakeGate({ path, searchParams, children }: {
  path: string;
  searchParams?: SupportSearchParams;
  children: ReactNode;
}) {
  const clubSlug = supportParam(searchParams, "club");
  if (clubSlug) {
    const response = await getPublicSupportIntakeStatus(clubSlug);
    if (!response.data) return <div role="status">
      <p>{response.status === 404 ? "This club could not be found." : "Support requests are temporarily unavailable."} Your request has not been submitted.</p>
      <Link href={path}>Choose a club</Link><EmailFallback />
    </div>;
    return <>
      <p><strong>Send to {response.data.club.name}</strong> · <Link href={path}>Change club</Link></p>
      {response.data.accepting_requests ? <>{children}<EmailFallback /></> : <div role="status">
        <p>Online requests for this club are temporarily unavailable. Your request has not been submitted.</p>
        <EmailFallback />
      </div>}
    </>;
  }

  const query = supportParam(searchParams, "q").slice(0, 120);
  const requestedOffset = Number(supportParam(searchParams, "offset"));
  const offset = Number.isSafeInteger(requestedOffset) && requestedOffset >= 0 ? requestedOffset : 0;
  let directory;
  try { directory = await getClubDirectory(query, offset); } catch { directory = null; }
  const pageHref = (nextOffset: number) => `${path}?${new URLSearchParams({ q: query, offset: String(nextOffset) })}`;
  return <div>
    <p>Choose the club that should review your request.</p>
    <form action={path} method="get" style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
      <label>Find your club <input name="q" defaultValue={query} maxLength={120} /></label>
      <button type="submit">Search clubs</button>
    </form>
    {directory ? <>
      <ul>{directory.clubs.map(club => <li key={club.slug}><Link href={`${path}?${new URLSearchParams({ club: club.slug })}`}>{club.name}</Link></li>)}</ul>
      {!directory.clubs.length ? <p>No matching clubs were found.</p> : null}
      <nav aria-label="Club search pages" style={{ display: "flex", gap: "1rem" }}>
        {offset > 0 ? <Link href={pageHref(Math.max(0, offset - directory.limit))}>Previous clubs</Link> : null}
        {offset + directory.limit < directory.total ? <Link href={pageHref(offset + directory.limit)}>More clubs</Link> : null}
      </nav>
    </> : <p>The club list is temporarily unavailable.</p>}
    <EmailFallback />
  </div>;
}
