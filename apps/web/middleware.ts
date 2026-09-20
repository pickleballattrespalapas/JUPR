import { NextResponse, type NextRequest } from "next/server";
import { privateShareLink, publicShareQuery } from "./lib/shareMetadataCore";

export function middleware(request: NextRequest) {
  const headers = new Headers(request.headers);
  // Always overwrite inbound headers: previews must describe the actual request.
  headers.set("x-pcs-club-path", request.nextUrl.pathname);
  headers.set("x-pcs-share-query", publicShareQuery(request.nextUrl.searchParams).toString());
  headers.set("x-pcs-share-origin", request.nextUrl.origin);
  headers.set("x-pcs-share-private", privateShareLink(request.nextUrl.pathname, request.nextUrl.searchParams) ? "1" : "0");
  return NextResponse.next({ request: { headers } });
}
export const config = { matcher: ["/((?!api/|_next/|share-image|favicon.ico|robots.txt|sitemap.xml).*)"] };
