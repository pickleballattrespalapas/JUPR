import { NextResponse, type NextRequest } from "next/server";

export function middleware(request: NextRequest) {
  const headers = new Headers(request.headers);
  // Overwrite supplied values. Metadata must follow the actual requested page.
  headers.set("x-pcs-club-path", request.nextUrl.pathname);
  return NextResponse.next({ request: { headers } });
}

export const config = { matcher: "/clubs/:path*" };
