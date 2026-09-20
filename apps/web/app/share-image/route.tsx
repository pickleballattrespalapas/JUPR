import { ImageResponse } from "next/og";
import { resolvePageShare } from "@/lib/shareMetadata";
import { plainText, publicShareQuery } from "@/lib/shareMetadataCore";

export const runtime = "edge";
export async function GET(request: Request) {
  const url = new URL(request.url);
  const path = url.searchParams.get("path") || "/";
  const share = await resolvePageShare(path, publicShareQuery(url.searchParams).toString());
  const accent = /^#[0-9a-f]{6}$/i.test(share.accent || '') ? share.accent! : '#0f766e';
  // Text-only fallback: no arbitrary URL fetches, no dependency on external fonts/images.
  return new ImageResponse(
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', background: '#f8fafc', color: '#0f172a', padding: 64, borderTop: `18px solid ${accent}` }}>
      <div style={{ display: 'flex', fontSize: 28, fontWeight: 700, color: '#475569', marginBottom: 38 }}>{plainText(share.siteName, 85)}</div>
      <div style={{ display: 'flex', fontSize: share.title.length > 70 ? 48 : 60, fontWeight: 700, lineHeight: 1.12, marginBottom: 30 }}>{plainText(share.title.replace(` | ${share.siteName}`, ''), 108)}</div>
      <div style={{ display: 'flex', fontSize: 27, lineHeight: 1.4, color: '#475569' }}>{plainText(share.description, 180)}</div>
      <div style={{ display: 'flex', marginTop: 'auto', paddingTop: 22, fontSize: 22, color: '#475569' }}>Pickleball · Events · Community</div>
    </div>,
    { width: 1200, height: 630, headers: { 'Cache-Control': share.noindex ? 'private, no-store' : 'public, max-age=300, s-maxage=300', 'X-Robots-Tag': 'noindex' } }
  );
}
