import type { MetadataRoute } from "next";
import { getClubDirectory, getPublicSite } from "@/lib/clubSiteServer";
import { publicClubLinks, clubPageHref } from "@/lib/clubSite";
export const dynamic = "force-dynamic";
export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const origin=(process.env.NEXT_PUBLIC_JUPR_WEB_BASE_URL||process.env.JUPR_WEB_BASE_URL||"https://pickleballclubsandwich.com").replace(/\/$/,"");
  const routes=["/","/clubs","/create-club","/site-map","/how-ratings-work","/faq","/privacy","/terms","/support"];
  try {
    let offset=0,total=1;
    while(offset<total){const data=await getClubDirectory("",offset);if(!data)break;total=data.total;
      for(const club of data.clubs){const site=await getPublicSite(club.slug);if(!site||site.document.visibility!=="listed")continue;
        routes.push(`/clubs/${club.slug}`,...publicClubLinks(site.document).map(([,p])=>`/clubs/${club.slug}/${p}`),...site.document.pages.filter(p=>p.slug!=="home" && p.in_navigation).map(p=>clubPageHref(club.slug,p.slug)));}
      if(!data.clubs.length)break;offset+=data.limit;
    }
  } catch { return routes.filter(p=>!p.startsWith("/clubs/")).map(path=>({url:`${origin}${path}`})); }
  return routes.map(path=>({url:`${origin}${path}`,changeFrequency:"weekly",priority:path==="/"?1:.7}));
}
