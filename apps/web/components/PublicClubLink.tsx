"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { createContext, useContext, type ComponentProps, type ReactNode } from "react";
import { canLinkClubPage, type PageNavigationSettings } from "@/lib/clubSite";

const NavigationContext = createContext<{ slug: string; settings: PageNavigationSettings } | null>(null);
export function ClubPageNavigationProvider({ slug, settings, children }: {
  slug: string; settings: PageNavigationSettings; children: ReactNode;
}) {
  return <NavigationContext.Provider value={{ slug, settings }}>{children}</NavigationContext.Provider>;
}

export default function PublicClubLink({ children, ...props }: ComponentProps<typeof Link>) {
  const context = useContext(NavigationContext);
  if (!context) return <Link {...props}>{children}</Link>;
  return <ClubLink context={context} {...props}>{children}</ClubLink>;
}

function ClubLink({ context, children, ...props }: ComponentProps<typeof Link> & {
  context: { slug: string; settings: PageNavigationSettings };
}) {
  const pathname = usePathname();
  const href = typeof props.href === "string" ? props.href : props.href.pathname || "";
  if (!canLinkClubPage(context.settings, context.slug, href, pathname || "")) {
    // Preserve names/scores in records while removing links to a private section.
    // Standalone navigation actions disappear rather than becoming dead buttons.
    const recordLink = /\/(players|matches)\/[^/?#]+/.test(href);
    return recordLink ? <span>{children}</span> : null;
  }
  return <Link {...props}>{children}</Link>;
}
