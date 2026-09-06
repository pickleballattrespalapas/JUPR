"use client";

import Image from "next/image";
import Link from "next/link";
import { Fragment, useState } from "react";
import { normalizeSponsorWebsite, sponsorTierLabels, type TournamentSponsor } from "@/lib/tournamentSponsors";
import styles from "./TournamentSponsorDisplay.module.css";

function Sponsor({ sponsor, presenting = false }: { sponsor: TournamentSponsor; presenting?: boolean }) {
  const [failedUrl, setFailedUrl] = useState("");
  let website = "";
  try { website = normalizeSponsorWebsite(sponsor.website); } catch { /* Invalid legacy links render as plain names. */ }
  const name = <strong>{sponsor.name}</strong>;
  const logo = sponsor.logo_url && failedUrl !== sponsor.logo_url ? <Image unoptimized src={sponsor.logo_url} alt="" width={presenting ? 240 : 160} height={presenting ? 96 : 56} referrerPolicy="no-referrer" className={styles.logo} onError={() => setFailedUrl(sponsor.logo_url || "")} /> : null;
  const content = presenting ? <>{name}{logo}</> : <>{logo}{name}</>;
  const brand = website ? <a className={styles.sponsor} href={website} target="_blank" rel="sponsored noopener noreferrer" referrerPolicy="no-referrer" aria-label={`Visit ${sponsor.name} (opens in a new tab)`}>{content}</a> : <span className={styles.sponsor}>{content}</span>;
  if (presenting || !sponsor.public_description?.trim()) return brand;
  return <div className={styles.describedSponsor}>{brand}<p className={styles.description}>{sponsor.public_description}</p></div>;
}

export type SponsorDisplayProps = {
  sponsors: TournamentSponsor[];
  placement: "header" | "footer";
  title?: string;
  titleHref?: string;
  compact?: boolean;
  headingLevel?: "h1" | "h2";
};

export default function TournamentSponsorDisplay({ sponsors, placement, title, titleHref, compact = false, headingLevel: Heading = "h1" }: SponsorDisplayProps) {
  const records = sponsors.filter(s => placement === "header" ? s.tier === "presenting" : s.tier !== "presenting");
  if (placement === "header") {
    const presenting = records.length ? <div className={styles.presenting} aria-label="Presenting sponsors"><span>Presented by </span><span className={styles.names}>{records.map((s, index) => <Fragment key={s.id}>{index > 0 ? <span>{index === records.length - 1 ? " and " : ", "}</span> : null}<Sponsor sponsor={s} presenting /></Fragment>)}</span></div> : null;
    const header = title ? <div className={styles.titleRow}><Heading className={styles.title}>{titleHref ? <Link className={styles.titleLink} href={titleHref}>{title}</Link> : title}</Heading>{presenting}</div> : presenting;
    const described = records.filter(s => s.public_description?.trim());
    const content = described.length ? <div>{header}<div className={styles.headerDescriptions}>{described.map(s => <p key={s.id} className={styles.description}>{records.length > 1 ? <><strong>{s.name}</strong>{": "}</> : null}{s.public_description}</p>)}</div></div> : header;
    return compact ? <div className={styles.compact}>{content}</div> : content;
  }
  if (!records.length) return null;
  const groups = new Map<string, { label: string; community: boolean; records: TournamentSponsor[] }>();
  for (const tier of ["premier", "supporting"] as const) for (const s of records.filter(row => row.tier === tier)) {
    const label = s.level || sponsorTierLabels[tier];
    const key = `${tier}:${label}`;
    if (!groups.has(key)) groups.set(key, { label, community: tier === "supporting", records: [] });
    groups.get(key)!.records.push(s);
  }
  return <footer className={styles.footer} aria-label="Tournament sponsors">{Array.from(groups, ([key, group]) => <section key={key} className={`${styles.group} ${group.community ? styles.community : ""}`}><h3>{group.label}</h3><div className={styles.grid}>{group.records.map(s => <Sponsor key={s.id} sponsor={s} />)}</div></section>)}</footer>;
}
