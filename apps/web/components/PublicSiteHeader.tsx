"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import styles from "./PublicSiteHeader.module.css";

type Props = {
  productName: string;
  isStaging: boolean;
  stagingBuildSha: string | null;
};

type NavigationItem = {
  label: string;
  href: string;
  active: (pathname: string) => boolean;
  staff?: boolean;
};


function Brand({ productName, isStaging, stagingBuildSha }: Props) {
  const shortBuildSha = stagingBuildSha?.slice(0, 7).toUpperCase() || null;

  return (
    <div className={styles.brandGroup}>
      <Link href="/?welcome=1" className={styles.brand}>
        {productName}
      </Link>
      {isStaging ? (
        <span className={styles.stagingIdentity}>
          <span className={styles.environment}>STAGING</span>
          <span
            className={styles.buildSha}
            role="note"
            data-staging-build-sha={stagingBuildSha || "unavailable"}
            aria-label={
              stagingBuildSha
                ? `Staging build commit ${stagingBuildSha}`
                : "Staging build commit unavailable"
            }
            title={
              stagingBuildSha
                ? `Staging deployment commit ${stagingBuildSha}`
                : "Staging deployment commit is unavailable"
            }
          >
            BUILD {shortBuildSha || "UNAVAILABLE"}
          </span>
        </span>
      ) : null}
    </div>
  );
}

export default function PublicSiteHeader({
  productName,
  isStaging,
  stagingBuildSha
}: Props) {
  const pathname = usePathname() || "/";

  if (pathname.startsWith("/clubs/")) {
    return isStaging ? <div className={styles.compactHeader}><span className={styles.environment}>STAGING</span><span className={styles.buildSha} data-staging-build-sha={stagingBuildSha || "unavailable"}>BUILD {stagingBuildSha?.slice(0,7).toUpperCase() || "UNAVAILABLE"}</span></div> : null;
  }

  if (pathname === "/admin" || pathname.startsWith("/admin/")) {
    return (
      <header className={styles.compactHeader}>
        <Brand
          productName={productName}
          isStaging={isStaging}
          stagingBuildSha={stagingBuildSha}
        />
      </header>
    );
  }

  return (
    <header className={styles.header}>
      <div className={styles.brandRow}>
        <Brand
          productName={productName}
          isStaging={isStaging}
          stagingBuildSha={stagingBuildSha}
        />
      </div>
      <nav className={styles.nav} aria-label="Primary navigation">
        {([
          {label: "About PCS", href: "/?welcome=1", active: (p: string) => p === "/"},
          {label: "Find my club", href: "/clubs", active: (p: string) => p === "/clubs"},
          {label: "Create a club", href: "/create-club", active: (p: string) => p === "/create-club"},
          {label: "Staff sign in", href: "/admin/login", active: (p: string) => p === "/admin/login", staff: true}
        ] as NavigationItem[]).map((item) => {
          const active = item.active(pathname);
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={`${styles.link} ${active ? styles.active : ""} ${item.staff ? styles.staff : ""}`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
