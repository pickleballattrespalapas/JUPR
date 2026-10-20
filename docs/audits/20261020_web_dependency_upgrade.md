# F02: isolated web dependency upgrade candidate

This candidate updates `apps/web` from staging commit `ac0dfdd4` and is now rebased onto the operational-fix candidate `2b9b12088bce04602a61507d4ca17561ed03a1ea`. The dependency upgrade has not been pushed, merged, deployed, or tested against production. It requires a separate staging review because the security update crosses framework majors.

## Version choice and changes

- Next.js and its ESLint preset: **14.2.35 → 16.4.2**. Next 14 is outside the published support policy. The October 14 security update requires at least 15.5.28 or 16.4.1; 16.4.2 is the stable registry release inspected on October 20. Next 15's published two-year maintenance window from October 21, 2024 is almost over, so a Next 15 bridge would immediately need another upgrade.
- React, React DOM and the existing test renderer: **18.3.1 → 19.3.1**, with matching React 19 type definitions. TypeScript remains 5.5.3, above Next 16's minimum.
- ESLint: **8.57.1 → 10.13.0**. ESLint 9 is also end-of-life. Replace `next lint` with the ESLint CLI and flat config; add an explicit CI lint step because Next 16 no longer lints during a build.
- Node: application `engines`, `.nvmrc`, and the four web-related workflows use **24.x**. Node 20 is end-of-life; Vercel supports 24.x and respects the package engine override. Local checks used Node 24.19.0, npm 11.9.0; deployment patch versions remain provider-controlled.
- Migrate request `params`, `searchParams`, `cookies()` and `headers()` to async access, including the admin workspace helper and every caller. Preserve workspace redirect behavior and check it with a truly async cookie mock.
- Update React 19 nullable ref types and explicitly initialize two timer refs. Adapt five existing test harnesses for async page invocation, Promise search parameters, effect flushing and automatic image-preload markup; retain their functional/security assertions.
- Preserve Webpack explicitly in development, build, and local browser-fixture scripts. Retain the existing Edge middleware and its request-header overwrite behavior. React Compiler and Cache Components remain disabled.
- Refresh permitted transitive lockfile ranges. Relevant resolved packages include PostCSS 8.5.23, DOMPurify 3.4.17, nanoid 3.3.20, source-map-js 1.2.2 and sharp 0.35.5. No forced audit fixes, prereleases or package overrides are used.

## Advisory assessment

The original lockfile audit returned 10 affected package entries: 1 critical, 8 high, 1 moderate. This is a dependency inventory finding, not proof that every advisory is exploitable through a PCS route.

The updated production dependency audit reports **zero known advisories** at the time checked. The full audit reports **five high entries from one remaining unpatched advisory**, GHSA-vfj7-8cjw-p6xm: `eslint-config-next → @next/eslint-plugin-next → fast-glob → micromatch → braces@3.0.3`.

`braces` is development-only in the lockfile; the advisory requires attacker-controlled deeply nested brace patterns. The Next lint plugin's `get-root-dirs` utility feeds explicitly configured `settings.next.rootDir` patterns to fast-glob. This app does not configure that setting, and its lint target paths are fixed. No runtime application import of these lint packages was found. Do not treat the five dependency-chain entries as five separate production exposures. Do not use npm's suggested downgrade to Next's old ESLint preset: it restores vulnerable/unsupported dependencies. Track the upstream patch and re-run the full audit before release; repository/configuration input remains a development trust boundary.

For the original Next advisories, PCS uses the App Router and a `next/og` share-image route, so framework security maintenance matters even when a feature-specific condition is absent. The source has no Server Actions, Cache Components, application rewrite rules or nonce-based CSP configuration. Both `next/image` uses are `unoptimized`, and no remote image patterns or SVG-optimizer opt-in are configured. The Windows-only RCE condition does not describe the Linux/Vercel deployment configuration reviewed in the audit. No exploit or load test was performed, and provider mitigations were not reassessed during this local fix. The new version also includes the October fixes rather than stopping at the older August minimum.

## Validation and remaining release gates

Completed locally for the initial dependency candidate, before the operational-fix rebase:

- Fresh `npm ci` from the updated lockfile.
- `next typegen`, `tsc --noEmit`, and production `next build --webpack`.
- All **80** component-test commands, followed by a successful complete `npm run test:component`.
- ESLint CLI: **0 errors, 191 warnings**. The new preset initially reported 174 errors under React Compiler recommendations that did not exist in the previous preset: set-state-in-effect (111), refs (51), purity (6), immutability (4), preserve-manual-memoization (2). These five rules remain visible as warnings while the compiler is disabled. Existing hooks rules retain their severity; no assertions or tests were disabled. Address this baseline in a separate component-quality change before enabling the compiler.
- Production and full lockfile audits, with the residual development advisory described above.
- Round-robin browser suite: **10 scenario groups passed**, including independent public/admin organizer, operator and spectator sessions, sharing/revocation, 4/5/8/9/12/16-player schedules, late arrivals, score saving, and preserved rotations. No browser errors or HTTP 500/async-request errors were observed in its fixture log.
- Full recap browser suite against the production build: passed generation/editing, independent email order, autosave, reusable defaults, publication/private revision isolation, result refresh/reload, photo failure recovery, PDF/PNG export, layout management, and desktop/mobile editing.
- `git diff --check`.

Standard Playwright Chromium download was blocked by a truncated CDN archive in this environment; a scratch-only Chromium binary from `@sparticuz/chromium@153.0.0` was used with the existing Playwright package. This runtime was not added to the repository's dependencies. Browser requests in the fixtures are restricted to loopback/synthetic assets and persistence is in memory.

The initial cold development-server recap run failed: compiling the first public preview reloaded the open admin editor, and the expected revision control disappeared. An isolated checkout of the exact Next 14 baseline passed with the same browser. The Next 16 **production** build also passed the entire suite without changing any assertions. The dev fixture now precompiles the public preview before opening the editor; the full suite passed again after clearing the development build cache. CI explicitly builds and starts the production server for recap acceptance. This separates development module-graph reloads from release behavior without masking a failed workflow assertion. The optional `RECAP_BROWSER_NEXT_MODE=start` requires a build with both API base variables set to `http://127.0.0.1:8042`.

Expected warnings: the Edge `middleware.ts` naming deprecation; a build warning about `process.cwd` in Next's own Edge import graph; test-renderer deprecation/act-environment notices; old peer ranges in transitive ESLint plugins even though the Next 16.4 preset supplies its ESLint 10 compatibility adapters. None was hidden. The local browser checks must validate actual navigation as well as the compile result.

Before treating F02 as released: review this isolated candidate, run its GitHub checks, deploy only to authorized staging, verify Vercel's actual Node/Next versions, then run staging authentication, admin workspace switching, public navigation, registration and export/share-image smoke checks. Source, lockfile, build and browser evidence must refer to the same candidate SHA. A clean scanner result alone does not establish security or production readiness.

## Integration with the operational fixes

The rebase preserves the new support-intake command in `test:component` (now 81 component-test commands) and the operational fixes in the player-profile and tournament-results pages. The three new support, data-correction and profile-privacy page boundaries now await Promise-based `searchParams`; the internal intake gate continues to receive a resolved query object.

The focused support-intake suite retains all existing assertions and additionally checks the three real page boundaries with Promise query parameters, explicit club propagation to forms and links, privacy-profile values, ambiguous club values, and absent query parameters. After integration, the support-intake suite, `next typegen`, `tsc --noEmit`, focused ESLint checks, production `next build --webpack`, package/lock consistency checks, and `git diff --check` pass. An independent source review found no integration blocker. The original full component and browser results above remain evidence for the initial candidate; unchanged browser fixtures were not repeated for this rebase.

These integrated web checks ran on core `b12260a8` plus the dependency and intake changes. The final rebase onto `2b9b1208` adds only backend migration, contract, test and documentation changes; the application and workflow trees are unchanged from that validated integration.

## Primary references

- [Next support policy](https://nextjs.org/support-policy)
- [October 14 security update](https://nextjs.org/blog/nextjs-security-update-october-14-2026)
- [Next 16 upgrade guide](https://nextjs.org/docs/app/guides/upgrading/version-16)
- [Node release status](https://nodejs.org/en/about/previous-releases)
- [ESLint support status](https://eslint.org/version-support/)
- [Vercel Node versions and package override](https://vercel.com/docs/functions/runtimes/node-js/node-js-versions)
- [Remaining braces advisory](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm)
- [App Router DoS advisory](https://github.com/vercel/next.js/security/advisories/GHSA-8h8q-6873-q5fj)
- [AVIF image-optimization advisory](https://github.com/vercel/next.js/security/advisories/GHSA-2xp9-vwfh-vxw4)
