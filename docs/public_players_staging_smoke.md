# Public players staging smoke

This slice closes automated parity evidence for the public player directory and profile while keeping Streamlit available as the production fallback until manual acceptance.

## Automated gate

Run from the repository root after deploying the stacked PR to the isolated staging API and Vercel Preview:

```bash
cd apps/web
STAGING_WEB_BASE_URL=<staging-preview-url> \
VERCEL_AUTOMATION_BYPASS_SECRET=<automation-secret> \
npx playwright test e2e/players.staging.spec.ts
```

The browser suite uses the real staging pages and API. It does not intercept or synthesize responses. It proves:

- `/clubs/tres-palapas/players` defaults to Active, visibly searches public display names, and emits deterministic row/profile links;
- filtered-empty and API/profile error states are explicit and privacy-safe;
- a real staging profile renders doubles/overall and singles fields, format breakdowns, rating trend or its authoritative empty state, awards, relationships, Club Social projection, and verified-update/privacy entry;
- recent/full history navigation is URL-stable and every rendered match says Singles or Doubles.

## Manual acceptance (deferred until all parity pages are ready)

Use one active doubles player, one active player with singles matches, and one inactive player. Compare with Streamlit for:

1. public display name and privacy/alias handling;
2. current/starting/high/low rating facts and trend points;
3. league and Singles/Doubles breakdowns;
4. recent and full history ordering, teams, result, score, and rating change;
5. trophy/badge ownership, prestige, and dates;
6. best partner, rival, frequent partner/opponent facts;
7. Club Social totals, skill buckets, and recent events;
8. mobile table scrolling and card wrapping;
9. verified-update request and profile-privacy links.

Do not retire the Streamlit player page or alter production routing from this PR.
