# JUPR Web (Next.js public shell)

This is a minimal, read-only Next.js app for public pages that can eventually serve `juprleagues.com` and club pages.

## Routes

- `/` public JUPR landing page.
- `/clubs/[clubSlug]` club landing page.
- `/clubs/[clubSlug]/leaderboards` public leaderboard page.

## Environment variables

Use one of the following:

- `JUPR_API_BASE_URL` (preferred for server-side runtime)
- `NEXT_PUBLIC_JUPR_API_BASE_URL` (fallback)

Example:

```bash
export JUPR_API_BASE_URL=http://localhost:8000
```

## Local development

```bash
cd apps/web
npm install
npm run dev
```

Then open `http://localhost:3000`.

If the API is unavailable, pages render graceful empty/error states instead of crashing.
