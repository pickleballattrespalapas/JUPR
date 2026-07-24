# Custom Domain Cutover

Pickleball Club Sandwich is the official public SaaS name and domain direction. The intended production domain set is:

- `pickleballclubsandwich.com` — primary public SaaS domain.
- `www.pickleballclubsandwich.com` — primary www alias.
- `juprleagues.com` — transitional alias during the JUPR-to-Pickleball Club Sandwich rename.
- `www.juprleagues.com` — transitional www alias.

## Current read-only inventory

On 2026-07-24, both apex domains returned HTTP 200 from the current Vercel
project. Both `www` domains returned HTTP 502 and were absent from the project's
domain assignments. FastAPI CORS preflight succeeded for each of the four listed
origins and returned the exact origin rather than a wildcard.

No domain assignment, DNS record, redirect, CORS setting, or production
environment variable was changed during this inventory. The steps below remain
owner-controlled cutover work and require explicit approval before execution.

## Application configuration

Use `pickleballclubsandwich.com` as the canonical public web base URL once the domain is verified in Vercel:

```text
NEXT_PUBLIC_JUPR_WEB_BASE_URL=https://pickleballclubsandwich.com
JUPR_WEB_BASE_URL=https://pickleballclubsandwich.com
```

Keep the FastAPI public base URL separate:

```text
NEXT_PUBLIC_JUPR_API_BASE_URL=<public FastAPI base URL>
JUPR_API_BASE_URL=<public FastAPI base URL>
```

Backend CORS should include both the primary and transitional domains while the rename is in progress:

```text
JUPR_ALLOWED_ORIGINS=https://pickleballclubsandwich.com,https://www.pickleballclubsandwich.com,https://juprleagues.com,https://www.juprleagues.com
```

Do not put Supabase service-role credentials in Vercel frontend variables.

## Vercel domain setup

Verify the two existing apex assignments, then add and verify the missing `www`
aliases in the Vercel project that serves `apps/web`:

- `pickleballclubsandwich.com`
- `www.pickleballclubsandwich.com`
- `juprleagues.com`
- `www.juprleagues.com`

Set `pickleballclubsandwich.com` as the primary/canonical production domain after verification.

## DNS setup

At the DNS provider:

- Point the apex `pickleballclubsandwich.com` to Vercel using the Vercel-provided apex record.
- Point `www.pickleballclubsandwich.com` to Vercel using the Vercel-provided CNAME target.
- Keep `juprleagues.com` and `www.juprleagues.com` pointed to the same Vercel app during the transition.

Use the exact DNS values Vercel displays for the project because Vercel may vary verification targets by account/project.

## Smoke validation

After DNS is verified and Vercel deploys production:

```bash
python scripts/smoke_public_web.py \
  --api-base-url <public FastAPI base URL> \
  --web-base-url https://pickleballclubsandwich.com
```

Then run the same command for the transitional alias:

```bash
python scripts/smoke_public_web.py \
  --api-base-url <public FastAPI base URL> \
  --web-base-url https://juprleagues.com
```

## Rollback

If the custom domain cutover fails:

1. Use the production candidate record to identify the exact previously working
   Vercel deployment and restore its domain assignment.
2. Remove or pause the custom domain assignment in Vercel if it is serving a broken deployment.
3. Re-point DNS to the previous working target or use the direct Vercel deployment URL in communications.
4. Keep the Streamlit app available as the admin fallback until the public web app passes smoke.
5. Re-run `scripts/smoke_public_web.py` before trying the custom domain again.

Before cutover, record the production Git SHA, immutable Vercel deployment,
production Fly image, Supabase project ref/migration head, and feature-flag
projection in one rollback packet. Do not infer any of them from a moving alias.

## Rename follow-up

Once the custom-domain website is stable, audit public copy, docs, metadata, emails, and admin labels to replace remaining public-facing JUPR naming with Pickleball Club Sandwich language where appropriate. Keep internal package names and historical rating terminology stable until a separate technical rename plan is reviewed.
