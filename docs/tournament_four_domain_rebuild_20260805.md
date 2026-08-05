# Tournament four-domain workspace rebuild

This staging-only rebuild organizes tournament administration into four product domains:

1. **Tournament** — basics, venue, registration settings, and public policies.
2. **Competition** — events, event policies, age and skill rules, and divisions.
3. **Commerce** — fees, extras, option presets, bundles, and giveaways.
4. **Review** — tournament preview, impact comparison, conflict resolution, publication, and registration opening.

The implementation includes venue-level court capacity, event-level age-policy inheritance and automatic age-split preview, four-player teams at the event level, resolved inherited defaults on division summaries, common option presets, automatic impact review, published-versus-proposed comparisons, and an audited force-change resolution workflow for affected registrations.

## Staging safety contract

- Target branch: `staging`
- Fly app: `juprleagues-api-staging`
- Supabase project: `sijpxjxvdtrehmqvirfi`
- Email mode: `dry_run`
- Staging write wave: `open`
- Production override: disabled

Production resources must not be accessed or changed by this work.
