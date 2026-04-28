# JUPR Private Operator README (Joe)

This README is for operating JUPR at Tres Palapas.
It is **not** written as a public contributor guide.

> Security rule: never paste secret values into this repo, commit history, issues, PRs, or ChatGPT.

## 1) What JUPR is

- JUPR is the player ratings and event operations app used at Tres Palapas.
- It tracks official player ratings and supports official Tres Palapas events.
- It is the source of truth for player-facing ratings and core tournament/admin workflows.

## 2) Branch model

Use this model consistently:

- `rollback-feb8` = **production** branch
- `Test` = **staging** branch
- `professional/*` = **work branches** that open PRs into `Test`

Typical flow:

1. Create/update a `professional/*` branch.
2. Open PR into `Test`.
3. Validate in staging.
4. Promote to production branch (`rollback-feb8`) only after staging is stable.

## 3) Streamlit deployment

- The **production Streamlit app** deploys from `rollback-feb8`.
- The **test/staging Streamlit app** deploys from `Test`.
- Secrets are stored in **Streamlit Community Cloud** secrets settings.
- Never put secrets in repo files, commit messages, PR comments, or ChatGPT prompts.

## 4) Supabase

- There is currently **one production Supabase project**.
- There is **no staging Supabase project yet**.
- SQL migrations are often manually pasted into the Supabase SQL editor.
- Preferred canonical location for migration files is: `supabase/migrations/`.

## 5) MailerSend / email

- Sender name: **JUPR Notifications**
- Reply-to address: **joe@juprleagues.com**
- Unsubscribe behavior must be enforced by the **JUPR database/state** (not only MailerSend provider state).

## 6) Local setup (minimal)

If you do not run locally, **skip this section**.

1. Install Python (use the same major/minor version configured in Streamlit settings; record that version here when confirmed).
2. Create and activate a virtual environment.
3. Install dependencies from `requirements.txt`.
4. Run the app locally with Streamlit for quick checks before pushing.

Example commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 7) Testing checklist before deploy

Before deploying any branch, smoke test these pages/flows:

- Public home page
- Leaderboards
- Player profile + player search
- Tournament registration page
- Admin login
- Match log/admin page (basic smoke test)

## 8) Migration checklist

When applying DB changes:

1. Review SQL carefully.
2. Apply SQL in Supabase SQL editor.
3. Verify expected tables/columns/indexes exist.
4. Deploy the correct app branch.
5. Run smoke tests in app after migration.

## 9) Rollback checklist

If a release causes issues:

1. Stop and document the issue briefly.
2. Re-deploy known-good branch (`rollback-feb8`) to production.
3. Confirm core user flows (home, leaderboards, login, registration).
4. If DB changes were part of incident, assess whether a DB rollback is safe before executing anything destructive.
5. Log what happened and what needs to be fixed before next release.

## 10) Making repo private

Order of operations:

1. Stabilize `Test` first.
2. Confirm Streamlit/GitHub integration has access to the private repository.
3. Then switch repo visibility to private.

## 11) What not to commit

Never commit any of the following:

- Supabase keys
- Streamlit secrets
- MailerSend API keys
- Private player exports
- Production DB dumps

If a secret is accidentally exposed, rotate it immediately and remove exposure from history where possible.
