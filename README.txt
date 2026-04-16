Apply these replacements on branch rollback-feb8:

1. Replace streamlit_app.py with the file in this bundle.
2. Add jupr_app/ui/admin_auth.py.
3. Update requirements.txt to include extra-streamlit-components>=0.1.71.
4. Redeploy.

What this patch does:
- Persists signed admin sessions to a browser cookie.
- Restores admin login on hard page loads and direct ?page=... navigation.
- Keeps the existing HMAC-signed TTL model.

Notes:
- This patch fixes the sitewide “looks logged out after internal navigation or reload” pattern.
- It does not yet refactor internal admin link buttons to in-app navigation, but with cookie-backed restore those links should no longer boot the user out.
