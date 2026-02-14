import streamlit as st

from jupr_app.ui.layout import page_shell

def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("📘 Admin Guide", "Operational playbook for admins.", mode_label=mode_label)

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    st.markdown(
        """
### 🏟️ League Manager (Live Ladder Night)
1) **Setup:** Select League + Week, set Total Rounds, paste names  
2) **Roster review:** Add missing players (starting rating)  
3) **Courts:** Choose court sizes (4s/5s) and preview  
4) **Play:** Enter scores; matches are saved with snapshots  
5) **Movement:** Review auto movement; override if needed  
6) **Next round:** Confirm and continue

### 🧾 Record Match (Unified Wizard)
- Use this wizard for all match entry workflows (league, challenge ladder, tournaments, round robin, and moneyball)
- Legacy quick-entry tools have been retired to keep one canonical match writer

### 📝 Match Log
- Filter matches, scan duplicates, bulk delete
- After deletions: run **Admin Tools → Replay History → ALL** for consistency

### 👥 Player Editor
- Edit player name / active flag
- Edit per-league active + rating
- Merge duplicates (then run Replay ALL)

### ⚙️ Admin Tools
- Diagnostics: snapshot column checks + null snapshot scans
- Replay History: rebuild ratings and rewrite snapshots
- Migrations: one-time maintenance tasks
- Reports: export standings snapshots
"""
    )
