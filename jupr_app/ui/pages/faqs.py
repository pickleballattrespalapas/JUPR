import streamlit as st

from jupr_app.ui.layout import page_shell


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("❓ JUPR Rating FAQs", "Answers about JUPR ratings and recorded play.", mode_label=mode_label)
    st.markdown(
        """
JUPR (Joe’s Unique Pickleball Ratings) is Tres Palapas’ **in-house rating system** used to create better matchups, seed events, and keep leveled play fair.

---

## What is a JUPR rating?
- JUPR is a **single player rating** displayed on a **1.000–7.000** scale.
- Ratings update based on **recorded match results** from JUPR-eligible events at Tres Palapas.
- The decimals help show *small* movement over time, but they are not meant to imply “perfect precision.”

---

## How do I get a JUPR rating?
You get a JUPR rating once you have **recorded matches** in the system from a JUPR-eligible event (examples below).
- If you haven’t played a recorded event yet, you may show as **Unrated / No JUPR**.
- The same calculation rules apply from your first rated match. There is no provisional period or special new-player multiplier.

---

## What matches count toward JUPR?
**Counts (JUPR-eligible):**
- Official JUPR Ladders and JUPR Round Robins
- Tournaments run through Tres Palapas with official score entry

**Does not count (not recorded / not JUPR-eligible):**
- Open Play
- Social Round Robins
- Drills and clinics

---

## What affects how much my rating moves?
JUPR is performance-based. Rating movement depends on:
- **Opponent strength** (beating stronger opponents moves you more)
- **Expected outcome** (results that surprise the system move you more)
- **Game score** (the score matters—not only win/loss)

JUPR does not add recency, reliability, or a changing new-player factor to the calculation.

---

## Can my rating go up after a loss?
Yes, it can happen.
If you **perform better than expected** (for example: a very close loss against a significantly higher-rated team), your rating may increase.

---

## Can my rating go down after a win?
No, winning is rewarded, not punished.
However, because scores matter, a win that is **far below expected performance** can result in minimal movement but will never result in a loss of rating. 

---

## How does JUPR work for doubles?
JUPR is an **individual rating**, but doubles results are used to update each player.
In doubles:
- the system evaluates the matchup based on **both teams** (each team’s strength is derived from the two players),
- then gives both partners the same rating change based on the outcome and score.

JUPR stays an in-house club rating; it is not intended to replace a national or universal rating.

---

## What is the difference between “Overall” and “League” ratings?
- **Overall JUPR**: your rating across all JUPR-eligible matches at Tres Palapas.
- **League JUPR** (if shown): your rating **within a specific league** or series.

If you only play one league, your league rating and overall rating may look similar. If you play multiple formats/events, they can differ.

---

## What if a score was entered wrong?
If you believe there is a data-entry error (wrong score, wrong partner, wrong opponent):
- report it to the organizer as soon as possible.
Once corrected, the system will recompute the rating impact from the accurate result.

---

## How should I use my JUPR rating?
Use it to:
- join the right leveled sessions,
- seed ladders and tournaments fairly,
- track improvement over time,
- create competitive, enjoyable matches.

JUPR is designed to reflect **performance at Tres Palapas** based on recorded play.

---
"""
    )
