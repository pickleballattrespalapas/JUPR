# Public round robin history

The club navigation links to `/clubs/{slug}/round-robins`. Completed round robins
and Moneyball can be filtered by session start date and type, with 24 sessions per
page. Dates use America/Mazatlan, matching the club's existing date helpers.

The archive includes public-created sessions, explicitly shared admin sessions,
and officially published or approved results. Unshared admin drafts, active or
archived sessions, ladders and unfinished supplemental drafts are excluded before
pagination. Each results URL checks the same eligibility and club scope again.

The directory selects only summary JSON fields. Opening one session computes its
standings with the existing generator rules and projects round scores. No badges,
rating replay, organizer writes, or database migrations run on these pages.
The website editor can hide the Round Robins navigation section, and results
respect the existing summary and score display settings.
