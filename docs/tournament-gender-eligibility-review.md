# Tournament gender eligibility review

Registrations with a gender conflict may be submitted. The supplied genders are
preserved, and age, rating, identity, and contact validation still apply.

- Same-gender mixed pairs and entries outside a men's or women's division see a
  nonblocking notice that they can still submit.
- Non-binary entries receive no gender notice.
- Gender review is private to organizers. It is not included in the public
  registration response and does not send a player notification.

In **Tournament Admin → Registration → Registrants**, flagged entries show
**Needs admin approval**. Open the registration to approve or decline eligibility.
Only a club administrator, club owner, or super admin can decide.

Pending and declined entries cannot be imported into a draw. Approval covers the
current division and exact participant pair, including both sides of a confirmed
partner link. Changes to the pair, submitted genders, or division invalidate the
old decision. Other eligibility checks and partner confirmation still apply.
An imported entry must be removed from its draw before its decision can change.

Decisions are stored in `public.tournament_gender_eligibility_reviews`, with
reviewer and timestamp. Apply migration
`20260923003009_tournament_gender_eligibility_reviews.sql` before deploying the
API. The table has RLS enabled and is accessible only to the API service role.

Regression coverage: `tests/test_api_contract_tournament_gender_review.py`,
public-registration and draw-import contracts, and the web eligibility,
partner-profile, and registration-save component checks.
