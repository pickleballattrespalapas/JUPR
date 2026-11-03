"""Edition-scoped activities. The same question bank is served to the editor."""
from datetime import date
from .weekly_report_public import text


def poll(key, question, options, tips, keywords):
    return {"id": key, "question": question, "options": options, "tips": tips, "keywords": keywords}


POLLS = [
    poll("drops-resets", "Which shot would you most like to practice?",
         ["Third-shot drops", "Transition resets"],
         ["Start at the baseline with a partner at the kitchen line. Land five third-shot drops in the kitchen, then switch.",
          "Start halfway to the kitchen. Have a partner volley gently at your feet and try to land five soft resets in the kitchen."],
         ["clinic", "drop", "reset"]),
    poll("serve-return", "Which start to a point would you like to improve?",
         ["A deeper serve", "A deeper return"],
         ["Place a target three feet inside the baseline. Try ten serves and count how many land beyond the service box midpoint.",
          "Have a partner serve ten balls. Aim your return deep through the middle and follow it toward the kitchen."],
         ["serve", "return"]),
    poll("partners", "Which doubles habit would help your team most?",
         ["Calling the middle ball", "Moving together"],
         ["Before each point, agree who covers the middle. Call the ball early and see how many points you play without hesitation.",
          "Start side by side at the kitchen. Have your opponents alternate targets while you and your partner shift together."],
         ["doubles", "partner", "team"]),
    poll("kitchen", "What would you work on first at the kitchen line?",
         ["Crosscourt dinks", "Quick volleys"],
         ["Dink crosscourt with a partner. Count how many consecutive balls both players can hit below net height.",
          "Stand at the kitchen line with a partner. Start with gentle volleys and build speed while keeping your paddle in front."],
         ["dink", "volley", "kitchen"]),
    poll("pressure", "Which point would you like another chance to practice?",
         ["Returning a lob", "Blocking a speed-up"],
         ["Have a partner feed comfortable lobs. Turn and move into position, then choose a controlled reply.",
          "Start with cooperative dinks. Let your partner occasionally speed one up while you practice a compact block."],
         ["defense", "block", "lob"]),
    poll("match", "Which part of match day would you like to rehearse?",
         ["Playing the first point", "Serving at game point"],
         ["Start six practice games at 0-0. Before each one, tell your partner your plan for the first three shots.",
          "Start at 10-10 and play out the game. Switch servers and repeat, using the same routine before each serve."],
         ["tournament", "playoff", "ladder"]),
]
MODES = {"automatic", "review", "off"}
FEATURES = {"poll", "shoutout", "none"}


def clean_engagement(raw):
    if not isinstance(raw, dict):
        return None
    question = raw.get("poll") or {}
    if not isinstance(question, dict):
        raise ValueError("Choose a valid poll.")
    options = question.get("options") or []
    tips = question.get("tips") or []
    if not isinstance(options, list) or len(options) != 2 or not isinstance(tips, list) or len(tips) != 2:
        raise ValueError("Choose two answers and a follow-up tip for each one.")
    options = [text(value, 80) for value in options]
    tips = [text(value, 450) for value in tips]
    title = text(question.get("question"), 200)
    if not title or not all(options + tips) or options[0].casefold() == options[1].casefold():
        raise ValueError("Give the poll a question, two different answers and their tips.")
    if str(raw.get("mode")) not in MODES or str(raw.get("featured")) not in FEATURES:
        raise ValueError("Choose an available engagement setting.")
    return {"mode": raw["mode"], "featured": raw["featured"], "approved": raw.get("approved") is True,
            "shoutouts": raw.get("shoutouts") is True,
            "poll": {"id": text(question.get("id"), 80), "question": title, "options": options, "tips": tips},
            "subject": text(raw.get("subject"), 140), "body": text(raw.get("body"), 1600)}


def new_engagement(db, club_id, week_start, recap):
    prefs = db.table("club_recap_preferences").select("settings").eq("club_id", club_id).limit(1).execute().data or []
    settings = (prefs[0].get("settings") or {}) if prefs else {}
    mode = settings.get("engagement_mode") or "automatic"
    report = recap.get("report") or {}
    history = (db.table("recap_engagement_editions").select("config").eq("club_id", club_id)
               .eq("scope_key", report.get("scope_label") or "Whole club").lt("week_start", week_start)
               .order("week_start", desc=True).limit(4).execute().data or [])
    recent = {(r.get("config") or {}).get("poll", {}).get("id") for r in history}
    corpus = " ".join(str(item.get(key) or "") for block in report.get("edition", []) for item in block.get("items", [])
                      for key in ("kind", "title", "text")).casefold()
    offset = date.fromisoformat(week_start).toordinal() // 7 % len(POLLS)
    ordered = POLLS[offset:] + POLLS[:offset]
    chosen = max(ordered, key=lambda item: (item["id"] not in recent, sum(word in corpus for word in item["keywords"])))
    return clean_engagement({"mode": mode, "featured": "poll", "approved": False, "shoutouts": settings.get("engagement_shoutouts") is True, "poll": chosen})
