"""Semantic identity for lifetime achievements across historical engine keys."""
SINGLE_LIFETIME = {"participant", "dedicated_participant_50", "lifetime_participant_200", "first_win", "social_butterfly", "network_builder", "high_roller", "clutch_performer", "nemesis_found"}


def award_key(row: dict) -> tuple:
    badge_id = str(row.get("badge_id"))
    pid = int(row["player_id"])
    if badge_id in SINGLE_LIFETIME:
        return pid, badge_id
    if badge_id == "level_up":
        value = row.get("value_json") or {}
        milestone = value.get("milestone")
        if milestone is None:
            try:
                milestone = str(row.get("context_id") or "").split("milestone:")[1].split(":")[0]
            except IndexError:
                milestone = None
        try:
            milestone = float(milestone)
            milestone = milestone / 400 if milestone > 20 else milestone
            return pid, badge_id, milestone
        except (TypeError, ValueError):
            pass
    return pid, badge_id, str(row.get("context_type")), str(row.get("context_id"))
