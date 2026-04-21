from __future__ import annotations

from io import BytesIO


def render_player_digest_chart_png(digest_json: dict) -> bytes | None:
    chart = (digest_json or {}).get("chart") or {}
    points = chart.get("points") or []
    parsed: list[tuple[int, float]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            x = int(point.get("match_number"))
            y = float(point.get("overall_after"))
        except Exception:
            continue
        if x < 1:
            continue
        parsed.append((x, y))

    if len(parsed) < 2:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 3.2), dpi=150)
    x, y = zip(*sorted(parsed, key=lambda row: row[0]))
    ax.plot(x, y, color="#1f77b4", linewidth=2.2, marker="o", markersize=3.5)
    ax.set_title(str(chart.get("title") or "Overall JUPR by Match"), fontsize=11)
    ax.set_xlabel("Match")
    ax.set_ylabel("JUPR")
    ax.grid(True, alpha=0.3)
    max_match = max(x)
    if max_match <= 10:
        tick_step = 1
    elif max_match <= 30:
        tick_step = 2
    elif max_match <= 75:
        tick_step = 5
    else:
        tick_step = 10
    tick_values = list(range(1, max_match + 1, tick_step))
    if tick_values[-1] != max_match:
        tick_values.append(max_match)
    ax.set_xticks(tick_values)
    ax.set_xlim(1, max_match)
    fig.tight_layout()

    output = BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight")
    plt.close(fig)
    return output.getvalue()
