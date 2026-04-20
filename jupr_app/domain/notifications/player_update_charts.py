from __future__ import annotations

from io import BytesIO


def render_player_digest_chart_png(digest_json: dict) -> bytes | None:
    chart = (digest_json or {}).get("chart") or {}
    points = chart.get("points") or []
    parsed: list[tuple[str, float]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        date_text = str(point.get("date") or "").strip()
        try:
            y = float(point.get("overall_after"))
        except Exception:
            continue
        if not date_text:
            continue
        parsed.append((date_text, y))

    if len(parsed) < 2:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    x_vals = pd.to_datetime([row[0] for row in parsed], utc=True, errors="coerce")
    y_vals = [row[1] for row in parsed]

    valid = [(x, y) for x, y in zip(x_vals, y_vals) if not pd.isna(x)]
    if len(valid) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 3.2), dpi=150)
    x, y = zip(*valid)
    ax.plot(x, y, color="#1f77b4", linewidth=2.2, marker="o", markersize=3.5)
    ax.set_title(str(chart.get("title") or "Overall JUPR Trend"), fontsize=11)
    ax.set_ylabel("JUPR")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()

    output = BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight")
    plt.close(fig)
    return output.getvalue()
