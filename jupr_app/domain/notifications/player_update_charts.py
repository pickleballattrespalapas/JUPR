from __future__ import annotations

from io import BytesIO
from datetime import datetime, timedelta


def render_player_digest_chart_png(digest_json: dict) -> bytes | None:
    chart = (digest_json or {}).get("chart") or {}
    points = chart.get("points") or []
    parsed: list[tuple[datetime | None, int, float]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            y = float(point.get("overall_after"))
        except Exception:
            continue
        match_number = int(point.get("match_number") or 0)
        dt = None
        try:
            raw_dt = str(point.get("date") or "").strip()
            if raw_dt:
                dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
        except Exception:
            dt = None
        parsed.append((dt, match_number, y))

    if not parsed:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 3.2), dpi=150)
    parsed = sorted(parsed, key=lambda row: (row[0] or datetime.min, row[1]))
    y = [row[2] for row in parsed]
    x_dates = [row[0] for row in parsed]
    if all(x_dates):
        ax.plot(x_dates, y, color="#1f77b4", linewidth=2.2, marker="o", markersize=3.5)
        ax.set_xlabel("Date")
        if len(x_dates) == 1:
            ax.set_xlim(x_dates[0] - timedelta(days=1), x_dates[0] + timedelta(days=1))
    else:
        x_idx = list(range(1, len(parsed) + 1))
        ax.plot(x_idx, y, color="#1f77b4", linewidth=2.2, marker="o", markersize=3.5)
        ax.set_xlabel("Match")
        if len(x_idx) == 1:
            ax.set_xlim(0, 2)
        else:
            ax.set_xlim(1, max(x_idx))
    ax.set_title(str(chart.get("title") or "Overall JUPR by Match"), fontsize=11)
    ax.set_ylabel("JUPR")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output = BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight")
    plt.close(fig)
    return output.getvalue()
