from __future__ import annotations

from io import BytesIO


def render_player_digest_chart_png(digest_json: dict) -> bytes | None:
    chart = (digest_json or {}).get("chart") or {}
    points = chart.get("points") or []
    point_rows: list[dict] = []
    for idx, point in enumerate(points, start=1):
        if not isinstance(point, dict):
            continue
        try:
            y = float(point.get("overall_after"))
        except Exception:
            continue
        match_number = None
        try:
            raw_match_number = point.get("match_number")
            if raw_match_number not in (None, ""):
                match_number = int(raw_match_number)
        except Exception:
            match_number = None
        point_rows.append(
            {
                "overall_after": y,
                "match_number": match_number,
                "fallback_x": idx,
                "is_anchor": bool(point.get("is_anchor")),
            }
        )

    if not point_rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 3.2), dpi=150)
    point_rows = sorted(
        point_rows,
        key=lambda row: (
            0 if row["is_anchor"] else 1,
            row["match_number"] if row["match_number"] is not None else row["fallback_x"],
            row["fallback_x"],
        ),
    )
    parsed: list[tuple[int, float]] = []
    for sequence_idx, row in enumerate(point_rows, start=1):
        x_value = row["match_number"] if row["match_number"] is not None else sequence_idx
        parsed.append((x_value, row["overall_after"]))

    x_values = [row[0] for row in parsed]
    y_values = [row[1] for row in parsed]
    ax.plot(x_values, y_values, color="#1f77b4", linewidth=2.2, marker="o", markersize=3.5)
    ax.set_xlabel("Match")

    if len(x_values) == 1:
        x_single = float(x_values[0])
        ax.set_xlim(x_single - 1, x_single + 1)
    else:
        x_min = min(x_values)
        x_max = max(x_values)
        if x_min == x_max:
            ax.set_xlim(float(x_min) - 1, float(x_max) + 1)
        else:
            ax.set_xlim(float(x_min), float(x_max))
    ax.set_title(str(chart.get("title") or "Overall JUPR by Match"), fontsize=11)
    ax.set_ylabel("JUPR")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output = BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight")
    plt.close(fig)
    return output.getvalue()
