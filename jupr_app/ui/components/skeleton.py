from __future__ import annotations


def skeleton_css() -> str:
    """Shared CSS for low-contrast shimmer skeleton placeholders."""
    return """
      @keyframes cc-skeleton-shimmer {
        0% {
          background-position: 180% 0;
        }
        100% {
          background-position: -80% 0;
        }
      }

      .cc-skeleton-block {
        border-radius: 10px;
        display: block;
        width: 100%;
        background: linear-gradient(
          100deg,
          color-mix(in srgb, var(--cc-muted) 10%, transparent) 30%,
          color-mix(in srgb, var(--cc-muted) 18%, transparent) 50%,
          color-mix(in srgb, var(--cc-muted) 10%, transparent) 70%
        );
        background-size: 220% 100%;
        animation: cc-skeleton-shimmer 1.8s ease-in-out infinite;
      }

      .cc-skeleton-header {
        display: grid;
        gap: 0.5rem;
      }

      .cc-skeleton-row {
        border: 1px solid var(--cc-border);
        border-radius: 12px;
        background: color-mix(in srgb, var(--cc-bg) 90%, var(--cc-panel));
        padding: 0.75rem 0.85rem;
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        gap: 0.7rem;
        min-height: 72px;
      }

      .cc-skeleton-row-left {
        display: flex;
        align-items: center;
        gap: 0.7rem;
      }

      .cc-skeleton-chip-wrap {
        display: inline-flex;
        gap: 0.45rem;
      }

      .cc-skeleton-card {
        border: 1px solid var(--cc-border);
        border-radius: 12px;
        background: var(--cc-bg);
        padding: 0.8rem 0.9rem;
        min-height: 114px;
        display: grid;
        align-content: space-between;
        gap: 0.7rem;
      }

      .cc-skeleton-card-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
      }
    """


def _block(height_px: int, width: str, class_name: str = "") -> str:
    klass = "cc-skeleton-block"
    if class_name:
        klass += f" {class_name}"
    return f'<span class="{klass}" style="height:{height_px}px; width:{width};"></span>'


def render_header_skeleton() -> str:
    return (
        '<div class="cc-skeleton-header" aria-hidden="true">'
        + _block(20, "42%")
        + _block(12, "68%")
        + "</div>"
    )


def render_table_row_skeleton() -> str:
    return (
        '<article class="cc-skeleton-row" aria-hidden="true">'
        '<div class="cc-skeleton-row-left">'
        + _block(24, "40px")
        + '<div style="display:grid; gap:0.4rem; width:100%;">'
        + _block(14, "52%")
        + _block(11, "36%")
        + "</div></div>"
        + '<div class="cc-skeleton-chip-wrap">'
        + _block(20, "44px")
        + _block(20, "56px")
        + "</div></article>"
    )


def render_card_skeleton() -> str:
    return (
        '<article class="cc-skeleton-card" aria-hidden="true">'
        '<div style="display:grid; gap:0.6rem;">'
        + _block(16, "60%")
        + _block(12, "38%")
        + "</div>"
        + '<div class="cc-skeleton-card-actions">'
        + _block(30, "116px")
        + _block(30, "128px")
        + "</div></article>"
    )
