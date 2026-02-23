from __future__ import annotations


def card_css(scope_selector: str = ".cc-root") -> str:
    """Return reusable card styles for Command Center surfaces."""
    return f"""
      {scope_selector} .cc-card {{
        border: 1px solid var(--cc-border);
        background: var(--cc-panel);
        color: var(--cc-text);
        border-radius: 16px;
      }}

      {scope_selector} .cc-card--e0 {{ box-shadow: none; }}
      {scope_selector} .cc-card--e1 {{ box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08); }}
      {scope_selector} .cc-card--e2 {{ box-shadow: var(--cc-shadow); }}
      {scope_selector} .cc-card--e3 {{ box-shadow: 0 18px 42px rgba(15, 23, 42, 0.2); }}

      {scope_selector}[data-theme='dark'] .cc-card--e1 {{ box-shadow: 0 3px 10px rgba(0, 0, 0, 0.42); }}
      {scope_selector}[data-theme='dark'] .cc-card--e3 {{ box-shadow: 0 22px 48px rgba(0, 0, 0, 0.66); }}

      {scope_selector} .cc-card--interactive {{
        transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
      }}

      {scope_selector} .cc-card--interactive:hover {{
        transform: translateY(-2px);
        border-color: var(--cc-accent-border);
      }}

      {scope_selector} .cc-card--interactive.cc-card--e0:hover {{
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.12);
      }}

      {scope_selector}[data-theme='dark'] .cc-card--interactive.cc-card--e0:hover {{
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.48);
      }}

      {scope_selector} .cc-card--interactive.cc-card--e1:hover {{
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.14);
      }}

      {scope_selector}[data-theme='dark'] .cc-card--interactive.cc-card--e1:hover {{
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.52);
      }}

      {scope_selector} .cc-card--interactive.cc-card--e2:hover {{
        box-shadow: 0 18px 38px rgba(15, 23, 42, 0.18);
      }}

      {scope_selector}[data-theme='dark'] .cc-card--interactive.cc-card--e2:hover {{
        box-shadow: 0 20px 42px rgba(0, 0, 0, 0.62);
      }}

      {scope_selector} .cc-card--interactive.cc-card--e3:hover {{
        box-shadow: 0 24px 50px rgba(15, 23, 42, 0.24);
      }}

      {scope_selector}[data-theme='dark'] .cc-card--interactive.cc-card--e3:hover {{
        box-shadow: 0 28px 56px rgba(0, 0, 0, 0.7);
      }}
    """


def Card(
    content: str,
    *,
    elevation: int = 1,
    interactive: bool = False,
    class_name: str = "",
    tag: str = "div",
    attrs: str = "",
) -> str:
    """Render an HTML card wrapper with elevation and interaction states."""
    if elevation not in (0, 1, 2, 3):
        raise ValueError("Card elevation must be in range 0-3.")

    classes = ["cc-card", f"cc-card--e{elevation}", "cc-card--interactive" if interactive else "cc-card--static"]
    if class_name:
        classes.extend(class_name.split())

    attrs_prefix = f" {attrs.strip()}" if attrs.strip() else ""
    return f"<{tag} class=\"{' '.join(classes)}\"{attrs_prefix}>{content}</{tag}>"
