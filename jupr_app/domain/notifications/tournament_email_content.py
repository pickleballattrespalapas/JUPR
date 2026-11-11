"""Bounded, passive HTML for an organizer's tournament email body."""
from __future__ import annotations

import re
from html import escape
from html.parser import HTMLParser
from urllib.parse import urlsplit

TEXT_LIMIT = 10_000
HTML_LIMIT = 100_000
_TAGS = {
    "a", "b", "br", "caption", "div", "em", "h1", "h2", "h3", "h4",
    "hr", "i", "li", "ol", "p", "small", "span", "strong", "table",
    "tbody", "td", "th", "thead", "tr", "u", "ul",
}
_ATTRS = {
    "align", "aria-hidden", "aria-label", "bgcolor", "border", "cellpadding",
    "cellspacing", "class", "colspan", "height", "lang", "role", "rowspan",
    "scope", "style", "title", "valign", "width",
}
_CSS = {
    "background", "background-color", "border", "border-bottom", "border-collapse",
    "border-color", "border-left", "border-radius", "border-right", "border-top",
    "border-spacing", "color", "display", "font-family", "font-size", "font-style",
    "font-weight", "height", "letter-spacing", "line-height", "list-style-type",
    "margin", "margin-bottom", "margin-left", "margin-right", "margin-top",
    "max-height", "max-width", "min-height", "min-width", "mso-hide", "mso-padding-alt",
    "opacity", "overflow", "overflow-wrap", "padding", "padding-bottom", "padding-left",
    "padding-right", "padding-top", "table-layout", "text-align", "text-decoration",
    "text-transform", "vertical-align", "white-space", "width", "word-break", "word-wrap",
}
_BLOCKS = {"br", "div", "h1", "h2", "h3", "h4", "hr", "li", "ol", "p", "table", "tr", "ul"}


def clean_tournament_message(message: str, message_format: str = "text") -> str:
    if message_format not in {"text", "html"}:
        raise ValueError("Choose plain text or HTML for the email format.")
    message = str(message or "").strip()
    limit = HTML_LIMIT if message_format == "html" else TEXT_LIMIT
    if "\x00" in message or len(message) > limit:
        raise ValueError(f"Use an email message up to {limit:,} characters without null characters.")
    if message_format == "html" and message:
        render_tournament_message_html(message)
    return message


class _EmailHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.html: list[str] = []
        self.text: list[str] = []
        self.stack: list[str] = []
        self.href: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _TAGS:
            raise ValueError(f"HTML emails cannot contain <{tag}>. Paste the email body only.")
        safe = []
        names = set()
        for name, raw_value in attrs:
            value = raw_value or ""
            if name in names or name not in _ATTRS | ({"href", "target"} if tag == "a" else set()):
                raise ValueError(f"HTML emails cannot use the {name} attribute.")
            names.add(name)
            if any(ord(character) < 32 and character not in "\t\r\n" for character in value):
                raise ValueError("Remove control characters from the HTML email.")
            if name == "href":
                parsed = urlsplit(value.strip())
                if parsed.scheme not in {"https", "http", "mailto"} or (parsed.scheme != "mailto" and (not parsed.hostname or parsed.username or parsed.password)):
                    raise ValueError("Email links must use https, http, or mailto addresses.")
                self.href.append(value)
            if name == "target" and value != "_blank":
                raise ValueError("HTML email links may only open a new window.")
            if name == "style":
                for declaration in value.split(";"):
                    if not declaration.strip():
                        continue
                    prop, separator, css_value = declaration.partition(":")
                    if not separator or prop.strip().lower() not in _CSS or not re.fullmatch(r"[a-zA-Z0-9\s#%,.!()/'\-:]*", css_value) or re.search(r"url\s*\(|expression\s*\(|javascript|vbscript", css_value, re.I):
                        raise ValueError("HTML emails support static inline formatting only.")
            safe.append(f' {name}="{escape(value, quote=True)}"')
        self.html.append("<" + tag + "".join(safe) + ">")
        if tag in _BLOCKS:
            self.text.append("\n")
        if tag not in {"br", "hr"}:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in {"br", "hr"}:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"br", "hr"}:
            return
        if not self.stack or self.stack.pop() != tag:
            raise ValueError("The HTML email has unbalanced tags.")
        self.html.append(f"</{tag}>")
        if tag == "a" and self.href:
            self.text.append(" (" + self.href.pop() + ")")
        if tag in _BLOCKS:
            self.text.append("\n")
        elif tag in {"td", "th"}:
            self.text.append(" · ")

    def handle_data(self, data: str) -> None:
        self.html.append(escape(data))
        # A hidden preheader is useful in HTML but is still harmless readable text.
        self.text.append(data)

    def handle_decl(self, decl: str) -> None:
        raise ValueError("Paste an HTML email body without a document declaration.")


def render_tournament_message_html(message: str) -> tuple[str, str]:
    parser = _EmailHTML()
    parser.feed(message)
    parser.close()
    if parser.stack:
        raise ValueError("The HTML email has unbalanced tags.")
    text = re.sub(r"[ \t]+", " ", "".join(parser.text))
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return "".join(parser.html), text
