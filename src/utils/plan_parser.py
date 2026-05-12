"""Shared plan parsing helpers for LiquidWorld outputs."""

import re

PLAN_TEXT_REPLACEMENTS = {
    "\u00a0": " ",
    "\u202f": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u2005": " ",
    "\u2003": " ",
    "\u2192": " to ",
    "\u21d2": " to ",
    "\u27f6": " to ",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2044": "/",
}

POUR_ACTION_RE = re.compile(
    r"\b[Pp]our\s+"
    r"(?P<volume>\d+(?:\.\d+)?|\d+/\d+)"
    r"\s*L\s*"
    r"(?:\([^)]*\)\s*)?"
    r"from\s+(?P<source>\w+)\s+to\s+(?P<dest>\w+)",
    re.IGNORECASE,
)


def normalize_plan_text(text: str) -> str:
    """Strip presentation formatting before extracting actions."""
    for old, new in PLAN_TEXT_REPLACEMENTS.items():
        text = text.replace(old, new)
    return re.sub(r"(\*\*|__|`)", "", text)


def format_pour_action(match: re.Match) -> str:
    """Normalize regex matches into canonical pour-action strings."""
    return f"Pour {match.group('volume')} L from {match.group('source')} to {match.group('dest')}"


def extract_pour_actions(text: str) -> list[str]:
    """Extract canonical pour actions from a text fragment."""
    normalized = normalize_plan_text(text)
    return [format_pour_action(match) for match in POUR_ACTION_RE.finditer(normalized)]


def find_last_plan_summary(text: str) -> str | None:
    """Return the final Plan summary block, stopping at a blank line or EOF."""
    matches = list(
        re.finditer(
            r"[Pp]lan [Ss]ummary:?\s*\n(.*?)(?=\n\s*\n|\Z)",
            text,
            re.DOTALL,
        )
    )
    if matches:
        return matches[-1].group(1)
    return None
