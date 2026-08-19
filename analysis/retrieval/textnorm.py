"""Text normalization for evidence containment checks."""

from __future__ import annotations

import re


_WS_RE = re.compile(r"\s+")
_QUOTE_RE = re.compile(r"[\"'`“”‘’]")


def normalize(s: str) -> str:
    """Lowercase, strip quotes, collapse whitespace."""
    if not s:
        return ""
    out = s.lower().strip()
    out = _QUOTE_RE.sub("", out)
    out = _WS_RE.sub(" ", out)
    return out.strip()
