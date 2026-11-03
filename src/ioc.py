"""Deterministic indicator-of-compromise matching.

Threat-feed URLs are exact-match data. Embedding them (as the old Milvus feed documents did)
made retrieval depend on shared boilerplate text rather than on whether the email actually
contains the indicator. Here we extract URLs from the email and look them up directly.

* exact URL match  -> `match_type="url"`  -> hard evidence (routes to a deterministic block)
* host-only match  -> `match_type="host"` -> soft evidence (shown to the LLM, can trigger the
                       guard veto, but never blocks on its own, because feeds often list
                       paths on otherwise legitimate hosts)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple
from urllib.parse import urlsplit

_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+", re.IGNORECASE)
_TRAILING = ".,;:!?"


def normalize_url(url: str) -> str:
    """Lower-case scheme+host, drop fragment and trailing punctuation/slash. Scheme-insensitive key."""
    u = url.strip().rstrip(_TRAILING)
    try:
        parts = urlsplit(u)
    except ValueError:
        return u.lower()
    host = (parts.hostname or "").lower()
    path = parts.path.rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{path}{query}"


def extract_urls(text: str) -> List[str]:
    return [m.group(0).rstrip(_TRAILING) for m in _URL_RE.finditer(text or "")]


def host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


class IOCIndex:
    def __init__(self, records: Iterable[Dict[str, Any]], fetched_at: str = ""):
        self.fetched_at = fetched_at
        self._by_url: Dict[str, Dict[str, Any]] = {}
        self._by_host: Dict[str, Dict[str, Any]] = {}
        for r in records:
            url = r.get("url", "")
            if not url:
                continue
            self._by_url.setdefault(normalize_url(url), r)
            h = host_of(url)
            if h:
                self._by_host.setdefault(h, r)

    def __len__(self) -> int:
        return len(self._by_url)

    @classmethod
    def from_json(cls, path: str) -> "IOCIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data.get("records", []), fetched_at=data.get("fetched_at", ""))

    def match(self, text: str) -> List[Dict[str, Any]]:
        """Return de-duplicated matches for every URL found in `text`."""
        out: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()
        for url in extract_urls(text):
            key = normalize_url(url)
            rec = self._by_url.get(key)
            if rec is not None and (key, "url") not in seen:
                seen.add((key, "url"))
                out.append({"match_type": "url", "url": url, **_public(rec)})
                continue
            h = host_of(url)
            rec = self._by_host.get(h)
            if rec is not None and (h, "host") not in seen:
                seen.add((h, "host"))
                out.append({"match_type": "host", "url": url, **_public(rec)})
        return out


def _public(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {"indicator_id": rec.get("indicator_id", ""), "source": rec.get("source", ""),
            "threat_type": rec.get("threat_type", "")}
