"""Shared normalization component (Sec. 3.4.1): exact -> partial -> model-based -> fallback, with a cache.

Maps informal, misspelled or non-standard inputs ("al-6061-t6", "Aluminum 6061", "SS304") to canonical IDs.
"""
from __future__ import annotations

import difflib
import re
import threading
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .llm import LLM


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


class Normalizer:
    def __init__(self, canonical: dict[str, list[str]], llm: "LLM | None" = None, kind: str = "term") -> None:
        """canonical: {canonical_id: [aliases...]}"""
        self.canonical = canonical
        self.kind = kind
        self.llm = llm
        self._cache: dict[str, tuple[str | None, str]] = {}
        self._lock = threading.Lock()
        self._alias_index = {}
        for cid, aliases in canonical.items():
            for a in [cid, *aliases]:
                self._alias_index[_key(a)] = cid

    def normalize(self, text: str | None) -> tuple[str | None, str]:
        """Returns (canonical_id or None, method)."""
        if not text:
            return None, "empty"
        k = _key(text)
        with self._lock:
            if k in self._cache:
                return self._cache[k]
        result = self._exact(k) or self._partial(k) or self._model(text) or (None, "fallback")
        with self._lock:
            self._cache[k] = result
        return result

    def _exact(self, k: str):
        cid = self._alias_index.get(k)
        return (cid, "exact") if cid else None

    def _partial(self, k: str):
        # substring containment either way, then fuzzy ratio
        for alias, cid in self._alias_index.items():
            if len(alias) >= 4 and (alias in k or k in alias):
                return cid, "partial"
        best = difflib.get_close_matches(k, list(self._alias_index), n=1, cutoff=0.82)
        if best:
            return self._alias_index[best[0]], "partial"
        return None

    def _model(self, text: str):
        if self.llm is None:
            return None
        ids = list(self.canonical)
        obj = self.llm.json_call(
            f"You map an informal {self.kind} name to one canonical ID from a fixed list. "
            "Reply with JSON {\"id\": <one of the ids or null>, \"confidence\": 0..1}. Use null if nothing matches.",
            f"input: {text!r}\ncanonical ids: {ids}", agent=f"normalizer.{self.kind}", thinking=False)
        cid = obj.get("id")
        if cid in self.canonical and float(obj.get("confidence", 0)) >= 0.5:
            return cid, "model"
        return None


def build_alias_map(entries: Iterable[dict], id_key: str = "id", alias_key: str = "aliases") -> dict[str, list[str]]:
    return {e[id_key]: list(e.get(alias_key, [])) + [e.get("name", "")] for e in entries}
