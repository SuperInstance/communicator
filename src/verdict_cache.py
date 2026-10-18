"""
verdict_cache — same utterance, same questions, same model: same control values.

WHY THIS EXISTS. jev-lattice measured that Jev is not bit-deterministic: 7% of
gates flipped GO<->OPEN at threshold 0.9 when the SAME question was re-asked
against the SAME evidence. Their fix was a proof cache, and the same defect
lands harder here: CC values drive a synthesiser continuously, so a controller
that jitters for no reason is an instrument nobody can play. A listener hears
the flip. An operator tuning a threshold against it tunes against noise.

CONTENT-ADDRESSED, NOT TIME-ADDRESSED. The key is sha1(model, utterance text,
question fingerprint). There is deliberately no TTL:

  * The utterance text does not change after it is spoken.
  * The questions change only by editing judged.py, which changes the
    fingerprint, which invalidates every reading under the old wording with
    no manual invalidation code anywhere.
  * The model is named explicitly (never a floating alias), so a generation
    bump is a different key, not a silent mixture.

Staleness here would mean "the model changed its mind about identical
evidence", which is exactly the behaviour being suppressed.

Only successful readings are stored. A network failure must never become a
sticky verdict: the next call retries for real. Hits and misses are counted
because call counts are a first-class output in this project -- "how many
Jev calls did that cost" is a question the voice loop must always be able to
answer.

Pure standard library. File persistence is one JSON document; the scale is
thousands of utterances, not millions, and it can be read with `cat`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

from .judged import question_fingerprint


def cache_key(text: str, model: str, fingerprint: str | None = None) -> str:
    """sha1 over model + question wording + utterance text, truncated."""
    fp = fingerprint if fingerprint is not None else question_fingerprint()
    canon = json.dumps({"model": model, "fingerprint": fp, "text": text},
                       sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(canon.encode()).hexdigest()[:16]


@dataclass
class VerdictCache:
    """
    Content-addressed store of judged-track readings.

    `path=None` is a pure in-memory cache (tests, one-shot runs). A path
    persists to a single JSON document, loaded eagerly and rewritten on every
    put -- the scale does not justify a database, and a cache you can read
    with a text editor is a cache you can audit.
    """

    path: str | None = None
    hits: int = 0
    misses: int = 0
    _entries: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.path is not None and os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    doc = json.load(fh)
                if doc.get("v") == 1:
                    self._entries = doc["entries"]
            except (ValueError, OSError):
                # A corrupt cache file is a miss, never a crash: the cost of
                # rebuilding is one API call, the cost of a dead voice loop
                # is the voice loop.
                self._entries = {}

    def get(self, text: str, model: str) -> dict | None:
        key = cache_key(text, model)
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(self, text: str, model: str, resolved_model: str,
            readings: list) -> None:
        """Store one successful reading. Caller filters ok=False; this method
        double-checks, because a cached failure is the one unrecoverable bug
        in a cache whose selling point is stability."""
        if not readings or not all(getattr(r, "ok", False) for r in readings):
            return
        key = cache_key(text, model)
        self._entries[key] = {
            "ts": time.time(),
            "model": resolved_model,
            "readings": [
                {"track": r.track, "value": r.value, "detail": dict(r.detail),
                 "model": r.model, "ok": r.ok}
                for r in readings
            ],
        }
        if self.path is not None:
            self._flush()

    def __len__(self) -> int:
        return len(self._entries)

    def _flush(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"v": 1, "entries": self._entries}, fh,
                      ensure_ascii=True, sort_keys=True)
        os.replace(tmp, self.path)
