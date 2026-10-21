#!/usr/bin/env python3
"""
gc.py -- the retention ladder's DRY-RUN garbage collector.

WHY (docs/research/BRIEF-RETENTION-GC.md): saving everything in every
form is unaffordable -- an hour of raw audio is ~350 MB, its score plus
transcript a few hundred KB. The GC is a CURATION system; deletion is
the side effect. Collection is the one sanctioned deletion in a
never-delete house, so it runs DRY FIRST: this script REPORTS what
would be collected, never collects. **There is no delete call in this
file, by construction**; live collection starts only when the
quintessential miss rate is measured near zero and Casey rules on it.

THE LADDER (v1 constants, stated because they are per-person and will
be re-learned):
    AUDIO  HOT < 48 h; collectible (dry) only when derivation is
           complete AND not pinned AND not quintessential AND age >
           72 h.
    MIDI   collectible (dry) after 14 days, and ONLY when its tone
           essentials have GRADUATED into transcript annotations
           (retention.json annotations) -- how-it-was-said survives as
           marks on the words before the score is let go.
    TRANSCRIPT  kept forever, always.
PINS (the addendum's error-driven retention): a session beneath a
conversation the system got wrong is PINNED via POST /api/pin and is
exempt at EVERY layer until shadow work completes and a LESSON exists
(POST /api/unpin with the lesson). The debt clears when the lesson
exists, not when time passes.

DERIVATION CONTRACT (files the analysis tier writes into a session
folder): the score as `score.mid` or `score.json`; the transcript as
`transcript.json`; a quintessential marker as `quintessential.json`.
Presence = complete; the liaison does not produce them.

The report lands at browser_capture/gc-report-latest.json, a GC_DRY_RUN
event rides the liaison's outbox to the digest (via POST /log), and
GET /api/gc-report serves it to agents.

USAGE:  python3 browser_capture/gc.py            # dry run, report only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "gc-report-latest.json"

AUDIO_HOT_H = 48
AUDIO_COLLECT_AFTER_H = 72
MIDI_COLLECT_AFTER_D = 14

DERIV_SCORE = ("score.mid", "score.json")
DERIV_TRANSCRIPT = "transcript.json"
QUINT_MARKER = "quintessential.json"


def retention_path(sess: Path) -> Path:
    return sess / "retention.json"


def load_retention(sess: Path) -> dict:
    try:
        r = json.loads(retention_path(sess).read_text())
        if isinstance(r, dict):
            return r
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def save_retention(sess: Path, r: dict) -> None:
    retention_path(sess).write_text(json.dumps(r, indent=2))


def active_pins(r: dict) -> list:
    return [p for p in r.get("pins", []) if isinstance(p, dict)
            and not p.get("cleared")]


def session_state(sess: Path, out: Path) -> dict:
    """Everything the ladder knows about one session, measured from the
    files: age from meta's wall clock, audio health from meta's
    page-asserted + the liaison's measured cache (never trusted alone),
    derivations by presence, pins/annotations from retention.json."""
    try:
        meta = json.loads((sess / "meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    r = load_retention(sess)
    wc = meta.get("wall_clock_end") or meta.get("wall_clock_start") or ""
    age_h = None
    try:
        age_h = (time.time() - time.mktime(
            time.strptime(wc[:19], "%Y-%m-%dT%H:%M:%S"))) / 3600
    except ValueError:
        pass
    files = {p.name for p in sess.iterdir()}
    audio_bytes = (sess / "audio.wav").stat().st_size \
        if "audio.wav" in files else 0
    return {
        "stamp": sess.name,
        "dir": f"free_sessions/{sess.name}",
        "age_h": round(age_h, 1) if age_h is not None else None,
        "mode": meta.get("mode"),
        "audio_measured_silent": bool(
            (meta.get("audio") or {}).get("audio_silent")),
        "deriv_score": any(f in files for f in DERIV_SCORE),
        "deriv_transcript": DERIV_TRANSCRIPT in files,
        "quintessential": QUINT_MARKER in files,
        "pins": active_pins(r),
        "annotations": len(r.get("annotations", [])),
        "audio_bytes": audio_bytes,
        "has_audio": "audio.wav" in files,
        "has_midi_candidate": meta.get("mode") in (None, "auto", "manual",
                                                   "drill"),
    }


def ladder(st: dict) -> dict:
    """The tier verdicts for one session, with the BLOCKING REASON when
    a layer is not collectible -- the reason is the report's product."""
    v = {"audio": {"state": "kept", "collectible": False, "reason": None},
         "midi": {"state": "kept", "collectible": False, "reason": None}}
    if not st:
        return v
    if st["pins"]:
        for layer in v.values():
            layer["reason"] = "pinned"
        return v
    if st["quintessential"]:
        for layer in v.values():
            layer["reason"] = "quintessential"
        return v
    age = st["age_h"]
    if age is None:
        for layer in v.values():
            layer["reason"] = "unreadable-clock"
        return v
    # AUDIO
    a = v["audio"]
    a["state"] = "hot" if age < AUDIO_HOT_H else "warm"
    if not st["deriv_score"] or not st["deriv_transcript"]:
        a["reason"] = "derivation-incomplete"
    elif age < AUDIO_COLLECT_AFTER_H:
        a["reason"] = f"younger-than-{AUDIO_COLLECT_AFTER_H}h"
    else:
        a["collectible"] = True
    # MIDI: only if there is a score to thin, and only after its tone
    # essentials graduated into transcript annotations
    m = v["midi"]
    if not st["deriv_score"]:
        m["reason"] = "no-score-to-thin"
    elif age < MIDI_COLLECT_AFTER_D * 24:
        m["reason"] = f"younger-than-{MIDI_COLLECT_AFTER_D}d"
    elif st["annotations"] < 1:
        m["reason"] = "essentials-not-graduated-to-annotations"
    else:
        m["collectible"] = True
    return v


def dry_run(out: Path) -> dict:
    root = out / "free_sessions"
    sessions = []
    if root.is_dir():
        for sess in sorted(root.iterdir()):
            if not (sess / "meta.json").is_file():
                continue
            st = session_state(sess, out)
            if st:
                st["ladder"] = ladder(st)
                sessions.append(st)
    would_audio = [s for s in sessions if s["ladder"]["audio"]["collectible"]]
    would_midi = [s for s in sessions if s["ladder"]["midi"]["collectible"]]
    def count_reason(layer: str) -> dict:
        c: dict = {}
        for s in sessions:
            rsn = s["ladder"][layer]["reason"]
            if rsn:
                c[rsn] = c.get(rsn, 0) + 1
        return c
    silent = [s for s in sessions if s["audio_measured_silent"]]
    report = {
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out": str(out),
        "dry_run": True,
        "live_collection": "BLOCKED BY DESIGN -- the quintessential miss "
                           "rate is unmeasured (no selection criteria "
                           "yet); Casey rules on live collection",
        "sessions": len(sessions),
        "would_collect": {
            "audio": [{"stamp": s["stamp"], "dir": s["dir"],
                       "age_h": s["age_h"],
                       "bytes": s["audio_bytes"]} for s in would_audio],
            "midi": [{"stamp": s["stamp"], "dir": s["dir"],
                      "age_h": s["age_h"]} for s in would_midi],
        },
        "blocked": {"audio": count_reason("audio"),
                    "midi": count_reason("midi")},
        "pinned": [{"stamp": s["stamp"],
                    "reasons": [p.get("reason") for p in s["pins"]]}
                   for s in sessions if s["pins"]],
        "measured_silent_audio": [s["stamp"] for s in silent],
        "totals": {
            "audio_bytes_would_free":
                sum(s["audio_bytes"] for s in would_audio),
            "audio_bytes_total":
                sum(s["audio_bytes"] for s in sessions),
        },
        "audit": {
            "samples_rechecked": 0, "quintessential_misses": None,
            "note": "the audit re-checks routine-pile samples for "
                    "quintessentials hiding in them; MiniMax's selection "
                    "study owns the criteria -- until it exists the miss "
                    "rate is UNMEASURABLE and live collection stays "
                    "blocked",
        },
    }
    return report


def post_log(report: dict, base: str) -> None:
    evt = {"kind": "GC_DRY_RUN", "page": "gc", "ts": report["ran_at"],
           "sessions": report["sessions"],
           "would_collect_audio": len(report["would_collect"]["audio"]),
           "would_collect_midi": len(report["would_collect"]["midi"]),
           "audio_bytes_would_free": report["totals"]
               ["audio_bytes_would_free"],
           "pinned": len(report["pinned"])}
    try:
        req = urllib.request.Request(
            base + "/log", data=json.dumps(evt).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:  # noqa: BLE001 -- reporting, never fatal
        print(f"[gc] /log unreachable ({exc}); the report stands alone",
              file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/mnt/c/proart_capture/corpus")
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    report = dry_run(Path(a.out))
    REPORT.write_text(json.dumps(report, indent=2))
    post_log(report, a.base)
    if not a.quiet:
        print(f"[gc] DRY RUN {report['ran_at']}: {report['sessions']} "
              f"sessions; would collect "
              f"{len(report['would_collect']['audio'])} audio / "
              f"{len(report['would_collect']['midi'])} midi; "
              f"{report['totals']['audio_bytes_would_free'] / 1e6:.1f} MB "
              f"audio would free; pinned {len(report['pinned'])}")
        print(f"[gc] blocked (audio): "
              f"{report['blocked']['audio']}")
        print(f"[gc] report: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
