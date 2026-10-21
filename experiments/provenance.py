"""Provenance: what KIND of data a session is, and what that is worth.

Casey's doctrine, in his words: the first take is real discovery, the
second take is acting like you discovered it. So the corpus is not one
kind of data and the portrait must not average the kinds together.

Two classes:

    DRILLED   performed. Somebody pressed the button and delivered
              something -- the potato drill, a recitation, a reading.
              The words were chosen and the delivery was on purpose.
    AMBIENT   caught. The page opened because there was motion or sound,
              and whatever was said was said first time, unposed.

They carry different evidentiary weight in the portrait. An ambient
session is evidence about the person; a drilled session is evidence
about the performance. Both are worth keeping. Averaging them into one
number would be evidence about neither.

**The provenance already carries the trigger.** The page's own
`meta.json` records WHY the session opened -- "motion" for the
auto-caught ones, "manual" for the ones somebody started -- and that is
the signal. This module reads it and names the class; it does not
guess.

The classes are the one piece of vocabulary shared by the speech tool,
the digestion and the person profile, so they live in their own small
module rather than in three copies that can drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The triggers that mean somebody DELIBERATELY performed. A session
#: that opened because of one of these is a drilled session, whatever
#: else the page recorded about it.
PERFORMED_TRIGGERS = {"manual", "drill", "potato", "perform", "performed",
                      "recite", "scripted"}

#: The modes that mean the session was performed rather than caught.
PERFORMED_MODES = {"manual", "drill", "potato", "perform", "recite"}

DRILLED = "drilled"
AMBIENT = "ambient"

CLASS_NOTES = {
    DRILLED: ("performed on purpose: the words were chosen and the "
              "delivery was on purpose. Evidence about the performance."),
    AMBIENT: ("caught first time: the page opened because there was "
              "motion or sound, and whatever was said was said unposed. "
              "Evidence about the person."),
}


def read_meta(session_dir) -> dict:
    """The session's own metadata, or an empty dict."""
    p = Path(session_dir) / "meta.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}


def data_class(meta: dict) -> dict:
    """Name the class of a session and say why.

    The trigger is the page's own record of why the session opened. A
    deliberate trigger means the session was performed, and everything
    else was caught.
    """
    trigger = str(meta.get("trigger", "") or "").lower()
    mode = str(meta.get("mode", "") or "").lower()
    close = str(meta.get("close_reason", "") or "").lower()

    if trigger in PERFORMED_TRIGGERS or mode in PERFORMED_MODES:
        why = f"the page recorded trigger={trigger!r} and mode={mode!r}"
        cls = DRILLED
    else:
        why = (f"the page recorded trigger={trigger!r} and mode={mode!r}, "
               "which is the page catching something rather than somebody "
               "starting a performance")
        cls = AMBIENT
    return {
        "data_class": cls,
        "why": why,
        "trigger": trigger,
        "mode": mode,
        "close_reason": close,
        "note": CLASS_NOTES[cls],
    }


def session_class(session_dir) -> dict:
    """The class of a session on disk, and its provenance."""
    return data_class(read_meta(session_dir))


def provenance(session_dir) -> dict:
    """The full provenance of a session: the class and the raw fields."""
    meta = read_meta(session_dir)
    d = data_class(meta)
    d["session"] = str(session_dir)
    d["n_frames"] = meta.get("n_frames")
    d["wall_clock_start"] = meta.get("wall_clock_start")
    d["wall_clock_end"] = meta.get("wall_clock_end")
    d["achieved_video_fps"] = meta.get("achieved_video_fps")
    d["audio_sample_rate"] = (meta.get("audio") or {}).get("sample_rate")
    return d
