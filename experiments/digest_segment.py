"""Digest a closed segment: score it first, then say what it was.

The liaison's page wants a DIGESTION -- one short paragraph, what was
said in essence -- and it wants the segment SAVED AS A SCORE, not as flat
text. This is the plumbing for both, and it is one tool because the
digestion is made OF the score: there is no second pass over the audio
and no separate description of what happened.

The stages, each small and each replaceable:

    segment  ->  SCORE BUNDLE  ->  DIGESTION

The score bundle is the currency. It is the same bundle the
speech-to-score tool produces -- transcript with timestamps, a MIDI
melody, and an alignment table with per-syllable frame indices -- and
every part of it is on the session's own clock, so the digestion can
point at the frames it is talking about.

The digestion itself is a PLUGGABLE STAGE, because the tiers differ: a
laptop may visit the fleet, and the boat must not. With
--digest-command the tool hands the transcript to a command and uses what
comes back; without one it writes an extractive digest from the score and
SAYS that is what it did. The fallback is not a stub, though -- it is a
real paragraph built from the measurement, and it is honestly labelled so
nobody mistakes it for a generated one.

The repository's parts are used, not re-implemented: src/wavio.py for the
audio, src/pitch.py for the contour, src/smf.py for the MIDI, and
src/score.py for the note numbers and the tick convention. The
transcription and the syllables come from the speech-to-score tool, which
is imported rather than copied.

    python experiments/digest_segment.py <session dir> --start 12.0 --end 24.0
    python experiments/digest_segment.py <session dir> --segment 3
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import speech_to_score as s2s  # noqa: E402
from src import wavio  # noqa: E402


# ============================================================================
# 1. THE SEGMENT
# ============================================================================

def load_session(session_dir: Path) -> dict:
    """The session's own facts: the clock, the metadata, the audio."""
    clock = s2s.load_clock(session_dir)
    meta_path = session_dir / "meta.json"
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:  # noqa: BLE001
            meta = {}
    audio_path = session_dir / "audio.wav"
    audio = wavio.read_wav(audio_path) if audio_path.is_file() else None
    return {"clock": clock, "meta": meta, "audio": audio,
            "audio_path": audio_path}


def slice_audio(audio, start_s: float, end_s: float) -> np.ndarray:
    """The segment's samples, at the analysis rate.

    `src/wavio.py` has already resampled to 16 kHz, and the time base is
    exact, so a time range is a sample range and nothing shifts.
    """
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    sr = audio.sample_rate
    a = int(round(start_s * sr))
    b = int(round(end_s * sr))
    a = max(0, min(len(audio.samples), a))
    b = max(a, min(len(audio.samples), b))
    return np.asarray(audio.samples[a:b], dtype=np.float32)


def segment_bounds(args, session_dir: Path) -> tuple[float, float, dict]:
    """Where the segment is, in the session's own seconds.

    Three ways to say it, in order of preference: an explicit range, an
    index into the analyzer's segment table, or the whole session.
    """
    if args.start is not None and args.end is not None:
        return args.start, args.end, {"source": "explicit range"}
    if args.segment is not None:
        if not args.segments:
            print("a segment index needs --segments (the analyzer's table)")
            raise SystemExit(2)
        table = json.loads(Path(args.segments).read_text())
        segs = table.get("segments", [])
        if args.segment >= len(segs):
            print(f"segment {args.segment} is past the end of the table "
                  f"({len(segs)} segments)")
            raise SystemExit(2)
        seg = segs[args.segment]
        return (float(seg["start_s"]), float(seg["end_s"]),
                {"source": "analyzer segment table", "index": args.segment,
                 "face_state": seg.get("face_state"),
                 "speech_state": seg.get("speech_state"),
                 "frame_start": seg.get("frame_start"),
                 "frame_end": seg.get("frame_end")})
    meta = {}
    meta_path = session_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:  # noqa: BLE001
            meta = {}
    dur = float(meta.get("duration_s", 0.0) or 0.0)
    if dur <= 0:
        dur = 0.0
    return 0.0, dur, {"source": "the whole session"}


# ============================================================================
# 2. THE SCORE BUNDLE
# ============================================================================

def score_segment(samples: np.ndarray, clock, start_s: float, out_dir: Path,
                   model: str, device: str, compute_type: str,
                   segments_path: Path | None, session: str | None = None) -> dict:
    """The segment's score: words, melody, syllables, frames.

    The frame indices are the session's, not the segment's. A syllable
    that starts at 1.5 s into a segment that began at 12.0 s starts at
    frame N in the SESSION, so the page can link straight to the video.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. the words
    words, wmeta = s2s.transcribe(samples, model_name=model, device=device,
                                  compute_type=compute_type)
    # 2. the contour
    frames = s2s.pitch_frames(samples)
    local_notes = s2s.note_events(frames)

    # 3. the alignment, on the SESSION's clock. The notes are offset too:
    # a note measured on the segment is a note on the session's timeline,
    # and the two have to be in the same unit before they can be joined.
    offset_words = []
    for w in words:
        offset_words.append(s2s.Word(w.text, w.start + start_s, w.end + start_s,
                                     w.probability))
    notes = []
    for k, nt in enumerate(local_notes):
        moved = s2s.NoteEvent(index=k, start=nt.start + start_s,
                              end=nt.end + start_s, hz=nt.hz, midi=nt.midi,
                              confidence=nt.confidence, n_frames=nt.n_frames)
        notes.append(moved)
    syllabifier = s2s.Syllabifier()
    al = s2s.align(notes, offset_words, clock, syllabifier)
    if segments_path is not None and Path(segments_path).is_file():
        seg_doc = json.loads(Path(segments_path).read_text())
        s2s.join_segments(al["rows"], seg_doc.get("segments", []))

    # 4. the score itself
    midi_info = s2s.write_midi(notes, offset_words, out_dir / "segment.mid")

    report = {
        "session": session,
        "segment_start_s": start_s,
        "segment_end_s": start_s + len(samples) / float(s2s.ANALYSIS_SR),
        "duration_s": len(samples) / float(s2s.ANALYSIS_SR),
        "clock": clock.describe(),
        "transcribe": wmeta,
        "pitch": {
            "tracker": "src/pitch.py (YIN, 50 ms window, 10 ms hop, 70-400 Hz)",
            "n_frames": len(frames),
            "n_voiced": sum(1 for f in frames if f.hz is not None),
            "n_notes": len(notes),
        },
        "notes": [n.as_dict() for n in notes],
        "alignment": al,
        "syllables": {
            "source": syllabifier.source,
            "n_words": len(words),
            "n_syllables": al["n_syllables"],
            "words_from_dictionary": sum(1 for r in al["rows"]
                                         if r["syllable_source"] == "cmudict"),
            "words_from_heuristic": sum(1 for r in al["rows"]
                                        if r["syllable_source"] == "heuristic"),
            "heuristic_words": sorted({r["word"] for r in al["rows"]
                                       if r["syllable_source"] == "heuristic"}),
        },
        "midi": midi_info,
        "note": ("the words are offset onto the SESSION's clock, so every "
                 "syllable carries the session's frame indices and the page "
                 "can link straight to the video"),
    }
    (out_dir / "alignment.json").write_text(json.dumps(report, indent=2))
    (out_dir / "transcript.json").write_text(json.dumps(
        {"words": [{"text": w.text, "start_s": round(w.start, 4),
                    "end_s": round(w.end, 4),
                    "probability": round(w.probability, 4),
                    "frame_start": clock.frames_for_span(w.start, w.end)[0],
                    "frame_end": clock.frames_for_span(w.start, w.end)[1]}
                   for w in offset_words], "meta": wmeta}, indent=2))
    with (out_dir / "transcript.txt").open("w", encoding="utf-8") as f:
        for w in offset_words:
            f.write(f"[{w.start:8.2f} - {w.end:8.2f}] {w.text.strip()}\n")
    return {"report": report, "words": offset_words, "notes": notes,
            "syllabifier": syllabifier}


# ============================================================================
# 3. THE DIGESTION
# ============================================================================

def describe_contour(notes: list) -> dict:
    """What the melody did, in words a paragraph can carry."""
    if not notes:
        return {}
    midis = [n.midi for n in notes]
    durs = [n.end - n.start for n in notes]
    first, last = midis[0], midis[-1]
    hi, lo = max(midis), min(midis)
    # the shape: does it fall, rise, or stay
    if last - first <= -2:
        shape = "falls"
    elif last - first >= 2:
        shape = "rises"
    else:
        shape = "stays roughly level"
    # the biggest single move
    steps = [midis[i + 1] - midis[i] for i in range(len(midis) - 1)]
    biggest = max(steps, key=abs) if steps else 0
    return {
        "first_midi": first, "last_midi": last,
        "centre_midi": int(round(float(np.median(midis)))),
        "low_midi": lo, "high_midi": hi, "span_semitones": hi - lo,
        "shape": shape,
        "longest_note_s": round(max(durs), 3),
        "median_note_s": round(float(np.median(durs)), 3),
        "n_notes": len(notes),
        "biggest_step_semitones": biggest,
        "duration_s": round(sum(durs), 3),
        "midi_names": [s2s.midi_name(m) for m in midis],
    }


def extractive_digest(report: dict, words: list, notes: list,
                      segment_info: dict) -> str:
    """The fallback digestion: one paragraph, built from the measurement.

    It is not a generated summary and it does not pretend to be one. It
    says what was said, how it was delivered, and where it sits.
    """
    dur = report["duration_s"]
    n_words = len(words)
    if not n_words:
        return (f"A {dur:.1f}-second stretch with no speech in it. "
                f"The audio was measured and there were no words to "
                f"transcribe and no pitch to report, so there is nothing "
                f"here to say beyond that.")

    text = " ".join(w.text.strip() for w in words)
    text = " ".join(text.split())
    c = describe_contour(notes)
    pieces = []
    pieces.append(f"You spoke for {dur:.1f} seconds, {n_words} words")
    if c:
        pieces.append(
            f", on a melody that {c['shape']} from {s2s.midi_name(c['first_midi'])} "
            f"to {s2s.midi_name(c['last_midi'])}, spanning "
            f"{c['span_semitones']} semitones around {s2s.midi_name(c['centre_midi'])}, "
            f"in {c['n_notes']} notes whose longest lasted "
            f"{c['longest_note_s']:.2f} seconds")
    if segment_info.get("face_state") or segment_info.get("speech_state"):
        pieces.append(
            f", while {segment_info.get('face_state') or 'of unknown pose'} "
            f"and {segment_info.get('speech_state') or 'of unknown speech'}")
    pieces.append(f". You said: “{text}”")
    if report["alignment"]["n_without_note"]:
        pieces.append(
            f". {report['alignment']['n_without_note']} of "
            f"{report['alignment']['n_syllables']} syllables carried no "
            f"measurable note")
    return "".join(pieces) + "."


def generated_digest(command: str, report: dict, words: list) -> str | None:
    """Hand the transcript to a command and take what it says back.

    This is the fleet hook. The command gets the transcript on stdin and
    is expected to print one paragraph on stdout. If it fails, the
    caller falls back to the extractive digest rather than losing the
    segment.
    """
    text = " ".join(w.text.strip() for w in words)
    payload = {
        "transcript": text,
        "words": [{"text": w.text.strip(), "start_s": round(w.start, 3),
                   "end_s": round(w.end, 3)} for w in words],
        "duration_s": report.get("duration_s"),
        "n_notes": report.get("pitch", {}).get("n_notes"),
        "notes": report.get("notes", []),
    }
    try:
        proc = subprocess.run(command, shell=True, input=json.dumps(payload),
                              capture_output=True, text=True, timeout=300)
    except Exception as exc:  # noqa: BLE001
        print(f"the digest command failed: {exc}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"the digest command exited {proc.returncode}: "
              f"{proc.stderr.strip()[:200]}", file=sys.stderr)
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    # one paragraph: collapse the newlines a model likes to add
    return " ".join(out.split())


# ============================================================================
# 4. THE DRIVER
# ============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", nargs="?", default=None,
                    help="the free-session folder")
    ap.add_argument("--start", type=float, default=None,
                    help="the segment's start, in the session's seconds")
    ap.add_argument("--end", type=float, default=None,
                    help="the segment's end, in the session's seconds")
    ap.add_argument("--segment", type=int, default=None,
                    help="the segment's index in the analyzer's table")
    ap.add_argument("--segments", default=None,
                    help="the analyzer's segment table (JSON)")
    ap.add_argument("--out", default=None, help="where to write the bundle")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--device", default="auto", help="cuda | cpu | auto")
    ap.add_argument("--compute-type", default="auto")
    ap.add_argument("--digest-command", default=None,
                    help="a command that digests the transcript; it gets "
                         "the transcript as JSON on stdin and prints one "
                         "paragraph on stdout. Without it, the extractive "
                         "digest is used and says so.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if not a.session:
        ap.error("give a session folder, or --selftest")

    session_dir = Path(a.session)
    if not session_dir.is_dir():
        print(f"no such session folder: {session_dir}", file=sys.stderr)
        return 2
    sess = load_session(session_dir)
    if sess["audio"] is None:
        print(f"no audio.wav in {session_dir}", file=sys.stderr)
        return 2

    start_s, end_s, segment_info = segment_bounds(a, session_dir)
    out_dir = Path(a.out) if a.out else session_dir / "digestions" / \
        f"{int(start_s):06d}-{int(end_s):06d}"

    samples = slice_audio(sess["audio"], start_s, end_s)
    if samples.size == 0:
        print(f"the segment {start_s}-{end_s} s is outside the recording "
              f"({len(sess['audio'].samples) / sess['audio'].sample_rate:.1f} s)")
        return 2

    print(f"session   {session_dir}")
    print(f"segment   {start_s:.2f} - {end_s:.2f} s  ({segment_info['source']})")
    print(f"audio     {len(samples) / float(s2s.ANALYSIS_SR):.2f} s at "
          f"{s2s.ANALYSIS_SR} Hz")

    t0 = time.time()
    scored = score_segment(samples, sess["clock"], start_s, out_dir,
                           a.model, a.device, a.compute_type,
                           Path(a.segments) if a.segments else None,
                           session=str(session_dir))
    report = scored["report"]
    print(f"scored    {report['transcribe'].get('n_words', 0)} words, "
          f"{report['pitch']['n_notes']} notes, "
          f"{report['alignment']['n_syllables']} syllables "
          f"({time.time() - t0:.1f}s)")

    # the digestion
    generator = "extractive"
    digest = None
    if a.digest_command:
        digest = generated_digest(a.digest_command, report, scored["words"])
        if digest:
            generator = "command"
    if not digest:
        digest = extractive_digest(report, scored["words"], scored["notes"],
                                   segment_info)
        generator = "extractive"
    report["digest"] = {
        "text": digest,
        "generator": generator,
        "command": a.digest_command if generator == "command" else None,
        "honesty": ("this paragraph was built by the fallback from the "
                    "measurement, not generated; it says what was said and "
                    "what the delivery did, and it does not interpret"
                    if generator == "extractive" else
                    "this paragraph was generated by the command named "
                    "above, from the score bundle"),
    }
    (out_dir / "digest.json").write_text(json.dumps(report["digest"], indent=2))
    (out_dir / "alignment.json").write_text(json.dumps(report, indent=2))

    print()
    print("DIGESTION")
    print("-" * 72)
    print(digest)
    print("-" * 72)
    print(f"generator  {generator}")
    print(f"bundle     {out_dir}")
    return 0


# ============================================================================
# 5. THE SELFTEST
# ============================================================================

def selftest() -> int:
    """The checks that need no model and no session."""
    import tempfile

    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  [{'ok' if got == want else 'FAIL'}] {name}: {got!r}")

    print("SELFTEST -- the slice, the clock, the digest")
    print("-" * 60)

    # 1. THE SLICE. The time base is exact, so a range is a sample range.
    class FakeAudio:
        sample_rate = 16000
        samples = [float(i) for i in range(16000)]  # one second

    seg = slice_audio(FakeAudio(), 0.25, 0.5)
    check("the slice has the right length", len(seg), 4000)
    check("and the right first sample", float(seg[0]), 4000.0)
    check("and the right last sample", float(seg[-1]), 7999.0)
    check("a slice past the end is empty",
          len(slice_audio(FakeAudio(), 5.0, 6.0)), 0)
    check("a slice that starts past the end is empty",
          len(slice_audio(FakeAudio(), 9.0, 10.0)), 0)

    # 2. THE CLOCK. A syllable inside a segment carries the SESSION's
    # frame indices, not the segment's.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "sess"
        d.mkdir()
        offs = [round(i * 1000.0 / 60.0, 3) for i in range(600)]  # 10 s
        (d / "timestamps.json").write_text(json.dumps(
            {"wall_clock_start": "2026-01-01T00:00:00Z", "frame_offsets_ms": offs}))
        clock = s2s.load_clock(d)
        check("the clock has the frames", clock.n_frames, 600)
        # the session is 10 s long, so 6.0-7.0 s is frames 360..419
        f0, f1 = clock.frames_for_span(6.0, 7.0)
        check("a session-time span finds session frames", (f0, f1), (360, 419))
        check("and a span past the session's end has no frames",
              clock.frames_for_span(12.0, 13.0), (None, None))

    # 3. THE DIGEST. It says what was said and what the delivery did.
    notes = [s2s.NoteEvent(0, 0.0, 0.3, 130.0, 57, 0.9, 30),
             s2s.NoteEvent(1, 0.35, 0.8, 146.0, 59, 0.9, 45)]
    words = [s2s.Word(" hello ", 0.0, 0.4, 0.9),
             s2s.Word(" there ", 0.4, 0.8, 0.9)]
    report = {"duration_s": 0.8, "pitch": {"n_notes": 2},
              "alignment": {"n_syllables": 2, "n_without_note": 0}}
    d = extractive_digest(report, words, notes, {"face_state": "facing",
                                                  "speech_state": "speaking"})
    check("the digest mentions the words", "hello" in d, True)
    check("the digest mentions the delivery", "melody" in d, True)
    check("the digest mentions the face state", "facing" in d, True)
    check("and it is one paragraph", "\n" in d, False)
    empty = extractive_digest({"duration_s": 2.0,
                                "alignment": {"n_syllables": 0, "n_without_note": 0}},
                               [], [], {})
    check("an empty segment says so", "no speech" in empty, True)

    # 4. THE COMMAND HOOK. A command that prints nothing falls back.
    got = generated_digest("printf ''", report, words)
    check("an empty command gives nothing", got, None)

    print("-" * 60)
    if failures:
        print(f"SELFTEST FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
