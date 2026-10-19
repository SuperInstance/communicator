#!/usr/bin/env python3
"""Fetch a small multi-speaker conversational slice from the AMI Meeting Corpus.

WHY THIS EXISTS
---------------
`docs/research/ADOPTION-PLAN.md` and the commit that measured it: the whole
`examples/realtwospeaker`-adjacent corpus (`examples/realaudio/corpus/`) is
single-speaker continuous recording.  Measured: after any silence >= 0.6 s the
same speaker resumes 98.1-100% of the time.  The positive class -- *a speaker
yields the floor and somebody else takes it* -- is unrepresentable, so no
turn-taking accuracy number can be computed from it.

AMI is real 4-person meetings, so it contains that class.  Measured on the
segment annotations alone (this file, `--analyse-only`): after a >= 0.6 s gap
a DIFFERENT speaker takes the floor 73.5% of the time in EN2002a and 63.4% in
EN2002b.

HOW IT FETCHES (no pip dependency, no registration, no token)
-------------------------------------------------------------
The HuggingFace mirror `edinburghcstr/ami` is parquet-only and would need
pyarrow plus ~250 MB per shard.  Instead we use the public datasets-server
`rows` endpoint, which serves the *same* rows as individual 16 kHz WAV assets
with no auth.  Stdlib `urllib` only.

HOW THE SLICE IS RECONSTRUCTED  (read this before trusting a gap)
-----------------------------------------------------------------
The dataset stores each meeting as per-utterance WAV *slices* of the single
SDM (far-field) microphone channel, each with `begin_time`/`end_time`/
`speaker_id`/`microphone_id`.  The `audio_id` encodes those times to the sample,
e.g. AMI_EN2002a_sdm_FEO072_0035475_0035713 -> [354.75, 357.13) s.

Because SDM is ONE microphone, overlapping segments are slices of the *same*
signal.  Summing them would comb-filter.  So this script PAINTS them in
begin_time order, writing only the part of each segment beyond the watermark
already painted: that reconstructs the original channel exactly, and each
instant of audio is written once.

Two consequences, both real limitations, repeated in the README:
  * A segment wholly inside an already-painted region contributes ZERO samples.
    Those overlapping utterances are still listed in the sidecar segments file
    (so ground truth is complete) but they are not separately audible.
  * A region with NO annotated segment is rendered as digital silence.  We do
    not know what was said or heard there; the dataset simply has no row for it.
    A silence in the output is therefore not always a silence in the meeting.

IHM (headset mics, one per speaker) is fetched for a short sub-window as an
independent per-speaker-channel reference.

Run:  python3 fetch_ami.py --analyse-only
       python3 fetch_ami.py --out ../ami_sdm_EN2002a_turns.wav
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET = "edinburghcstr/ami"
SPLIT = "test"          # fixed for reproducibility
SR = 16000              # the analysis rate; the dataset is already at 16 kHz
PAGE = 100              # datasets-server hard page size
PAGE_DELAY = 2.0        # the API answers 429 under bursts; pace every page
NEAR_FAR = "SDM"        # far-field single mic -- what a peripheral actually hears
NEAR_HEADSET = "IHM"
FLOOR_S = 0.6           # the silence after which the existing corpus is 98-100% same-speaker
MERGE_S = 0.3           # same-speaker segments closer than this are one turn


def get_json(url: str, label: str, attempts: int = 6) -> dict:
    """GET + json, with real backoff for 429 (the API throttles bursts)."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < attempts - 1:
                wait = min(60.0, 5.0 * 2 ** attempt)
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after and retry_after.isdigit():
                    wait = float(retry_after)
                print(f"[{label}] 429, sleeping {wait:.0f}s "
                      f"(attempt {attempt + 1}/{attempts})")
                time.sleep(wait)
                continue
            if 400 <= exc.code < 500 and exc.code != 429:
                raise
            if attempt == attempts - 1:
                raise
            print(f"[{label}] retry {attempt + 1} after {exc.code}")
            time.sleep(3.0 * (attempt + 1))
        except (urllib.error.URLError, OSError) as exc:
            if attempt == attempts - 1:
                raise
            print(f"[{label}] retry {attempt + 1} after {type(exc).__name__}")
            time.sleep(3.0 * (attempt + 1))
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- metadata


def fetch_rows(config: str, cache: Path, want_meetings: list[str]) -> list[dict]:
    """Page the public rows API until every wanted meeting is fully covered.

    Rows come grouped by meeting_id, so we stop as soon as the last wanted
    meeting has been passed.  Cached to disk so re-runs do not re-hit the API.
    """
    if cache.exists():
        rows = json.loads(cache.read_text())
        # A cache is only reusable if it ran PAST every wanted meeting. A cache
        # whose last row still belongs to a wanted meeting was truncated by an
        # earlier run and must be re-fetched.
        tail_meeting = rows[-1]["meeting_id"] if rows else None
        covered = all(m in {r["meeting_id"] for r in rows} for m in want_meetings) \
            and tail_meeting not in want_meetings
        if not want_meetings or covered:
            print(f"[meta] {cache.name}: {len(rows)} cached rows")
            return rows
        print(f"[meta] {cache.name}: {len(rows)} rows but stops inside "
              f"{tail_meeting!r}; re-fetching")

    rows: list[dict] = []
    seen: list[str] = []
    tail = None
    for offset in range(0, 12000, PAGE):
        url = (f"{ROWS_API}?dataset={DATASET.replace('/', '%2F')}"
               f"&config={config.lower()}&split={SPLIT}&offset={offset}&length={PAGE}")
        payload = get_json(url, f"meta:{config}")
        for row in payload["rows"]:
            r = row["row"]
            rows.append({k: v for k, v in r.items() if k != "audio"})
            if not seen or seen[-1] != r["meeting_id"]:
                seen.append(r["meeting_id"])
            tail = r["meeting_id"]
        # rows arrive grouped by meeting_id, so once every wanted meeting has
        # been seen AND the page tail has moved past them, we are done. Testing
        # only "all wanted seen" would stop one page into the first meeting.
        if want_meetings and all(m in seen for m in want_meetings) and tail not in want_meetings:
            break
        if offset % 500 == 0:
            print(f"[meta] {config} offset={offset} rows={len(rows)} meetings={seen}")
        time.sleep(PAGE_DELAY)
    cache.write_text(json.dumps(rows))
    print(f"[meta] {config}: {len(rows)} rows -> {cache.name}")
    return rows


def to_turns(rows: list[dict]) -> list[dict]:
    """Merge same-speaker segments separated by < MERGE_S into one turn."""
    ordered = sorted(rows, key=lambda r: (r["begin_time"], r["end_time"]))
    turns: list[dict] = []
    for r in ordered:
        if (turns and turns[-1]["speaker_id"] == r["speaker_id"]
                and r["begin_time"] - turns[-1]["end_time"] < MERGE_S):
            turns[-1]["end_time"] = r["end_time"]
            turns[-1]["segments"].append(r)
        else:
            turns.append({"speaker_id": r["speaker_id"],
                          "begin_time": r["begin_time"],
                          "end_time": r["end_time"],
                          "segments": [r]})
    return turns


def analyse(rows: list[dict], meeting: str) -> dict:
    """The positive-class count.  This is the reason the slice exists."""
    turns = to_turns([r for r in rows if r["meeting_id"] == meeting])
    gaps = [(turns[i + 1]["begin_time"] - turns[i]["end_time"],
             turns[i]["speaker_id"], turns[i + 1]["speaker_id"])
            for i in range(len(turns) - 1)]
    floor = [g for g in gaps if g[0] >= FLOOR_S]
    yield_ = [g for g in floor if g[1] != g[2]]
    return {"meeting": meeting, "turns": len(turns), "gaps": len(gaps),
            "floor_gaps": len(floor), "floor_gap_then_other_speaker": len(yield_)}


def pick_window(rows: list[dict], meeting: str, length_s: float) -> dict:
    """Pick the `length_s` window with the most "silence then different speaker".

    This is a cherry-pick and it is a real bias: the slice is deliberately the
    turn-richest part of the meeting, so its switch rate is NOT the meeting's
    average.  README says so; the analyse() numbers are the meeting average.
    """
    turns = to_turns([r for r in rows if r["meeting_id"] == meeting])
    if not turns:
        raise SystemExit(f"no rows for meeting {meeting}")
    origin = turns[0]["begin_time"]
    best, best_score = 0, -1
    for start in [origin + i * 5.0 for i in range(int((turns[-1]["end_time"] - origin - length_s) // 5.0) + 1)]:
        end = start + length_s
        score = 0
        for i in range(len(turns) - 1):
            if start <= turns[i]["end_time"] and turns[i + 1]["begin_time"] <= end:
                if (turns[i + 1]["begin_time"] - turns[i]["end_time"] >= FLOOR_S
                        and turns[i]["speaker_id"] != turns[i + 1]["speaker_id"]):
                    score += 1
        if score > best_score:
            best, best_score = start, score
    return {"meeting": meeting, "start": best, "end": best + length_s,
            "yield_events_in_window": best_score}


# ---------------------------------------------------------------- audio


def decode_wav_f32(blob: bytes) -> tuple[np.ndarray, int]:
    """Decode a RIFF/WAVE blob, honouring wFormatTag 3 (float32).

    The datasets-server assets are 32-bit float WAV.  `wave` in the stdlib
    raises `unknown format: 3` on those, so the chunks are walked by hand.
    """
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE blob")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(blob):
        cid = blob[pos:pos + 4]
        size = struct.unpack("<I", blob[pos + 4:pos + 8])[0]
        body = blob[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError("missing fmt or data chunk")
    tag, channels, rate, _, _, bits = fmt
    if tag != 3 or bits != 32:
        raise ValueError(f"expected float32 (tag 3), got tag={tag} bits={bits}")
    samples = np.frombuffer(data, dtype="<f4").astype(np.float64)
    if channels > 1:                      # avg to mono, never take channel 0
        usable = (samples.size // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return samples, rate


def download_clip(row: dict, retries: int = 4) -> np.ndarray:
    url = row["audio"][0]["src"]
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                blob = r.read()
            samples, rate = decode_wav_f32(blob)
            if rate != SR:
                raise ValueError(f"expected {SR} Hz, got {rate}")
            return samples
        except Exception as exc:                     # noqa: BLE001 - reported below
            if attempt == retries - 1:
                raise RuntimeError(f"{row['audio_id']}: {type(exc).__name__}: {exc}")
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def paint(segments: list[dict], start_s: float, length_s: float) -> tuple[np.ndarray, list[dict]]:
    """Reconstruct one microphone channel over [start_s, start_s+length_s).

    Segments are ordered by begin_time and each contributes only the samples
    beyond the watermark already painted -- they are slices of ONE channel, so
    summing overlapping slices would comb-filter.  Every segment is recorded in
    the sidecar whether or not it contributed samples.
    """
    buf = np.zeros(int(length_s * SR), dtype=np.float64)
    placed: list[dict] = []
    painted_until = start_s
    for row in sorted(segments, key=lambda r: (r["begin_time"], r["end_time"])):
        b, e = row["begin_time"], row["end_time"]
        if e <= start_s or b >= start_s + length_s:
            continue
        samples = row["_samples"]
        # crop to window, then to the not-yet-painted remainder
        lo = max(b, start_s, painted_until)
        if e <= lo:
            placed.append({**{k: v for k, v in row.items() if k != "_samples"},
                           "audible": False, "audible_start_s": None,
                           "audible_end_s": None})
            continue
        src_from = int(round((lo - b) * SR))
        dst_from = int(round((lo - start_s) * SR))
        n = min(samples.size - src_from, buf.size - dst_from)
        buf[dst_from:dst_from + n] += samples[src_from:src_from + n]
        painted_until = lo + n / SR
        placed.append({**{k: v for k, v in row.items() if k != "_samples"},
                       "audible": True, "audible_start_s": round(lo, 4),
                       "audible_end_s": round(lo + n / SR, 4)})
    return buf, placed


def write_wav_pcm16(path: Path, samples: np.ndarray) -> None:
    """16-bit PCM mono 16 kHz -- what src/wavio.py reads."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def gather(rows: list[dict], meeting: str, start_s: float, end_s: float,
           config: str, label: str) -> list[dict]:
    """Attach freshly downloaded clip audio to each segment row."""
    # The cached metadata deliberately drops the signed asset URL, so the rows
    # are paged again here purely to recover the `audio[0].src` field.
    print(f"[dl:{config}] fetching asset URLs for {meeting} "
          f"[{start_s:.1f}, {end_s:.1f})")
    todo: list[dict] = []
    seen_meeting = False
    for offset in range(0, 12000, PAGE):
        url = (f"{ROWS_API}?dataset={DATASET.replace('/', '%2F')}&config={config.lower()}"
               f"&split={SPLIT}&offset={offset}&length={PAGE}")
        payload = get_json(url, f"dl:{config}")
        page_rows = [row["row"] for row in payload["rows"]]
        for r in page_rows:
            if r["meeting_id"] == meeting:
                seen_meeting = True
                if r["begin_time"] < end_s and r["end_time"] > start_s:
                    todo.append(r)
        # rows arrive grouped by meeting, so the first non-matching page after
        # the meeting has started means the meeting is behind us
        if seen_meeting and all(r["meeting_id"] != meeting for r in page_rows):
            break
        time.sleep(PAGE_DELAY)
    print(f"[dl:{config}] rows API returned {len(todo)} matching segments")
    if not todo:
        raise SystemExit(f"no {config} segments matched {meeting} "
                         f"[{start_s:.1f}, {end_s:.1f})")
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_clip, r): r for r in todo}
        for fut in concurrent.futures.as_completed(futures):
            r = futures[fut]
            r["_samples"] = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"[dl:{config}] {done}/{len(todo)}")
    return todo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="output .wav path for the far-field slice")
    ap.add_argument("--meeting", default="EN2002a")
    ap.add_argument("--length", type=float, default=420.0, help="slice seconds")
    ap.add_argument("--ihm-length", type=float, default=90.0,
                    help="headset-mic per-speaker-channel reference window, "
                         "seconds (0 to skip)")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--ihm-only", action="store_true",
                    help="skip the far-field fetch; build the headset-channel "
                         "reference from an existing slice sidecar")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    corpus = here.parent

    sdm = fetch_rows(NEAR_FAR, here / f"meta_{NEAR_FAR.lower()}_{SPLIT}.json",
                     [args.meeting])
    if args.analyse_only:
        print(json.dumps({m: analyse(sdm, m)
                          for m in sorted({r["meeting_id"] for r in sdm})}, indent=2))
        return 0
    if not args.out:
        ap.error("--out is required unless --analyse-only")

    win = pick_window(sdm, args.meeting, args.length)
    print(f"[win] {win}")

    out = Path(args.out).resolve()
    sidecar = out.with_suffix(".segments.json")
    if args.ihm_only:
        # Resume path: the far-field slice already exists on disk and the rows
        # API is rate-limited, so recover the window from the sidecar instead of
        # re-downloading ~300 clips.
        if not sidecar.exists():
            ap.error(f"--ihm-only needs an existing {sidecar}")
        prior = json.loads(sidecar.read_text())
        win = {"meeting": prior["meeting_id"],
               "start": prior["meeting_window_start_s"],
               "end": prior["meeting_window_end_s"]}
        print(f"[ihm-only] reusing window from {sidecar.name}: {win}")
    else:
        segments = gather(sdm, win["meeting"], win["start"], win["end"], NEAR_FAR, args.out)
        buf, placed = paint(segments, win["start"], win["end"] - win["start"])
        audible = sum(1 for p in placed if p["audible"])
        print(f"[mix] {len(placed)} segments, {audible} audible, "
              f"{len(placed) - audible} wholly inside an overlapping region")
        print(f"[mix] peak={np.abs(buf).max():.4f} rms={np.sqrt((buf**2).mean()):.5f}")

        write_wav_pcm16(out, buf)
        sidecar.write_text(json.dumps({
            "file": out.name,
            "config": NEAR_FAR,
            "dataset": DATASET, "split": SPLIT, "meeting_id": win["meeting"],
            "meeting_window_start_s": win["start"], "meeting_window_end_s": win["end"],
            "sample_rate": SR, "channels": 1, "encoding": "PCM 16-bit little-endian",
            "speakers": sorted({p["speaker_id"] for p in placed}),
            "segments": [{k: v for k, v in p.items() if k != "audio"} for p in placed],
        }, indent=1))

    # --- IHM per-speaker-channel reference over a short sub-window -----------
    # Each AMI headset mic is a physically separate channel, so these files are
    # channel-separated by construction: that is the strongest available
    # speaker ground truth, and it is independent of any F0 measurement.
    if args.ihm_length > 0:
        sub_start = win["start"] + 60.0
        ihm = gather(fetch_rows(NEAR_HEADSET,
                                here / f"meta_{NEAR_HEADSET.lower()}_{SPLIT}.json",
                                [win["meeting"]]),
                     win["meeting"], sub_start, sub_start + args.ihm_length,
                     NEAR_HEADSET, "ihm")
        ch_dir = corpus / f"ihm_{win['meeting']}_channels_{int(args.ihm_length)}s"
        ch_dir.mkdir(exist_ok=True)
        for spk in sorted({r["speaker_id"] for r in ihm}):
            mine = [r for r in ihm if r["speaker_id"] == spk]
            b, placed_spk = paint(mine, sub_start, args.ihm_length)
            write_wav_pcm16(ch_dir / f"spk_{spk}.wav", b)
            (ch_dir / f"spk_{spk}.segments.json").write_text(json.dumps({
                "file": f"spk_{spk}.wav", "config": NEAR_HEADSET, "speaker_id": spk,
                "dataset": DATASET, "split": SPLIT, "meeting_id": win["meeting"],
                "window_start_s": sub_start,
                "window_end_s": sub_start + args.ihm_length,
                "sample_rate": SR, "encoding": "PCM 16-bit little-endian",
                "segments": [{k: v for k, v in p.items() if k != "audio"} for p in placed_spk],
            }, indent=1))
            print(f"[ihm] {spk}: {len(placed_spk)} segments, "
                  f"peak={np.abs(b).max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())