#!/usr/bin/env python3
"""
browser_capture_speech_probe.py -- run the PAGE's OWN speech gate over a WAV,
the way ZCode runs the page's own maths over the frames.

WHY THIS FILE EXISTS.

`browser_capture/verify.py` checks the browser's change-map and drift
mathematics by extracting the shipped JavaScript out of `index.html` and
running it under node -- the right idea, applied to the maths. It does NOT do
that for the speech gate. Instead it posts hardcoded `speech_frac`
constants, so the one number in the take metadata that the corpus's
achievement flags depend on is never measured by the verification.

This is the referee doing for the speech gate what the verifier does for the
maps: take the page's OWN `speechStats` function, verbatim, and run it over
the WAV that was actually written.

    python3 experiments/browser_capture_speech_probe.py <wav> [<wav> ...]
    python3 experiments/browser_capture_speech_probe.py --page <index.html> <wav>

It prints, for each WAV, what the page's gate says and what the pipeline's
own Silero gate says, and whether they agree.

The Silero side is optional: it needs the repo's model weights and
onnxruntime. Without them the Silero column reads "(unavailable)" and the
probe still reports the page's number.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAGE = Path("/home/proart/projects/communicator/browser_capture/index.html")

HARNESS = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
// The page's OWN gate, taken verbatim out of the shipped file.
const start = html.indexOf('function speechStats');
if (start < 0) { console.error('no speechStats in the page'); process.exit(2); }
const end = html.indexOf('/* A 16-bit PCM mono WAV');
const fn = html.slice(start, end);
const speechStats = new Function(fn + '\nreturn speechStats;')();
// The WAV the server actually wrote, read back as the page's float samples.
const b = fs.readFileSync(process.argv[3]);
const rate = b.readUInt32LE(24);
const n = (b.length - 44) / 2;
const s = new Float32Array(n);
for (let i = 0; i < n; i++) s[i] = b.readInt16LE(44 + 2 * i) / 32767;
const r = speechStats(s, n, rate);
console.log(JSON.stringify({rate, n, frac: r.frac, floor: r.floor, thr: r.thr}));
"""


def page_gate(page: Path, wav: Path) -> dict:
    """What the page's own speech gate says about a WAV."""
    node = shutil.which("node")
    if not node:
        return {"error": "node is not on PATH"}
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(HARNESS)
        script = f.name
    try:
        out = subprocess.run([node, script, str(page), str(wav)],
                             capture_output=True, text=True, timeout=300)
        if out.returncode != 0:
            return {"error": out.stderr.strip()[:300]}
        return json.loads(out.stdout)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


def silero_gate(wav: Path) -> dict:
    """What the PIPELINE's gate says about the same WAV.

    The reference computes `speech_frac` with `src.vad.SileroGate` at
    threshold 0.5 (corpus_capture_win.py:501-504), so that is the number the
    corpus gate's 0.15 / 0.05 thresholds are defined against.
    """
    model = os.environ.get("SILERO_VAD_MODEL")
    if not model:
        for cand in (ROOT / "models" / "silero_vad.onnx",):
            if cand.is_file():
                model = str(cand)
                break
    if not model:
        base = ROOT.name.split("-")[0]
        try:
            for sib in ROOT.parent.iterdir():
                cand = sib / "models" / "silero_vad.onnx"
                if sib.is_dir() and sib.name.startswith(base) and cand.is_file():
                    model = str(cand)
                    break
        except OSError:
            pass
    if not model or not Path(model).is_file():
        return {"error": "the Silero weights were not found"}
    try:
        os.environ["SILERO_VAD_MODEL"] = model
        sys.path.insert(0, str(ROOT))
        from src import wavio
        from src.vad import SileroGate
        audio = wavio.read_wav(str(wav))
        # A FRESH gate per file: it carries an LSTM state across frames.
        probs = SileroGate().push(list(audio.samples))
        if not probs:
            return {"error": "the gate returned no probabilities"}
        frac = sum(1 for v in probs if v >= 0.5) / len(probs)
        return {"frac": frac, "n_frames": len(probs),
                "source_sample_rate": audio.source_sample_rate}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wav", nargs="+", help="the WAV(s) to measure")
    ap.add_argument("--page", default=str(DEFAULT_PAGE),
                    help="the shipped index.html holding the speech gate")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="write the measurements to this path")
    a = ap.parse_args(argv)

    page = Path(a.page)
    if not page.is_file():
        print(f"page not found: {page}", file=sys.stderr)
        return 2

    results = []
    for w in a.wav:
        p = Path(w)
        entry = {"wav": str(p),
                 "page_gate": page_gate(page, p),
                 "pipeline_gate": silero_gate(p)}
        # The recorded value, if the take's metadata is beside the WAV.
        meta = p.with_suffix(".json")
        if meta.is_file():
            try:
                entry["recorded_speech_frac"] = json.loads(
                    meta.read_text()).get("speech_frac")
            except Exception:  # noqa: BLE001
                pass
        results.append(entry)
        pg = entry["page_gate"].get("frac")
        sg = entry["pipeline_gate"].get("frac")
        rec = entry.get("recorded_speech_frac")
        print(f"{p.name}")
        print(f"   the page's gate (speechStats) : "
              f"{pg if pg is None else round(pg, 4)}")
        print(f"   the pipeline's gate (Silero) : "
              f"{sg if sg is None else round(sg, 4)}")
        print(f"   recorded in the take's meta  : {rec}")
        if pg is not None and sg is not None and abs(pg - sg) > 0.25:
            print("   -> the two gates disagree by more than 0.25")
    if a.json_path:
        out = Path(a.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
