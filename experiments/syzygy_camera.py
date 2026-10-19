"""
syzygy_camera.py -- capture camera + microphone together and render the camera
through Syzygy, so the two modalities meet in a shared symbolic substrate.

WHY THIS EXISTS.

Everything measured so far says the turn question cannot be answered from audio
alone. Silence duration gives AUC 0.529 over 1396 pauses and every threshold
loses to the trivial baseline (0.6225); pitch separates two voice CLASSES but
not speakers within one (0.5 and 0.3 sigma apart). At the moment of decision,
END and RESUME are byte-identical audio. The information is not in the audio.

`docs/research/SPECTROGRAM-TURN.md` then closed the obvious escape: the only
image-capable model available locally answers a constant 'different' to every
spectrogram AND to a blank rectangle, so "render it as a picture and ask a
model" does not work here.

Syzygy is the remaining option, and it is a different kind of thing entirely. It
converts a camera frame into Braille characters, edge glyphs, a tone ramp and a
16-point spectrum in ONE pass, using integer arithmetic only, with no
allocation and a byte-deterministic output. Verified on this machine:

    219 checks, 0 failures; golden FNV-1a 0x6dbdd1a8; full-output 0x463de14b
    a 160x96 frame -> 80x24 cells, 0.1 ms in a browser, 1.5-4.8 KB on an MCU

So it needs no model, no torch, no GPU, and it is ~0.1% of a 60 Hz frame budget.
It also measures and does not judge -- which is the same division of labour
this codebase already uses between src/watcher.py and src/turn.py.

WHY THE THREECONDITION DESIGN IS THE ONLY ONE THAT PROVES ANYTHING.

"Is someone facing the camera" is not the claim. The claim is that the camera
ADDS information the microphone does not have, so the capture is built around
three conditions:

    A  speaking, facing the camera
    B  speaking, head turned away   <- identical audio, different vision
    C  silent, facing

A and B differ ONLY in vision. If audio separates them, the camera is
redundant. If vision separates them and audio does not, the join is real.
Condition C is what stops "loud" from masquerading as "attending", since it is
facing-but-quiet.

WHAT THIS FILE DOES, AND WHAT IT DELIBERATELY DOES NOT.

It captures, renders, and records. It does NOT decide anything: it emits the
glyph field, the spectrum peak, and the audio, plus a manifest naming the
condition. Judging is a separate step, because a harness that both measures and
concludes is how a rigged result gets reported as a clean one -- the mistake
this project has now made and caught twice.

THE HARDWARE IS NOT IN THIS REPO'S CONTROL.

WSL exposes no audio device and no /dev/video*, so capture runs on the Windows
side and lands on disk for WSL to read. Measured here:

    /dev/snd contains only a timer
    PulseAudio exposes only RDPSink.monitor and RDPSource (an RDP bridge)
    Windows sees "Microphone Array (Realtek(R) Audio)" and "ASUS FHD webcam"

Run `experiments/syzygy_capture_win.py` on the Windows python, then this one.

Reproduce:
    python experiments/syzygy_camera.py --seconds 40 --out examples/realfacing/
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".scratch"
SYZYGY_URL = "https://github.com/SuperInstance/Syzygy.git"
FRAME_W, FRAME_H = 160, 96          # Syzygy's own benchmark size
CELLS_W, CELLS_H = FRAME_W // 2, FRAME_H // 4

RENDER_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include "syz_fused.h"

/* argv: <nv12 path> <w> <h> ; prints one JSON-ish line per frame:
   <cols> <rows> <peak_bin> then hex masks and tone chars. */
int main(int argc, char **argv) {
    const char *path = argv[1];
    uint32_t W = (uint32_t)atoi(argv[2]), H = (uint32_t)atoi(argv[3]);
    size_t fb = (size_t)W * H * 3 / 2;
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); return 1; }
    uint8_t *all = (uint8_t *)malloc(fb * 4096);
    size_t got = fread(all, 1, fb * 4096, f);
    fclose(f);
    size_t nframes = got / fb;
    fprintf(stderr, "frames=%zu\n", nframes);
    static uint8_t pool[1 << 20];
    static uint8_t mask[4096], tone[4096];
    static uint32_t glyph[4096];
    for (size_t i = 0; i < nframes; i++) {
        SyzArena a; syz_arena_init(&a, pool, sizeof pool);
        SyzNv12 nv = { .y = all + i * fb, .uv = all + i * fb + (size_t)W * H,
                       .w = W, .h = H, .y_stride = W, .uv_stride = W };
        SyzFusedParams p = { .braille_thresh = 100, .edge_thresh2 = 4000, .fft_row = 1 };
        SyzFusedOut o = { .cap = 4096, .mask = mask, .glyph = glyph, .tone = tone };
        int rc = syz_fused(&a, &nv, &p, &o);
        if (rc) { fprintf(stderr, "rc=%d\n", rc); return 1; }
        printf("F %u %u %u\n", o.cols, o.rows, o.peak_bin);
        for (uint32_t k = 0; k < o.cols * o.rows; k++) printf("%02x", mask[k]);
        printf("\n");
        for (uint32_t k = 0; k < o.cols * o.rows; k++) printf("%02x", tone[k]);
        printf("\n");
    }
    return 0;
}
"""


def ensure_syzygy() -> Path:
    """Clone and build Syzygy into .scratch, verifying its own suite first.

    Fetched, not vendored -- the repo does not carry third-party code, and
    Syzygy is MIT with a golden hash, so verification is cheap and the receipt
    is the hash itself rather than a claim in a README.
    """
    d = SCRATCH / "Syzygy"
    if not (d / "include" / "syz_fused.h").is_file():
        SCRATCH.mkdir(exist_ok=True)
        if not d.exists():
            subprocess.run(["git", "clone", "--depth", "1", SYZYGY_URL, str(d)],
                           check=True)
    binp = d / "syz_render"
    src = d / "syz_render.c"
    if not binp.is_file() or src.read_text() != RENDER_C:
        src.write_text(RENDER_C)
        subprocess.run(["cc", "-O2", f"-I{d/'include'}", str(src), "-o", str(binp)],
                       check=True)
    return d


def verify(d: Path) -> str:
    """Run Syzygy's own suite and pull the golden hash out of it."""
    r = subprocess.run(["sh", "tools/suite-total.sh"], cwd=d,
                       capture_output=True, text=True, timeout=600)
    h = ""
    for line in r.stdout.splitlines():
        if "golden fnv1a" in line:
            h = line.split("=", 1)[-1].strip()
    if r.returncode != 0:
        raise SystemExit("Syzygy's own suite FAILED -- refusing to use a kernel "
                         "whose tests do not pass here.\n" + r.stdout[-2000:])
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True,
                    help="path to the .nv12 written by the Windows capture")
    ap.add_argument("--audio", help="path to the matching .wav")
    ap.add_argument("--meta", help="JSON sidecar naming the condition per frame")
    ap.add_argument("--out", default=str(ROOT / "examples/realfacing"))
    ap.add_argument("--keep-hash", action="store_true")
    a = ap.parse_args()

    d = ensure_syzygy()
    golden = verify(d)
    print(f"Syzygy suite passed; golden {golden}")

    out = Path(a.out)
    (out / "glyphs").mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(d / "syz_render"), a.capture, str(FRAME_W), str(FRAME_H)],
                       capture_output=True, text=True, check=True)
    lines = r.stdout.splitlines()
    cols = rows = peak = 0
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("F "):
            i += 1
            continue
        _, cs, rs, pk = lines[i].split()
        cols, rows, peak = int(cs), int(rs), int(pk)
        frames.append({"mask": lines[i + 1], "tone": lines[i + 2], "peak_bin": peak})
        i += 3

    meta = {}
    if a.meta and Path(a.meta).is_file():
        meta = json.loads(Path(a.meta).read_text())

    manifest = {
        "note": ("Syzygy glyph field per camera frame, alongside the matching "
                 "audio. This file MEASURES and does not judge: no verdict is "
                 "recorded here by design."),
        "syzygy_golden": golden,
        "frame_px": [FRAME_W, FRAME_H],
        "cells": [cols, rows],
        "n_frames": len(frames),
        "capture_nv12": str(a.capture),
        "audio": a.audio,
        "conditions": meta,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    for idx, fr in enumerate(frames):
        (out / "glyphs" / f"{idx:05d}.json").write_text(json.dumps(fr))

    print(f"cells {cols}x{rows}  frames {len(frames)}  -> {out}")
    print(f"tone field of frame 0:")
    for r_ in range(rows):
        print("   ", frames[0]["tone"][r_ * cols * 2:(r_ + 1) * cols * 2])


if __name__ == "__main__":
    main()