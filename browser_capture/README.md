# browser_capture — the corpus recording, as a browser page

The same 16 takes `CAPTURE-NEXT.md` asks for, the same files the Windows
script writes, but driven from a page you can see yourself in.

## Start it (one command, from the repo root, inside WSL)

The server runs as a **systemd user service** — it starts with WSL and
restarts itself if it dies, so there is no terminal to accidentally
Ctrl-C (that happened once; never again). The bench command is now a
VIEW of the running server, never the server itself:

```
python3 browser_capture/bench.py        # status + live log tail
```

The one status command, if that is all you need:

```
systemctl --user status browser-capture
```

Then open **http://localhost:8765** in your Windows browser.
(If the page doesn't load, WSL's localhost forwarding is off — restarting
WSL usually brings it back. Nothing left this machine: the server binds
127.0.0.1 only.)

Takes are written to `/mnt/c/proart_capture/corpus` — the **same folder the
Windows script writes to** — so the gate and every existing tool see them
unchanged. Use `--out somewhere` to record somewhere else, `--port` to move
the page.

## What you click

1. The page asks for **camera + microphone** on load — allow both. You
   should see yourself, large. The MIRROR button flips the preview to your
   liking; it never flips the recorded data.
2. Click **START PLAN**. From there the page drives: a 5-second countdown,
   then a big prompt for each take — *FACE THE CAMERA — RECITE THE POEM*,
   *TURN 90 DEGREES — STAY SILENT*, and so on — with the poem on screen for
   the recite takes (glance at it, don't read it), a live timer, and a live
   SPEAKING / SILENT indicator fed by your actual microphone.
3. Each take **uploads the moment it ends** and the page shows the same
   numbers the script prints — achieved frame rate, mic rate, drift, speech
   percent — with `VOICE_MISSING` / `NOT_SILENT` flags if the label didn't
   match what happened. Flagged takes are kept, exactly as the script keeps
   them.
4. After each take there's a small **"what I actually did"** note box —
   optional, type while you remember it, click *Save note*.
5. **PAUSE** stops after the current take. **SKIP TAKE** marks one skipped.
   **REDO THIS TAKE** re-records. Click any take in the list to record it
   out of order. Re-records never overwrite: every take gets its own
   timestamp, same as the script.

Keep the tab in the foreground while recording — background tabs throttle
the timers the capture runs on.

## What lands on disk (per take)

Next to each other in the corpus folder, all in the script's exact format:

| file | what it is |
|---|---|
| `A1_<stamp>.npz` | `maps`, `maps_raw`, `luma` + both scales |
| `A1_<stamp>.wav` | 16-bit mono PCM at the rate the mic really ran |
| `A1_<stamp>_timing.json` | per-frame timestamps, achieved rates |
| `A1_<stamp>.json` | label, speech fraction, drift, provenance |
| `A1_<stamp>_full.webm` | full-resolution video of the take, with audio |
| `rich/A1_<stamp>_rich.json` | instruction shown, wall clocks, devices, your note |
| `rich/session.json` | every take, its files, labels and flags |

## After recording

Run the gate, same as after the Windows script:

```
python experiments/corpus_integrity.py
```

## Proving it works without a camera

```
python3 browser_capture/verify.py
/home/proart/.local/audio-venv/bin/python browser_capture/verify.py
```

The second one adds the numpy round-trip and runs the gate itself on a
synthetic corpus (145 checks, all green as of 2026-10-20 — see DONE.md).
It never touches the real corpus folder.
