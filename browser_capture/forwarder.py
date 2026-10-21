#!/usr/bin/env python3
"""
forwarder.py -- the LIVE FORWARDER sidecar for the Muse+Jetson pair.

WHY (relayed 2026-10-21): while AUTO runs, the Jetson seat is building
an analyzer that wants the newest luma batch and the latest heartbeat,
pushed every few seconds to jetson-lan:~/work/pair/live/. This sidecar
does exactly that, and ONLY that -- it reads public state over the
liaison's HTTP API and ships it over the existing ssh alias. It is a
SEPARATE PROCESS on purpose: additive only, zero coupling. If it dies,
hangs, or the Jetson is off, the capture path never notices; it just
logs and retries on the next cycle.

WHAT IT SHIPS (atomic rename on the far side, so the analyzer never
reads a torn file):
    ~/work/pair/live/latest_luma.raw   the newest ~90-frame luma batch
    ~/work/pair/live/latest_luma.json  {n, offsets_ms, ts, frame_shape}
    ~/work/pair/live/heartbeat.json    the newest page heartbeat

It pushes only while a capture page is live (a heartbeat fresher than
10 s in state armed/recording/paused).

USAGE (installed as browser-capture-forwarder.service):
    python3 browser_capture/forwarder.py            # defaults
    python3 browser_capture/forwarder.py --interval 3
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8765"
DEST = "jetson-lan"
DEST_DIR = "~/work/pair/live"
HERE = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"[forwarder] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def get_json(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return json.loads(r.read().decode())


def ssh_write(dest: str, name: str, data: bytes) -> bool:
    """Write to dest:DEST_DIR/name via temp+mv -- the reader either sees
    the whole old file or the whole new one, never a torn one."""
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", dest,
           f"mkdir -p {DEST_DIR} && cat > {DEST_DIR}/.{name}.tmp && "
           f"mv {DEST_DIR}/.{name}.tmp {DEST_DIR}/{name}"]
    try:
        p = subprocess.run(cmd, input=data, capture_output=True, timeout=15)
        return p.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def latest_heartbeat(events: list):
    for ev in reversed(events):
        if isinstance(ev, dict) and ev.get("kind") == "heartbeat":
            return ev
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--dest", default=DEST)
    a = ap.parse_args()

    log(f"watching {a.base}, shipping to {a.dest}:{DEST_DIR} every "
        f"{a.interval:g}s")
    last_ship_ok = True
    while True:
        time.sleep(a.interval)
        try:
            state = get_json(a.base, "/live-state")
            hb = latest_heartbeat(state.get("events", []))
            if not hb or hb.get("state") not in ("armed", "recording",
                                                 "paused"):
                continue   # no live page: nothing to forward
            received = hb.get("received_at", "")
            age = time.time() - (time.mktime(
                time.strptime(received[:19], "%Y-%m-%dT%H:%M:%S"))
                if received else 0)
            if age > 10:
                continue
            batch = get_json(a.base, "/luma-latest")
            if not batch.get("n"):
                continue
            luma = base64.b64decode(batch.get("luma_b64", ""))
            meta = json.dumps({
                "n": batch["n"], "offsets_ms": batch.get("offsets", []),
                "ts": batch.get("ts"), "frame_shape": [96, 160],
                "frame_bytes": 15360, "format": "uint8 BT.601 luma, "
                "row-major, frames concatenated",
                "heartbeat_state": hb.get("state"),
            }).encode()
            ok = ssh_write(a.dest, "latest_luma.raw", luma) and \
                ssh_write(a.dest, "latest_luma.json", meta) and \
                ssh_write(a.dest, "heartbeat.json", json.dumps(hb).encode())
            if not ok and last_ship_ok:
                log(f"ship to {a.dest} refused (Jetson off? retrying)")
            last_ship_ok = ok
        except Exception as exc:  # noqa: BLE001 -- the sidecar never dies
            if last_ship_ok:
                log(f"cycle failed ({exc}); retrying")
            last_ship_ok = False
    return 0


if __name__ == "__main__":
    sys.exit(main())
