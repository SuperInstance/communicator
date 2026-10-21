#!/usr/bin/env python3
"""
bench.py -- the BENCH VIEW for the browser_capture service.

WHY THIS EXISTS (live incident, 2026-10-21): the bench used to BE the
server -- `python3 browser_capture/server.py` in a terminal -- and one
accidental Ctrl-C killed localhost:8765 mid-session. The server is now a
systemd USER service (browser-capture.service, starts with WSL, restarts
on failure) and this script is only a WINDOW onto it: service status,
the live /info answer, and a tail of the server log. Ctrl-C here leaves
the view; the service keeps serving. Nothing this script does can stop
the server.

USAGE (from anywhere):

    python3 browser_capture/bench.py

The one status command, if that is all you need:

    systemctl --user status browser-capture
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

SERVICE = "browser-capture.service"
URL = "http://127.0.0.1:8765/info"
LOG = Path(__file__).resolve().parent / "server.log"


def main() -> int:
    print("=" * 62)
    print(f"browser_capture BENCH VIEW -- the service serves; this only watches")
    print("=" * 62)

    r = subprocess.run(
        ["systemctl", "--user", "status", SERVICE, "--no-pager", "-l"],
        capture_output=True, text=True)
    print(r.stdout.rstrip())
    if r.stderr.strip():
        print(r.stderr.rstrip())

    try:
        with urllib.request.urlopen(URL, timeout=3) as resp:
            print(f"\nserver answers /info: {resp.read().decode().strip()}")
    except Exception as exc:  # noqa: BLE001 -- the view must never crash
        print(f"\nserver NOT answering on {URL}: {exc}")
        print("fix with:  systemctl --user restart browser-capture")

    print("\n-- last log lines (following server.log; Ctrl-C exits the "
          "VIEW only) --")
    if not LOG.exists():
        LOG.touch()
    # tail -F, not exec: keep the Ctrl-C semantics ours (view, not server)
    sys.exit(subprocess.call(["tail", "-n", "40", "-F", str(LOG)]))


if __name__ == "__main__":
    main()
