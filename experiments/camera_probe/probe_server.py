#!/usr/bin/env python3
"""
probe_server.py -- the server half of the camera-exposure probe.

WHAT THIS IS.

Casey's front camera comes up REALLY dark in Chrome on this machine. The
capture page's own live luma chip reads it, and the reference point is the
old Windows-stack captures: ~106-110 mean luma with Casey in frame. The
question is which constraint set the PRODUCT page should ship, and the
answer has to be measured rather than reasoned about, because two plausible
fixes (pin the frame rate harder; stop pinning it) predict opposite
outcomes.

This server hosts one page that offers the candidate constraint sets, shows
the live luma each produces, and records the numbers.

WHY IT IS STDLIB ONLY, AND WHY IT BINDS 127.0.0.1.

The brief is explicit: a local probe, no pip installs, nothing leaves the
machine. The server only serves a page and appends a line to a JSONL file.
It is the same shape as `browser_capture/server.py`, deliberately, so that
nothing about the probe depends on anything the product does not already
depend on.

THE RESULTS FILE.

Every LOG PRESET click appends one JSON object to `results.jsonl`, beside
this script, in this worktree. One line per measurement, so the file can be
appended, diffed and read with `jsonlines` without a parser. The fields are
the ones the brief asks for -- preset, settings, a capabilities summary, and
the luma mean/min/max -- plus the manual slider state, so a hand-found
winning combination is recorded as numbers rather than as a description.

USAGE:

    python3 experiments/camera_probe/probe_server.py            # port 8766
    python3 experiments/camera_probe/probe_server.py --port 8767

Then open http://localhost:8766 in Chrome on the Windows side.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "probe.html"
RESULTS = HERE / "results.jsonl"

HOST = "127.0.0.1"
DEFAULT_PORT = 8766

# A record larger than this is a mistake or an attack, not a measurement.
MAX_BODY = 4 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))
        sys.stderr.flush()

    # -- helpers ---------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, indent=2).encode("utf-8"),
                   "application/json")

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            if not PAGE.is_file():
                self._json(500, {"error": f"{PAGE} is missing"})
                return
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/health":
            self._json(200, {"ok": True, "results": str(RESULTS),
                             "results_exists": RESULTS.is_file()})
            return
        if path == "/results":
            try:
                text = RESULTS.read_text() if RESULTS.is_file() else ""
            except OSError as exc:
                self._json(500, {"error": str(exc)})
                return
            self._send(200, text.encode("utf-8"), "application/x-ndjson")
            return
        self._json(404, {"error": f"no route {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/log":
            self._json(404, {"error": f"no route {path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "bad Content-Length"})
            return
        if length <= 0:
            self._json(400, {"error": "empty body"})
            return
        if length > MAX_BODY:
            self._json(413, {"error": f"body over {MAX_BODY} bytes"})
            return
        raw = self.rfile.read(length)
        try:
            record = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"error": f"not JSON: {exc}"})
            return
        if not isinstance(record, dict):
            self._json(400, {"error": "the record must be a JSON object"})
            return
        record.setdefault("received_at", time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                       time.gmtime()))
        try:
            RESULTS.parent.mkdir(parents=True, exist_ok=True)
            with RESULTS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as exc:
            self._json(500, {"error": f"could not append to {RESULTS}: {exc}"})
            return
        n = sum(1 for _ in RESULTS.open("r", encoding="utf-8")) \
            if RESULTS.is_file() else 0
        print(f"logged {record.get('preset', '?')} -> {RESULTS.name} "
              f"({n} records)", flush=True)
        self._json(200, {"ok": True, "records": n, "path": str(RESULTS)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default=HOST,
                    help=f"default {HOST}: localhost only")
    a = ap.parse_args(argv)
    if not PAGE.is_file():
        print(f"probe page missing: {PAGE}", file=sys.stderr)
        return 2
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"camera probe on http://{a.host}:{a.port}  "
          f"results -> {RESULTS}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
