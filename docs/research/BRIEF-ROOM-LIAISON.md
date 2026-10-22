# BRIEF - Room Liaison v1 (Casey via Muse, 2026-10-20)

## The upgrade Casey ordered
The local host (browser_capture server + page) stops being a web page and
becomes the LIAISON IN THE ROOM for Muse: a resident local agent that
(a) syncs with Muse both ways - it can send Muse messages, Muse can
send it messages; (b) exposes APIs ON THE LAPTOP meant for agents (the
local coder seats and Casey-at-the-laptop call it directly); (c) is
MAINTAINED by Muse (Muse is its maintenance agent, not its occupant;
Casey owns it).

## Build v1 (browser_capture/server.py, ZCode; after the audio sick-lane)
1. AGENT API (JSON, localhost):
   - GET /api/status: page state, open segment, last heartbeat values,
     sessions saved, uptime.
   - GET /api/sessions: every free session with meta PLUS measured audio
     stats (peak abs sample, nonzero count, RMS) - artifact health is
     part of the API, computed from the files, never asserted.
   - POST /api/inbox {from, text} appends liaison_inbox.jsonl;
     GET /api/inbox reads it. Local agents message the room through this.
2. OUTBOX + SELF-CHECKS: the liaison appends events to liaison_outbox.jsonl
   and ships each to a rendezvous dir on the Oracle (the tunnels already
   terminate there) where the rounds of Muse read them. Event classes:
   session_saved, segment_closed, AND health alerts it computes itself -
   first one: any saved WAV whose measured peak is 0 raises SILENT_AUDIO
   immediately. The bug class from tonight becomes a self-report.
3. Muse-side contract (document, do not build yet): Muse polls the
   rendezvous on rounds; replies from Muse land via /api/inbox. State
   the latency honestly in the doc: minutes-scale, maintenance-grade.
Keep the page exactly as usable as today; the API is additive. Verify:
curl every endpoint, POST+GET an inbox round-trip, and a forced
silent-WAV event reaching the outbox. Commit, push, DONE.md it.
