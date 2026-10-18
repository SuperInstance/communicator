"""
smf — a hand-rolled Standard MIDI File writer and parser.

mido is not installable on this machine (checked; and the task allows a
hand-rolled writer). The SMF spec needed for the five measured tracks is
small:

    header chunk   "MThd" + 6 bytes: format(2) ntrks(2) division(2)
    track chunk    "MTrk" + length + events as (vlq-delta, event)
    channel event  status byte + data bytes (note_on 0x9n, note_off 0x8n,
                   control_change 0xBn, program_change 0xCn, pitchwheel 0xEn)
    meta event     0xFF type vlq-len data (track_name 0x03, set_tempo 0x51
                   with a 3-byte microseconds-per-quarter, end_of_track 0x2F)

Delta times are variable-length quantities (big-endian, 7 bits per byte,
max 4 bytes). Pitch bend carries a 14-bit value offset by 8192, sent
LSB-then-MSB. The writer does not use running status (legal, and simpler);
the parser DOES accept it, so files from other writers decode too.

Everything here works on ints and bytes — no floats in the wire format
except the tempo's integer microseconds-per-quarter derived by the caller.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

TICKS_PER_BEAT = 480  # slackwater's MultiTrackEncoder default

# status bytes on channel 0
NOTE_OFF = 0x80
NOTE_ON = 0x90
CONTROL_CHANGE = 0xB0
PROGRAM_CHANGE = 0xC0
PITCHWHEEL = 0xE0

# meta types
META_TRACK_NAME = 0x03
META_SET_TEMPO = 0x51
META_END_OF_TRACK = 0x2F


class SmfError(ValueError):
    """Malformed MIDI data."""


@dataclass
class MidiEvent:
    """One event at an absolute tick.

    kind is one of: note_on, note_off, control_change, program_change,
    pitchwheel, track_name, set_tempo, end_of_track.
    """
    tick: int
    kind: str
    data: dict = field(default_factory=dict)

    def key(self) -> tuple:
        return (self.tick, self.kind, tuple(sorted(self.data.items())))


@dataclass
class MidiTrackData:
    name: str
    events: list[MidiEvent]


def vlq(n: int) -> bytes:
    """Encode a non-negative delta time as a variable-length quantity."""
    if n < 0:
        raise SmfError(f"negative delta time: {n}")
    groups = [n & 0x7F]
    n >>= 7
    while n:
        groups.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(groups))


def write_track_chunk(events: list[MidiEvent]) -> bytes:
    """Encode absolute-tick events into one MTrk chunk.

    Events are sorted (tick, fixed event-order key) before delta encoding;
    callers may pass events in any order. end_of_track is appended if absent.
    """
    ordered = sorted(events, key=_event_sort_key)
    body = bytearray()
    prev_tick = 0
    for ev in ordered:
        body += vlq(ev.tick - prev_tick)
        body += _encode_event(ev)
        prev_tick = ev.tick
    if not ordered or ordered[-1].kind != "end_of_track":
        body += vlq(0) + bytes([0xFF, META_END_OF_TRACK, 0x00])
    return b"MTrk" + struct.pack(">I", len(body)) + bytes(body)


def _event_sort_key(ev: MidiEvent) -> tuple:
    # end_of_track strictly last within the same tick
    last = 1 if ev.kind == "end_of_track" else 0
    return (ev.tick, last)


def _encode_event(ev: MidiEvent) -> bytes:
    if ev.kind == "note_on":
        return bytes([NOTE_ON, _u7(ev.data["note"]), _u7(ev.data["velocity"])])
    if ev.kind == "note_off":
        return bytes([NOTE_OFF, _u7(ev.data["note"]), _u7(ev.data.get("velocity", 0))])
    if ev.kind == "control_change":
        return bytes([CONTROL_CHANGE, _u7(ev.data["control"]),
                      _u7(ev.data["value"])])
    if ev.kind == "program_change":
        return bytes([PROGRAM_CHANGE, _u7(ev.data["program"])])
    if ev.kind == "pitchwheel":
        raw = int(ev.data["value"]) + 8192
        if not 0 <= raw <= 16383:
            raise SmfError(f"pitch bend out of range: {ev.data['value']}")
        return bytes([PITCHWHEEL, raw & 0x7F, (raw >> 7) & 0x7F])
    if ev.kind == "track_name":
        name = ev.data["name"].encode("ascii")
        return bytes([0xFF, META_TRACK_NAME]) + vlq(len(name)) + name
    if ev.kind == "set_tempo":
        tempo = int(ev.data["tempo"])  # microseconds per quarter note
        if not 0 < tempo <= 0xFFFFFF:
            raise SmfError(f"tempo out of range: {tempo}")
        return bytes([0xFF, META_SET_TEMPO, 0x03,
                      (tempo >> 16) & 0xFF, (tempo >> 8) & 0xFF, tempo & 0xFF])
    if ev.kind == "end_of_track":
        return bytes([0xFF, META_END_OF_TRACK, 0x00])
    raise SmfError(f"unknown event kind: {ev.kind!r}")


def _u7(v: int) -> int:
    v = int(v)
    if not 0 <= v <= 127:
        raise SmfError(f"data byte out of 0..127: {v}")
    return v


def write_smf(tracks: list[MidiTrackData],
              ticks_per_beat: int = TICKS_PER_BEAT) -> bytes:
    """Assemble a format-1 SMF from named tracks."""
    if not 1 <= ticks_per_beat <= 0x7FFF:
        raise SmfError(f"ticks_per_beat out of range: {ticks_per_beat}")
    chunks = b"".join(write_track_chunk(t.events) for t in tracks)
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), ticks_per_beat)
    return header + chunks


# -- parsing ----------------------------------------------------------------

def parse_smf(data: bytes) -> tuple[int, list[MidiTrackData]]:
    """Parse a format 0/1 SMF back into named tracks with absolute ticks.

    Used by the tests to prove the writer's output is valid by the same
    rules a reader applies. Handles running status. Raises SmfError on the
    first structural violation instead of guessing.
    """
    if len(data) < 14 or data[:4] != b"MThd":
        raise SmfError("not an SMF: missing MThd")
    header_len, fmt, ntrks, division = struct.unpack(">IHHH", data[4:14])
    if header_len != 6:
        raise SmfError(f"unexpected header length {header_len}")
    if division & 0x8000:
        raise SmfError("SMPTE timing not supported")
    pos = 8 + header_len
    tracks: list[MidiTrackData] = []
    while pos < len(data) and len(tracks) < ntrks:
        if data[pos:pos + 4] != b"MTrk":
            raise SmfError(f"expected MTrk at byte {pos}")
        chunk_len = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        chunk = data[pos + 8:pos + 8 + chunk_len]
        if len(chunk) != chunk_len:
            raise SmfError("truncated track chunk")
        tracks.append(_parse_track(chunk))
        pos += 8 + chunk_len
    if len(tracks) != ntrks:
        raise SmfError(f"header promises {ntrks} tracks, found {len(tracks)}")
    return division, tracks


def _parse_track(chunk: bytes) -> MidiTrackData:
    events: list[MidiEvent] = []
    tick = 0
    pos = 0
    name = ""
    running_status: int | None = None
    n = len(chunk)
    while pos < n:
        delta, pos = _read_vlq(chunk, pos)
        tick += delta
        if pos >= n:
            raise SmfError("event truncated after delta")
        b = chunk[pos]
        if b == 0xFF:
            pos += 1
            meta_type = chunk[pos]
            pos += 1
            meta_len, pos = _read_vlq(chunk, pos)
            payload = chunk[pos:pos + meta_len]
            if len(payload) != meta_len:
                raise SmfError("truncated meta event")
            pos += meta_len
            running_status = None
            if meta_type == META_END_OF_TRACK:
                events.append(MidiEvent(tick, "end_of_track"))
                break
            if meta_type == META_TRACK_NAME:
                name = payload.decode("ascii")
                events.append(MidiEvent(tick, "track_name", {"name": name}))
            elif meta_type == META_SET_TEMPO:
                if meta_len != 3:
                    raise SmfError("set_tempo length != 3")
                tempo = (payload[0] << 16) | (payload[1] << 8) | payload[2]
                events.append(MidiEvent(tick, "set_tempo", {"tempo": tempo}))
            else:
                events.append(MidiEvent(tick, f"meta_0x{meta_type:02x}",
                                        {"payload": bytes(payload)}))
        else:
            if b & 0x80:
                status = b
                pos += 1
                running_status = status if status < 0xF0 else None
            else:
                if running_status is None:
                    raise SmfError("data byte with no running status")
                status = running_status
            kind, nbytes = {
                NOTE_OFF: ("note_off", 2),
                NOTE_ON: ("note_on", 2),
                CONTROL_CHANGE: ("control_change", 2),
                PROGRAM_CHANGE: ("program_change", 1),
                PITCHWHEEL: ("pitchwheel", 2),
            }.get(status & 0xF0, (None, 0))
            if kind is None:
                raise SmfError(f"unsupported status byte 0x{status:02x}")
            args = chunk[pos:pos + nbytes]
            if len(args) != nbytes:
                raise SmfError("truncated channel event")
            pos += nbytes
            ch = status & 0x0F
            if kind == "note_on" and args[1] == 0:
                kind = "note_off"  # canonical: note_on vel 0 IS note_off
                data = {"note": args[0], "velocity": 0}
            elif kind == "note_on":
                data = {"note": args[0], "velocity": args[1]}
            elif kind == "note_off":
                data = {"note": args[0], "velocity": args[1]}
            elif kind == "control_change":
                data = {"control": args[0], "value": args[1]}
            elif kind == "program_change":
                data = {"program": args[0]}
            else:  # pitchwheel
                raw = args[0] | (args[1] << 7)
                data = {"value": raw - 8192}
            data["channel"] = ch
            events.append(MidiEvent(tick, kind, data))
    return MidiTrackData(name=name, events=events)


def _read_vlq(buf: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if pos >= len(buf):
            raise SmfError("truncated VLQ")
        b = buf[pos]
        pos += 1
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            return value, pos
    raise SmfError("VLQ longer than 4 bytes")
