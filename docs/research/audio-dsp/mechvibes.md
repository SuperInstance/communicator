# mechvibes — a keyboard-typing sound app, not the Rust synth we were pointed at

Investigated 2026-10-18.

## First: the assigned repo is not the Rust synth

`hainguyents13/mechvibes` is an **Electron/JavaScript desktop app**, not Rust:

```
git ls-files | grep -c '\.rs$'   →  0
```

`package.json` is an Electron manifest (`"main": "src/main.js"`, `electron-builder`, a global
`iohook` config, scripts `electron .`). The tracked files are 19 JS, 66 mp3, 34 wav, 20 json, 11
ogg — it is a player with a bundled sample library. The 19 `src/audio/<switch>/` directories are
keyboard sound packs.

**The Rust is in a separate repo:** `hainguyents13/mechvibes-dx` — Rust, **MIT**, 332★,
last push 2026-10-15, described as "an upgraded version of Mechvibes". I cloned that too and did
the real analysis there.

The browser IDE `maximecb/noisecraft` is the third repo in this family; see `noisecraft.md`.

## mechvibes-dx (the Rust part) — real numbers

| fact | value |
|---|---|
| license | **MIT** (Hai Nguyen) |
| HEAD | `dca17c1` 2026-10-15 — actively developed |
| `.rs` files | 132, **33,295 lines** total |
| of which vendored `patches/rdev/` | a forked global-input-hook library, ~4k lines |
| real application code | `src/`, ~29,344 lines across ~100 files |
| audio engine | `src/libs/audio/` — **2,083 lines** |
| CI | `.github/workflows/ci.yml`, `AppImageBuilder.yml` |

Dependencies (`Cargo.toml:28-32`): `rodio 0.20.1`, `cpal 0.15`, `symphonia 0.5`, `serde`.

## The audio engine is a voice-pool mixer, not a modular synth graph

`src/libs/audio/engine.rs` (767 lines). The brief asked about "its graph model". The honest
answer: **there isn't one in the synthesis sense.** The structure is:

```rust
pub enum AudioCommand { ... }        // engine.rs:27
pub enum UiEvent { ... }             // engine.rs:46
pub struct AudioEngineHandle { ... } // engine.rs:59
pub fn spawn_engine(...)             // engine.rs:84
struct EngineState { ... }           // engine.rs:174
```

A command channel between a UI thread and an audio thread, plus a fixed-size pool of playback
sinks. Key functions:

- `handle_key_event` (`:213`), `handle_mouse_event` (`:233`)
- `debounce_press` (`:303`) — per-key debouncing of auto-repeat
- `lookup_timing` (`:315`) — reads per-key `[offset, duration]` timing windows
- `play_segment` (`:331`) — trims and plays a sample
- `apply_fade` (`:382`)
- `manage_active_sinks` (`:420`) — voice cap with click-free eviction
- `resample_if_needed` (`:438`)

There is no oscillator, no filter graph, no modular patching. The "sounds" are 100 pre-recorded
WAV/MP3 samples; `src/audio/<pack>/config.json` is a JSON map from keycode to filename with
`{0-4}` wildcard fan-out and separate `press`/`release` (down/up) entries.

The nearest thing to a graph model in the family is **noisecraft's** node graph (JSON `nodes` with
`ins`/`outNames`/`params`), not mechvibes'. See `noisecraft.md`.

## The two stealable mechanisms (MIT — reusable)

### 1. Click-free voice stealing — `engine.rs:418-436`

```rust
/// Removes finished sinks, then evicts the oldest voice (ramped down to
/// avoid a click) if the pool is still at or above `max_voices`.
fn manage_active_sinks(sinks: &mut Vec<Sink>, max_voices: usize) {
    sinks.retain(|s| !s.empty());
    if sinks.len() >= max_voices {
        let old_sink = sinks.remove(0);
        std::thread::spawn(move || {
            const STEPS: u32 = 10;
            let starting_volume = old_sink.volume();
            for step in 1..=STEPS {
                let gain = starting_volume * (1.0 - (step as f32) / (STEPS as f32));
                old_sink.set_volume(gain.max(0.0));
                std::thread::sleep(Duration::from_millis(EVICT_RAMP_MS / (STEPS as u64)));
            }
            old_sink.stop();
        });
    }
}
```

The idea worth taking: **when you must drop a voice because you hit a polyphony cap, do not cut
it — ramp it to zero over ~10 steps first.** Cutting a live sample at full gain produces an
audible click, which in a room with a mic in it is not just ugly, it contaminates the recording.
This is the "bounded resource" discipline our peripheral needs and is trivially reimplementable
in pure Python: a fixed voice budget, oldest-first eviction, linear gain ramp to zero.

### 2. Symmetric edge fade — `engine.rs:382-416`

`apply_fade` applies a linear fade-in over `FADE_IN_MS` and a linear fade-out over `FADE_OUT_MS`,
each **clamped to half the buffer** (`:393-398`) so short samples do not have the two ramps
collide and double-multiply the same samples. The fade-out loop walks backwards from the last
frame (`:408-415`), which is the easy way to get the direction wrong.

### 3. Minor: UI/audio thread separation is *tested*, not just intended

`engine.rs:663-664` — `#[test] fn the_engine_loop_never_polls_the_audio_device()` and
`engine.rs:685` `fn no_device_presence_polling_remains_anywhere_in_the_engine()`. These are
regression tests asserting an architectural invariant (the audio thread never blocks on device
enumeration). Cheap, and worth copying when we split our own worker/UI boundary.

## Can we run it?

No. It needs `cpal` → a live audio output device, plus `rdev` → a **global keyboard hook**
requiring platform input privileges (on Linux, X11/Wayland access). We are in WSL with no audio
hardware. We can read it; we cannot execute it here.

## Verdict

**Mostly a dead end; two small clean wins.**

The app itself is irrelevant to a voice peripheral — it is a keyboard-typing soundboard built
around pre-recorded samples and a global input hook. Its "modular synth in Rust" framing does not
survive contact with the source.

But the MIT license and the two mechanisms in §The two stealable mechanisms are genuinely good
and directly applicable to bounded-resource audio on our side: click-free voice eviction and
edge fades. Neither is hard, and both are things our renderer will eventually need. Take the ideas,
skip the project.

## Unverified

- `soundpack_loader.rs` (915 lines) was not read; it parses the JSON packs and may contain timing
  logic worth a look.
- `resampler.rs` (222 lines) not read; likely a thin `rodio` wrapper.
- The exact `FADE_IN_MS` / `FADE_OUT_MS` / `EVICT_RAMP_MS` constants were not read.
- No code was compiled or executed (no audio device; `cargo` build not attempted).