# Sonic Pi — scheduling without threads, and teaching through the trace

- **Repo**: https://github.com/sonic-pi-net/sonic-pi
- **Branch audited**: `dev` @ `351bb64949d4e2096d0f5573e01f8be5e6ef223d`
- **Stars / forks**: 12210 / 998 · **disk**: 468 MB
- **Last push**: 2026-10-18 (very active)

## LICENSE — read this before you copy anything

| Scope | License |
|---|---|
| `app/` (main source, incl. the classic Ruby runtime) | **MIT** (`LICENSE.md:3-8`) |
| `app/web/` (the new web runtime) | **AGPL-3.0-or-later** (`LICENSE.md:35-41`) |
| `app/web/runtime/data/`, `app/web/web/data/` (tutorial content) | MIT (`LICENSE.md:41`) |
| SuperSonic (audio engine) + Clockwork (timing library) | **AGPL-3.0-or-later** (`LICENSE.md:48-52`) |
| bundled synthdefs | MIT, with GPL-3.0 exceptions (`LICENSE.md:115-132`) |
| GUI binaries | "must comply with the terms of the GPL" because of linked GPLv3 software (`LICENSE.md:30-34`) |

GraphQL reports `NOASSERTION` because the license is composite — read `LICENSE.md`, do
not trust the API field.

**Consequence: the single most interesting file in this repo — the scheduler — is AGPL.**
The parts that are MIT (`runtime.rb`, `event_history.rb`, `clockwork_arena.rb`) are the
older, wall-clock-based versions. Read all of it; copy almost none of it.

## THE surprise: the C++ audio server is no longer in this tree

The `app/server/native/` directory contains exactly one file: `.gitkeep`. The historical
`app/server/sonic-pi-server/` (C++ `main.cpp` / `sequencer.cpp` / `player.cpp` /
`oscsender.cpp`) that every writeup about Sonic Pi describes **does not exist at either
`dev` or `stable` HEAD**. I checked both branches with recursive tree listings.

What replaced it is a git submodule (`.gitmodules`):

```
[submodule "app/external/supersonic"]
	path = app/external/supersonic
	url = https://github.com/samaaron/supersonic
```

So the timing/audio engine now lives outside this repository entirely, under AGPL. If you
want to audit the real-time scheduling you must audit `samaaron/supersonic` separately —
**I did not, and it is out of my assigned repo list.** Flagging as an unverified gap.

## THE second surprise: a wall-clock-free scheduler

`app/web/runtime/lib/sonic_pi/scheduler.rb:1-15` is the header comment, and it is the
thesis of the whole file:

```ruby
# SPDX-License-Identifier: AGPL-3.0-or-later
# Sonic Pi's threads without threads: every thread is a fiber with its own
# logical clock, and one scheduler runs whichever is due next. Nothing waits
# on a wall clock; a sleep is a jump in logical time, so a whole program runs
# in an instant and its trace is the same every time.
#
# What is kept faithful to Sonic Pi:
#   * a thread's identity is its spawn path (the run is [0], the program's
#     body [0, 0], its children [0, 0, n] in spawn order);
#   * a spawned thread runs at once, until it first sleeps, syncs or ends,
#     and only then does the parent carry on (CRuby's threads do this, and
#     programs can see it);
#   * threads due at the same instant run in the order they became due;
#   * cues and syncs order by (time, priority, thread path, delta), so a
#     sync sees a cue at its own instant only when the cue sorts after it.
```

**This is the answer to our no-audio-hardware constraint.** A scheduler whose time is a
number rather than a clock can be exercised in full, deterministically, in microseconds,
with no audio device, no clock skew, and no flaky tests. "a whole program runs in an
instant and its trace is the same every time."

### Two modes, one queue (`:311-318`)

```ruby
# Two modes over one queue. NRT (run): resume the next due thread at
# once; logical time is only a number, and a program that never ends is
# traced up to max_time. RT (start_job + step): a thread is resumed when
# the host's clock reaches its wake time less its schedule-ahead, and
# step says when to come back. The scheduler never sleeps itself; the
# host owns the one real wait.
```

Note the last two lines: **the scheduler does not sleep. It tells the host when to come
back.**

### The trace mode is bounded, so it cannot hang (`:320-323`)

```ruby
# A trace has a horizon: a program that never ends (a live_loop with no
# stop) is traced up to max_time of logical time, and a thread that
# spins without sleeping is cut off by the event cap. Both are recorded
# as a runtime error so the trace says it was cut short.
```

Two independent backstops — a logical-time horizon and an event count — plus an explicit
error record. A test-mode scheduler that can wedge is a scheduler you will not run in CI.

### RT mode returns its own next deadline (`:393-419`)  ← the stealable line

```ruby
# RT: runs every thread whose wake time the host's clock has reached, and
# a record it makes sounds its schedule-ahead after that (Sonic Pi's
# spider time plus sched_ahead_time): so a run's first beat leaves with
# the same headroom as every other. Returns when the host should call
# again (on that clock), or nil when nothing is waiting.
def step(now)
  @now = now
  loop do
    break if @queue.empty?
    @queue.sort! do |a, b|
      ta = a[2].start + a[0]
      tb = b[2].start + b[0]
      ta == tb ? a[1] <=> b[1] : ta <=> tb
    end
    wake, _, p = @queue[0]
    break if p.start + wake > now
    @queue.shift
    resume(@lang, p, nil)
  end
  send_fx_frees(now)
  free_jobs(now) if @live
  prune_history!
  nexts = @queue.map { |(wake, _, p)| p.start + wake } +
          @fx_frees.map { |(at, _)| at - FREE_LEAD } +
          @node_frees.map { |(at, _)| at - FREE_LEAD }
  nexts.min
end
```

Line 417 is the mechanism. The scheduler **publishes the timestamp of the next thing that
matters**, by taking the min over three independent deadline sources, and the host sleeps
exactly that long. No polling, no fixed tick, no wakeup that finds nothing to do. If your
"peripheral nervous system" polls at a fixed 10 ms and nothing is scheduled, it is burning
CPU to discover it has nothing to do. Compute the horizon instead.

`FREE_LEAD` is the same idea for teardown: node and fx frees are scheduled with a lead
time so the *bookkeeping* happens ahead of the *event*.

### Ordering is a 4-tuple, not a timestamp (`:144-147`)

```ruby
# Ordered on the session's clock: a job's logical time from its start, so a
# cue from one run and a sync in another compare as Sonic Pi's wall-clock
# spider times do. A trace's one job starts at 0, so nothing changes there.
def key(priority = @priority) = [@start + @time, priority, @path, @delta]
```

`[absolute_time, priority, thread_path, delta]` — and `@seq` is a monotonic insertion
counter used as the final tiebreak in the sort at `:407`. Same idea as SuperCollider's
`mStabilityCounter`: **make ties impossible to get wrong by construction.**

## THE classic runtime: EventHistory — MIT, and directly usable

`app/server/ruby/lib/sonicpi/event_history.rb` is MIT. It is a **time-indexed state
store**: "what was the value of this thing at time t?" and "what is the next value?".

Bounded by construction (`:178-189`):

```ruby
def initialize
  @trim_history = true
  @min_history_size = 20
  @history_depth = 32
  @state = EventHistoryNode.new
  @event_matchers = EventMatchers.new
  @process_mut = Mutex.new
  @matcher_mut = Mutex.new
  ...
```

Two queries (`:196-209`):

```ruby
# Get the last seen version (at or before the current time)
def get(t, p, i, d, b, m, path, val_matcher=nil, get_next=false)
  ...
# Get next version (after current time)
# return nil if nothing found
def get_next(t, p, i, d, b, m, path, val_matcher=nil)
  get(t, p, i, d, b, m, path, val_matcher, true)
end
```

Insert, newest-first, with automatic eviction (`:401-417`):

```ruby
def __insert_event!(e, idx=0, sn=@state)
  if idx == e.path_size
    # we are at the leaf node
    sn.events.unshift(e)
    bubble_up_sort!(sn.events)
    # Auto-trim history Keep at least @min_history_size elements and
    # only remove elements older than @history_depth seconds ago
    if @trim_history
      cutoff_t = (Time.now - @history_depth).to_i
      while (sn.events.size > @min_history_size) && (sn.events.last.time.to_i < cutoff_t)
        sn.events.pop
      end
    end
    return sn
  end
  path_segment = e.path_segment(idx)
  child_node = sn.children[path_segment] ||= EventHistoryNode.new
```

The `EventHistoryNode` trie (`:29-35`) keys on OSC-address path segments, so each
independently-controlled parameter gets its own sorted event list with its own bound.

**This is precisely the structure our 9-track MIDI score needs.** Rendering a score
"as of time T", scrubbing backwards, asking "when does this CC next change?" — all three
are `get` / `get_next` here.

### Blunt criticism: it mixes clocks

Line 412 evicts on `Time.now - @history_depth` — **wall clock** — while the events being
evicted carry `event.time` in **logical** time. Any environment where the two disagree
(non-realtime mode, tests, time-warp, negative `time_warp`) evicts the wrong things.
Correct version: evict on `e.time - @history_depth`, and let the caller pass the cutoff
in. Do not copy this line.

## Schedule-ahead: never schedule at "now"

`runtime.rb:1441`: `Kernel.sleep default_sched_ahead_time`. And `scheduler.rb:393-397`:

```
# RT: runs every thread whose wake time the host's clock has reached, and
# a record it makes sounds its schedule-ahead after that ...
```

`:1147` makes it concrete:

```ruby
Native.gui(:sleep, p.uid, p.start + from + p.sched_ahead + @time_warp, ...)
```

The event is handled at wall time `T` but is *stamped* `T + sched_ahead`. The audio
engine always has a window of future, already-committed events. **A voice system that
emits MIDI at the moment it detects a syllable is already too late; it must emit for a
time slightly in the future, sized to the worst-case engine latency.**

## How it teaches newcomers

`etc/doc/tutorial/` — 100 markdown files. The progression is worth noting as a
pedagogy, not as DSP:

1. **Beep first** (`02.1-Your-First-Beeps.md`, 3.4 KB). Audible result before any syntax.
2. **One axis at a time**: Synths → Samples → Randomisation → Programming Structures →
   Threads → FX → Control → Data Structures → MIDI → OSC → Multichannel.
3. **`tick` and `look`** (`A.11-beat-tracking.md`): `tick` increments and returns the
   current beat; `look` reads it without advancing. Then a ring's `.tick` does both —
   advance the beat *and* index into the sequence.

```ruby
puts tick #=> 0
puts tick #=> 1
puts look #=> 1     # same value twice
```

**Stealable:** separate *the counter* from *the lookup*. Our peripheral system needs the
same split — a monotonic event counter that advances on every detected speech event,
plus readers that observe without consuming. It makes the "advance" side trivially
testable and the "read" side freely re-readable.

4. **`live_loop`** (`09.2-Live-Loops.md`) — a block that restarts in place on edit. The
   pedagogical bet is that **the cheapest possible edit-while-running loop is what makes
   a system feel alive**, and you build the whole language around protecting that loop.

5. **Every tutorial file shows its expected output inline** (`#=>`). Deterministic output
   is a teaching device, not just a test device.

Also relevant to us: `13.2-Live-Audio.md` exists — Sonic Pi does treat live input as a
first-class citizen, which is the closest thing in this audit to our peripheral system.

## What to take, what to skip

**Take:**
- `step()` returning `nexts.min` — publish your own deadline (AGPL, but it is a one-liner
  of an idea, not code)
- `EventHistory`'s bounded time-indexed store, as a *design* — and fix its clock bug
- `tick` / `look` separation
- schedule-ahead as a first-class parameter
- deterministic trace mode with two independent termination guards

**Skip:**
- everything in `app/web/` (AGPL)
- `runtime.rb` is 2113 lines and `scheduler.rb` is 1708 lines. This is the accumulated
  weight of two architectures (classic Ruby runtime and web runtime) coexisting. Neither
  file is a model of restraint.
- the scheduler uses `@queue.sort!` on every `step` — O(n log n) per tick with n = number
  of live threads. Fine for dozens, wrong for thousands. Use a heap (see SuperCollider's
  `PriorityQueue.h`) if the count grows.

## Verification notes

- Tree listings obtained for both `dev` and `stable`; both lack the C++ server directory.
- `app/server/ruby/test/` contains 134 test files including `test_ring.rb` (7.3 KB),
  `test_clockwork_arena.rb` (17.2 KB), `test_node_kill_scheduling.rb`, `test_link_beats.rb`.
- CI is extensive: 33 build/test workflow yml files (`test.yml`, `tests.yml`, `ci.yml`,
  `nightly.yml`, `ruby.yml`, per-platform builds). Notably
  `app/web/specs/**.expected.json` — golden-file testing of traced program output. That is
  the "trace is the same every time" claim made into an enforced CI contract.
- `app/web/runtime/` is an mruby (submodule) build — I did not build it.
