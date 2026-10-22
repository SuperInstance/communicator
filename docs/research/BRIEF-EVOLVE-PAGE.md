# BRIEF - The evolve page (Casey via Muse, 2026-10-20) - a lab the fleet can iterate all night

## What Casey asked for
A page he loads ONCE, from which the fleet can evolve: capture running,
experiments iterating against his real room and hardware overnight,
results flowing back without him touching anything.

## The design: stable shell, hot-swappable probes
- SHELL (stable, rarely changes): served by the liaison server at
  /evolve. Opens camera + mic (raw, processing off), runs the ambient
  AUTO capture as usual, shows an honest lab status (current probe,
  last result, next run) and the pause/kill switch. Nothing else.
- PROBES (the evolving part): versioned JS modules under
  browser_capture/probes/, served by the liaison. The shell re-fetches
  the probe MANIFEST every cycle, so an agent can rewrite a probe and
  the already-open page picks up the new version WITHOUT a reload -
  that reload-free pickup is the whole point; build it first-class.
- RESULTS: every probe POSTs to /log (kind: probe_result, probe name +
  version) and may save artifacts under the session corpus. Agents
  read results, rewrite probes, iterate. The loop is the deliverable.

## First tenants, in order
1. sonar-band (BRIEF-ROOM-SONAR step 1): sweep 16k-22k + pink burst,
   per-frequency return levels - the usable band on this hardware.
2. audio-health: periodic saved-WAV measurement (the SILENT_AUDIO
   class, patrolled forever, not remembered).
3. Future probes arrive as new modules; nothing in the shell changes.

## Safety rails (hard, in the shell, not the probes)
- Audible-band probes: OFF by default between 22:00 and 07:00 local.
- Near-ultrasonic probes: modest level ceiling + short duty cycle,
  enforced by the shell regardless of what a probe requests.
- One kill switch kills capture AND probes. Status text always says
  what ran, when, and what it measured - no silent experiments, ever.
