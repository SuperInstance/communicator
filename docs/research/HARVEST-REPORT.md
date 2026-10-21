# HARVEST REPORT — what the room gave us

**Appended by:** `experiments/free_session_analyze.py --harvest`
**Corpus:** `/mnt/c/proart_capture/corpus` (the capture page's output
directory; its free sessions arrive under `free_sessions/`)
**Derived takes:** `<corpus>/derived/` — its own corpus root, because the
gate's stimulus check fails any take whose stimulus differs from the
corpus majority, so a derived take cannot sit beside a script take.

This report is APPENDED, once per run. Each section is one pass of the
harvester over the corpus: the minutes it harvested per condition, the
confidence of those labels, the coverage against the target of 63
independent sessions per class, and what is still missing.

The steady state this buys: **Casey does nothing.** The page records, the
harvester derives, and this file says what the room produced.

**The first section below is a DRY RUN**, over the captures that exist
today: the script's A, B, C and empty1 recordings, re-encoded in the
free-record layout so the harvester could be exercised over the room that
is already on disk. Its numbers are real measurements of real footage. The
minutes are small because the captures are thirty seconds each.

**To run it:**

```bash
# one pass (this is the cron line)
/home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \
    --harvest /mnt/c/proart_capture/corpus

# or continuously
/home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \
    --harvest /mnt/c/proart_capture/corpus --watch 60
```

---

## Run 2026-10-21T03:01:15Z — 4 new session(s), 0 failed

**corpus** `/tmp/harvest_dry3`  
**derived takes** `/tmp/harvest_dry3/derived`  
**sessions found** 4, **new** 4, **already harvested** 0, **failed** 0  
**takes written** 4  
**sessions known to the harvester** 4

### Minutes harvested this run

| condition | takes | minutes | mean conf | min conf | max conf |
|---|---|---|---|---|---|
| A | 2 | 0.15 | 0.735 | 0.689 | 0.780 |
| B | 1 | 0.06 | 0.709 | 0.709 | 0.709 |
| C | 1 | 0.49 | 0.894 | 0.894 | 0.894 |

### Minutes harvested in total, from the derived corpus

| condition | takes | minutes | mean conf | min conf | max conf | sessions |
|---|---|---|---|---|---|---|
| A1 | 1 | 0.08 | 0.689 | 0.689 | 0.689 | 1 |
| A2 | 1 | 0.07 | 0.780 | 0.780 | 0.780 | 1 |
| B1 | 1 | 0.06 | 0.709 | 0.709 | 0.709 | 1 |
| C1 | 1 | 0.49 | 0.894 | 0.894 | 0.894 | 1 |

**0.69 minutes in total, from 3 independent session(s).**

### The gate over the derived corpus

**4 takes, 0 failed, ok=True**

### Coverage against the target

Target: **63 independent sessions per class** (experiments/teacher_data_requirement.py -- n = ceil(2*(z*s/d)^2) with z=2, s=0.01022 (between-session sd within a label) and d=0.00366 (between-label separation) on the conc_mean region feature; recorded in experiments/teacher_data_requirement_results.json).

| class | takes | distinct sessions | usable sessions | gap to target |
|---|---|---|---|---|
| SILENT | 1 | 1 | 1 | 62 |
| SPEAK | 3 | 2 | 2 | 61 |

| condition | takes | distinct sessions | usable sessions | gap to target |
|---|---|---|---|---|
| A1 | 1 | 1 | 1 | 62 |
| A2 | 1 | 1 | 1 | 62 |
| B1 | 1 | 1 | 1 | 62 |
| C1 | 1 | 1 | 1 | 62 |

### What is still missing

- **SPEAK is 61 sessions short of the 63 target** (2 recorded). Each free session that contains talking fills it.
- **SILENT is 62 sessions short of the 63 target** (1 recorded). It is the harder half: the room has to be quiet on purpose.
- The conditions still short of the target: A (2 takes), B (1 take), C (1 take). The target is per CLASS, so these are a record of which behaviour the room has been asked for, not a separate quota.

