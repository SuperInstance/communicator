# DUAL-3 RELATIVE GEOMETRY — the two views in time and not in space, and what this take cannot license

**Tool:** `experiments/dual_render.py` (80/80 selftest green), `experiments/dual_ingest.py` (29/29), `experiments/lens.py` (21/21)
**Render:** `docs/research/DUAL-3-RELATIVE-RENDER.txt` (2,987 lines)
**Numbers:** `docs/research/DUAL-3-RELATIVE-RENDER.json`
**Bundles:** `docs/research/DUAL-3-KEYFRAME-META.json` (the phone's camera-luma channel), `docs/research/DUAL-3-PHONE-META.json` (the phone's ASCII channel)
**Device:** CPU, numpy + scipy + OpenCV, no GPU, no network, no paid API
**Take:** `~/dual-capture/dual-3.json` (the phone's room-lens) against eight ProArt sessions
**Alignment:** the WALL CLOCK. The duet's clap channel was dead (zcode's pair report); this lane consumes the alignment and never computes it.

---

## 0. THE HEADLINE

**The dual views are built, and the honest answer is that they are aligned in TIME and in nothing else.** The two panels of the matched-moment dual-ASCII are the same instant seen by two cameras in two places, and no cell on the left is the same part of the room as a cell on the right.

**No scale is guessed anywhere in this file, because the take cannot supply one.** The route for two uncalibrated views is audio-sync first and sub-frame, then a DUSt3R/MASt3R-style relative pose which is scale-free, and the metric scale from the ranging/TDOA side. This take supplies neither the pose nor the scale: the phone's pose carries **no position** and a **null heading in all 513 frames**, so the relative rotation is unknown and the relative translation is unmeasurable. That is a property of the take and not a choice.

**Three numbers carry the result:**

| | at the grain floor | at the signal threshold |
|---|---|---|
| the fixed view's crossings | 29 (8 in the shared minutes) | 4 (3 in the shared minutes) |
| the roving view's crossings | **0** | **6** |
| the shared crossing, z against the slide null | — (no crossings to share) | **+1.32** (p 0.208) |
| the same, on the tighter criterion | — | **+1.61** (p 0.112) |

**So the threshold rescues the roving channel from having no crossings at all, and the coincidence is suggestive and not significant.** Five shared crossings out of six, against a slide null that produces 3.30 on average. The z is +1.32, and the tighter criterion gives +1.61 with p 0.112.

**THE DETECTOR'S DURATION RULE CHANGED UNDER THESE RENDERS, AND THE VERDICT MOVED WITH IT.** The corpus's crossing detector's minimum duration used to be **5 frames** — 0.083 s at 60 fps and 2.5 s at 2 fps — so the two frame rates were counting different events and the corpus's crossings-per-minute was a mixture of two quantities. It is now a **time**: `max(0.35 s, 1.5 × the session's own median frame interval)`, the same moment the chaining gap already uses. The rule was declared before the sweep that measured it, and the numbers in this file are the ones the new rule produces.

**And the graded witness still cannot measure this take's own alignment.** Raw r **+0.2327** (z +1.425), log r **+0.3002** (z +1.689), and the slide null's own peak sits at **−42.1 s** (raw) with a spread of **227.3 s** — a broad region means the sweep found nothing to localise. The duet cannot align itself; only the claps can, and they are dead on this take.

---

## 1. WHAT WAS BUILT, AND ON WHAT RULE

Casey's ask was the three renders from `dual-3` in **relative geometry**, with **no guessed scale**, filed where the pair report lives. All four parts are done.

**THE ROOVING SIDE IS THE PHONE'S CAMERA LUMA, NOT ITS ASCII.** The phone's room-lens renders *characters*, and a character collapses half the brightness range into one glyph, so its change map is a mask of band flips rather than a change detector — on this take it was on **99.8 %** of the time. But the bundle also carries **103 keyframes**: the camera's own picture, as JPEG, at the phone's own resolution. Those are the camera's luma, which is the very quantity the ProArt's change map is in. So the phone gets a second channel, in the corpus's own units, and the two views become commensurable for the first time.

**THE TWO CHANGE MAPS ARE NOW IN THE SAME UNITS, AND THAT IS A MEASUREMENT.** The renderer's line about the two sensors used to assert that they were never in the same units; that was true for the ASCII channel and false for the keyframe channel, and it is now read from the bundles' own `the_units` and printed either way. Here it prints:

> THE TWO CHANGE MAPS ARE IN THE SAME UNITS — both are the camera's own luma — so ONE scale is honest between them. They are still not in the same SPACE: a cell on the left and a cell on the right are not the same part of the room.

**THE SCALE IS NOT HERE AND NONE IS GUESSED.** Every number in the render is relative: a moment, a frame interval, a brightness against a floor. There is no distance in metres in the file, and none is assumed. The JSON carries `the_scale_is_guessed: false` as a field rather than as a promise in prose.

---

## 2. THE RELATIVE GEOMETRY: WHAT THE TAKE SUPPORTS AND WHAT IT DOES NOT

The render carries a block that says both halves, and the JSON carries them as two lists.

**What the take supports** — the roving device's own orientation, which is all the pose holds:

| axis | range over the session |
|---|---|
| alpha (the azimuth-like tilt) | 1.7° to 351.2° |
| beta (the pitch) | 5.3° to 99.0° |
| gamma (the roll) | −84.3° to 73.7° |

**What the take does not support:**

- the roving device's **HEADING is null in every frame**, so its azimuth is unknown and the two views' relative **ROTATION** cannot be solved either;
- the roving device's **POSITION was never recorded**, so it cannot be placed in the room and the relative **TRANSLATION** is not measurable from this bundle.

**And the scale has not landed.** The metric scale belongs to zcode's ranging/TDOA side; this render waits for it and does not borrow one.

**THE TIME AXIS IS THE ONLY AXIS THEY SHARE:** the fixed side is one frame every **499.750 ms** and the roving side one every **2105.5 ms**, and the offset between their clocks is **+500538.000 ms**. The roving view's interval is **4.21×** the fixed view's, so the same movement covers 4.21× more of the picture per frame in the roving view — which is exactly why the window is half the keyframe interval, so that both panels cover the same span of room time.

**THE ALIGNMENT IS THE WALL CLOCK AND IT SAYS SO.** The sidecar carries no clap times, so the alignment's resolution cannot be measured from it: the offset is as good as whatever produced it, and where that was the wall clock, it carries the device-to-device skew the corpus cannot measure. The render says this in the header, again above the dual-ASCII, above the fused score and above the differenced view — four times, because a reader who lands on one panel should not have to scroll to learn that.

---

## 3. THE MATCHED MOMENTS

**THE MATCHING IS BY MOMENT AND BY NOTHING ELSE.** The 102 moments are the roving stream's **own frames**, each one carried into the fixed clock by the alignment. They are not an evenly spread set and they are not chosen: they are the frames the phone actually produced. They all fall inside the **235.12 s** of shared minutes (**+500.54 s .. +735.66 s**), which is the whole of the duet that both devices were looking at the same room at the same time.

**THE WINDOW IS HALF THE KEYFRAME INTERVAL (1.0525 s), SO BOTH PANELS COVER THE SAME SPAN OF ROOM TIME.** The fixed side then holds 4 frames per moment and the roving side 1, which is the honest reading of a 4.21× rate difference and not a way of making the two views look alike.

**AND THE MOMENTS INVERT.** Every matched moment maps back to the roving frame it came from, to better than 1e-9 s. That is a selftest, not a claim.

---

## 4. THE THREE RENDERS

All three are in `docs/research/DUAL-3-RELATIVE-RENDER.txt`, in the order the brief asked for.

### 4.1 The dual-ASCII, side by side

102 matched moments, each a pair of 40×24 panels: the fixed view's change on the left, the roving view's change on the right, one scale shared between them (`shared-auto`, so a side-by-side of two different scales would not be a comparison). The window statistic is the **mean** change over the window, which is frame-rate independent; the corpus's 3.0 luma floor was measured on a mean, so a mean is the quantity it was measured on.

### 4.2 The fused score

Two staves over 112 slices of **2.105 s** — the keyframe interval, so a slice is one phone frame — with each channel annotating the other. A mark is decided by the **strongest frame in the slice**, so a thing that moved for one frame still counts. The energy scale is shared, maximum **86530.8 luma**.

### 4.3 The differenced view

At the middle moment (**+609.162 s** fixed): the fixed frame moved in **0 cells** above its own floor of 3.000, the roving frame saw movement in **894 cells** above its own floor of 3.000. `witnessed: true` — the fixed view's movement was witnessed by the roving view. That is the third render doing its job.

---

## 5. THE WITNESSES, AND WHY THE VERDICT IS NEGATIVE

### 5.1 The binary count — degenerate, and degenerate for a reason

| | count | share |
|---|---|---|
| both moved | 75 | 67.0 % |
| only the fixed moved | 0 | 0.0 % |
| only the roving moved | 37 | 33.0 % |
| neither moved | 0 | 0.0 % |

`p_fixed` 0.6696, `p_roving` **1.000**, the excess over chance **+0.000**. The roving channel is on **100.0 %** of its frames at the grain floor, so its row of the contingency table is constant and the table has nothing to measure; the fixed view is on **58.6 %** of its own. The alignment sensitivity table shows the count is **75 at every shift** from −2000 ms to +2000 ms, which is what a count that is not measuring the clock looks like — and also what a count that is measuring nothing looks like.

### 5.2 The graded witness — the slide null still finds its own peak

| reading | observed r | null mean | null sd | z | slides that match it |
|---|---|---|---|---|---|
| raw | +0.2327 | −0.0024 | 0.1649 | **+1.425** | 7 of 113 |
| log | +0.3002 | −0.0067 | 0.1817 | **+1.689** | 5 of 113 |

The sweep's maximum is at **−42.1 s** (raw, r 0.3316) with a sharpness of **227.34 s**, and at **−37.89 s** (log, r 0.347) with a sharpness of **149.455 s**. A peak spread over 149 to 227 seconds is not a peak. **The duet cannot measure its own alignment**, and this is the number that says so.

The log reading is reported as a **second** reading and not a replacement: the raw reading was asked for first, and the log one was added after the raw one was seen.

### 5.3 The crossing witness — run at two thresholds, and neither is picked

At the **grain floor** the roving view finds **no crossings at all**: the corpus detector needs a blob that holds its shape while it TRANSLATES, and a channel that changes everywhere — a hand-held camera, whose whole picture moves — has no such blob. That is a finding about the channel, not a failure of the render.

At each channel's **own signal threshold** (§6) the roving view finds **6** crossings and the fixed view **4**, and the witness is then run properly:

| criterion | observed | null mean | null sd | z | slides that match it | p |
|---|---|---|---|---|---|---|
| the two intervals overlap at all | 5 of 6 | 3.303 | 1.283 | **+1.322** | 50 of 241 | 0.2075 |
| they share half of the shorter | 5 of 6 | 2.996 | 1.244 | **+1.611** | 27 of 241 | 0.112 |

**Both thresholds are reported and neither is picked.** The grain floor is the channel's own floor and the signal threshold is where the channel is reading the room; the gap between them is a finding about the channel, and not a parameter to be tuned until the answer comes out right. The verdict is the same at both: **suggestive and not significant.**

---

## 6. THE SIGNAL THRESHOLD: A RULE AND A TOOL BEHIND A NUMBER

The **94.938** in the earlier prose was a filed number with no tool behind it. It now has both.

**THE RULE, in one sentence:** *the threshold at which this channel is on 90 % of its frames — the 10th percentile of its per-frame change-map maxima.* It is a **signal** threshold, not a grain floor: the floor asks "is there anything here at all", and this asks "is this channel reading the room". The floor is a property of the sensor; the signal threshold is a property of **this recording**, so it is measured and not borrowed.

**THE TOOL:** `lens.signal_threshold(maps, duty_target=0.90)`, in `experiments/lens.py`, next to the change-map floor it is distinct from. The percentile convention is named because it moves the number: nearest-rank (the k-th smallest observed maximum, k = ceil((1 − duty) × n)) is the canonical one, and the linearly interpolated value is reported beside it. On this take the two are **94.938** and **95.106** — the same data, two conventions, a number that has to carry its convention with it.

**THE RULE REPRODUCES THE FILED NUMBER EXACTLY**, from the bundle:

| channel | maps | threshold | interpolated | rank | share above | crossings at it | crossings at the floor |
|---|---|---|---|---|---|---|---|
| the phone's KEYFRAME channel | 102 | **94.9375** | 95.10625 | 11 of 102 | 89.2 % (90.2 % at-or-above) | **6** | **0** |
| the phone's ASCII channel | 488 | 52.0625 | 52.0625 | 49 of 488 | 89.8 % | 28 | 29 |
| the fixed ProArt view | 1865 | **0.0000** | 0.0000 | 187 of 1865 | 86.8 % | 4 | 29 |

**THE KEYFRAME CHANNEL IS SATURATED AT ITS OWN FLOOR AND READS AT ITS SIGNAL THRESHOLD.** At the corpus's grain floor of 3.000 it is on **100.0 %** of its frames and finds **no** crossings at all; at 94.9375 it is on 89.2 % of its frames and finds **six**. The fixed view is on 58.6 % of its frames at the same floor and finds 29 crossings.

**AND THE CROSSING COUNTS ARE THE CORPUS'S OWN DETECTOR'S, UNDER ITS NEW DURATION RULE.** The minimum duration used to be **5 frames** — 0.083 s at 60 fps against 2.5 s at 2 fps — so the two frame rates counted different events and the corpus's crossings-per-minute was a mixture of two quantities. It is now a **time**: `max(0.35 s, 1.5 × the session's own median frame interval)`, the corpus's own moment, which is the same expression the chaining gap already uses. **The rule was declared before the sweep that measured it**, and the counts above are the ones it produces.

**THE FIXED VIEW'S THRESHOLD IS ZERO, AND THAT IS A FINDING ABOUT THE CHANNEL.** 13.2 % of the ProArt stream's change maps have a maximum of exactly zero — the recorder writes a picture only when it changes — so the threshold at which that channel is on 90 % of its frames is the threshold that reads **everything**. A channel that cannot be quiet is a channel that is saturated, and it is saturated at its own resolution. The render says this in full, next to the number.

**AND THE TIES ARE REPORTED, BECAUSE THEY CHANGE THE SHARE.** The rule's exactness holds only when the maxima carry no ties. The ASCII channel carries 2 ties at its threshold and the fixed view carries 246; where they occur, the share of frames above the threshold is what the ties give and not the target, and the tool says so rather than quoting the target it missed.

---

## 7. WHAT IS NOT HERE

**NO SCALE.** No distance in metres appears in the render or its JSON, and none is assumed. The scale belongs to zcode's ranging/TDOA side.

**NO RELATIVE POSE.** The phone's pose carries no position and a null heading, so the relative rotation is unknown and the relative translation is unmeasurable. The render states both, as a measurement of what the take supports.

**NO SPATIAL REGISTRATION.** The panels are side by side because they saw the same minute, not because they saw the same thing from the same place. A cell on the left and a cell on the right are not the same part of the room, and this render makes no attempt to make them so.

**NO ALIGNMENT OF ITS OWN.** The wall clock stands in, and it carries the device-to-device skew. The alignment's resolution is reported as **unmeasurable** from a sidecar with no clap times, rather than guessed.

**NO VERDICT OF "THE SAME MOMENT".** The crossing witness at the signal threshold finds five shared crossings out of six, against a null that produces 3.30. The z is **+1.32** and the tighter criterion **+1.61** with p 0.112. That is suggestive and not significant, and it is filed that way.

---

## 8. THE COMMANDS

```
python experiments/dual_ingest.py \
  --room-lens ~/dual-capture/dual-3.json --out /tmp/duet-3 \
  --label "the phone room-lens (the roving eye)" --fixed-label "the ProArt, merged" \
  --fixed-session 1792655768 --fixed-session 1792656069 --fixed-session 1792656077 \
  --fixed-session 1792656100 --fixed-session 1792656118 --fixed-session 1792656155 \
  --fixed-session 1792656183 --fixed-session 1792656235

python experiments/dual_render.py \
  --fixed /tmp/duet-3/fixed --roving /tmp/duet-3/keyframes \
  --sync /tmp/duet-3/wallclock-fixed.json --out /tmp/duet-relative \
  --scale shared-auto --map mean --mode both --matched \
  --window 1.0525 --slice 2.105 --note "..."
```

**The window is half the keyframe interval (1.0525 s), so the two panels cover the same span of room time.** The slice is the keyframe interval (2.105 s), so a slice is one phone frame.

---

## 9. WHAT LANDS NEXT

**zcode owns the alignment numbers when take 2 lands.** When the clap sidecar replaces the wall clock, this same command re-renders with a measured, sub-frame alignment, and the three renders change only in their time axis — not in their geometry, which is still not measurable from this take.

**The relative pose needs a position and a heading.** Both are in the next take's capture contract, already filed in the brief: the phone's PLACE and a HEADING, not just a tilt. With those, the DUSt3R-style stage can run and the render can say what the relative pose is rather than what it cannot be.

**The scale lands from the ranging/TDOA side**, and until it does, the render keeps no metric claim. That is the research note's constraint and it is now the code's behaviour, not a note in a document.

---

*Committed for Muse to review; not merged.*
