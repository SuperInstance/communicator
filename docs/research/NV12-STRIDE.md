# The captures were being read at the wrong stride

**Date:** 2026-10-19
**Experiment:** `experiments/nv12_stride_audit.py`
**Results:** `experiments/nv12_stride_audit_results.json`
**Reader:** `src/nv12.py` · **Tests:** `tests/test_nv12.py`

---

## 1. What happened

Every video experiment in this repo read a camera capture with a hard-coded
frame stride of `W*H*3//2` = **23,040 bytes** for a 160×96 frame. That is the
size of a standard NV12 or I420 frame, and the assumption was written into
the readers, the docstrings and the seeded corpus metadata.

The captures on disk are **not** standard frames. They were written by
`C:\proart_capture\nv12small.py`, which writes, per frame:

```
Y plane   15,360 bytes   (160 × 96)
U,V        3,840 bytes   (80 × 24 samples, INTERLEAVED)
──────────────────────────────
total     19,200 bytes   = W*H*5/4
```

The chroma is subsampled 2:1 across but **4:1 down**, so each frame is 20%
shorter than a standard one. Reading it at 23,040 therefore does not merely
shift the frames — it **splices two different frames into one**: rows 0..71
come from frame *k* and rows 72..95 from frame *k+1*. Only every sixth frame
survives intact.

**80% of every "frame" this repo has measured was a tear between two
different moments.**

## 2. The evidence, three ways

### 2.1 The video's own length has to agree with the audio's

The captures were made as two separate recordings, so nothing about them
tells you the video's frame count. The audio's duration does, because a
capture that records audio and video for the same period must produce a
video of that same length.

| capture | stride | frames | duration @30 fps | the WAV | match |
|---|---:|---:|---:|---:|:--:|
| A | 23,040 | 750 | 25.00 s | 29.98 s | ✗ |
| A | **19,200** | **900** | **30.00 s** | 29.98 s | **✓** |
| A | 15,360 | 1,125 | 37.50 s | 29.98 s | ✗ |
| B | **19,200** | **900** | **30.00 s** | 29.98 s | **✓** |
| C | **19,200** | **900** | **30.00 s** | 29.98 s | **✓** |
| empty1 | **19,200** | **4,500** | **150.00 s** | 149.91 s | **✓** |

Only the 19,200-byte stride reproduces the audio's duration, on every
capture, to within 0.09 s.

### 2.2 The frames stop being garbage

A real video's consecutive frames are similar. A mis-strided read mixes
unrelated bytes together. The mean consecutive |diff| of the luma plane:

| capture | stride 23,040 | stride 19,200 | ratio |
|---|---:|---:|---:|
| A | 43.560 | **0.545** | 80× |
| B | 37.740 | **0.433** | 87× |
| C | 42.935 | **0.367** | 117× |
| empty1 | 39.295 | **0.005** | 7,446× |

### 2.3 You can look at it

Reading A.nv12 at 19,200 and rendering frame 0 at 4× gives a sharp, stable
photograph of Casey facing the camera in his workshop. A filmstrip of the
mouth region across the take shows the mouth **visibly opening and closing**.
The same bytes read at 23,040 give alternating bands of ceiling and
chroma-stripe garbage.

> The image was the check that mattered. Every arithmetic check above can be
> satisfied by a clever mistake; an eye cannot be.

### 2.4 The capture script settles it

`C:\proart_capture\nv12small.py` is still on disk and says:

```python
W, H = 160, 96
hr = H // 4   # I420 chroma planes are H/4 rows each in a width-W array
y = i[:H, :W]
u = i[H:H+hr, :W//2].ravel(); v = i[H+hr:, :W//2].ravel()
uv = np.empty((u.size, 2), np.uint8); uv[:, 0] = u; uv[:, 1] = v
bufs.append(y.tobytes() + uv.tobytes())
```

19,200 bytes per frame. The `hr = H // 4` is the whole story.

## 3. The three layouts

| layout | stride | structure | who writes it |
|---|---:|---|---|
| `standard_nv12` | 23,040 | luma + **interleaved** UV (80×48 samples) | a standard camera pipeline |
| `y_uv_halfres` | 19,200 | luma + **interleaved** UV (80×24 samples) | **the old captures** |
| `luma_only` | 15,360 | luma only, no chroma | `experiments/corpus_capture_win.py` |

> **CORRECTION (2026-10-20) — the chroma is INTERLEAVED, not planar, and
> this file first said otherwise.** `nv12small.py` writes
> `uv[:, 0] = u; uv[:, 1] = v` on an `(N, 2)` array, which in row-major order
> is **U, V, U, V** — interleaved. An independent lane measured it: in
> A.nv12's chroma region the even-indexed bytes average **125.05** and the
> odd-indexed bytes **135.88** (a U−V difference of 10.82 counts) with **no
> step at the midpoint** (1.16 counts). Planar chroma would show the step
> and no even/odd difference.
>
> The stride and the luma are identical either way, so **no measurement in
> this file changes**. What changed is the layout's **name**: it was
> `y_u_v_planar`, which asserted the falsehood. It is now `y_uv_halfres` —
> named for what actually distinguishes it, the chroma's half vertical
> resolution. `y_u_v_planar` survives only as a deprecated alias, so that
> nothing breaks and nothing should use it.
>
> **The lesson repeats itself.** A wrong name is a wrong name; the fact that
> it did not change a number is not a reason to keep it. A reader who greps
> for the layout in a results file should not be handed a description of the
> bytes that is false.

Note the third one: **the new capture tool writes the luma plane only**
(`ys.append(yuv[:a.height, :a.width].tobytes())`). So the corpus that is
about to be recorded would have been mis-read too, in the same way, by the
same readers.

**A Y+U+V file and a Y+interleaved-UV file are byte-identical when U == V.**
So a file written with neutral grey chroma cannot be attributed to one
layout or the other by any reader — including this one. With real chroma they
are distinguishable, which is the case the detector has to get right.

## 4. The reader

`src/nv12.py` replaces the hand-rolled stride arithmetic. It **detects** the
layout from the content — the candidate whose luma has the smallest mean
consecutive |diff| — and reports the margin over the runner-up, because a
detector that cannot say how sure it is has no business being trusted:

```
A.nv12      -> y_uv_halfres  stride 19200  900 frames  mean|d| 0.545  margin 79.9x
B.nv12      -> y_uv_halfres  stride 19200  900 frames  mean|d| 0.433  margin 87.1x
C.nv12      -> y_uv_halfres  stride 19200  900 frames  mean|d| 0.367  margin 117.1x
empty1.nv12 -> y_uv_halfres  stride 19200  4500 frames mean|d| 0.005  margin 7446.1x
```

A file whose layout is **ambiguous** (every candidate scores the same) is an
**error**, not a guess. `read_luma` raises rather than pick.

## 5. The payoff: the speaking signal was there all along

The video does carry the speaking signal. It was in the mouth, and it was
invisible to a torn read. Measured on the mouth region (rows 46–58, cols
62–86, read off a 5× render with a coordinate grid) against a **cheek
control** at the same rows to the right of the mouth:

| | mouth A−C | effect | cheek A−C | effect | mouth/cheek |
|---|---:|---:|---:|---:|---:|
| **correct stride (19,200)** | +0.8044 | **+0.733 sd** | +0.3267 | +0.477 sd | **1.54×** |
| wrong stride (23,040) | −2.4699 | **−0.183 sd** | +0.7442 | +0.028 sd | −6.68× |

* Permutation null for the mouth effect size: **z = +10.9** (400 shuffles).
* AUC (A > C) on the mouth region: **0.675** at the correct stride, 0.460 at
  the wrong one.
* The **empty room** carries **0.015%** of A's mouth-region motion — 6,600×
  quieter. (Over the whole frame it is 103× quieter.)
* The mouth-specific excess over the cheek is **+0.256 sd** at the correct
  stride and **−0.211 sd** — the wrong sign — at the wrong one.

So: **the mouth moves 1.54× more than the cheek when Casey speaks, and at
the wrong stride that signal does not exist at all.**

## 6. What still does not work: frame-level alignment

The session-level signal is real. The **frame-level** relationship between
the old captures' audio and their video is not recoverable, and no amount of
correct reading changes that.

* The correlation between the speech envelope and the mouth-motion envelope
  peaks at **+0.13** over a ±20 s lag search, and at every assumed video time
  base from 10 Hz to 60 Hz. **At every one of those settings the wrong-session
  controls are as strong as or stronger than the real pairing.** A statistic
  whose controls beat it is not a measurement of speech.
* The independent MediaPipe lane agrees and goes further: mouth aperture vs
  speech is **+0.2052** against a scrambled-audio null of **+0.0902**, so it
  *does* beat the null — but it **fails its controls**. Mouth **width**
  scores **+0.3038** (higher than the aperture; a mouth that opens does not
  get wider) and head **height** scores **+0.3316** with a sharper peak
  (z +3.77 vs +2.70). The correlation is about head motion, not about the
  mouth.

This is consistent with the capture script's own account: the old captures
were made by a script that recorded audio and video **separately and
started them independently**. Nothing in the data can undo that.

**The new capture tool fixes it by construction** — one video frame, one
audio block and one clock reading per iteration — so the corpus that is
about to be recorded is the thing that unblocks this lane.

> **CORRECTION (2026-10-19).** `docs/research/TEACHER-LOO-FOLDS.md` §3
> reported the audio↔video correlation as A **+0.0776**, B **+0.0247**,
> C **+0.1130**, and concluded *"the video does not follow the audio"*. Those
> numbers were computed on **torn frames**. On correct frames the same
> whole-frame instrument returns **+0.1536** for A — twice the published
> value — so the published number was not merely imprecise, it was
> measuring a defect.
>
> The *conclusion* survives, for a better reason and with different
> numbers: on correct frames the mouth signal is real at the session level
> (+0.733 sd) and the frame-level correlation still fails its controls. But
> the quoted figures are withdrawn. §3's *arguments* stand; its *tables* were
> computed on torn features and need re-running.
>
> Separately, `docs/research/TEACHER-PIPELINE.md` and
> `docs/research/ROW-BAND-REMOVAL.md` describe the artefacts of the wrong
> stride as properties of the *scene* — "row banding", "LED flicker
> banding", "chroma-plane-as-luma". They are properties of the *reader*.
> The row-band-removal pass was removing a tear.

## 7. The blast radius

Every one of these was computed on torn frames and is now wrong:

| artefact | what it measured | status |
|---|---|---|
| `row_band_removal.py` | "LED flicker banding" | the tear at row 72 |
| `pose_test.py` | facing vs turned | to be re-run |
| `room_map.py` | the room's motion map | to be re-run |
| `ascii_motion_cause.py`, `ascii_motion_render.py` | the ASCII motion maps | to be re-run |
| `seed_corpus_from_captures.py` | the corpus's 600 frames / 20.0 s | **wrong on both counts** |
| `cloud_multimodal.py`, `cloud_vision_probe.py` | the cloud vision bench | to be re-run |
| `teacher_loo_control.py`, `teacher_static_control.py` | the LOO fold and its controls | to be re-run |

The seeded corpus metadata said `"n_frames": 600, "seconds": 20.0`. That was
**two** errors stacked: the seeder's `--max-frames 600` cap *and* the wrong
stride. The truth is **900 frames, 30.0 s**.

The teacher's *structural* findings — that the LOO fold is degenerate because
the label is session-level, that a blank control scores 1.000, that fold A is
unidentifiable because pose ≡ speaking in the training set — are properties
of the **task construction**, not of the frames, and they stand. The
numbers that were computed *from* the frames do not.

## 8. What to do

1. **Read the captures through `src/nv12.py`.** Every reader in the repo now
   does; the layout is detected and recorded in each results file.
2. **Re-seed the corpus** into `corpus_v2` (900 frames, 30.0 s, 19,200-byte
   frames) and re-run everything downstream of it.
3. **Re-record.** The old captures cannot give frame-level audio↔video
   labels, because their audio and video were independent takes. The new
   tool's shared clock fixes that, and **condition D** (turned *and* silent)
   is still the highest-value take because it is what makes all four pose
   folds identifiable.
4. **Watch the third format.** `corpus_capture_win.py` writes luma-only
   frames. The readers are now correct for it; keep them that way.

## 9. How this was found, for the next time

Three independent lines agreed: the file size and the audio's duration, the
collapse of the frame-to-frame difference, and the capture script still
sitting on disk. Any one of them alone would have been arguable.

The failure was not exotic. A constant was written into a reader, the
constant was plausible, and **the constant was never checked against
anything that could disagree with it** — because the checks that existed
were checks on numbers that had already been produced with the same
constant in them. The frame count was "verified" against a frame count
computed from the same wrong stride. The "20.0 s" in the corpus metadata was
`600 / 30`, where 600 came from the same wrong read.

**A number that is derived from the assumption it is supposed to test is not
a check.** The check has to come from outside: the file size, the script on
disk, the audio's own duration, or a human eye.
