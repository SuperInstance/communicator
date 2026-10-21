/* over-detector -- BRIEF-WAKE-AND-TURN.md mechanism 3, as an evolve probe.
 *
 * OVER is a TRAINED transmit word: it fires ONLY as a conjunction --
 *   (a) the word "over" in terminal position (ending a speech run),
 *   (b) spoken with its own prosody (terminal, falling -- the way a
 *       question has a melody, OVER has one),
 *   (c) followed by ~1.5-2 s of trailing silence.
 * Mid-sentence "over" never fires: no terminal position, no trailing
 * pause. This probe measures the conjunction's components separately
 * and reports the headline numbers the brief asks for: FALSE-FIRES on
 * ambient talk (fires per hour with no OVER meant), plus the component
 * counts so a miss is diagnosable. (Missed yields need meant-OVER
 * ground truth -- the drill-deck class the brief plans; this probe
 * exposes the pieces for it.)
 *
 * PAGE-TIER ONLY, per the tiers law: 20-year-old tools, decomposed --
 *   energy envelope (speech runs + trailing silence),
 *   YIN pitch on the run's tail (terminal falling prosody),
 *   template matching for the word (multi-duration fixed-grid cosine
 *   distance against operator-calibrated templates; no model, no cloud).
 *
 * Modes, self-selecting: no stored templates -> CALIBRATION (the lab
 * rows say what to do: say "over" alone, five times, ~1/s apart, then
 * wait); templates present -> AMBIENT MEASURE (tap, detect, count).
 * Templates live in localStorage (bc_over_templates) so every later
 * run -- including other nights -- reuses the same calibration.
 *
 * Bump meta.version to hot-swap into already-open pages (no reload).
 */
export const meta = { name: "over-detector", version: 4 };

const SR = 48000;
const FFT_N = 512;                 // 10.7 ms at 48k
const HOP = 256;                   // 5.3 ms feature hop
const BANDS = 16;                  // mel-ish triangular filters, speech band
const DURS = [0.35, 0.45, 0.55];   // candidate word durations, seconds
const MIN_RUN_S = 0.15;            // speech run minimum
const FIRE_SILENCE_S = 1.5;        // the brief's trailing-silence bar
const NEAR_SILENCE_S = 1.2;        // near-misses reported separately
const TAIL_PROSODY_S = 0.30;       // pitch window at the run's end

/* -- features: FFT (radix-2, magnitude), mel band energies -------------- */
function fftMag(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(ang * k), wi = Math.sin(ang * k);
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * wr - im[i + k + len / 2] * wi;
        const vi = re[i + k + len / 2] * wi + im[i + k + len / 2] * wr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr;
        im[i + k + len / 2] = ui - vi;
      }
    }
  }
  const mag = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) mag[i] = Math.hypot(re[i], im[i]);
  return mag;
}

const MEL = (() => {
  const lo = 100, hi = 4000, f = [];
  const mel = (hz) => 1127 * Math.log(1 + hz / 700);
  for (let b = 0; b <= BANDS + 1; b++)
    f.push(700 * (Math.exp((mel(lo) + (mel(hi) - mel(lo)) * b / (BANDS + 1)) / 1127) - 1));
  const filters = [];
  for (let b = 1; b <= BANDS; b++) {
    const bins = [];
    for (let k = Math.floor(f[b - 1] / (SR / FFT_N));
         k <= Math.ceil(f[b + 1] / (SR / FFT_N)) && k < FFT_N / 2; k++)
      bins.push(k);
    filters.push(bins.map(k => {
      const hz = k * SR / FFT_N;
      let w = 0;
      if (hz >= f[b - 1] && hz <= f[b]) w = (hz - f[b - 1]) / (f[b] - f[b - 1]);
      else if (hz > f[b] && hz <= f[b + 1]) w = (f[b + 1] - hz) / (f[b + 1] - f[b]);
      return { k, w };
    }));
  }
  return filters;
})();

function wordFeatures(x, off, durS) {
  // resample the window to a FIXED 20 frames, per band log-energy,
  // band-mean removed, frame-normalized: duration goes into the grid,
  // not the distance. Each FFT point is the BOX AVERAGE of the samples
  // mapping to it (a crude lowpass), applied identically to templates
  // and candidates so the decimation is consistent, not aliasing noise.
  const FRAMES = 20;
  const win = Math.round(durS * SR);
  if (off < 0 || off + win > x.length) return null;
  const out = [];
  for (let f = 0; f < FRAMES; f++) {
    const start = off + Math.round(f * win / FRAMES);
    const span = Math.max(1, Math.round(win / FRAMES));
    const re = new Float64Array(FFT_N), im = new Float64Array(FFT_N);
    for (let i = 0; i < FFT_N; i++) {
      // point i averages the samples in [start + i*span/FFT_N,
      // start + (i+1)*span/FFT_N)
      const a = start + Math.floor(i * span / FFT_N);
      const b = start + Math.ceil((i + 1) * span / FFT_N);
      let s = 0, c = 0;
      for (let q = a; q < b && q < x.length; q++) { s += x[q]; c++; }
      const v = c ? s / c : 0;
      re[i] = v * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / FFT_N));
    }
    const mag = fftMag(re, im);
    const bands = new Float64Array(BANDS);
    for (let b = 0; b < BANDS; b++) {
      let s = 0;
      for (const { k, w } of MEL[b]) s += mag[k] * w;
      bands[b] = Math.log10(Math.max(s, 1e-10));
    }
    let m = 0;
    for (const v of bands) m += v;
    m /= BANDS;
    let nrm = 0;
    for (let b = 0; b < BANDS; b++) { bands[b] -= m; nrm += bands[b] * bands[b]; }
    nrm = Math.sqrt(nrm) || 1;
    for (let b = 0; b < BANDS; b++) bands[b] /= nrm;
    out.push(bands);
  }
  return out;
}

function gridDistance(a, b) {
  // cosine distance on normalized frames, averaged over the fixed grid
  // (the band index is BI: an inner `b` once shadowed the parameter and
  // turned every distance into a crash -- found by the node repro)
  if (!a || !b) return Infinity;
  let s = 0;
  for (let f = 0; f < a.length; f++) {
    let dot = 0;
    for (let bi = 0; bi < BANDS; bi++) dot += a[f][bi] * b[f][bi];
    s += 1 - dot;
  }
  return s / a.length;
}

function matchTemplates(x, endSample, templates) {
  // best template distance over candidate durations ending at the
  // speech-run end (terminal position)
  let best = Infinity;
  for (const dur of DURS) {
    const off = endSample - Math.round(dur * SR);
    if (off < 0) continue;
    const feats = wordFeatures(x, off, dur);
    if (!feats) continue;
    for (const t of templates) best = Math.min(best, gridDistance(feats, t));
  }
  return best;
}

/* -- YIN on the tail (the bit-equal port from the seed proof) ------------ */
function yinF0Track(x, start, len) {
  // 48k GEOMETRY: tau for 70-400 Hz is 120-686, so the frame must be
  // >= 806 samples; the seed-proof port's 512 was 16k geometry and made
  // every track empty here (found by the synthetic OVER not firing)
  const FRAME = 1024, HOPP = 320, FMIN = 70, FMAX = 400, THR = 0.15;
  const tauMin = Math.max(1, Math.floor(SR / FMAX));
  const tauMax = Math.ceil(SR / FMIN);
  const W = FRAME - tauMax;
  if (W < tauMin || start < 0 || start + len > x.length) return [];
  const out = [];
  for (let p = 0; p + FRAME <= len; p += HOPP) {
    const frame = x.subarray(start + p, start + p + FRAME);
    let s = 0;
    for (const v of frame) s += v * v;
    if (Math.sqrt(s / FRAME) < 1e-5) continue;
    let eHead = 0;
    for (let j = 0; j < W; j++) eHead += frame[j] * frame[j];
    const d = new Float64Array(tauMax + 1);
    for (let tau = 1; tau <= tauMax; tau++) {
      let c = 0, e = 0;
      for (let j = 0; j < W; j++) {
        const a = frame[j], b = frame[tau + j];
        c += a * b; e += b * b;
      }
      const raw = eHead + e - 2 * c;
      d[tau] = raw > 0 ? raw : 0;
    }
    const dp = new Float64Array(tauMax + 1).fill(1);
    let run = 0;
    for (let tau = 1; tau <= tauMax; tau++) {
      run += d[tau];
      dp[tau] = run > 0 ? d[tau] * tau / run : 1;
    }
    let tau = null;
    for (let t = tauMin; t <= tauMax; t++) {
      if (dp[t] < THR) {
        while (t + 1 <= tauMax && dp[t + 1] < dp[t]) t++;
        tau = t; break;
      }
    }
    if (tau === null) {
      let best = tauMin;
      for (let t = tauMin; t <= tauMax; t++) if (dp[t] < dp[best]) best = t;
      tau = best;
    }
    const conf = Math.max(0, Math.min(1, 1 - dp[tau]));
    if (conf < 0.5) continue;
    const den = dp[tau - 1] - 2 * dp[tau] + dp[tau + 1];
    let refined = tau;
    if (tau > 0 && tau + 1 <= tauMax && Math.abs(den) > 1e-12) {
      const shift = 0.5 * (dp[tau - 1] - dp[tau + 1]) / den;
      refined = tau + Math.max(-1, Math.min(1, shift));
    }
    const hz = SR / refined;
    if (FMIN <= hz && hz <= FMAX) out.push({ p, hz });
  }
  return out;
}

function terminalFalling(x, runEnd, runMeanHz) {
  // terminal prosody: the run's final voiced stretch falls -- slope or
  // final-below-mean rule, stated, not tuned
  const track = yinF0Track(x, runEnd - Math.round(TAIL_PROSODY_S * SR),
                           Math.round(TAIL_PROSODY_S * SR));
  if (track.length < 2) return { falling: false, n: track.length,
                                 slope: null };
  const t0 = track[0].hz, t1 = track[track.length - 1].hz;
  const slope = 1200 * Math.log2(t1 / t0) / (track[track.length - 1].p - track[0].p) * SR / 1000;
  const finalBelow = t1 < runMeanHz * Math.pow(2, -2 / 12);
  return { falling: slope < -20 || finalBelow, n: track.length,
           slope: +slope.toFixed(1) };
}

/* -- speech runs + trailing silence (energy envelope) -------------------- */
function speechRuns(x) {
  const WIN = Math.round(0.02 * SR);
  const env = [];
  for (let p = 0; p + WIN <= x.length; p += WIN) {
    let s = 0;
    for (let i = p; i < p + WIN; i++) s += x[i] * x[i];
    env.push(Math.sqrt(s / WIN));
  }
  const sorted = env.slice().sort((a, b) => a - b);
  const thr = Math.max(1e-4, 0.1 * sorted[Math.floor(sorted.length * 0.95)]);
  const raw = [];
  let start = -1;
  for (let i = 0; i < env.length; i++) {
    if (env[i] >= thr && start < 0) start = i;
    if ((env[i] < thr || i === env.length - 1) && start >= 0) {
      const end = (env[i] < thr ? i : i + 1);
      if ((end - start) * 0.02 >= MIN_RUN_S)
        raw.push({ startS: start * 0.02, endS: end * 0.02,
                   start: start * WIN, end: end * WIN });
      start = -1;
    }
  }
  // MERGE runs separated by < 80 ms: an intra-word syllable gap ("OH -
  // ver") is not a turn boundary, and a split OVER found no candidates
  // in the first E2E (10 half-word runs of 0). Found by the synthetic
  // calibration: 5 said words, 10 runs, 0 candidates.
  const runs = [];
  for (const r of raw) {
    const last = runs[runs.length - 1];
    if (last && (r.startS - last.endS) < 0.08) last.endS = r.endS,
                                                last.end = r.end;
    else runs.push({ startS: r.startS, endS: r.endS,
                     start: r.start, end: r.end });
  }
  return { runs, thr };
}

function runMeanF0(x, run) {
  const track = yinF0Track(x, run.start, Math.min(run.end - run.start,
                                                  Math.round(1.0 * SR)));
  if (!track.length) return null;
  const hzs = track.map(t => t.hz).sort((a, b) => a - b);
  return hzs[Math.floor(hzs.length / 2)];
}

/* -- templates in localStorage -------------------------------------------- */
function loadTemplates() {
  try {
    const t = JSON.parse(localStorage.bc_over_templates || "null");
    if (t && Array.isArray(t.templates) && t.templates.length >= 2 &&
        typeof t.threshold === "number")
      return { templates: t.templates.map(m => m.map(f => Float64Array.from(f))),
               threshold: t.threshold, when: t.when };
  } catch (e) {}
  return null;
}
function saveTemplates(templates, threshold) {
  try {
    localStorage.bc_over_templates = JSON.stringify({
      when: new Date().toISOString(), threshold: threshold,
      templates: templates.map(m => m.map(f => Array.from(f))),
    });
  } catch (e) {}
}

/* -- calibration: record 10 s, take the speech runs as OVER templates ---- */
async function calibrate(ctx) {
  const x = await ctx.mic.tapFor(10000);
  const { runs } = speechRuns(x);
  const cands = runs.filter(r => (r.endS - r.startS) >= 0.25 &&
                                 (r.endS - r.startS) <= 1.0);
  if (cands.length < 2)
    return { summary: "CALIBRATION INCOMPLETE: said-word bursts not found " +
              "(need 2+; said " + cands.length + " of " + runs.length +
              " runs) -- say OVER alone, five times, ~1/s apart, then wait" };
  const feats = cands.map(r => wordFeatures(x, r.end - Math.round(0.45 * SR), 0.45))
    .filter(f => f !== null);
  let maxD = 0, pairs = 0;
  for (let i = 0; i < feats.length; i++)
    for (let j = i + 1; j < feats.length; j++) {
      maxD = Math.max(maxD, gridDistance(feats[i], feats[j])); pairs++;
    }
  // Threshold floor 0.25, set by MEASURED separation through the real
  // browser capture chain (the first E2E: over-shaped burst 0.104,
  // every non-over burst >= 0.416) -- identical synthetic calibrations
  // otherwise produce a degenerate 0.02 that misses real variation.
  const threshold = +(Math.max(0.25, maxD * 1.25 + 0.02)).toFixed(4);
  saveTemplates(feats, threshold);
  const summary = "CALIBRATED: " + feats.length + " OVER templates " +
    "(max pairwise distance " + maxD.toFixed(3) + ", threshold " +
    threshold + ") stored; next runs measure ambient false-fires";
  await ctx.result({ summary: summary, calibrated: feats.length,
                     threshold: threshold });
  return { summary: summary, calibrated: feats.length,
           threshold: threshold };
}

/* -- ambient measure: tap, find run-ends, evaluate the conjunction -------- */
async function measure(ctx, cal) {
  const TAP_S = 45;
  const x = await ctx.mic.tapFor(TAP_S * 1000);
  const { runs } = speechRuns(x);
  const events = [];
  for (let i = 0; i < runs.length; i++) {
    const r = runs[i];
    const nextStart = i + 1 < runs.length ? runs[i + 1].startS : x.length / SR;
    const silence = nextStart - r.endS;
    const dist = matchTemplates(x, r.end, cal.templates);
    const matched = dist <= cal.threshold;
    const meanHz = runMeanF0(x, r);
    const pros = meanHz ? terminalFalling(x, r.end, meanHz)
                        : { falling: false, n: 0, slope: null };
    const fire = matched && pros.falling && silence >= FIRE_SILENCE_S;
    events.push({
      run_end_s: +r.endS.toFixed(2), dur_s: +(r.endS - r.startS).toFixed(2),
      template_distance: +dist.toFixed(4), word_matched: matched,
      terminal_falling: pros.falling, prosody_slope_st_per_s: pros.slope,
      trailing_silence_s: +silence.toFixed(2),
      near_miss: matched && pros.falling && silence >= NEAR_SILENCE_S &&
                 silence < FIRE_SILENCE_S,
      FIRE: fire,
    });
  }
  const fires = events.filter(e => e.FIRE).length;
  const nears = events.filter(e => e.near_miss).length;
  const matched = events.filter(e => e.word_matched).length;
  const falling = events.filter(e => e.terminal_falling).length;
  const hours = TAP_S / 3600;
  let wav = null, dir = null;
  try {
    wav = encodeWav(x);
    dir = await ctx.saveArtifact({
      probe: { name: meta.name, version: meta.version },
      over: { mode: "ambient_measure", tap_s: TAP_S,
              threshold: cal.threshold, calibrated_at: cal.when,
              events: events, fires: fires, near_misses: nears },
    }, wav);
  } catch (e) { /* the counts stand; the artifact save is best-effort */ }
  const summary = "ambient " + TAP_S + "s: " + runs.length + " runs, " +
    matched + " word-matched, " + falling + " terminal-falling, " +
    fires + " FIRE" + (fires === 1 ? "" : "s") + " -> " +
    (fires / hours).toFixed(1) + " false-fires/hour" +
    (nears ? " (" + nears + " near-miss at 1.2-1.5s)" : "") +
    (dir ? "  [saved " + dir + "]" : "");
  await ctx.result({ summary: summary, fires: fires, runs: runs.length,
                     false_fires_per_hour: +(fires / hours).toFixed(2),
                     near_misses: nears, events: events, artifact: dir });
  return { summary: summary };
}

function encodeWav(x) {
  const n = x.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++)
    dv.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); ws(8, "WAVE");
  ws(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true); dv.setUint32(24, SR, true);
  dv.setUint32(28, SR * 2, true); dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true); ws(36, "data"); dv.setUint32(40, n * 2, true);
  const pcm = new Int16Array(buf, 44, n);
  for (let i = 0; i < n; i++) {
    let v = Math.round(x[i] * 32767);
    pcm[i] = v > 32767 ? 32767 : v < -32768 ? -32768 : v;
  }
  return new Uint8Array(buf);
}

export async function run(ctx) {
  const cal = loadTemplates();
  if (!cal) return await calibrate(ctx);
  return await measure(ctx, cal);
}
