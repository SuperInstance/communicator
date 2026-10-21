/* sonar-band -- ROOM SONAR STEP 1 (docs/research/BRIEF-ROOM-SONAR.md).
 * The first tenant of the evolve lab. Plays an exponential sine sweep
 * 16k -> 22k (3 s) and a pink-noise burst (0.5 s) through the REAL
 * speakers (railed by the shell: gain ceiling, duty ceiling), taps the
 * REAL mic around them, and measures what survives the chain per 250-Hz
 * band. Verdict rule stated, not tuned: a band is USABLE when its sweep
 * SNR >= 10 dB and its transfer is within 40 dB of the best band; the
 * usable band is the widest contiguous run in 15-24 kHz. If this
 * hardware is deaf above 16 kHz, the lane closes with a number.
 * The raw capture + this report save as a probe artifact session for
 * MiniMax's step-2 deconvolution.
 *
 * Bump meta.version to hot-swap into already-open pages (no reload).
 */
export const meta = { name: "sonar-band", version: 2 };

const F0 = 16000, F1 = 22000, SWEEP_S = 3.0, PINK_S = 0.5, GAP_S = 0.4;
const FFT_N = 8192, BAND_HZ = 250, SNR_MIN_DB = 10, TRANS_SPAN_DB = 40;
const GAIN = 0.25;

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

function spectrumOf(buf, offset, len, sr) {
  const nWin = Math.max(1, Math.floor(len / FFT_N));
  const acc = new Float64Array(FFT_N / 2);
  for (let w = 0; w < nWin; w++) {
    const off = offset + w * FFT_N;
    if (off + FFT_N > buf.length) break;
    const re = new Float64Array(FFT_N), im = new Float64Array(FFT_N);
    for (let i = 0; i < FFT_N; i++)
      re[i] = buf[off + i] *
        (0.5 - 0.5 * Math.cos(2 * Math.PI * i / FFT_N));   // Hann
    const m = fftMag(re, im);
    for (let i = 0; i < FFT_N / 2; i++) acc[i] += m[i];
  }
  for (let i = 0; i < acc.length; i++) acc[i] /= nWin;
  return acc;
}

function makeSweep(sr) {
  const n = Math.round(SWEEP_S * sr);
  const out = new Float32Array(n);
  const ratio = F1 / F0, ln = Math.log(ratio);
  for (let i = 0; i < n; i++) {
    const t = i / n;
    const ph = 2 * Math.PI * F0 * (SWEEP_S * (Math.pow(ratio, t) - 1) / ln);
    let v = Math.sin(ph);
    const f = Math.min(i, n - 1 - i) / (0.01 * sr);
    v *= Math.min(1, Math.max(0, f));
    out[i] = v;
  }
  return out;
}

function makePink(sr) {
  const n = Math.round(PINK_S * sr);
  const out = new Float32Array(n);
  let b0 = 0, b1 = 0, b2 = 0;
  for (let i = 0; i < n; i++) {
    const w = Math.random() * 2 - 1;
    b0 = 0.99765 * b0 + w * 0.0990460;
    b1 = 0.96300 * b1 + w * 0.2965164;
    b2 = 0.57000 * b2 + w * 1.0526913;
    out[i] = (b0 + b1 + b2 + w * 0.1848) * 0.16;
  }
  for (let i = 0; i < n; i++) {
    const f = Math.min(i, n - 1 - i) / (0.01 * sr);
    out[i] *= Math.min(1, Math.max(0, f));
  }
  return out;
}

function buildWav(samples, sr) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const ws = (off, s) => { for (let i = 0; i < s.length; i++)
    dv.setUint8(off + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); ws(8, "WAVE");
  ws(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true); dv.setUint32(24, sr, true);
  dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true); ws(36, "data");
  dv.setUint32(40, n * 2, true);
  const pcm = new Int16Array(buf, 44, n);
  for (let i = 0; i < n; i++) {
    let v = Math.round(samples[i] * 32767);
    pcm[i] = v > 32767 ? 32767 : v < -32768 ? -32768 : v;
  }
  return new Uint8Array(buf);
}

export async function run(ctx) {
  const sr = ctx.shell.state().rate || 48000;
  const micName = ctx.shell.state().mic;

  // 0.4 s of room, then the sweep, a gap, then the pink burst
  const noise = await ctx.mic.tapFor(400);
  const sweep = makeSweep(sr);
  const pink = makePink(sr);

  const playSweepP = ctx.playStimulus(sweep, { gain: GAIN });
  const sweepRet = await ctx.mic.tapFor(SWEEP_S * 1000 + 300);
  const playSweep = await playSweepP;
  if (playSweep.refused)
    return { summary: "REFUSED by shell rail: " + playSweep.reason };

  await ctx.mic.tapFor(GAP_S * 1000);   // the gap, tapped and discarded
  const playPinkP = ctx.playStimulus(pink, { gain: GAIN });
  const pinkRet = await ctx.mic.tapFor(PINK_S * 1000 + 300);
  const playPink = await playPinkP;

  // analysis: transfer = |return| / |reference stimulus|, per band
  const noiseSpec = spectrumOf(noise, 0, noise.length, sr);
  const sweepSpec = spectrumOf(sweepRet, 0, sweepRet.length, sr);
  const refSweep = spectrumOf(sweep, 0, sweep.length, sr);
  const pinkSpec = spectrumOf(pinkRet, 0, pinkRet.length, sr);
  const refPink = spectrumOf(pink, 0, pink.length, sr);
  const binHz = sr / FFT_N;
  const med = (a) => a.slice().sort((x, y) => x - y)
    [Math.floor(a.length / 2)];
  // v2 HONESTY FIX, taught by the first real run: a band the sweep never
  // excited must score NO EXCITATION, not garbage-through-a-floor -- the
  // v1 verdict "22-24 kHz" was an artifact of dividing by a ~zero
  // reference above the sweep's top. Only bands the reference actually
  // drove (within 60 dB of the reference's own peak) are scored.
  let refMax = 0;
  for (let b = Math.floor(15000 / binHz);
       b < Math.min(FFT_N / 2, Math.floor(24000 / binHz)); b++)
    refMax = Math.max(refMax, refSweep[b] || 0);
  const refFloor = refMax * Math.pow(10, -60 / 20);
  const rows = [];
  for (let f = 15000; f < 24000; f += BAND_HZ) {
    const tr = [], pn = [], nz = [], rf = [];
    for (let b = Math.floor(f / binHz);
         b < Math.floor((f + BAND_HZ) / binHz); b++) {
      if (b < 1 || b >= FFT_N / 2) continue;
      tr.push(sweepSpec[b] / Math.max(refSweep[b], 1e-12));
      pn.push(pinkSpec[b] / Math.max(refPink[b], 1e-12));
      nz.push(Math.max(noiseSpec[b], 1e-12));
      rf.push(refSweep[b] || 0);
    }
    if (!tr.length) continue;
    if (med(rf) < refFloor) {
      rows.push({ f0: f, f1: f + BAND_HZ, no_excitation: true,
        note: "the sweep drove no energy here; not scored" });
      continue;
    }
    rows.push({
      f0: f, f1: f + BAND_HZ,
      transfer_db: + (20 * Math.log10(Math.max(med(tr), 1e-12))).toFixed(1),
      snr_db: + (20 * Math.log10(Math.max(med(tr) / med(nz), 1e-6))).toFixed(1),
      pink_snr_db: + (20 * Math.log10(
        Math.max(med(pn) / med(nz), 1e-6))).toFixed(1),
    });
  }
  const bestTr = Math.max(...rows.map(r => r.transfer_db));
  for (const r of rows)
    r.usable = r.snr_db >= SNR_MIN_DB &&
               r.transfer_db >= bestTr - TRANS_SPAN_DB;
  let bestRun = null, i = 0;
  while (i < rows.length) {
    if (!rows[i].usable) { i++; continue; }
    let j = i;
    while (j < rows.length && rows[j].usable) j++;
    if (!bestRun || j - i > bestRun[1] - bestRun[0]) bestRun = [i, j];
    i = j;
  }
  const band = bestRun ?
    (rows[bestRun[0]].f0 / 1000) + "-" +
    (rows[bestRun[1] - 1].f1 / 1000) + " kHz" : "NONE";
  const summary = "usable band: " + band + "  (SNR>=10dB & within " +
    TRANS_SPAN_DB + "dB of best; mic: " + micName + ")";

  // save the raw capture for the step-2 deconvolution (MiniMax)
  const raw = new Float32Array(noise.length + sweepRet.length +
    Math.round(GAP_S * sr) + pinkRet.length);
  raw.set(noise, 0);
  raw.set(sweepRet, noise.length);
  raw.set(pinkRet, noise.length + sweepRet.length +
    Math.round(GAP_S * sr));
  let dir = null;
  try {
    dir = await ctx.saveArtifact({
      probe: { name: meta.name, version: meta.version },
      sonar: { stimulus: { sweep: { type: "exponential_sine",
        f0_hz: F0, f1_hz: F1, duration_s: SWEEP_S, gain: playSweep.gainUsed,
        fades_ms: 10 }, pink: { type: "pink_burst", duration_s: PINK_S,
        gain: playPink.gainUsed || 0, fades_ms: 10 },
        schedule_s: { pre: 0.4, sweep: SWEEP_S, gap: GAP_S, pink: PINK_S },
        sample_rate: sr },
        analysis: { fft_n: FFT_N, window: "hann", band_hz: BAND_HZ,
          rule: "usable = sweep SNR >= " + SNR_MIN_DB +
            " dB and transfer within " + TRANS_SPAN_DB +
            " dB of best; widest contiguous run in 15-24 kHz" },
        bands: rows, usable_band: band },
    }, buildWav(raw, sr));
  } catch (err) { /* the report stands; note the save failure */ }

  await ctx.result({ summary: summary, usable_band: band, bands: rows,
    artifact: dir });
  return { summary: summary + (dir ? "  [saved " + dir + "]" : "") };
}
