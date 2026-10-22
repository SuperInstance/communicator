/* chirp-cal -- TAKE-2's bench measurement (BRIEF-DUAL-CAPTURE-TAKE2
 * rule 3): "a bench pass sets the chirp level from measurement (what
 * the witness actually receives at room distance), not from a
 * constants file." This probe plays the ANSWER chirp (the same 1-4 kHz
 * 0.6 s linear chirp the evolve shell answers with) at a railed ladder
 * of levels, measures what the witness MIC receives for each, and
 * writes the lowest level achieving >= 30 dB loop-SNR into the
 * speakers lens file (chirp_gain) via /api/lens. Runs on the LIVE
 * witness page via the probe manifest -- no reload, one quiet pass
 * (3 chirps x 0.6 s at modest gains).
 *
 * Bump meta.version to hot-swap into already-open pages (no reload).
 */
export const meta = { name: "chirp-cal", version: 1 };

const LADDER = [0.05, 0.12, 0.25];
const TARGET_SNR_DB = 30;

export async function run(ctx) {
  const sr = ctx.shell.state().rate || 48000;
  const results = [];
  for (const gain of LADDER) {
    const play = await ctx.playStimulus(makeChirp(0.6, sr), { gain });
    if (play.refused)
      return { summary: "REFUSED by shell rail: " + play.reason };
    const ret = await ctx.mic.tapFor(1000);
    let peak = 0, floor = 1e9;
    const W = 960;
    for (let i = 0; i + W <= ret.length; i += W) {
      let s = 0;
      for (let j = i; j < i + W; j++) s += ret[j] * ret[j];
      const r = Math.sqrt(s / W);
      peak = Math.max(peak, r);
      floor = Math.min(floor, r);
    }
    const snr = 20 * Math.log10((peak + 1e-12) / (floor + 1e-12));
    results.push({ gain, played: play.gainUsed, snr_db: +snr.toFixed(1) });
  }
  const ok = results.filter(r => r.snr_db >= TARGET_SNR_DB);
  const chosen = ok.length ? ok[0].gain : null;
  let lensNote = "";
  if (chosen) {
    try {
      const r = await fetch("/api/lens", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sensor: "speakers",
          updates: { normals: {
            ...(await (await fetch("/api/lenses")).json())
              .lenses.speakers.normals, chirp_gain: chosen } },
          verified_by: "chirp-cal probe" }) });
      lensNote = r.ok ? "" : " (lens write failed)";
    } catch (e) { lensNote = " (lens write failed)"; }
  }
  const summary = "bench chirp levels: " +
    results.map(r => `g${r.played}=${r.snr_db}dB`).join(", ") +
    (chosen ? ` -> chirp_gain ${chosen} (>=${TARGET_SNR_DB}dB at the mic)`
            : ` -> none reached ${TARGET_SNR_DB}dB; rail ceiling too low`) +
    lensNote;
  await ctx.result({ summary: summary, ladder: results, chosen: chosen });
  return { summary: summary };
}

function makeChirp(durS, sr) {
  const n = Math.round(durS * sr);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / n;
    out[i] = Math.sin(2 * Math.PI * (1000 + 3000 * t / 2) * t * durS) *
      Math.min(1, Math.min(i, n - 1 - i) / (0.01 * sr));
  }
  return out;
}
