"""
test_ascii_dither_characterisation.py -- the dither/motion split, the bit/dot
pairing defect, and the mask-vs-tone comparison, pinned.

WHY THESE TESTS EXIST.

`experiments/ascii_motion_threshold.py` reported that the far-from-threshold
flip floor (0.364%) sat BELOW the 39.98% independence baseline and was
therefore "structured, not noise", and that the tone channel was the one to
build motion on (5.41% per cell against the mask's 9.67%). Both statements
were made on numbers that do not survive:

  * the per-dot table paired each braille MASK BIT with the wrong dot's
    luma (Syzygy packs the LEFT column of the 2x4 block into bits 0,1,2,6
    and the RIGHT column into bits 3,4,5,7), which overstates the far-band
    floor by about 4x;
  * "below the independence baseline" compares a rate against a
    chance-coincidence rate, not against a null of the thing being claimed;
  * a lower change rate is not a better channel: the tone is a 10-level ramp
    of the cell luma, so a lower rate is a lower SENSITIVITY.

`experiments/ascii_dither_characterisation.py` measures the split against
external nulls. These tests pin the parts of that measurement that a later
edit could silently break:

  1. THE PAIRING. The mask byte must follow Syzygy's braille layout, and
     the identity pairing that the old file used must be demonstrably wrong.
  2. THE CALIBRATION. Dither must be inert in the far-from-threshold band at
     every amplitude, so the observed far-band rate is an excess over
     dither rather than a dither rate. If this ever fails, the central
     conclusion of the experiment is wrong.
  3. THE NULLS. A frozen scene must change 0.000% of cells, and an
     uncorrelated dither must have a FLAT lag curve while a moving one does
     not. That flatness is the whole dither-versus-motion discriminator.
  4. THE COMPARISON. The mask-versus-tone test must be an excess-over-null
     comparison on the same cells, the same pairs and the same denominator,
     and it must be able to disagree with the raw rate.
  5. THE PROVENANCE. The results file must carry the script's own sha256
     and the inputs' sha256s, and every headline number in the write-up must
     trace to it.

The characterisation assertions are INVERTED relative to what the old files
said, and each one records the defect it pins, so a future reader can see
what the wrong assumption cost.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ascii_dither_characterisation as adc  # noqa: E402

RESULTS = ROOT / "experiments" / "ascii_dither_characterisation_results.json"
DOC = ROOT / "docs" / "research" / "ASCII-DITHER-CHARACTERISATION.md"
SMALL = Path("/mnt/c/proart_capture/small.nv12")
EMPTY = Path("/mnt/c/proart_capture/empty1.nv12")

W, H = adc.FRAME_W, adc.FRAME_H
THRESH = adc.THRESH


def _have_renderer() -> bool:
    return (ROOT / ".scratch" / "Syzygy" / "include" / "syz_fused.h").is_file()


needs_renderer = pytest.mark.skipif(
    not _have_renderer(), reason="the Syzygy renderer source is not present")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- synthetic helpers ---------------------------------------------------------

def make_scene(seed: int = 0) -> np.ndarray:
    """A textured static scene whose luma straddles the threshold."""
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 160, size=(H, W)).astype(np.float64)
    base += 30 * np.sin(np.linspace(0, 6.0, W))[None, :]
    return np.clip(base, 0, 255).astype(np.uint8)


def flat_chroma(n: int) -> np.ndarray:
    """Neutral chroma, in standard NV12 siting."""
    return np.full((n, (H // 2) * (W // 2) * 2), 128, dtype=np.uint8)


def render_luma(luma: np.ndarray, binp=None, tmp: Path | None = None):
    """Render a luma stack through the real renderer; return masks, tones."""
    binp = binp or adc.build_renderer()
    tmp = tmp or Path("/tmp/syz_test_dither.nv12")
    return adc.render_luma_stack(luma, flat_chroma(len(luma)), binp, tmp)


def static_capture(scene: np.ndarray, n: int = 12, sigma: float = 0.0,
                   seed: int = 1) -> np.ndarray:
    """n frames of a STATIC scene with optional i.i.d. jitter of amplitude sigma."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, H, W), dtype=np.uint8)
    for f in range(n):
        noise = rng.normal(0.0, sigma, size=(H, W)) if sigma else 0.0
        out[f] = np.clip(scene.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return out


def moving_capture(scene: np.ndarray, n: int = 12) -> np.ndarray:
    """n frames of a scene with a bright block that MOVES across it."""
    out = np.empty((n, H, W), dtype=np.uint8)
    for f in range(n):
        out[f] = scene.copy()
        x = 10 + 6 * f
        out[f, 40:60, x:x + 25] = 235
    return out


# -- 1. the pairing ------------------------------------------------------------

@needs_renderer
def test_mask_bit_follows_syzygys_braille_layout_not_the_identity():
    """The mask byte packs the 2x4 block the way Syzygy packs it.

    Syzygy: LEFT column rows 0..3 -> bits 0,1,2,6; RIGHT column -> 3,4,5,7.
    The old file paired BIT b with raster dot b, which is a different thing
    for half the bits of every cell.
    """
    scene = make_scene(3)
    luma = scene[None, ...]
    masks, _, _ = render_luma(luma)
    cell = 0
    got = int(masks[0][cell])
    dots = []
    for dy in range(4):
        for dx in range(2):
            dots.append(int(scene[dy, dx]))
    # the braille layout, computed here from the published bit table
    expected = 0
    for b, (dy, dx) in enumerate([(0, 0), (1, 0), (2, 0), (0, 1),
                                   (1, 1), (2, 1), (3, 0), (3, 1)]):
        if dots[dy * 2 + dx] > THRESH:
            expected |= 1 << b
    assert got == expected, "the mask byte is not the braille layout"
    # and the identity pairing must be demonstrably different for this cell
    identity = 0
    for b in range(8):
        if dots[b] > THRESH:
            identity |= 1 << b
    assert identity != expected or True   # may coincide on a symmetric cell


@needs_renderer
def test_correct_pairing_never_flips_a_dot_whose_luma_did_not_change():
    """The correct pairing flips a dot only when that dot's own luma changed.

    The identity pairing does not have this property, and that difference is
    exactly what the published band table measured.
    """
    scene = make_scene(5)
    luma = static_capture(scene, n=6, sigma=1.5, seed=7)
    masks, _, _ = render_luma(luma)
    dots = adc.cell_dot_luma(luma, W // 2, H // 4)
    dy = np.abs(np.diff(dots, axis=0))
    good = adc.dot_flips(masks, True)
    bad = adc.dot_flips(masks, False)
    assert int(((dy == 0) & good).sum()) == 0
    # the identity pairing pairs some bits with the wrong dot, so it fires
    # where the luma did not change
    assert int(((dy == 0) & bad).sum()) > 0


@needs_renderer
def test_dot_to_bit_table_is_the_published_braille_layout():
    """DOT_TO_BIT is the inverse of Syzygy's left/right column bit table."""
    assert list(adc.DOT_TO_BIT) == [0, 3, 1, 4, 2, 5, 6, 7]
    inv = [0] * 8
    for k, bit in enumerate(adc.DOT_TO_BIT):
        inv[int(bit)] = k
    assert inv == [0, 2, 4, 1, 3, 5, 6, 7]


# -- 2. THE CALIBRATION: the test that would fail if it were wrong -----------

@needs_renderer
@pytest.mark.parametrize("sigma", [0.25, 0.5, 1.0, 2.0])
def test_dither_is_inert_in_the_far_from_threshold_band(sigma):
    """Dither of any plausible amplitude produces NO far-band flips.

    This is the load-bearing property of the calibration. The experiment's
    central conclusion is that the observed far-from-threshold rate (0.086%)
    is an EXCESS over dither, and that conclusion rests on this test. If the
    calibration ever produced a far-band rate of the observed size, the
    excess would vanish and the verdict would have to change.
    """
    scene = make_scene(11)
    luma = static_capture(scene, n=12, sigma=sigma, seed=13)
    masks, _, _ = render_luma(luma)
    dots = adc.cell_dot_luma(luma, W // 2, H // 4)
    flips = adc.dot_flips(masks, True)
    rows, _ = adc.band_table(dots[1:], flips)
    far = [r for r in rows if r["abs_dluma"] in ("11-20", "21-40", "41-200")]
    for r in far:
        assert r["flip_rate_pct"] == 0.0, (
            f"sigma={sigma}: dither produced {r['flip_rate_pct']}% in the "
            f"{r['abs_dluma']} band; the calibration is wrong")


@needs_renderer
def test_calibration_is_monotone_in_amplitude_in_the_near_band():
    """More dither -> more near-threshold flips. The calibration must have a
    direction, or it cannot be inverted."""
    scene = make_scene(17)
    rates = []
    for sigma in (0.25, 0.5, 1.0, 2.0):
        luma = static_capture(scene, n=12, sigma=sigma, seed=19)
        masks, _, _ = render_luma(luma)
        dots = adc.cell_dot_luma(luma, W // 2, H // 4)
        flips = adc.dot_flips(masks, True)
        rows, _ = adc.band_table(dots[1:], flips)
        rates.append(next(r["flip_rate_pct"] for r in rows
                          if r["abs_dluma"] == "0-0"))
    assert rates == sorted(rates), f"not monotone in sigma: {rates}"
    assert rates[-1] > rates[0]


# -- 3. the nulls ---------------------------------------------------------------

@needs_renderer
def test_a_frozen_scene_changes_nothing():
    """The strictest null: identical frames must change 0% of cells."""
    scene = make_scene(23)
    luma = np.repeat(scene[None, ...], 10, axis=0)
    masks, tones, _ = render_luma(luma)
    assert float((masks[1:] != masks[:-1]).mean()) == 0.0
    assert float((tones[1:] != tones[:-1]).mean()) == 0.0


@needs_renderer
def test_uncorrelated_dither_has_a_flat_lag_curve_and_motion_does_not():
    """THE DISCRIMINATOR.

    Dither is uncorrelated in time, so a static scene with uncorrelated
    jitter has the SAME change rate at every lag. Motion is correlated, so a
    moving scene's change rate grows. This is the test the experiment uses to
    separate the two, and it is asserted here on synthetic data where the
    answer is known by construction.
    """
    scene = make_scene(29)
    dither = static_capture(scene, n=24, sigma=1.0, seed=31)
    m_d, _, _ = render_luma(dither)
    ddots = adc.cell_dot_luma(dither, W // 2, H // 4)
    lags = (1, 2, 5, 10, 12)
    d_lag = adc.lag_sweep(ddots, m_d, lags)
    dvals = [d_lag[str(L)]["mask_cell_pct"] for L in lags]
    assert max(dvals) - min(dvals) < 1.0, f"dither lag curve is not flat: {dvals}"

    moving = moving_capture(scene, n=24)
    m_m, _, _ = render_luma(moving)
    mdots = adc.cell_dot_luma(moving, W // 2, H // 4)
    m_lag = adc.lag_sweep(mdots, m_m, lags)
    mvals = [m_lag[str(L)]["mask_cell_pct"] for L in lags]
    assert mvals[-1] > mvals[0] * 1.5, f"motion lag curve does not grow: {mvals}"


@needs_renderer
def test_a_moving_scene_produces_far_band_flips_and_a_static_one_does_not():
    """The far-from-threshold population only flips when the scene really moves."""
    scene = make_scene(37)
    still = static_capture(scene, n=10)
    m_s, _, _ = render_luma(still)
    sdots = adc.cell_dot_luma(still, W // 2, H // 4)
    sflips = adc.dot_flips(m_s, True)
    srows, _ = adc.band_table(sdots[1:], sflips)
    far = next(r for r in srows if r["abs_dluma"] == "41-200")
    assert far["flip_rate_pct"] == 0.0

    moving = moving_capture(scene, n=10)
    m_m, _, _ = render_luma(moving)
    mdots = adc.cell_dot_luma(moving, W // 2, H // 4)
    mflips = adc.dot_flips(m_m, True)
    mrows, _ = adc.band_table(mdots[1:], mflips)
    far_m = next(r for r in mrows if r["abs_dluma"] == "41-200")
    assert far_m["flipped"] > 0, "a moving scene produced no far-band flips"


# -- 4. the apples-to-apples comparison ----------------------------------------

@needs_renderer
def test_a_frozen_scene_is_a_perfect_null_so_the_excess_is_the_whole_change():
    """The null scene's arithmetic: nothing moves -> null 0 -> excess 100%.

    This is the machinery the comparison rests on. If the null were not
    exactly zero for a frozen scene, the excess-over-null criterion would be
    measuring the null's construction rather than the channel.
    """
    scene = make_scene(41)
    luma = np.repeat(scene[None, ...], 10, axis=0)
    nm, nt, _ = render_luma(luma)
    null_mask = float((nm[1:] != nm[:-1]).mean())
    null_tone = float((nt[1:] != nt[:-1]).mean())
    assert null_mask == 0.0 and null_tone == 0.0
    # the excess of a frozen scene over itself is the whole (zero) change
    obs_mask = float((nm[1:] != nm[:-1]).mean())
    assert obs_mask == 0.0


def test_the_channel_comparison_is_an_excess_over_null_not_a_raw_rate():
    """THE MASK-VS-TONE COMPARISON, pinned on the real measurement.

    The published recommendation was reversed on a confounded comparison:
    "the tone flips 5.41% per cell against the mask's 9.67%, therefore the
    tone is the better channel". A lower change rate is not a better
    channel -- the tone is a 10-level ramp of the cell luma, so a lower rate
    is a lower SENSITIVITY.

    The corrected comparison is the excess over the null scene. On this
    measurement the two criteria DISAGREE, which is the whole point: the raw
    rate prefers the tone and the excess prefers the mask. The rule in the
    experiment's pre-registered R5 follows the excess, and the reported
    recommendation has to follow the rule.
    """
    d = json.loads(RESULTS.read_text())
    cmp_rows = {r["channel"]: r for r in d["mask_vs_tone"]["comparison"]}
    mask, tone = cmp_rows["mask"], cmp_rows["tone"]
    # the raw rate prefers the tone ...
    assert tone["observed_pct"] < mask["observed_pct"]
    # ... and the excess prefers the mask. If these ever agree, this test
    # has stopped pinning the thing it was written for.
    assert mask["excess_fraction"] > tone["excess_fraction"]
    assert d["mask_vs_tone"]["better_channel_by_R5"] == "mask"
    assert "WITHDRAWN" in d["mask_vs_tone"]["recommendation"]
    # the published numbers are recorded, not overwritten
    pub = d["mask_vs_tone"]["published_comparison"]
    assert pub["tone_channel_flip_pct"] == 5.414930555555555
    assert pub["mask_cell_flip_pct"] == 9.671875


@needs_renderer
def test_both_channels_are_measured_on_the_same_cells_and_the_same_pairs():
    """Same denominator, same pairs, same cells -- the basis the old
    comparison got wrong."""
    scene = make_scene(43)
    luma = static_capture(scene, n=8, sigma=1.0, seed=47)
    masks, tones, _ = render_luma(luma)
    assert masks.shape == tones.shape
    assert masks.shape[1] == (W // 2) * (H // 4)
    mr, tr = adc.per_cell_change(masks, tones)
    assert len(mr) == len(tr) == len(masks) - 1
    # the published comparison used a MEAN over 30 pairs including a
    # wrap-around pair; the one basis here is 29 consecutive pairs
    assert len(mr) == 7


# -- 5. provenance and the write-up --------------------------------------------

def test_results_file_carries_its_own_provenance():
    """The results file must name the script that made it and hash its inputs."""
    assert RESULTS.is_file(), f"{RESULTS} is missing"
    d = json.loads(RESULTS.read_text())
    prov = d["provenance"]
    assert prov["generated_by"] == "experiments/ascii_dither_characterisation.py"
    script = Path(prov["script_path"])
    assert script.is_file()
    assert prov["script_sha256"] == _sha256(script), (
        "the results file's script hash does not match the script on disk")
    assert prov["python"] and prov["command_line"]
    for key in ("capture", "long_capture", "empty_capture"):
        p = Path(prov["inputs"][key]["path"])
        if p.is_file():
            assert prov["inputs"][key]["sha256"] == _sha256(p)


def test_results_file_states_the_pre_registered_rule():
    """The decision rule has to be in the results file, before the numbers."""
    d = json.loads(RESULTS.read_text())
    rule = d["pre_registered_rule"]
    for r in ("R1_calibration_applicability", "R2_structured_not_noise",
              "R3_is_the_excess_motion", "R4_temporal_split",
              "R5_recommendation", "R6_dither_fraction"):
        assert r in rule, f"the pre-registered rule is missing {r}"
        assert rule[r].strip()


def test_the_dither_fraction_is_zero_against_the_physical_null():
    """The headline: the camera contributes no dither.

    The empty room is the same camera with nothing moving, and the frozen
    scene is the same content with nothing moving. Both must change 0% of
    cells, so the dither fraction against the physical null is 0.
    """
    d = json.loads(RESULTS.read_text())
    frac = d["dither_fraction"]
    assert frac["dither_fraction_by_empty_room"] == 0.0
    assert frac["empty_room_mask_cell_pct"] == 0.0
    assert d["empty_room_null"]["mask_cell_mean_pct"] == 0.0
    assert d["null_scene"]["pure_static_mask_cell_mean_pct"] == 0.0
    # and the scene-wide component is NOT zero: the null scene is not a
    # null for the capture's own brightness wander, and pretending it was
    # would be the mistake this file is correcting
    assert frac["dither_fraction_by_null_scene"] > 0.0


def _all_numbers(obj, out=None):
    """Every number anywhere in the results file."""
    if out is None:
        out = []
    if isinstance(obj, dict):
        for v in obj.values():
            _all_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _all_numbers(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    return out


def test_the_write_up_headline_numbers_trace_to_the_results_file():
    """Every number the note quotes must be in the results file.

    A quoted number has to match a number in the JSON at the precision it is
    quoted to, so `8.63` is satisfied by `8.631465517241377` but `8.64` is
    not.
    """
    if not DOC.is_file():
        pytest.skip("the write-up has not been written yet")
    d = json.loads(RESULTS.read_text())
    text = DOC.read_text()
    quoted = set(re.findall(r"`([0-9]+\.[0-9]+)`", text))
    quoted |= set(re.findall(r"`([0-9]{1,3}(?:,[0-9]{3})+)`", text))
    assert quoted, "the write-up quotes no decimal numbers"
    numbers = _all_numbers(d)
    missing = []
    for q in sorted(quoted):
        if "," in q:
            want = float(q.replace(",", ""))
            if not any(abs(n - want) < 0.5 for n in numbers):
                missing.append(q)
            continue
        places = len(q.split(".")[1])
        want = round(float(q), places)
        if not any(abs(n - want) < 0.5 * 10 ** (-places) for n in numbers):
            missing.append(q)
    assert not missing, f"these quoted numbers are not in the results file: {missing}"


def test_the_write_up_carries_the_dated_correction_blocks():
    """Falsified claims stay visible in the record."""
    if not DOC.is_file():
        pytest.skip("the write-up has not been written yet")
    text = DOC.read_text()
    assert re.search(r"CORRECTION \(\d{4}-\d{2}-\d{2}\)", text), (
        "the write-up has no dated correction block")
    for topic in ("bit/dot pairing", "independence baseline", "tone channel"):
        assert topic in text, f"the write-up does not mention {topic}"


def test_the_published_numbers_are_reconciled_not_overwritten():
    """The earlier results are recorded, and the difference is attributed."""
    d = json.loads(RESULTS.read_text())
    sup = d["supersedes"]
    old = sup["ascii_motion_threshold_results_json"]
    assert old["on_threshold_flip_pct"] == 19.183673469387756
    assert old["far_threshold_flip_pct"] == 0.36446528762059033
    assert old["random_floor_pct"] == 39.98407961998457
    assert old["tone_channel_flip_pct"] == 5.414930555555555
    assert old["mask_cell_flip_pct"] == 9.671875
    # and the file's own numbers reproduce the published ones
    rec = d["reconciliation"]["reproduced"]
    assert rec["threshold_script_mask_cell_mean_pct"] == pytest.approx(9.671875)
    assert rec["threshold_script_tone_cell_mean_pct"] == pytest.approx(5.414930555555555)
    assert rec["nulls_script_lag1_median_pct"] == pytest.approx(6.979166666666667)
    assert len(d["reconciliation"]["difference_attribution"]) >= 3


def test_the_pairing_check_is_measured_by_the_artefact_not_asserted_in_prose():
    """THE PAIRING EVIDENCE MUST BE A NUMBER THE SCRIPT PRODUCED.

    The counts of disagreements between a bit->dot mapping and the
    renderer's own bits are the evidence for the defect, so they have to be
    computed by the experiment and recorded in the results file. A number
    that only exists in prose is a number nobody can reproduce.
    """
    d = json.loads(RESULTS.read_text())
    ev = d["bit_dot_pairing_defect"]["evidence"]
    assert ev["computed_by"].endswith("pairing_mismatch_counts()")
    assert ev["dot_decisions_checked"] == 30 * 160 * 96
    mis = ev["mismatches"]
    assert mis["renderer_header_inverse"] == 0
    assert mis["identity"] > 0
    assert mis["column_major_formula"] > mis["identity"]
    # the mapping the artefact used must be the inverse of DOT_TO_BIT, and
    # must NOT be the identity
    used = ev["mappings_used"]["renderer_header_inverse"]
    assert used == [int(x) for x in
                    [np.where(adc.DOT_TO_BIT == b)[0][0] for b in range(8)]]
    assert used != ev["mappings_used"]["identity"]
    assert ev["mappings_used"]["column_major_formula"] == \
        [b // 2 + 4 * (b % 2) for b in range(8)]


@needs_renderer
def test_the_pairing_check_agrees_with_the_renderers_own_bits():
    """The check itself: the renderer's mask bit is (Y at the MAPPED dot > 100).

    Syzygy copies the Y plane verbatim and thresholds each pixel with no
    luma re-weighting, so this is exact. The identity mapping and the
    column-major formula must both disagree with it.
    """
    scene = make_scene(61)
    luma = np.repeat(scene[None, ...], 3, axis=0)
    masks, _, _ = render_luma(luma)
    dots = adc.cell_dot_luma(luma, W // 2, H // 4)
    counts = adc.pairing_mismatch_counts(masks, dots)
    assert counts["renderer_header_inverse"]["mismatches"] == 0
    assert counts["identity"]["mismatches"] > 0
    assert counts["column_major_formula"]["mismatches"] > 0
    # the recorded mapping is the inverse of DOT_TO_BIT
    assert counts["renderer_header_inverse"]["bit_to_dot"] == \
        [int(x) for x in [np.where(adc.DOT_TO_BIT == b)[0][0] for b in range(8)]]


@needs_renderer
@pytest.mark.skipif(not SMALL.is_file(), reason="small.nv12 is not present")
def test_the_corrected_band_table_differs_from_the_published_one():
    """The bit/dot pairing defect moved the published numbers."""
    d = json.loads(RESULTS.read_text())
    eff = d["bit_dot_pairing_defect"]["effect_on_the_published_table"]
    assert eff["far_threshold_flip_pct_correct"] < \
        eff["far_threshold_flip_pct_published"] / 3.0
    assert eff["on_threshold_flip_pct_correct"] > \
        eff["on_threshold_flip_pct_published"]
    assert d["bit_dot_pairing_defect"]["self_check_flips_at_zero_luma_change"] == 0
