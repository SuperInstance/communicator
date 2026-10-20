"""
Tests for experiments/pose_feature_search.py -- can any statistic separate
facing from turned at 160x96?

WHY THESE TESTS EXIST.

`pose_test.py` reported a facing-vs-turned difference of 9.182 against a
time-shifted control of 9.178 and concluded "no pose signal". That result was
ambiguous: the number was equally consistent with the head turn never having
happened and with the statistic being blind to head pose. A feature search
is the way to settle it, and a feature search is worthless unless the
instrument is known to work.

So the tests here pin three things:

1. THE INSTRUMENT WORKS ON A KNOWN DIFFERENCE. A synthetic frontal image
   must score MORE frontal than a synthetic asymmetric one, and the whole
   panel must find a frontal-vs-turned difference in synthetic captures whose
   answer is known. A panel that cannot see a difference that is definitely
   there tells you nothing when it reports that it cannot see one.

2. THE CONTROLS BITE. The empty room must show NO separation: a statistic
   that separates two blocks of an empty room is reading the banding, not
   the person, and the empty room is the control that has already killed
   one project claim (the ASCII character delta).

3. THE VERDICT CAN COME OUT THE OTHER WAY. A separation that does not
   exceed its own within-condition split must produce the verdict "no
   signal", and a verdict must cite the number behind it. A verdict function
   that cannot fail is not a result.

Every fixture here is synthetic, so the suite runs anywhere with no capture
drive attached.
"""

from __future__ import annotations
import pytest as _pytest  # noqa: E402
_pytest.importorskip("numpy", reason="exercises code needing numpy")  # noqa: E402

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import pose_feature_search as pfs      # noqa: E402

H, W = pfs.FRAME_H, pfs.FRAME_W


# -- synthetic fixtures, built here rather than borrowed from the real
# -- captures, so the suite runs anywhere --------------------------------

def _grid():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    return yy, xx


def frontal_image(cx=80.0, rx=15.0, ry=20.0, amp=60.0):
    """A blob that is symmetric about the vertical line x = cx."""
    yy, xx = _grid()
    blob = (((xx - cx) / rx) ** 2 + ((yy - H / 2) / ry) ** 2) <= 1.0
    return amp * blob


def asymmetric_image(cx=100.0, rx=15.0, ry=20.0, amp=60.0):
    """The same blob plus a one-sided bump: no vertical axis makes it match
    its own mirror."""
    yy, xx = _grid()
    blob = (((xx - cx) / rx) ** 2 + ((yy - H / 2) / ry) ** 2) <= 1.0
    bump = (((xx - (cx - 45.0)) / 8.0) ** 2
            + ((yy - (H / 2 + 12.0)) / 9.0) ** 2) <= 1.0
    return amp * blob + 45.0 * bump


def still_capture(image, n=6):
    """n identical frames of one image, as a capture."""
    return np.repeat(np.asarray(image, dtype=np.float32)[None, ...], n, axis=0)


def noise_capture(n=60, seed=0, band=True):
    """A room with nobody in it: per-row flicker banding plus pixel noise."""
    rng = np.random.default_rng(seed)
    yy, _ = _grid()
    rows = np.arange(H, dtype=np.float64)
    out = np.empty((n, H, W), dtype=np.float32)
    for t in range(n):
        if band:
            b = 20.0 * np.sin(2 * np.pi * (t % 4) / 4.0
                               + 6.0 * np.pi * rows / H)
        else:
            b = np.zeros(H)
        out[t] = 100.0 + b[:, None] + rng.normal(0.0, 1.0, (H, W))
    return out


def static_features(image):
    """The symmetry features of one still image, via the module's own code."""
    R = pfs.row_remove(still_capture(image))
    ff = pfs.frame_features(R)
    return {k: float(np.median(v)) for k, v in ff.items() if v.size}


# -- 1: the instrument must work on a known difference --------------------

class TestInstrumentWorks:
    def test_mirrored_image_scores_more_frontal_than_an_asymmetric_one(self):
        """
        THE positive control, at the level of a single image.

        A mirrored (frontal) subject must be scored MORE frontal than an
        asymmetric (turned) one. If a symmetry statistic cannot tell a
        symmetric picture from an asymmetric one, then its verdict on the
        real captures is worth nothing -- and "no signal" would mean "the
        instrument is blind", not "the pose is not there".
        """
        frontal = static_features(frontal_image())
        asym = static_features(asymmetric_image())

        # "more frontal" = more symmetric about its own best axis
        assert frontal["mirror_resid_static"] < asym["mirror_resid_static"], (
            f"the mirrored image scores {frontal['mirror_resid_static']:.5f} "
            f"mirror residual against the asymmetric image's "
            f"{asym['mirror_resid_static']:.5f}; the mirrored one should "
            f"score LOWER (more frontal)")
        # "more frontal" = higher correlation with its own horizontal mirror
        assert frontal["mirror_cov"] > asym["mirror_cov"], (
            f"the mirrored image scores mirror_cov "
            f"{frontal['mirror_cov']:.5f} against the asymmetric image's "
            f"{asym['mirror_cov']:.5f}; the mirrored one should score HIGHER")
        # "more frontal" = less asymmetric about the frame centre
        assert frontal["asym_static_centre"] < asym["asym_static_centre"], (
            f"the mirrored image is {frontal['asym_static_centre']:.5f} "
            f"asymmetric about the centre against the asymmetric image's "
            f"{asym['asym_static_centre']:.5f}; the mirrored one should "
            f"score LOWER")

    def test_the_best_fit_mirror_axis_finds_a_known_symmetry_axis(self):
        """
        The axis is SEARCHED, not assumed to be the frame centre, so it has
        to be able to find an axis that is not the centre.
        """
        prof = np.zeros(W)
        prof[95:106] = 1.0          # a symmetric bump about x = 100
        resid, axis = pfs.mirror_fit(prof[None, :])
        assert abs(int(axis[0]) - 100) <= 1, (
            f"the best-fit mirror axis of a bump symmetric about x=100 was "
            f"found at x={int(axis[0])}")
        assert resid[0] < 0.05, (
            f"a perfectly symmetric profile should have a near-zero mirror "
            f"residual, got {resid[0]:.5f}")

    def test_the_mirror_fit_is_not_trivially_won_by_the_frame_edge(self):
        """
        A degenerate guard on the mirror fit.

        Mirroring about column 0 compares one sample with itself, so an
        implementation that SKIPS the out-of-range terms finds that axis
        perfect for every profile and reports a symmetry that is not there.
        The mirrored value must be treated as zero outside the frame, so an
        edge mirror is penalised.
        """
        prof = np.zeros(W)
        prof[0:6] = 1.0              # a bump hard against the left edge
        resid, axis = pfs.mirror_fit(prof[None, :])
        assert axis[0] != 0 or resid[0] > 0.5, (
            "mirroring about column 0 must not look perfect for a profile "
            f"that is not symmetric (residual {resid[0]:.5f})")
        assert resid[0] > 0.0

    def test_the_panel_finds_a_known_difference_in_synthetic_captures(self):
        """
        The whole pipeline, end to end, on frames whose answer is known:
        a synthetic frontal blob against a synthetic turned one.

        If the panel cannot find THIS difference, its negative result on
        the real captures would be a statement about the panel, not about
        the capture.
        """
        feats = pfs.run_synthetic_panel(n=120)
        signals = [f["name"] for f in feats.values() if f["verdict"] == "signal"]
        assert signals, (
            "the panel found no difference between synthetic frontal and "
            "synthetic turned captures whose answer is known; the instrument "
            "is blind and its verdict on the real captures means nothing")
        assert "mirror_cov" in signals, (
            "the panel should find the mirror-covariance difference between "
            f"synthetic frontal and turned captures; it found {signals}")

    def test_the_synthetic_panel_is_reproducible(self):
        """A positive control that changes between runs is not a control."""
        a = pfs.run_synthetic_panel(n=80)
        b = pfs.run_synthetic_panel(n=80)
        for name, f in a.items():
            assert f["sep_ab"] == pytest.approx(b[name]["sep_ab"], rel=1e-9), (
                f"{name} moved between two identical synthetic runs "
                f"({f['sep_ab']} vs {b[name]['sep_ab']})")


# -- 2: the controls must bite -------------------------------------------

class TestControlsBite:
    def test_the_empty_room_control_path_returns_no_separation(self):
        """
        THE empty-room control, on synthetic noise.

        Two blocks of an empty room must not separate, and the panel must
        report "no signal" for a comparison between two identical captures.
        The empty room is the control that has already killed one project
        claim, so the path that computes it has to be exercised.
        """
        room = noise_capture(n=80, seed=7)
        conds = {"A": pfs.condition_features(room),
                 "B": pfs.condition_features(room.copy()),
                 "C": pfs.condition_features(room.copy())}
        blocks = [pfs.condition_features(noise_capture(n=80, seed=100 + i))
                  for i in range(3)]
        feats = pfs.build_analysis(conds, blocks)

        assert set(feats) == {n for n, _, _ in pfs.FEATURE_SPECS}
        for name, f in feats.items():
            assert f["sep_ab"] == pytest.approx(0.0, abs=1e-12), (
                f"{name}: two identical captures separated by "
                f"{f['sep_ab']}")
            assert f["verdict"] == "no signal", (
                f"{name}: two identical captures were given the verdict "
                f"'{f['verdict']}'")

    def test_the_empty_room_control_grows_when_the_empty_room_changes(self):
        """
        The control has to be able to SEE something, or "no separation in the
        empty room" would be true of any statistic at all.

        Here one block of the empty room gets a bright patch that moves, so
        the empty-room separation for the profile features must become
        clearly non-zero.
        """
        quiet = pfs.condition_features(noise_capture(n=80, seed=11))
        busy_frames = noise_capture(n=80, seed=12)
        yy, xx = _grid()
        patch = (((xx - 60) / 10.0) ** 2 + ((yy - H / 2) / 12.0) ** 2) <= 1.0
        for t in range(len(busy_frames)):
            busy_frames[t] += 50.0 * patch
        busy = pfs.condition_features(busy_frames)

        scales = pfs.compute_scales({"A": quiet}, [quiet, busy])
        empty = pfs.max_pair_sep(
            [pfs._values("asym_static_centre", "frame", quiet),
             pfs._values("asym_static_centre", "frame", busy)],
            scales["asym_static_centre"])
        assert empty > 0.5, (
            "the empty-room control did not notice a bright moving patch "
            f"appearing in one block of an empty room (separation {empty:.3f})")

    def test_row_removal_flattens_an_empty_room(self):
        """
        The banding is a per-row offset, so removing each row's median
        across the columns must leave an empty room flat. If this fails,
        every feature in the panel is reading the banding.
        """
        R = pfs.row_remove(noise_capture(n=40, seed=3))
        col = R.reshape(len(R), -1).std(axis=1)
        assert float(np.median(col)) < 2.0, (
            "row removal left an empty room with a column structure of "
            f"{float(np.median(col)):.3f} counts; it should be flat")

    def test_the_permutation_null_bounds_a_separation_that_is_not_there(self):
        """
        Two IDENTICAL samples must not separate, and the null must not be
        smaller than the separation it is supposed to bound.
        """
        rng = np.random.default_rng(5)
        a = rng.normal(0.0, 1.0, 400)
        b = a.copy()
        scale = pfs.robust_scale(np.concatenate([a, b]))
        observed = pfs.sep(a, b, scale)
        p95 = pfs.permutation_null(a, b, scale, trials=100, seed=1)
        assert observed == pytest.approx(0.0, abs=1e-12)
        assert p95 >= observed

    def test_a_real_difference_beats_the_permutation_null(self):
        """The other direction: the null must not swallow a real effect."""
        rng = np.random.default_rng(6)
        a = rng.normal(0.0, 1.0, 400)
        b = rng.normal(4.0, 1.0, 400)
        scale = pfs.robust_scale(np.concatenate([a, b]))
        observed = pfs.sep(a, b, scale)
        p95 = pfs.permutation_null(a, b, scale, trials=100, seed=2)
        assert observed > p95, (
            f"a separation of {observed:.3f} between two well-separated "
            f"samples did not beat the permutation null of {p95:.3f}")

    def test_the_permutation_null_actually_bites(self):
        """
        The null must have TEETH: a separation drawn from the null itself
        should exceed its own 95th percentile only about five percent of the
        time. A null that everything passes is not a null.

        (It is not exactly five percent -- the observed separation is one
        draw from the same distribution as the null, so the rate is a
        property of the estimator -- but it must not be "always passes".)
        """
        rng = np.random.default_rng(17)
        trials = 40
        beats = 0
        for i in range(40):
            a = rng.normal(0.0, 1.0, 200)
            b = rng.normal(0.0, 1.0, 200)
            scale = pfs.robust_scale(np.concatenate([a, b]))
            observed = pfs.sep(a, b, scale)
            p95 = pfs.permutation_null(a, b, scale, trials=trials,
                                       seed=1000 + i)
            if observed > p95:
                beats += 1
        assert beats <= 8, (
            f"{beats} of 40 separations from the null beat their own 95th "
            "percentile; the permutation null is not constraining anything")

    def test_a_constant_feature_cannot_separate_anything(self):
        """
        A feature that does not vary has no scale, so it must report zero
        separation rather than an infinite one.
        """
        const = np.full(50, 3.0)
        assert pfs.robust_scale(const) == 0.0
        assert pfs.sep(const, const + 1.0, pfs.robust_scale(const)) == 0.0


# -- 3: the verdict must be able to come out the other way ---------------

class TestVerdict:
    def test_verdict_is_no_signal_when_the_separation_does_not_beat_the_split(self):
        """
        THE requirement: a separation that does not exceed the
        within-condition split must produce "no signal", not a hedge.
        """
        v = pfs.feature_verdict("some_feature", sep_ab=1.000, within=1.000,
                                empty=0.200)
        assert v["verdict"] == "no signal"
        assert v["margin"] == pytest.approx(0.0)

    def test_verdict_is_no_signal_when_the_empty_room_beats_it(self):
        """The other required control: the empty room."""
        v = pfs.feature_verdict("some_feature", sep_ab=0.400, within=0.100,
                                empty=0.900)
        assert v["verdict"] == "no signal"
        assert v["margin"] < 0

    def test_verdict_cites_the_number_behind_it(self):
        """
        A verdict that cannot be checked against the table is not a result.
        Every verdict string must carry the numbers that produced it.
        """
        v = pfs.feature_verdict("some_feature", sep_ab=1.234, within=0.500,
                                empty=0.200)
        assert "1.234" in v["detail"] and "0.500" in v["detail"]
        assert "some_feature" in v["detail"] or "A-vs-B" in v["detail"]

    def test_verdict_is_no_signal_when_the_permutation_null_is_not_beaten(self):
        """The label-permutation null gates the verdict."""
        v = pfs.feature_verdict("some_feature", sep_ab=1.000, within=0.100,
                                empty=0.100, perm95=1.500)
        assert v["verdict"] == "no signal"
        assert "permutation" in v["detail"]

    def test_verdict_is_no_signal_when_the_spatial_null_is_not_beaten(self):
        """The spatial null gates the verdict: a statistic that only works
        where the subject happened to sit is not a pose statistic."""
        v = pfs.feature_verdict("some_feature", sep_ab=1.000, within=0.100,
                                empty=0.100, spatial=1.200)
        assert v["verdict"] == "no signal"
        assert "spatial" in v["detail"]

    def test_a_feature_that_also_splits_the_two_facing_conditions_is_confounded(self):
        """
        A and C are BOTH facing. A statistic that separates them as well is
        measuring motion or the difference between two sessions, not pose.
        """
        v = pfs.feature_verdict("some_feature", sep_ab=2.000, within=0.100,
                                empty=0.100, perm95=0.100, spatial=0.100,
                                sep_ac=1.500, sep_bc=2.000)
        assert v["verdict"] == "confounded"
        assert "pose" in v["detail"]

    def test_a_pose_consistent_feature_is_a_signal(self):
        """
        The other direction, so the function is not stuck: B differs from
        both facings, the two facings agree, and every control is cleared.
        """
        v = pfs.feature_verdict("some_feature", sep_ab=2.000, within=0.100,
                                empty=0.100, perm95=0.100, spatial=0.100,
                                sep_ac=0.100, sep_bc=2.000)
        assert v["verdict"] == "signal"
        assert v["margin"] == pytest.approx(1.9)

    def test_the_summary_can_say_no_signal(self):
        """The summary is allowed -- and required -- to print "no signal"."""
        feats = {name: {"name": name, "verdict": "no signal", "margin": -0.5,
                        "sep_ab": 0.1, "within": 0.9, "empty": 0.2}
                 for name, _, _ in pfs.FEATURE_SPECS}
        s = pfs.summarise(feats)
        assert s["signals"] == []
        assert "NO SIGNAL" in s["verdict"]

    def test_the_margin_is_the_separation_minus_the_largest_control(self):
        """
        The headline is the MARGIN over the controls, not the raw
        separation, so the arithmetic of the margin is pinned.
        """
        feats = pfs.run_synthetic_panel(n=80)
        for name, f in feats.items():
            assert f["margin"] == pytest.approx(
                f["sep_ab"] - max(f["within"], f["empty"]), abs=1e-9), (
                f"{name}: margin {f['margin']} is not the separation "
                f"{f['sep_ab']} minus the largest control "
                f"{max(f['within'], f['empty'])}")

    def test_every_feature_reports_every_control(self):
        """
        A feature that quietly omits a control is the failure mode this
        project has already produced three times: a verdict that
        contradicts its own printed output.
        """
        feats = pfs.run_synthetic_panel(n=80)
        for name, kind, _ in pfs.FEATURE_SPECS:
            f = feats[name]
            for key in ("sep_ab", "within", "within_a", "within_b", "empty",
                        "margin", "ratio", "verdict", "detail", "scale",
                        "median_a", "median_b", "median_c"):
                assert key in f, f"{name}: the panel does not report {key}"
            assert f["verdict"] in ("signal", "confounded", "no signal")
            if kind == "frame":
                assert f["perm95"] is not None, (
                    f"{name}: a per-frame feature must report its "
                    "label-permutation null")
            assert f["margin"] == pytest.approx(
                f["sep_ab"] - max(f["within"], f["empty"]), abs=1e-9)


# -- 4: the script runs, and says so when the captures are missing -------

class TestScriptEndToEnd:
    def test_the_script_runs_and_says_the_captures_are_absent(self, tmp_path,
                                                               monkeypatch,
                                                               capsys):
        """
        The script must run to completion even when the real captures are
        not there, and it must say clearly that they are absent. A silent
        fallback that prints a table of synthetic numbers as if they were
        measurements would be the worst possible failure.
        """
        missing = tmp_path / "not_here.nv12"
        monkeypatch.setattr(sys, "argv", [
            "pose_feature_search.py",
            "--a", str(missing), "--b", str(missing), "--c", str(missing),
            "--empty", str(missing),
        ])
        rc = pfs.main()
        out = capsys.readouterr().out
        assert rc == 0
        assert "THE REAL CAPTURES ARE ABSENT" in out
        assert "SYNTHETIC ONLY" in out

    def test_the_script_writes_no_results_file_by_default(self, tmp_path,
                                                          monkeypatch):
        """
        The owning task for this experiment restricts writes to the script
        and its test, so nothing may be written unless it is asked for.
        """
        before = set(p.name for p in tmp_path.iterdir())
        monkeypatch.setattr(sys, "argv", [
            "pose_feature_search.py",
            "--a", str(tmp_path / "x.nv12"), "--b", str(tmp_path / "x.nv12"),
            "--c", str(tmp_path / "x.nv12"), "--empty", str(tmp_path / "x.nv12"),
        ])
        pfs.main()
        assert set(p.name for p in tmp_path.iterdir()) == before

    def test_the_feature_panel_covers_the_required_candidates(self):
        """
        The panel is not free to shrink. It has to contain the candidates
        the experiment was specified to try, so that "no signal" is a
        statement about that panel and not about a smaller one.
        """
        names = {n for n, _, _ in pfs.FEATURE_SPECS}
        for required in ("asym_static_centre", "asym_motion_centre",
                         "mirror_resid_motion", "mirror_axis_offset",
                         "mirror_resid_static", "active_width",
                         "active_sharpness", "motion_centroid_x",
                         "motion_centroid_x_spread", "lr_motion_ratio",
                         "grad_vertical_share", "column_energy_share",
                         "mirror_cov"):
            assert required in names, (
                f"the panel is missing the required candidate {required}")
        assert len(names) == len(pfs.FEATURE_SPECS)

    def test_every_feature_has_a_justification(self):
        """
        Every candidate has to say why it is in the panel, and the reason
        has to be more than a name.
        """
        for name, kind, why in pfs.FEATURE_SPECS:
            assert kind in ("frame", "session")
            assert len(why) > 30, f"{name}: no justification given"

    def test_the_capture_loader_reads_the_luma_plane(self, tmp_path):
        """
        The loader must take the luma plane and reshape it to (n, H, W).

        The frame layout is DETECTED from the file's content, not assumed --
        see `src/nv12.py`. This fixture is a STANDARD NV12 file (15,360 luma
        bytes then 7,680 bytes of interleaved chroma per 160x96 frame,
        23,040 bytes per frame) and it is the only candidate stride that
        divides its own length, so the loader has exactly one layout to
        choose and takes it.
        """
        fb = W * H * 3 // 2
        frames = np.arange(3 * fb, dtype=np.uint8).reshape(3, fb)
        p = tmp_path / "t.nv12"
        p.write_bytes(frames.tobytes())
        Y = pfs.load_y(p, 100)
        assert Y.shape == (3, H, W)
        assert Y.dtype == np.float32
        assert np.allclose(Y[0], frames[0, :W * H].reshape(H, W))

    def test_a_mis_strided_read_tears_the_frames(self, tmp_path):
        """
        CHARACTERISATION: reading a 19,200-byte-per-frame capture at the
        standard-NV12 stride of 23,040 produces TORN frames, not merely wrong
        ones -- and the tearing is SILENT, because the array still has the
        right shape.

        WHY THIS TEST EXISTS. Every experiment in this repo used to assume the
        frame stride was W*H*3//2 = 23,040. The captures on disk carry
        19,200 bytes per 160x96 frame (15,360 luma + 1,920 U + 1,920 V), so
        a reader at 23,040 tears every frame but every sixth. The damage is
        invisible to any check that looks at shapes, which is why it survived
        review and turned the "drift", the "row banding" and the "motion" this
        project measured into a stride artefact.

        The assertions below pin the exact tearing pattern, so the behaviour is
        documented rather than remembered, and so that a regression to a
        hard-coded stride is caught here rather than in a result file.

        The measured effect on the real captures was an 80-fold difference in
        mean consecutive |diff| (43.560 at the wrong stride against 0.545 at
        the right one, on A.nv12). The synthetic fixture here shows a smaller
        factor because its square genuinely moves on every frame, which puts a
        floor under the correct stride's |diff| that the real captures do not
        have.
        """
        from src.nv12 import luma_view, mean_consecutive_diff

        stride_true = W * H * 5 // 4          # 19,200: Y + U + V
        stride_wrong = W * H * 3 // 2        # 23,040: standard NV12
        n = 4
        rng = np.random.default_rng(7)

        planes = []
        for i in range(n):
            y = np.full((H, W), 40, dtype=np.uint8)
            x0 = 10 + (i * 2) % 100           # a square that really moves
            y[40:56, x0:x0 + 16] = 220
            # A light sensor dither, so no two frames are byte-identical --
            # but small enough that the CORRECT stride still has a tiny mean
            # consecutive |diff|, the way the real captures do.
            y = np.clip(y.astype(np.int16)
                        + rng.integers(0, 2, size=(H, W)), 0, 255).astype(np.uint8)
            planes.append(y)
        buf = bytearray()
        for y in planes:
            buf += y.tobytes()
            buf += bytes(W * H // 8)          # U
            buf += bytes(W * H // 8)          # V
        assert len(buf) == n * stride_true
        p = tmp_path / "capture.nv12"
        p.write_bytes(bytes(buf))

        # -- the CORRECT read ------------------------------------------------
        Y = pfs.load_y(p, None)
        assert Y.shape == (n, H, W)
        for i, y in enumerate(planes):
            assert np.array_equal(Y[i], y), f"frame {i} is not the luma plane"

        # -- the WRONG read, at the stride this repo used to assume ---------
        data = np.frombuffer(bytes(buf), dtype=np.uint8)
        wrong = luma_view(data, "standard_nv12", W, H)
        assert wrong.shape[0] == len(buf) // stride_wrong == 3
        # frame 2 of the wrong read starts 48 rows into true frame 2, runs
        # through true frame 2's chroma, and ends with true frame 3's FIRST
        # 24 ROWS. That is the tear: one "frame" assembled from two of them.
        assert np.array_equal(wrong[2][-24:], planes[3][:24])
        # Frame 0 is the one frame a wrong stride cannot spoil: it starts at
        # byte 0, which is the first luma plane either way. Every LATER frame
        # is a collage of two true frames.
        assert np.array_equal(wrong[0], planes[0])
        for i in range(1, wrong.shape[0]):
            assert not np.array_equal(wrong[i], planes[i]), (
                f"mis-strided frame {i} is still intact -- the tear moved")

        # -- and the consequence, which is the number that broke the project
        good = mean_consecutive_diff(Y.astype(np.int32))
        bad = mean_consecutive_diff(wrong.astype(np.int32))
        assert bad > 5 * good, (
            f"a mis-strided read should tear the frames hard: "
            f"{bad:.3f} vs {good:.3f} mean consecutive |diff|")
