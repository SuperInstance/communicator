"""
test_capture_stride_readers.py -- the lane's readers, pinned to the RIGHT stride.

WHY THESE TESTS EXIST.

Seven experiment scripts in this lane used to read a `.nv12` capture with a
hand-rolled stride of `W*H*3//2` = 23,040 bytes per frame. The captures on
disk carry 19,200 bytes per frame -- a 160x96 luma plane followed by 80x24
INTERLEAVED U,V samples -- so at the old stride all but every sixth frame was
torn: rows 0..71 from frame k, rows 72..95 from frame k+1.

Two of the lane's scripts are the C character renderers' hosts, and their C
code reads a STANDARD NV12 frame of 23,040 bytes. That C code is not wrong for
standard NV12, and it was left alone. What changed is that the captures are
re-wrapped into standard NV12 before the renderer sees them, and the luma is
copied verbatim.

These tests pin three things:

  1. EVERY READER IN THE LANE READS THE TRUE FRAMES. A synthetic capture with
     a known, position-dependent luma is written in the `y_uv_halfres`
     layout, and every reader must return exactly that luma -- not a
     mis-strided slice of it.

  2. THE RE-WRAP CHANGES NO LUMA BYTE. A renderer that is handed standard
     NV12 must see the same luma the file was written with.

  3. THE OLD STRIDE IS GONE FROM THE LANE, and the committed results files
     carry the identity of their inputs. A number with no recorded layout,
     stride, frame count and input hash is not a result.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from src import nv12  # noqa: E402

W, H = 160, 96
LUMA = W * H
TRUE_STRIDE = 19200          # 15,360 luma + 3,840 interleaved U,V at 80x24
STANDARD_STRIDE = 23040      # 15,360 luma + 7,680 interleaved U,V at 80x48

LANE = [
    "experiments/emptyroom_null.py",
    "experiments/ascii_motion_nulls.py",
    "experiments/ascii_motion_threshold.py",
    "experiments/av_sync.py",
    "experiments/drift_structure.py",
    "experiments/combined_filter.py",
    "experiments/temporal_median_fix.py",
]

RESULTS = {
    "experiments/emptyroom_null.py": "experiments/emptyroom_null_results.json",
    "experiments/ascii_motion_nulls.py": "experiments/ascii_motion_nulls_results.json",
    "experiments/ascii_motion_threshold.py": "experiments/ascii_motion_threshold_results.json",
    "experiments/av_sync.py": "experiments/av_sync_results.json",
    "experiments/drift_structure.py": "experiments/drift_structure_results.json",
    "experiments/combined_filter.py": "experiments/combined_filter_results.json",
    "experiments/temporal_median_fix.py": "experiments/temporal_median_fix_results.json",
}


def make_luma(n_frames: int, seed: int = 7) -> np.ndarray:
    """A synthetic scene where every pixel value encodes its own position.

    Position-dependent on purpose: a mis-strided read cannot accidentally
    reproduce it, so a reader that returns the wrong frames fails loudly
    rather than by coincidence.
    """
    rng = np.random.default_rng(seed)
    base = (rng.integers(40, 90, size=(H, W))).astype(np.uint8)
    frames = np.empty((n_frames, H, W), dtype=np.uint8)
    for i in range(n_frames):
        f = base.copy()
        x = 10 + (i * 3) % (W - 30)
        y = 20 + (i * 1) % (H - 30)
        f[y:y + 10, x:x + 12] = 230
        frames[i] = f
    return frames


def write_halfres(path: Path, luma: np.ndarray) -> Path:
    """Write frames in the layout the captures on disk actually use.

    Luma, then INTERLEAVED U,V at 80x24 -- half the vertical resolution of a
    standard NV12 frame's chroma, which is what makes the frame 20% shorter.
    U and V are different constants so the chroma is not ambiguous with a
    planar layout.
    """
    out = bytearray()
    for f in luma:
        u = np.full((H // 4, W // 2), 90, np.uint8)
        v = np.full((H // 4, W // 2), 160, np.uint8)
        uv = np.empty((u.size, 2), np.uint8)
        uv[:, 0] = u.ravel()
        uv[:, 1] = v.ravel()
        out += f.tobytes() + uv.tobytes()
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def capture(tmp_path):
    """A 40-frame capture in the layout the real captures use."""
    luma = make_luma(40)
    return write_halfres(tmp_path / "cap.nv12", luma), luma


# --- 1. every reader in the lane reads the true frames -----------------------

def test_the_capture_really_is_19200_bytes_per_frame(capture):
    path, luma = capture
    assert path.stat().st_size == 40 * TRUE_STRIDE
    got, layout = nv12.read_luma(path, W, H)
    assert layout.name == "y_uv_halfres"
    assert layout.stride == TRUE_STRIDE
    assert layout.n_frames == 40
    np.testing.assert_array_equal(got, luma)


def test_emptyroom_null_loads_the_true_luma(capture):
    import emptyroom_null
    path, luma = capture
    got, layout = emptyroom_null.load(path)
    assert layout.stride == TRUE_STRIDE
    np.testing.assert_array_equal(got, luma)


def test_drift_structure_loads_the_true_luma(capture):
    import drift_structure
    path, luma = capture
    got, layout = drift_structure.load_y(path, 40)
    assert layout.stride == TRUE_STRIDE
    np.testing.assert_array_equal(got, luma.astype(np.float64))


def test_av_sync_loads_the_true_luma(capture):
    import av_sync
    path, luma = capture
    got, layout = av_sync.load_y(path)
    assert layout.stride == TRUE_STRIDE
    np.testing.assert_array_equal(got, luma.astype(np.float64))


def test_combined_filter_loads_the_true_luma(capture):
    import combined_filter
    path, luma = capture
    got, layout = combined_filter.load_raw(path, 40)
    assert layout.stride == TRUE_STRIDE
    np.testing.assert_array_equal(got, luma.astype(np.float64))


def test_temporal_median_fix_loads_the_true_luma_and_standard_chroma(capture):
    import temporal_median_fix
    path, luma = capture
    got, uv = temporal_median_fix.load(path, 40)
    np.testing.assert_array_equal(got, luma)
    # the chroma handed to the renderer must be STANDARD NV12: 80x48
    # interleaved samples, i.e. 7,680 bytes a frame
    assert uv.shape == (40, STANDARD_STRIDE - LUMA)
    # and each chroma row must be the source row duplicated
    src = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    src_uv = src[:40 * TRUE_STRIDE].reshape(40, TRUE_STRIDE)[:, LUMA:]
    src_uv = src_uv.reshape(40, H // 4, W // 2, 2)
    expect = np.repeat(src_uv, 2, axis=1).reshape(40, -1)
    np.testing.assert_array_equal(uv, expect)


def test_ascii_motion_nulls_histogram_reads_the_true_luma(capture):
    import ascii_motion_nulls
    path, luma = capture
    hist = ascii_motion_nulls.luma_histogram(luma)
    assert hist["n_pixels"] == 40 * LUMA
    flat = np.sort(luma.reshape(-1))
    assert hist["p50"] == int(flat[len(flat) // 2])
    assert hist["max"] == int(flat[-1])


# --- 2. the re-wrap changes no luma byte -------------------------------------

def test_standard_nv12_rewrap_preserves_every_luma_byte(capture, tmp_path):
    import ascii_motion_threshold
    path, luma = capture
    dst = tmp_path / "std.nv12"
    written, layout = ascii_motion_threshold.standard_nv12(path, dst)
    assert layout.stride == TRUE_STRIDE
    raw = np.frombuffer(dst.read_bytes(), dtype=np.uint8)
    n = 40
    assert len(raw) == n * STANDARD_STRIDE
    got = raw.reshape(n, STANDARD_STRIDE)[:, :LUMA].reshape(n, H, W)
    np.testing.assert_array_equal(got, luma)


def test_rewrapped_capture_is_readable_as_standard_nv12(capture, tmp_path):
    """The renderer reads standard NV12, so the re-wrapped file must be so."""
    import ascii_motion_threshold
    path, _ = capture
    dst = tmp_path / "std.nv12"
    ascii_motion_threshold.standard_nv12(path, dst)
    got, layout = nv12.read_luma(dst, W, H)
    assert layout.name == "standard_nv12"
    assert layout.stride == STANDARD_STRIDE
    assert layout.n_frames == 40


# --- 3. the old stride is gone from the lane, and the results carry identity -

def test_no_lane_script_slices_a_capture_with_a_three_halves_stride():
    """Banning the one-line idiom that caused the defect.

    `W*H*3//2` is still correct as the RENDERER's frame size and as the
    size of a re-wrapped standard-NV12 file, and the lane uses it for both.
    What must not come back is the idiom that caused the defect: computing a
    three-halves stride and slicing a capture with it in the same breath.
    """
    reads = re.compile(r"read_bytes|frombuffer|reshape|\[:")
    three_halves = re.compile(r"\*\s*3\s*(?://|/)")
    for rel in LANE:
        for i, line in enumerate((ROOT / rel).read_text().splitlines(), 1):
            if reads.search(line) and three_halves.search(line):
                pytest.fail(
                    f"{rel}:{i} slices a capture with a three-halves stride: "
                    f"{line.strip()!r}")


def test_every_lane_reader_goes_through_the_detector():
    """A reader that never mentions `src.nv12` is a reader that guesses."""
    for rel in LANE:
        text = (ROOT / rel).read_text()
        assert ("read_luma" in text or "standard_nv12" in text), (
            f"{rel} reads a capture without going through src.nv12")


@pytest.mark.parametrize("rel", LANE)
def test_results_do_not_report_the_wrong_frame_count(rel):
    """900 frames at 30 fps is 30.00 s. 750 frames is the torn-frame count.

    The `supersedes` block is allowed to name the old numbers -- that is
    where they belong. The CURRENT results may not.
    """
    results_path = ROOT / RESULTS[rel]
    if not results_path.is_file():
        pytest.skip(f"{RESULTS[rel]} has not been produced on this machine")
    doc = json.loads(results_path.read_text())
    current = {k: v for k, v in doc.items() if k != "supersedes"}

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from walk(v, f"{path}[{i}]")
        else:
            yield path, node

    for name, value in walk(current):
        low = name.lower()
        if "frame" in low and isinstance(value, int) and value == 750:
            pytest.fail(f"{RESULTS[rel]}: {name} is 750, the torn-frame count")
        if "seconds" in low and isinstance(value, (int, float)):
            assert abs(value - 25.0) > 1e-9, (
                f"{RESULTS[rel]}: {name} is 25.0 s, the wrong-stride duration")
        if "mismatch" in low and isinstance(value, (int, float)):
            assert abs(value) < 0.5, (
                f"{RESULTS[rel]}: {name} is {value} s; the video and the audio "
                "are the same length to within half a frame")


@pytest.mark.parametrize("rel", LANE)
def test_results_carry_the_identity_of_their_inputs(rel):
    results_path = ROOT / RESULTS[rel]
    if not results_path.is_file():
        pytest.skip(f"{RESULTS[rel]} has not been produced on this machine")
    doc = json.loads(results_path.read_text())
    prov = doc.get("provenance")
    assert prov is not None, f"{RESULTS[rel]} has no provenance block"
    assert prov.get("script_sha256"), f"{RESULTS[rel]} records no script hash"
    assert prov.get("generated_by") == rel, (
        f"{RESULTS[rel]} says it was generated by {prov.get('generated_by')}")
    # the layout, the stride and the frame count have to be somewhere in the
    # provenance, for the capture or for every capture
    blob = json.dumps(prov)
    assert "stride_bytes" in blob, (
        f"{RESULTS[rel]} records no stride in its provenance")
    assert "layout" in blob, (
        f"{RESULTS[rel]} records no layout in its provenance")
    assert "sha256" in blob, (
        f"{RESULTS[rel]} records no input hash in its provenance")


@pytest.mark.parametrize("rel", LANE)
def test_results_record_a_supersedes_block(rel):
    results_path = ROOT / RESULTS[rel]
    if not results_path.is_file():
        pytest.skip(f"{RESULTS[rel]} has not been produced on this machine")
    doc = json.loads(results_path.read_text())
    sup = doc.get("supersedes")
    assert sup is not None, f"{RESULTS[rel]} records no supersedes block"
    assert sup.get("previous_stride_bytes") == 23040, (
        f"{RESULTS[rel]} does not say the previous run read at 23,040 bytes")
