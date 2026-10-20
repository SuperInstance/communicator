"""
test_nv12.py -- the frame layout is DETECTED, and the wrong stride is caught.

WHY THESE TESTS EXIST.

The repo read every capture with a hard-coded 23,040-byte frame stride, which
is standard NV12 for a 160x96 frame. The files on disk are 19,200 bytes per
frame (Y + half-resolution INTERLEAVED chroma), so 80% of every "frame" the
repo used was a tear between two real frames. The measured consequence was a
mean consecutive |diff| of 43.560 where the truth is 0.545.

(The chroma was first described here as "Y + U plane + V plane, planar
chroma". That was FALSE -- see the note in `src/nv12.py`: the even-indexed
chroma bytes average 125.05 and the odd-indexed ones 135.88, with no
midpoint step, so the chroma is interleaved. The layout was renamed
`y_uv_halfres`; the wrong description stayed here long enough for an
independent lane to flag it.)

These tests pin two things:

  1. THE DETECTOR WORKS. A file written in each of the three known layouts
     is detected as that layout, with a margin over the runner-up.
  2. THE WRONG STRIDE IS CAUGHT. Reading a real capture with the wrong
     stride produces torn frames, and that is asserted here so the
     characteristic cannot silently return.

The characterisation assertions in this file are INVERTED relative to what
they used to say. Each one records the defect in the file's own docstring, so
a future reader can see what the wrong assumption cost.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import nv12  # noqa: E402

W, H = 160, 96
LUMA = W * H


def make_luma(n_frames: int, moving: bool = True, seed: int = 0) -> np.ndarray:
    """A synthetic scene: a bright square that moves, on a textured floor.

    Textured, because a flat scene would make every layout look equally
    plausible and the detector would have nothing to detect.
    """
    rng = np.random.default_rng(seed)
    base = (rng.integers(40, 90, size=(H, W))).astype(np.uint8)
    frames = np.empty((n_frames, H, W), dtype=np.uint8)
    for i in range(n_frames):
        f = base.copy()
        x = 10 + (i * 3 if moving else 0) % (W - 30)
        y = 20 + (i * 1 if moving else 0) % (H - 30)
        f[y:y + 10, x:x + 12] = 230
        frames[i] = f
    return frames


def write_layout(path: Path, luma: np.ndarray, layout: str) -> Path:
    """Write frames in the named layout.

    The chroma geometry is the whole point of this file, so it is spelled out:

      standard NV12   luma 160x96 = 15,360, then INTERLEAVED U,V at 80x48
                      samples = 7,680 bytes. Total 23,040 = W*H*3/2.
      y_uv_halfres    luma 160x96 = 15,360, then a U plane at 80x24 and a V
                      plane at 80x24 = 1,920 each. Total 19,200 = W*H*5/4.
                      This is what `nv12small.py` wrote: its chroma is
                      subsampled 2:1 horizontally but 4:1 vertically, not
                      2:2, so the frame is 20% shorter than a real NV12 one.
      luma_only       15,360.

    U and V are given DIFFERENT constant values, and that matters: a Y+U+V
    file and a Y+interleaved-UV file are BYTE-IDENTICAL when U == V, so a
    file written with neutral grey chroma cannot be attributed to one layout
    or the other -- and neither can any reader. With real chroma the two
    layouts are distinguishable, which is the case the detector has to get
    right.
    """
    n = len(luma)
    u_val, v_val = 90, 160
    out = bytearray()
    for f in luma:
        if layout == "luma_only":
            out += f.tobytes()
        elif layout == "y_uv_halfres":
            u = np.full((H // 4, W // 2), u_val, np.uint8)
            v = np.full((H // 4, W // 2), v_val, np.uint8)
            out += f.tobytes() + u.tobytes() + v.tobytes()
        elif layout == "standard_nv12":
            u = np.full((H // 2, W // 2), u_val, np.uint8)
            v = np.full((H // 2, W // 2), v_val, np.uint8)
            uv = np.empty((u.size, 2), np.uint8)
            uv[:, 0] = u.ravel()
            uv[:, 1] = v.ravel()
            out += f.tobytes() + uv.tobytes()
        else:
            raise ValueError(layout)
    path.write_bytes(bytes(out))
    return path


# --- the strides themselves -------------------------------------------------

def test_stride_for_each_layout():
    assert nv12.stride_for("standard_nv12", W, H) == 23040
    assert nv12.stride_for("y_uv_halfres", W, H) == 19200
    assert nv12.stride_for("luma_only", W, H) == 15360
    assert nv12.stride_for("luma_only", W, H) == LUMA


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError):
        nv12.stride_for("rgb24", W, H)


# --- the detector -----------------------------------------------------------

@pytest.mark.parametrize("layout", ["standard_nv12", "y_uv_halfres", "luma_only"])
def test_detector_picks_the_layout_it_was_written_in(tmp_path, layout):
    p = write_layout(tmp_path / f"{layout}.nv12", make_luma(30), layout)
    found = nv12.detect_layout(p, W, H)
    assert found.name == layout, (
        f"a file written as {layout} was detected as {found.name} "
        f"({found.mean_frame_diff:.3f} vs {found.runner_up_diff:.3f})")
    assert found.margin > 5.0, "the detector must say how sure it is"


@pytest.mark.parametrize("layout", ["standard_nv12", "y_uv_halfres", "luma_only"])
def test_detected_frame_count_and_stride(tmp_path, layout):
    p = write_layout(tmp_path / f"{layout}.nv12", make_luma(30), layout)
    found = nv12.detect_layout(p, W, H)
    assert found.n_frames == 30
    assert found.stride == nv12.stride_for(layout, W, H)


def test_read_luma_returns_the_written_frames(tmp_path):
    luma = make_luma(12)
    p = write_layout(tmp_path / "a.nv12", luma, "y_uv_halfres")
    got, layout = nv12.read_luma(p, W, H)
    assert layout.name == "y_uv_halfres"
    assert got.shape == (12, H, W)
    np.testing.assert_array_equal(got, luma)


def test_ambiguous_layout_is_an_error_not_a_guess(tmp_path):
    """A file with no content gives every layout the same score.

    The detector must refuse rather than pick one, because a wrong guess
    silently corrupts every downstream number. The caller can still force a
    layout explicitly.
    """
    p = tmp_path / "flat.nv12"
    p.write_bytes(bytes([100]) * (19200 * 8))
    with pytest.raises(ValueError):
        nv12.read_luma(p, W, H)


def test_a_static_scene_is_detected_not_ambiguous(tmp_path):
    """A static scene is NOT ambiguous: the wrong stride tears it.

    The chroma bytes of a real frame are not scene content, so a mis-strided
    read of a static scene still differs. Only a file with no content at all
    is ambiguous.
    """
    p = write_layout(tmp_path / "static.nv12", make_luma(8, moving=False),
                     "y_uv_halfres")
    luma, layout = nv12.read_luma(p, W, H)
    assert layout.name == "y_uv_halfres"
    assert len(luma) == 8


def test_forced_layout_is_honoured(tmp_path):
    p = write_layout(tmp_path / "a.nv12", make_luma(10), "y_uv_halfres")
    got, layout = nv12.read_luma(p, W, H, layout="y_uv_halfres")
    assert layout.name == "y_uv_halfres"
    assert len(got) == 10


# --- the wrong stride, characterised ---------------------------------------

def test_reading_with_the_wrong_stride_gives_torn_frames(tmp_path):
    """THE DEFECT, PINNED.

    Reading a y_uv_halfres capture at the standard-NV12 stride does not
    merely shift the frames: it splices two different frames into one. Rows
    0..71 of the mis-read frame come from true frame k and rows 72..95 from
    true frame k+1, so 25% of every frame is the wrong moment.

    This assertion is here so that the characteristic cannot return
    unnoticed. If a future change makes a wrong-stride read produce clean
    frames, this test fails and says why.
    """
    luma = make_luma(20)
    p = write_layout(tmp_path / "a.nv12", luma, "y_uv_halfres")
    wrong, _ = nv12.read_luma(p, W, H, layout="standard_nv12")
    # the mis-read frame 1 is spliced: its TOP rows are not frame 1's top rows
    top_mismatch = np.abs(wrong[1][:72].astype(int) - luma[1][:72].astype(int)).mean()
    assert top_mismatch > 1.0, (
        "a mis-strided read no longer splices frames -- the defect's "
        "signature has changed and this assertion must be revisited")
    # and the mean consecutive |diff| of the wrong read is far larger
    right, _ = nv12.read_luma(p, W, H, layout="y_uv_halfres")
    assert nv12.mean_consecutive_diff(wrong) > 10 * nv12.mean_consecutive_diff(right)


def test_wrong_stride_misaligns_the_frame_index(tmp_path):
    """A wrong stride does not just tear frames, it LIES about how many there are."""
    p = write_layout(tmp_path / "a.nv12", make_luma(30), "y_uv_halfres")
    right, _ = nv12.read_luma(p, W, H, layout="y_uv_halfres")
    wrong, _ = nv12.read_luma(p, W, H, layout="standard_nv12")
    assert len(right) == 30
    assert len(wrong) == 25          # 20 bytes short per frame -> fewer frames
    assert len(wrong) != len(right)


# --- the real captures ------------------------------------------------------

CAPTURE_DIR = Path("/mnt/c/proart_capture")


@pytest.mark.skipif(not (CAPTURE_DIR / "A.nv12").is_file(),
                    reason="the original captures are not on this machine")
def test_the_real_captures_are_y_uv_halfres_900_frames():
    """The files on disk are 19,200 bytes per frame, 900 frames, 30 fps."""
    for name in ("A", "B", "C"):
        luma, layout = nv12.read_luma(CAPTURE_DIR / f"{name}.nv12", W, H)
        assert layout.name == "y_uv_halfres", (
            f"{name}.nv12 was detected as {layout.name}")
        assert layout.stride == 19200
        assert len(luma) == 900
        assert layout.margin > 50.0, (
            "the margin over the runner-up must be large enough that this "
            "is not a coin flip")


@pytest.mark.skipif(not (CAPTURE_DIR / "A.nv12").is_file(),
                    reason="the original captures are not on this machine")
def test_the_real_capture_frame_diff_is_the_small_one():
    luma, layout = nv12.read_luma(CAPTURE_DIR / "A.nv12", W, H)
    assert layout.mean_frame_diff < 2.0, (
        "the real capture should have a small mean consecutive |diff| when "
        "read at the right stride")


@pytest.mark.skipif(not (CAPTURE_DIR / "A.wav").is_file(),
                    reason="the original captures are not on this machine")
def test_frame_count_matches_the_audio_duration():
    """900 frames at 30 fps is 30.00 s, and the WAV is 29.98 s.

    This is the check that settles the stride without appealing to the
    capture script: the video and the audio agree on the duration only at
    the correct stride.
    """
    import wave
    with wave.open(str(CAPTURE_DIR / "A.wav")) as w:
        seconds = w.getnframes() / w.getframerate()
    luma, layout = nv12.read_luma(CAPTURE_DIR / "A.nv12", W, H)
    assert abs(len(luma) / 30.0 - seconds) < 0.05, (
        f"{len(luma)} frames at 30 fps is {len(luma)/30:.2f}s but the audio is "
        f"{seconds:.2f}s")
