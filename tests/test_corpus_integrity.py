"""
Tests for the corpus integrity gate.

WHY THIS FILE EXISTS.

The corpus is the thing every later result rests on, and the corpus has
already been wrong once: the earlier captures recorded 30 s of audio and
25 s of video separately, wrote a 16000 Hz WAV header over audio that
implies ~19188 Hz, and assumed 30 fps whatever the camera was delivering.
`experiments/av_sync.py` found no correspondence between the audio and the
video of those takes at any lag, at any candidate sample rate, with any of
three motion features.

So the corpus needs a report BEFORE anything is trained on it, and that
report has to be able to FAIL. A gate that always passes is a gate that
never fires, which is the same as no gate.

WHAT IS PINNED HERE.

  - a clean corpus passes and exits 0 (so the gate is not a nuisance);
  - a speaking condition with a measured speech fraction of 0.02 is caught,
    with the number in the message (a take that did not contain the poem
    must not be filed as a speaking take);
  - a camera delivering 25 fps against an assumed 30 is caught (the 1.2x
    time-base defect, in the form this project's own captures took);
  - a missing npz is caught (the take has no change maps, so it has no
    features);
  - a mixed-stimulus corpus is reported, with the majority named, because a
    corpus with two stimuli in it cannot be pooled;
  - a take with no timing.json is counted as an OLD-FORMAT take and NOT as
    a failure, so the two numbers are never confused;
  - a missing corpus directory exits non-zero with a message instead of
    crashing the pipeline.

Every corpus here is synthetic and lives in tmp_path. The tests do not
depend on /mnt/c being mounted, and they do not need the Silero weights:
the gate's inputs are the metadata the capture script wrote, not the audio.
"""

from __future__ import annotations
import pytest as _pytest  # noqa: E402
_pytest.importorskip("numpy", reason="exercises code needing numpy")  # noqa: E402

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import corpus_integrity              # noqa: E402

SCRIPT = ROOT / "experiments" / "corpus_integrity.py"


def write_take(d: Path, name: str, stamp: int, *,
               n_frames: int = 30,
               kind: str = "condition",
               speak=True,
               speech_frac=0.40,
               assumed_fps: float = 30.0,
               achieved_fps: float = 30.0,
               audio_sample_rate: int = 16000,
               requested_sample_rate: int = 16000,
               stimulus: str = "invictus",
               drift: float = 1.2,
               write_npz: bool = True,
               write_timing: bool = True,
               video_span_s: float | None = None,
               audio_span_s: float | None = None) -> Path:
    """Write one synthetic take, in the schema corpus_capture_win.py writes.

    The defaults are a GOOD take: the camera delivers what was assumed,
    the mic delivers what was requested, the two spans are the same length,
    the drift is low, and the change maps are there.
    """
    d.mkdir(parents=True, exist_ok=True)
    seconds = n_frames / assumed_fps
    if video_span_s is None:
        video_span_s = seconds
    if audio_span_s is None:
        audio_span_s = seconds

    meta = {
        "condition": name,
        "kind": kind,
        "note": "synthetic",
        "pose": "facing",
        "speak": speak,
        "stimulus": stimulus,
        "stimulus_title": "INVICTUS",
        "stimulus_estimated_s": 40.0,
        "speech_frac": speech_frac,
        "n_frames": n_frames,
        "seconds": seconds,
        "achieved_fps": achieved_fps,
        "audio_sample_rate": audio_sample_rate,
        "audio_seconds": audio_span_s,
        "map_shape": [48, 80],
        "map_scale": 1.0,
        "drift_raw": 12.0,
        "drift_after_row_removal": drift,
        "timestamp": stamp,
        "rep": 1,
        "filter": "row+median3",
    }
    meta_path = d / f"{name}_{stamp}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    if write_npz:
        maps = np.zeros((n_frames - 1, 48, 80), dtype=np.uint8)
        np.savez_compressed(d / f"{name}_{stamp}.npz", maps=maps,
                            scale=np.float32(1.0))

    if write_timing:
        timing = {
            "condition": name,
            "n_frames": n_frames,
            "frame_offsets_s": [round(i / achieved_fps, 6)
                                for i in range(n_frames)],
            "assumed_fps": assumed_fps,
            "achieved_fps": achieved_fps,
            "video_span_s": video_span_s,
            "audio_span_s": audio_span_s,
            "audio_sample_rate": audio_sample_rate,
            "requested_sample_rate": requested_sample_rate,
            "mic": "Synthetic Microphone",
            "capture_wallclock_s": 1.0,
        }
        (d / f"{name}_{stamp}_timing.json").write_text(json.dumps(timing))

    return meta_path


def failed_checks(take: dict):
    return [f["check"] for f in take["failures"]]


def take_named(report: dict, condition: str) -> dict:
    hits = [t for t in report["takes"] if t["condition"] == condition]
    assert hits, f"no take named {condition} in the report"
    return hits[0]


class TestCleanCorpus:
    def test_a_clean_corpus_passes_and_exits_zero(self, tmp_path):
        """The gate must not fire on good data, or it will be ignored."""
        corpus = tmp_path / "corpus"
        write_take(corpus, "A1", 1710000001, speak=True, speech_frac=0.42)
        write_take(corpus, "C1", 1710000002, speak=False, speech_frac=0.0)

        report = corpus_integrity.analyse(corpus)
        assert report["n_takes"] == 2
        assert report["n_failed"] == 0, report["takes"]
        assert report["ok"] is True
        assert all(failure_counts == 0
                   for failure_counts in report["failure_counts"].values())

        rc = corpus_integrity.main([str(corpus)])
        assert rc == 0

    def test_the_json_summary_is_written_and_carries_the_coverage(self, tmp_path):
        """The coverage number is the point of the report, so it must be in
        the JSON, with the 63 stated as the denominator it comes from."""
        corpus = tmp_path / "corpus"
        write_take(corpus, "A1", 1710000001)
        write_take(corpus, "A1", 1710000002)      # a second session of A1
        out = tmp_path / "ci.json"

        rc = corpus_integrity.main([str(corpus), "--json", str(out)])
        assert rc == 0
        assert out.is_file()
        doc = json.loads(out.read_text())

        cov = doc["coverage"]
        assert cov["target_sessions_per_class"] == 63
        assert "teacher_data_requirement.py" in cov["target_source"]
        a1 = cov["per_condition"]["A1"]
        assert a1["distinct_sessions"] == 2
        assert a1["gap_to_target"] == 61
        assert a1["target_met"] is False
        # SPEAK and SILENT are the classes the 63 applies to; both takes
        # here are speaking, so SPEAK holds 2 distinct sessions.
        assert cov["per_class"]["SPEAK"]["distinct_sessions"] == 2
        assert cov["per_class"]["SPEAK"]["gap_to_target"] == 61
        assert cov["per_class"]["SPEAK"]["target_met"] is False

    def test_the_report_separates_old_format_takes_from_failures(self, tmp_path):
        """A take with no timing.json is an old-format take. It is reported
        separately so it is never confused with a take that FAILED."""
        corpus = tmp_path / "corpus"
        write_take(corpus, "A1", 1710000001)
        write_take(corpus, "B1", 1710000002, write_timing=False)

        report = corpus_integrity.analyse(corpus)
        assert report["n_missing_timing"] == 1
        assert report["n_failed"] == 0

        text = corpus_integrity.format_report(report)
        assert "old-format" in text
        assert "SUMMARY: 0 of 2 takes failed" in text


class TestSpeechPlausibility:
    def test_a_speaking_take_with_no_speech_is_caught(self, tmp_path):
        """speech_frac 0.02 while speak=True: the poem was not recited, so
        the take must not be filed as a speaking take. The number goes in
        the message, because 'it failed' is not actionable."""
        corpus = tmp_path / "corpus"
        write_take(corpus, "A1", 1710000001, speak=True, speech_frac=0.02)

        report = corpus_integrity.analyse(corpus)
        assert report["n_failed"] == 1
        take = take_named(report, "A1")
        assert "speech" in failed_checks(take)
        message = take["failures"][0]["message"]
        assert "0.0200" in message
        assert report["failure_counts"]["speech"] == 1

        assert corpus_integrity.main([str(corpus)]) == 1

    def test_a_silent_take_that_talked_is_caught(self, tmp_path):
        """The other direction of the same check."""
        corpus = tmp_path / "corpus"
        write_take(corpus, "C1", 1710000001, speak=False, speech_frac=0.40)

        report = corpus_integrity.analyse(corpus)
        assert report["n_failed"] == 1
        assert "speech" in failed_checks(take_named(report, "C1"))


class TestRateConsistency:
    def test_a_camera_at_25_fps_against_an_assumed_30_is_caught(self, tmp_path):
        """THE time-base check. A camera delivering 25 fps when the frame
        index says 30 means every frame timestamp is 20% wrong -- the
        1.2x defect this project's earlier captures carried."""
        corpus = tmp_path / "corpus"
        write_take(corpus, "A1", 1710000001, assumed_fps=30.0,
                   achieved_fps=25.0)

        report = corpus_integrity.analyse(corpus)
        assert report["n_failed"] == 1
        take = take_named(report, "A1")
        assert "rate" in failed_checks(take)
        message = take["failures"][0]["message"]
        assert "25.000" in message and "30.000" in message

        assert corpus_integrity.main([str(corpus)]) == 1

    def test_a_mic_that_did_not_deliver_the_requested_rate_is_caught(self,
                                                                     tmp_path):
        """The half of the 1.2x defect that lives in the audio: the audio
        was captured at a rate the pipeline did not ask for, so every audio
        timestamp derived from it is wrong."""
        corpus = tmp_path / "corpus"
        write_take(corpus, "A1", 1710000001, audio_sample_rate=19188,
                   requested_sample_rate=16000)

        report = corpus_integrity.analyse(corpus)
        assert report["n_failed"] == 1
        take = take_named(report, "A1")
        assert "rate" in failed_checks(take)
        assert "19188" in take["failures"][0]["message"]

    def test_a_take_within_five_percent_of_the_assumed_rate_passes(self,
                                                                    tmp_path):
        """The tolerance is 5%, and 29.5 fps is inside it."""
        corpus = tmp_path / "corpus"
        write_take(corpus, "A1", 1710000001, assumed_fps=30.0,
                   achieved_fps=29.5)

        report = corpus_integrity.analyse(corpus)
        assert report["n_failed"] == 0


class TestFileIntegrity:
    def test_a_take_with_no_npz_is_caught(self, tmp_path):
        """No change maps means no features, and no features means the take
        cannot be trained on whatever it looks like."""
        corpus = tmp_path / "corpus"
        write_take(corpus, "A1", 1710000001, write_npz=False)

        report = corpus_integrity.analyse(corpus)
        assert report["n_failed"] == 1
        take = take_named(report, "A1")
        assert "file" in failed_checks(take)
        assert "npz" in take["failures"][0]["message"]

        assert corpus_integrity.main([str(corpus)]) == 1

    def test_a_wrong_map_count_is_caught(self, tmp_path):
        """The npz holds one map per frame PAIR, so n_frames-1 maps."""
        corpus = tmp_path / "corpus"
        meta_path = write_take(corpus, "A1", 1710000001, n_frames=30)
        npz_path = meta_path.with_suffix(".npz")
        np.savez_compressed(npz_path, maps=np.zeros((10, 48, 80), np.uint8),
                            scale=np.float32(1.0))

        report = corpus_integrity.analyse(corpus)
        assert report["n_failed"] == 1
        assert "file" in failed_checks(take_named(report, "A1"))


class TestStimulus:
    def test_a_mixed_stimulus_corpus_is_reported(self, tmp_path):
        """Two stimuli in one corpus cannot be pooled, so the minority take
        is flagged and the majority is named."""
        corpus = tmp_path / "corpus"
        write_take(corpus, "A1", 1710000001, stimulus="invictus")
        write_take(corpus, "A2", 1710000002, stimulus="invictus")
        write_take(corpus, "B1", 1710000003, stimulus="other-poem")

        report = corpus_integrity.analyse(corpus)
        assert report["stimulus"]["mixed"] is True
        assert report["stimulus"]["majority"] == "invictus"

        minority = take_named(report, "B1")
        assert "stimulus" in failed_checks(minority)
        assert "invictus" in minority["failures"][0]["message"]
        # the majority takes are not failed by this
        assert take_named(report, "A1")["ok"] is True

        text = corpus_integrity.format_report(report)
        assert "MIXED" in text

    def test_a_single_stimulus_corpus_is_not_flagged(self, tmp_path):
        corpus = tmp_path / "corpus"
        write_take(corpus, "A1", 1710000001)
        write_take(corpus, "A2", 1710000002)

        report = corpus_integrity.analyse(corpus)
        assert report["stimulus"]["mixed"] is False
        assert report["n_failed"] == 0


class TestLabelConsistency:
    def test_a_metadata_condition_that_disagrees_with_its_filename_is_caught(
            self, tmp_path):
        """The file says A1; the metadata says B1. One of them is a lie."""
        corpus = tmp_path / "corpus"
        meta_path = write_take(corpus, "A1", 1710000001)
        meta = json.loads(meta_path.read_text())
        meta["condition"] = "B1"
        meta_path.write_text(json.dumps(meta))

        report = corpus_integrity.analyse(corpus)
        assert report["n_failed"] == 1
        assert "label" in failed_checks(take_named(report, "B1"))

    def test_a_free_take_that_claims_to_speak_is_caught(self, tmp_path):
        """kind 'free' carries no label, so `speak` must be null."""
        corpus = tmp_path / "corpus"
        write_take(corpus, "FREE", 1710000001, kind="free", speak=True)

        report = corpus_integrity.analyse(corpus)
        assert report["n_failed"] == 1
        assert "label" in failed_checks(take_named(report, "FREE"))


class TestSpan:
    def test_a_video_and_audio_of_different_lengths_is_caught(self, tmp_path):
        """The old captures were exactly this: 25 s of video and 30 s of
        audio, recorded separately."""
        corpus = tmp_path / "corpus"
        write_take(corpus, "A1", 1710000001, n_frames=30, assumed_fps=30.0,
                   video_span_s=1.0, audio_span_s=1.2)

        report = corpus_integrity.analyse(corpus)
        assert report["n_failed"] == 1
        take = take_named(report, "A1")
        assert "span" in failed_checks(take)


class TestCli:
    def test_a_missing_corpus_directory_exits_non_zero_without_crashing(
            self, tmp_path):
        """The gate runs in a pipeline. A missing directory must be a clean
        non-zero exit with a message, not a traceback."""
        missing = tmp_path / "not_here"
        rc = corpus_integrity.main([str(missing)])
        assert rc != 0
        assert rc == 2

    def test_the_script_runs_as_a_program_and_exits_non_zero_on_a_failure(
            self, tmp_path):
        """End to end, through the real process exit code."""
        corpus = tmp_path / "corpus"
        write_take(corpus, "A1", 1710000001, speak=True, speech_frac=0.02)

        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(corpus)],
            capture_output=True, text=True)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "SUMMARY: 1 of 1 takes failed" in proc.stdout
        assert "speech=1" in proc.stdout

    def test_help_works(self):
        proc = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                              capture_output=True, text=True)
        assert proc.returncode == 0
        assert "--json" in proc.stdout
