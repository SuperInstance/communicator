"""
Tests for the local-VLM determinism lane.

These are deliberately network-free, GPU-free and download-free. The
MEASUREMENT needs an RTX 4050 and a 4 GB model download; what needs testing is
the RECORD.

The rule this file follows, from `docs/research/HOW-WE-KEEP-FOOLING-OURSELVES.md`:
every verdict must be able to fail. So a test that asserts a number copied
out of the artefact is a test that cannot fail. Every spread here is
RECOMPUTED from the per-run values the artefact records and compared with the
artefact's own summary -- if the summary were wrong, or were hardcoded, or
were edited by hand to agree with a conclusion, the recomputation would
disagree with it and this file would fail.

The controls are asserted as well, because they are the thing that makes the
rest of the artefact mean anything: a harness that cannot produce 0.000 on a
deterministic input is not measuring anything.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

RESULTS = ROOT / "experiments" / "local_vlm_vision_results.json"
HOSTED = ROOT / "experiments" / "director_repeatability_blind.json"


@pytest.fixture(scope="module")
def res():
    assert RESULTS.is_file(), f"missing artefact: {RESULTS}"
    return json.loads(RESULTS.read_text())


# ------------------------------------------------------------------ identity

def test_artefact_parses_and_names_its_question(res):
    assert res["experiment"] == "local_vlm_vision"
    assert "serving stack" in res["question"]
    assert res["date"]


def test_model_identity_is_complete(res):
    """Model id, REVISION and a content hash. A revision is a name; the
    hash says the bytes were the ones measured."""
    ident = res["model_identity"]
    for key in ("model_id", "revision", "weights_sha256", "weight_files"):
        assert key in ident, f"identity is missing {key}"
    assert ident["model_id"].startswith("Qwen/")
    assert len(ident["revision"]) == 40, "the revision must be a commit sha"
    assert len(ident["weights_sha256"]) == 64
    assert ident["weight_files"], "no weight files hashed"
    for f in ident["weight_files"]:
        assert len(f["sha256"]) == 64
        assert f["bytes"] > 0


def test_every_model_tried_is_recorded_with_its_outcome(res):
    """Which models were tried, and what happened to each."""
    attempts = res["model_attempts"]
    assert attempts, "no model attempts recorded"
    ids = [a.get("model_id") for a in attempts if a.get("model_id")]
    assert "Qwen/Qwen2-VL-2B-Instruct" in ids
    assert "Qwen/Qwen2.5-VL-3B-Instruct" in ids
    for a in attempts:
        assert a.get("outcome"), a
        assert a.get("role")
    primary = next(a for a in attempts if a["role"] == "PRIMARY")
    assert primary["weights_bytes"] > 0
    # a model that needs CPU offload is not the clean primary
    secondary = next(a for a in attempts if a["role"] == "SECONDARY")
    assert secondary.get("why_not_primary")


def test_runtime_identity_is_complete(res):
    """dtype, device and the versions, so the number can be attributed to a
    stack rather than to a model name."""
    rt = res["model_runtime"]
    for key in ("dtype", "resolved_device", "resolved_dtype", "class",
                "torch_version", "cuda_version", "gpu_name"):
        assert key in rt, f"runtime identity is missing {key}"
    assert rt["dtype"] in ("float16", "bfloat16", "float32")
    assert rt["gpu_total_mib"] and rt["gpu_total_mib"] > 0


def test_prompt_and_generation_parameters_are_recorded(res):
    p = res["prompt"]
    assert p["vision"].startswith("Is the person in this image SPEAKING")
    assert "exactly one word" in p["vision"]
    assert len(p["vision_sha256"]) == 64
    assert set(res["regimes"]) == {"greedy", "sampled"}
    assert res["regimes"]["greedy"]["do_sample"] is False
    assert res["regimes"]["sampled"]["do_sample"] is True
    assert res["regimes"]["sampled"]["temperature"] > 0
    assert isinstance(res["seed"], int)
    assert res["n_runs"] >= 8


def test_every_image_is_named_by_its_bytes(res):
    """The SHA-256 of the image bytes, for every frame of every condition.

    A frame whose bytes cannot be named cannot be re-checked."""
    for sname, stim in res["stimulus_by_stride"].items():
        for name, cond in stim["conditions"].items():
            assert len(cond["image_sha256"]) == cond["n"], (sname, name)
            for sha in cond["image_sha256"]:
                assert len(sha) == 64, (sname, name)
    conds = res["stimulus"]["conditions"]
    # the speak and the silent conditions must NOT be the same picture
    assert (conds["speak"]["image_sha256"]
            != conds["silent"]["image_sha256"])
    # the blank control is the same picture every time, by construction
    assert len(set(conds["blank"]["image_sha256"])) == 1


def test_frame_indices_are_recorded(res):
    conds = res["stimulus"]["conditions"]
    assert len(conds["speak"]["frame_indices"]) == conds["speak"]["n"]
    assert len(conds["silent"]["frame_indices"]) == conds["silent"]["n"]
    assert conds["blank"]["frame_indices"] is None
    for i in conds["speak"]["frame_indices"]:
        assert 0 <= i < res["stimulus"]["n_frames"]
    # the speaking frames must be spread over the session, not clustered
    spread = conds["speak"]["frame_indices"]
    assert spread[0] < spread[-1]
    assert spread[-1] - spread[0] > res["stimulus"]["n_frames"] // 4


def test_the_frame_labels_come_from_the_gate(res):
    """The rule this project uses: a frame label comes from the gate at that
    frame's OWN timestamp."""
    vad = res["stimulus"]["vad"]
    assert vad["threshold"] == 0.5
    assert vad["block_s"] == 0.032
    assert "centre" in vad["rule"]
    # A.nv12 is ~70% speech by the gate, and C.nv12 is silent throughout
    assert vad["A.wav_speech_fraction"] > 0.5
    assert vad["C.wav_speech_fraction"] == 0.0
    probs = vad["gate_prob_at_chosen_speak_frames"]
    assert len(probs) == res["stimulus"]["conditions"]["speak"]["n"]
    assert all(p >= 0.5 for p in probs), \
        "a chosen 'speaking' frame is not labelled speaking by the gate"


# ------------------------------------------------------------------ the stride

def test_the_stride_is_detected_and_recorded(res):
    """The stride is a physical property of the file, measured, not a
    constant in the source. The captures are 19,200 bytes per frame; the
    repo's readers assume 23,040."""
    det = res["stride_detection"]
    assert det, "no stride detection recorded"
    for name, d in det.items():
        scores = d["mean_abs_frame_diff_by_stride"]
        assert scores, name
        best = min(scores, key=lambda k: scores[k])
        assert str(d["detected_stride"]) == best, \
            f"{name}: the recorded stride is not the lowest-tornness one"
        assert d["detected_stride_name"] == "clean_19200", name
        # the wrong stride is catastrophically worse, and the record says so
        assert scores["19200"] < 1.0 < scores["23040"], name
        assert d["n_frames_at_detected_stride"] == 900, name


def test_both_strides_are_measured_and_they_differ(res):
    """The clean reading and the repo's torn reading are two cells of one
    table, and they are genuinely different pictures."""
    by_stride = res["stimulus_by_stride"]
    assert set(by_stride) == {"clean_19200", "torn_23040"}
    clean = by_stride["clean_19200"]
    torn = by_stride["torn_23040"]
    assert clean["n_frames"] == 900
    assert torn["n_frames"] == 750
    assert clean["fps_implied_by_this_stride"] > torn["fps_implied_by_this_stride"]
    # the same frame INDEX means a different picture at a different stride
    clean_sha = clean["conditions"]["speak"]["image_sha256"]
    torn_sha = torn["conditions"]["speak"]["image_sha256"]
    assert set(clean_sha).isdisjoint(set(torn_sha)), \
        "the clean and the torn arms are the same images; the control is void"
    assert res["primary_stride"] == "clean_19200"


def test_the_correction_and_the_superseded_run_are_kept_visible(res):
    """A corrected number is reported as a dated correction, not a silent
    overwrite."""
    corr = res["corrections"]
    assert corr, "no correction recorded"
    c = corr[0]
    assert c["date"]
    assert "23,040" in c["wrong"] and "19,200" in c["right"]
    assert "evidence" in c and "effect" in c
    sup = res["superseded_first_run"]
    assert sup["status"].startswith("SUPERSEDED")
    assert sup["stride"] == 23040
    assert "vision_task" in sup and "audit_strict_parser" in sup


# ------------------------------------------------------------------ the task

def test_conditions_have_equal_denominators(res):
    """EQUAL. Two conditions with different denominators is the shape of
    every wrong number in this project's history."""
    for sname, vt in res["vision_task_by_stride"].items():
        conds = vt["conditions"]
        ns = {name: c["n"] for name, c in conds.items()}
        assert set(ns) == {"speak", "silent", "blank"}, sname
        assert len(set(ns.values())) == 1, f"{sname}: denominators differ: {ns}"
        assert ns["speak"] == res["n_frames_per_condition"], sname
        for name, c in conds.items():
            assert len(c["texts"]) == c["n"], (sname, name)
            assert c["said_speaking"] + c["unparsed"] + c["errors"] == c["n"], \
                (sname, name)


def test_the_vision_rates_follow_from_the_counts(res):
    for sname, vt in res["vision_task_by_stride"].items():
        for name, c in vt["conditions"].items():
            assert c["said_speaking_rate"] == pytest.approx(
                c["said_speaking"] / c["n"]), (sname, name)
        assert vt["sensitivity_says_speaking_on_speech_frames"] == pytest.approx(
            vt["conditions"]["speak"]["said_speaking_rate"]), sname
        assert vt["margin_over_silent_person_present"] == pytest.approx(
            vt["sensitivity_says_speaking_on_speech_frames"]
            - vt["false_positive_rate_on_silent_person_present"]), sname


def test_degenerate_baselines_are_recorded(res):
    """always-SPEAKING scores FPR 1.000; always-NOT-SPEAKING scores
    sensitivity 0.000. Computed, not asserted by hand."""
    for sname, vt in res["vision_task_by_stride"].items():
        base = vt["degenerate_baselines"]
        assert base["always_speaking"]["fpr_silent"] == 1.0, sname
        assert base["always_speaking"]["fpr_blank"] == 1.0, sname
        assert base["always_speaking"]["sensitivity"] == 1.0, sname
        assert base["always_not_speaking"]["sensitivity"] == 0.0, sname
        assert base["always_not_speaking"]["fpr_silent"] == 0.0, sname
    # both degenerate directors must score 0.0 on the audit metric
    deg = res["controls"]["degenerate_directors"]
    assert deg["always_accept"]["headline"] == 0.0
    assert deg["always_reject"]["headline"] == 0.0


def test_the_stub_controls_are_zero(res):
    """THE CONTROL. If the deterministic stub moves, the harness is adding
    variance and every spread in this artefact is worthless."""
    ctl = res["controls"]
    assert ctl["stub_director"]["headline_spread"] == 0.0
    assert ctl["stub_director"]["n_distinct_headlines"] == 1
    assert len(ctl["stub_director"]["headlines"]) == res["n_runs"]
    assert len(set(ctl["stub_director"]["headlines"])) == 1
    assert ctl["stub_vision"]["spread"] == 0.0
    assert ctl["stub_vision"]["n_distinct_answers"] == 1


# ------------------------------------------------------------------ the spread

def test_vision_spread_is_recomputed_from_the_per_run_values(res):
    """RECOMPUTED, not read back. The recorded summary must agree with the
    per-run answers it claims to summarise."""
    for sname, per_cond in res["determinism_vision_by_stride"].items():
        for cond, r in per_cond.items():
            for regime in ("greedy", "sampled"):
                block = r[regime]
                assert len(block["per_run"]) == res["n_runs"], (sname, cond, regime)
                words = [p["answer"] for p in block["per_run"]]
                assert words == block["answers"]
                assert block["n_distinct_answers"] == len(set(words))
                scores = []
                for p in block["per_run"]:
                    if p["answer"] == "SPEAKING":
                        scores.append(1.0)
                    elif p["answer"] == "NOT SPEAKING":
                        scores.append(0.0)
                    else:
                        assert p["score"] is None
                        scores.append(None)
                live = [s for s in scores if s is not None]
                expected = (max(live) - min(live)) if live else None
                if expected is None:
                    assert block["spread"] is None
                else:
                    assert block["spread"] == pytest.approx(expected), \
                        (sname, cond, regime)
                n_speaking = sum(1 for w in words if w == "SPEAKING")
                assert block["n_saying_speaking"] == n_speaking
                assert block["said_speaking_rate"] == pytest.approx(
                    n_speaking / len(words))


def test_audit_spread_is_recomputed_from_the_per_run_headlines(res):
    """The SAME statistic the repo uses: max - min over the per-run
    headlines."""
    au = res["determinism"]["audit"]
    for regime in ("greedy", "sampled"):
        block = au[regime]
        assert len(block["runs"]) == res["n_runs"], regime
        headlines = [r["headline"] for r in block["runs"]
                     if r.get("headline") is not None]
        assert headlines == block["headlines"], regime
        assert block["n_scored"] == len(headlines)
        if headlines:
            assert block["headline_spread"] == pytest.approx(
                max(headlines) - min(headlines)), regime
            assert block["n_distinct_headlines"] == len(set(headlines))
        else:
            assert block["headline_spread"] is None


def test_the_audit_prompt_is_byte_identical_to_the_hosted_one(res):
    """The local model must be asked EXACTLY what the hosted model was
    asked. The SHA-256 is recomputed from the repo's own builder."""
    import debrief
    pairs = debrief.audit_pairs(20261019)
    prompt, materials = debrief.build_audit_prompt("blind", pairs, None)
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    assert sha == res["determinism"]["audit"]["prompt_sha256"]
    assert res["determinism"]["audit"]["prompt_chars"] == len(prompt)
    # and it carries NO material at all, which is why the frame-stride
    # defect cannot reach the audit arm
    assert materials == 0
    assert res["determinism"]["audit"]["materials_chars"] == 0


def test_the_audit_prompt_matches_the_committed_hosted_prompt(res):
    import debrief
    pairs = debrief.audit_pairs(20261019)
    prompt, _ = debrief.build_audit_prompt("blind", pairs, None)
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    hosted = json.loads(HOSTED.read_text())
    assert sha == hosted["results"][0]["prompt_sha256"], \
        "the local audit prompt is NOT byte-identical to the hosted one"
    assert sha == res["determinism"]["audit"]["prompt_sha256"]


# ------------------------------------------------------------------ the table

def test_the_hosted_row_is_the_committed_measurement(res):
    """The hosted number in the comparison table is the repo's own committed
    artefact, copied -- not re-measured here, and not a new claim."""
    hosted = json.loads(HOSTED.read_text())
    e = hosted["results"][0]
    ref = res["hosted_reference"]
    assert ref["model"] == e["model"]
    assert ref["mode"] == e["mode"]
    assert ref["n_runs"] == e["n_requested"]
    assert ref["headline_spread"] == pytest.approx(e["result"]["headline_spread"])
    assert ref["n_distinct_headlines"] == e["result"]["n_distinct_headlines"]
    assert ref["headlines"] == e["result"]["headlines"]
    assert ref["control_stub_headline_spread"] == e["control"]["headline_spread"]


def test_the_hosted_sampled_cell_is_not_guessed(res):
    """The hosted model at temperature > 0 was never run by this project and
    cannot be measured from this machine. The cell must be EMPTY, not
    filled in with a guess."""
    ref = res["hosted_reference"]
    assert ref["sampled"]["headline_spread"] is None
    assert "NOT MEASURED" in ref["sampled"]["note"]
    table = res["comparison_table"]["audit_headline_spread"]
    assert table["sampled"]["hosted"] is None


def test_the_comparison_table_agrees_with_the_blocks_it_summarises(res):
    au = res["determinism"]["audit"]
    table = res["comparison_table"]["audit_headline_spread"]
    assert table["greedy"]["local"] == au["greedy"]["headline_spread"]
    assert table["sampled"]["local"] == au["sampled"]["headline_spread"]
    assert table["greedy"]["hosted"] == res["hosted_reference"]["headline_spread"]
    # the vision cells must carry BOTH strides
    assert set(res["comparison_table"]["vision_answer_spread"]["by_stride"]) == \
        {"clean_19200", "torn_23040"}


def test_the_verdict_follows_from_the_numbers(res):
    """A verdict that could not come out the other way is not a verdict."""
    au = res["determinism"]["audit"]
    local_greedy = au["greedy"]["headline_spread"]
    v = res["verdict"]
    assert v["local_greedy_audit_spread"] == local_greedy
    assert v["hosted_greedy_audit_spread"] == \
        res["hosted_reference"]["headline_spread"]
    if local_greedy is None or au["greedy"]["n_scored"] == 0:
        assert v["hypothesis_A_hosting_not_task"] is None
    elif local_greedy == 0.0 and v["local_model_discriminates"]:
        assert v["hypothesis_A_hosting_not_task"] is True
        assert v["hypothesis_B_task_ambiguous"] is False
    elif local_greedy == 0.0:
        assert v["hypothesis_A_hosting_not_task"] is None
    else:
        assert v["hypothesis_A_hosting_not_task"] is False
        assert v["hypothesis_B_task_ambiguous"] is True
    assert v["reason"]
    # the vision verdict must also be present for both strides
    assert set(v["vision_spread_by_stride"]) == {"clean_19200", "torn_23040"}


def test_the_secondary_model_is_measured_the_same_way(res):
    """The second model is not a footnote: it is the same measurement, and
    its spreads recompute the same way."""
    sec = res.get("secondary_model")
    assert sec, "no secondary model block in the artefact"
    ident = sec["model_identity"]
    assert ident["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert len(ident["revision"]) == 40
    assert len(ident["weights_sha256"]) == 64
    assert ident["weights_sha256"] != res["model_identity"]["weights_sha256"]
    # the same two strides, the same equal denominators
    assert set(sec["vision_task_by_stride"]) == {"clean_19200", "torn_23040"}
    for sname, vt in sec["vision_task_by_stride"].items():
        ns = {n: c["n"] for n, c in vt["conditions"].items()}
        assert len(set(ns.values())) == 1, sname
        assert vt["conditions"]["speak"]["said_speaking"] + \
            vt["conditions"]["speak"]["unparsed"] + \
            vt["conditions"]["speak"]["errors"] == vt["conditions"]["speak"]["n"]
    # and its spreads recompute from its own per-run answers
    for sname, per in sec["determinism_vision_by_stride"].items():
        for cond, r in per.items():
            for regime in ("greedy", "sampled"):
                words = [p["answer"] for p in r[regime]["per_run"]]
                assert words == r[regime]["answers"]
                live = [1.0 if w == "SPEAKING" else 0.0 if w == "NOT SPEAKING"
                        else None for w in words]
                live = [s for s in live if s is not None]
                assert r[regime]["spread"] == pytest.approx(
                    max(live) - min(live)), (sname, cond, regime)
    # and its AUDIT spread recomputes from its own per-run headlines, on the
    # same byte-identical prompt as the primary and as the hosted run
    au = sec["determinism"].get("audit")
    assert au, "the secondary model has no audit block"
    assert au["prompt_sha256"] == res["determinism"]["audit"]["prompt_sha256"]
    for regime in ("greedy", "sampled"):
        block = au[regime]
        headlines = [r["headline"] for r in block["runs"]
                     if r.get("headline") is not None]
        assert headlines == block["headlines"], regime
        assert block["n_scored"] == len(headlines)
        if headlines:
            assert block["headline_spread"] == pytest.approx(
                max(headlines) - min(headlines)), regime


def test_both_local_models_are_stable_at_temperature_zero(res):
    """The claim rests on BOTH local models being stable under greedy
    decoding, on the byte-identical hosted prompt. If either moved, the
    comparison would be a property of one model rather than of the task."""
    au = res["determinism"]["audit"]
    assert au["greedy"]["headline_spread"] == 0.0
    assert au["greedy"]["n_distinct_headlines"] == 1
    assert au["greedy"]["n_scored"] == res["n_runs"]
    sec_au = res["secondary_model"]["determinism"]["audit"]
    assert sec_au["greedy"]["headline_spread"] == 0.0
    assert sec_au["greedy"]["n_distinct_headlines"] == 1
    assert sec_au["greedy"]["n_scored"] == res["n_runs"]
    # and both DISCRIMINATE, so the constant headline is not a degenerate one
    assert res["verdict"]["local_model_discriminates"] is True
    assert res["secondary_model"]["verdict"]["local_model_discriminates"] is True
    # and both DO move when allowed to sample, so the spread is not an
    # artefact of a harness that cannot measure anything
    assert au["sampled"]["headline_spread"] > 0.0
    assert sec_au["sampled"]["headline_spread"] > 0.0


def test_the_secondary_model_agrees_with_the_primary_about_the_frames(res):
    """The clean-versus-torn question was asked of BOTH models. If the two
    models disagreed about whether the stride moves the answer, the
    artefact would have to say which is right -- so it records that they
    agree."""
    sec = res["secondary_model"]
    for sname in ("clean_19200", "torn_23040"):
        for cond in ("speak", "silent", "blank"):
            p = res["determinism_vision_by_stride"][sname][cond]
            q = sec["determinism_vision_by_stride"][sname][cond]
            assert p["greedy"]["spread"] == q["greedy"]["spread"], (sname, cond)
            assert p["sampled"]["spread"] == q["sampled"]["spread"], (sname, cond)
    # and the clean and the torn arms of the PRIMARY agree too
    for cond in ("speak", "silent", "blank"):
        c = res["determinism_vision_by_stride"]["clean_19200"][cond]
        t = res["determinism_vision_by_stride"]["torn_23040"][cond]
        assert c["greedy"]["spread"] == t["greedy"]["spread"], cond
        assert c["sampled"]["spread"] == t["sampled"]["spread"], cond


def test_no_credentials_in_the_artefact(res):
    """No secrets in a committed file."""
    raw = RESULTS.read_text()
    for needle in ("CF_TOKEN", "Bearer ", "api_key", "apiKey", "sk-"):
        assert needle not in raw, f"{needle} appears in the artefact"
