import json

import pytest

from tacticllm import annotate as annotate_mod
from tacticllm.annotate import PROMPT, TACTIC_GUIDE, OllamaAnnotator, annotate, frame_times, make_annotator
from tacticllm.models import PointAnnotation, Segment, load_annotations


def ann(winner="near"):
    return PointAnnotation(is_rally=True, winner_side=winner, ending="winner", last_shot="smash")


def segs(n):
    return [Segment(index=i, start_s=i * 10, end_s=i * 10 + 5) for i in range(n)]


def test_frame_times_short_point_has_minimum_frames():
    times = frame_times(5.0, 5.5, fps=3, max_frames=64)
    assert len(times) == 4 and times[0] == 5.0 and times[-1] == pytest.approx(5.5)


def test_unknown_backend():
    with pytest.raises(ValueError, match="Nieznany backend"):
        make_annotator("abc", None, None)


def test_prompt_lists_every_tactic_tag():
    from typing import get_args

    from tacticllm.models import TacticTag

    text = PROMPT.format(source="x", tactics=TACTIC_GUIDE)
    for tag in get_args(TacticTag):
        assert tag in text, tag


def test_annotation_parses_model_json():
    raw = json.dumps({"is_rally": True, "winner_side": "far", "ending": "forced_error", "last_shot": "vibora",
                      "tactics_near": ["chiquita"], "tactics_far": ["smash_x3_x4"], "scoreboard": "6-4 2-1"})
    a = PointAnnotation.model_validate_json(raw)
    assert a.winner_side == "far" and a.tactics_far == ["smash_x3_x4"] and a.shots is None


def test_annotation_rejects_unknown_tactic():
    with pytest.raises(ValueError):
        PointAnnotation(is_rally=True, winner_side="near", ending="winner", last_shot="smash", tactics_near=["tiki_taka"])


def test_context_size_grows_with_frames(monkeypatch):
    monkeypatch.setattr(annotate_mod.settings, "ollama_frame_width", 896)
    annotator = OllamaAnnotator.__new__(OllamaAnnotator)  # bez połączenia z Ollamą
    small, big = annotator._context_size(4), annotator._context_size(64)
    assert small == 8192
    assert big % 4096 == 0 and big >= 64 * (896 * 504) / (28 * 28)


def test_annotate_saves_progress_resumes_and_skips_errors(tmp_path, monkeypatch):
    calls = []

    def fake_annotator(seg):
        calls.append(seg.index)
        if seg.index == 2:
            raise RuntimeError("timeout")
        return ann("near" if seg.index % 2 == 0 else "far")

    monkeypatch.setattr(annotate_mod, "make_annotator", lambda *a, **k: fake_annotator)
    out = tmp_path / "annotations.json"
    out.write_text(json.dumps({"0": ann("far").model_dump()}), encoding="utf-8")  # wcześniej przerwany przebieg

    result = annotate(None, segs(4), tmp_path / "clips", out, backend="ollama")

    assert calls == [1, 2, 3]  # punkt 0 był już gotowy
    assert sorted(result) == [0, 1, 3]  # punkt 2 pominięty po błędzie
    assert result[0].winner_side == "far"
    assert load_annotations(out) == result
