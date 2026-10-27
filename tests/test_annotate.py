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


def test_context_size_grows_with_frames():
    annotator = OllamaAnnotator.__new__(OllamaAnnotator)  # bez połączenia z Ollamą
    small, big = annotator._context_size(4), annotator._context_size(24)
    assert small == 8192
    # zmierzone: 24 klatki qwen2.5vl w Ollamie to ~25 tys. tokenów
    assert big % 4096 == 0 and big >= 24 * 1040 + 1500


def test_gemini_retries_on_rate_limit(monkeypatch):
    errors = pytest.importorskip("google.genai.errors")
    from tacticllm.annotate import GeminiAnnotator

    calls = []

    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
            if len(calls) == 2:
                raise errors.ServerError(503, {"error": {"message": "high demand", "status": "UNAVAILABLE"}})
            return "ok"

    annotator = GeminiAnnotator.__new__(GeminiAnnotator)  # bez klucza i połączenia
    annotator.client = type("Client", (), {"models": Models()})()
    monkeypatch.setattr(annotate_mod.time, "sleep", lambda s: None)
    assert annotator._generate(model="m") == "ok" and len(calls) == 3


def test_gemini_does_not_retry_other_errors(monkeypatch):
    errors = pytest.importorskip("google.genai.errors")
    from tacticllm.annotate import GeminiAnnotator

    class Models:
        def generate_content(self, **kwargs):
            raise errors.ClientError(400, {"error": {"message": "bad", "status": "INVALID_ARGUMENT"}})

    annotator = GeminiAnnotator.__new__(GeminiAnnotator)
    annotator.client = type("Client", (), {"models": Models()})()
    with pytest.raises(errors.ClientError):
        annotator._generate(model="m")


def test_gemini_daily_quota_stops_whole_run(tmp_path, monkeypatch):
    errors = pytest.importorskip("google.genai.errors")
    from tacticllm.annotate import GeminiAnnotator

    calls = []

    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            raise errors.ClientError(429, {"error": {
                "message": "Generate content requests per day limit exceeded on Free Tier",
                "status": "RESOURCE_EXHAUSTED",
                "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}})

    gemini = GeminiAnnotator.__new__(GeminiAnnotator)
    gemini.client = type("Client", (), {"models": Models()})()
    monkeypatch.setattr(annotate_mod, "make_annotator", lambda *a, **k: lambda seg: gemini._generate(model="m"))
    monkeypatch.setattr(annotate_mod.time, "sleep", lambda s: pytest.fail("nie czekamy na limit dzienny"))

    result = annotate(None, segs(3), tmp_path / "clips", tmp_path / "out.json", backend="gemini")
    assert result == {} and len(calls) == 1  # bez ponowień i bez kolejnych punktów


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
