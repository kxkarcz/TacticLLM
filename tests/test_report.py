import pytest

from tacticllm import report
from tacticllm.models import MatchInfo, PlayerProfile, TeamProfile


@pytest.fixture
def provider(monkeypatch):
    def set_provider(name, **extra):
        monkeypatch.setattr(report.settings, "report_provider", name)
        for key, value in extra.items():
            monkeypatch.setattr(report.settings, key, value)
    return set_provider


def test_gemini_report_model(provider):
    pytest.importorskip("agno.models.google")
    provider("gemini", gemini_report_model="gemini-test", google_api_key="g-key")
    model = report._model()
    assert type(model).__name__ == "Gemini" and model.id == "gemini-test" and model.api_key == "g-key"


def test_claude_report_model(provider):
    pytest.importorskip("agno.models.anthropic")
    provider("anthropic", report_model="claude-test", anthropic_api_key="")
    model = report._model()
    assert type(model).__name__ == "Claude" and model.id == "claude-test" and model.api_key is None  # pusty -> brak


def test_unknown_provider(provider):
    provider("openai")
    with pytest.raises(ValueError, match="Nieznany dostawca raportu"):
        report._model()


def test_compact_drops_notes_only_for_local_model(provider):
    stats = {"points": [{"index": 0, "notes": "długi opis", "decisive_moment": "smecz"}]}
    provider("gemini")
    assert report._compact(stats)["points"][0]["notes"] == "długi opis"
    provider("ollama")
    assert report._compact(stats)["points"][0] == {"index": 0, "decisive_moment": "smecz"}


class FakeAgent:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def run(self, prompt):
        from agno.run.agent import RunOutput

        self.calls += 1
        status, content = self.outputs.pop(0)
        return RunOutput(status=status, content=content)


def test_report_retries_when_model_is_overloaded(monkeypatch):
    base = pytest.importorskip("agno.run.base")
    monkeypatch.setattr("time.sleep", lambda s: None)
    agent = FakeAgent([(base.RunStatus.error, '{"error": {"code": 503}}'), (base.RunStatus.completed, "# Raport")])
    assert report.run_with_retry(agent, "prompt") == "# Raport" and agent.calls == 2


def test_report_failure_is_an_error_not_a_report(monkeypatch):
    base = pytest.importorskip("agno.run.base")
    monkeypatch.setattr("time.sleep", lambda s: None)
    agent = FakeAgent([(base.RunStatus.error, '{"error": {"code": 503}}')] * 3)
    with pytest.raises(RuntimeError, match="Nie udało się wygenerować raportu"):
        report.run_with_retry(agent, "prompt", attempts=3)


def test_match_text_lists_both_pairs():
    info = MatchInfo(title="Finał", teams=[
        TeamProfile(name="Niebiescy", start_side="near", players=[PlayerProfile(name="Jan", hand="left", lane="reves")]),
    ])
    text = report._match_text(info)
    assert "Mecz: Finał" in text and "Para A = Niebiescy: Jan (ręka: left, strona: reves)" in text
    assert "Para B: brak danych o zawodnikach" in text
