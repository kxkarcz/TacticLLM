from tacticllm.config import Settings


def test_api_keys_read_from_env_file_without_prefix(tmp_path, monkeypatch):
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("GOOGLE_API_KEY=g-test\nANTHROPIC_API_KEY=a-test\nTACTIC_ANNOTATOR=gemini\n", encoding="utf-8")
    s = Settings(_env_file=env)
    assert s.google_api_key == "g-test" and s.anthropic_api_key == "a-test" and s.annotator == "gemini"


def test_gemini_key_alias(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gem")
    assert Settings(_env_file=None).google_api_key == "gem"
