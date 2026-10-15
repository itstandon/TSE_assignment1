import pytest
from housing_trends.llm import llm_from_env


@pytest.fixture
def env(monkeypatch):
    for k in ("LLM_PROVIDER", "LLM2_API_KEY", "GOOGLE_API_KEY", "LLM2_MODEL", "LLM2_BASE_URL", "LLM_SEED"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_default_is_local_qwen_without_key(env):
    assert llm_from_env().name == "qwen2.5-7b-instruct@seed42"


def test_hosted_provider_still_needs_its_key(env):
    env.setenv("LLM_PROVIDER", "groq")
    with pytest.raises(RuntimeError, match="LLM2_API_KEY"):
        llm_from_env()


def test_unknown_provider_rejected(env):
    env.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(RuntimeError, match="qwen"):
        llm_from_env()


def test_unreachable_server_is_a_clear_runtime_error(env):
    env.setattr("housing_trends.llm.time.sleep", lambda s: None)   # skip post()'s back-off between retries
    env.setenv("LLM2_BASE_URL", "http://127.0.0.1:9/v1")      # discard port: nothing listens
    with pytest.raises(RuntimeError, match="cannot reach the LLM server"):
        llm_from_env()("system", "user")
