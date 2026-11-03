"""Configuration QA: env parsing, defaults, and consistency between config.py, .env.example and the README."""
import re
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.config import ROOT, Settings

ENV_EXAMPLE = (ROOT / ".env.example").read_text()
README = (ROOT / "README.md").read_text()


def env_names_in_config():
    src = (ROOT / "src" / "config.py").read_text()
    return set(re.findall(r'"([A-Z][A-Z0-9_]{3,})"', src)) & {n for n in re.findall(r'"([A-Z][A-Z0-9_]{3,})"', src) if n.startswith(("GEMINI_", "GROQ_", "LLM_"))}


def example_values():
    out = {}
    for line in ENV_EXAMPLE.splitlines():
        m = re.match(r"^#?\s*([A-Z][A-Z0-9_]+)=(.*?)(\s+#.*)?$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def test_every_env_var_read_by_config_is_documented_in_env_example_and_readme():
    names = env_names_in_config()
    assert {"LLM_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL", "LLM_MAX_RETRIES"} <= names
    for n in names:
        assert n in example_values(), f"{n} missing from .env.example"
        assert n in README, f"{n} missing from README"


def test_env_example_has_no_secrets_and_defaults_match_code(monkeypatch):
    ex = example_values()
    assert ex["GROQ_API_KEY"] == "" and ex["GEMINI_API_KEY"] == ""
    for var in ("LLM_PROVIDER", "GROQ_MODEL", "GEMINI_MODEL", "LLM_MAX_RETRIES", "GEMINI_MAX_OUTPUT_TOKENS", "GEMINI_MAX_OUTPUT_TOKENS_SHORT", "LLM_MAX_OUTPUT_TOKENS",
                "LLM_MAX_OUTPUT_TOKENS_SHORT", "GEMINI_THINKING_LEVEL", "GEMINI_THINKING_BUDGET"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.llm_provider == ex["LLM_PROVIDER"] == "groq"
    assert s.groq_model == ex["GROQ_MODEL"] and s.gemini_model == ex["GEMINI_MODEL"]
    assert s.llm_max_retries == int(ex["LLM_MAX_RETRIES"]) and s.max_output_tokens == int(ex["GEMINI_MAX_OUTPUT_TOKENS"])
    assert s.max_output_tokens_short == int(ex["GEMINI_MAX_OUTPUT_TOKENS_SHORT"]) and s.gemini_thinking_level is None and s.gemini_thinking_budget is None


def test_env_parsing_normalisation_and_precedence(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "  GROQ ")
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "500")
    assert Settings().llm_provider == "groq" and Settings().max_output_tokens == 500
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "700")                 # provider-neutral name wins over the legacy GEMINI_ name
    assert Settings().max_output_tokens == 700
    monkeypatch.setenv("LLM_PROVIDER", "")
    assert Settings().llm_provider == "groq"                           # empty falls back to the default


@pytest.mark.parametrize("var,val,needle", [("LLM_MAX_RETRIES", "abc", "LLM_MAX_RETRIES must be an integer"), ("LLM_MAX_RETRIES", "-1", ">= 0"),
                                            ("GEMINI_MAX_OUTPUT_TOKENS", "0", ">= 1"), ("LLM_MAX_OUTPUT_TOKENS_SHORT", "x1", "must be an integer"),
                                            ("GEMINI_THINKING_BUDGET", "-5", ">= 0")])
def test_invalid_numeric_env_values_give_clear_errors(monkeypatch, var, val, needle):
    monkeypatch.setenv(var, val)
    with pytest.raises(ValueError, match=needle):
        Settings()


def test_app_shows_configuration_error_instead_of_a_stack_trace(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "not-a-number")
    st.cache_resource.clear()
    at = AppTest.from_file(str(Path(ROOT / "app.py")), default_timeout=60).run()
    assert not at.exception and any("Configuration error" in e.value and "LLM_MAX_RETRIES" in e.value for e in at.error)
    st.cache_resource.clear()


def test_active_model_and_key_flag_follow_selected_provider():
    s = Settings(llm_provider="groq", groq_api_key="k" * 12, gemini_api_key="", groq_model="gm", gemini_model="mm")
    assert s.has_api_key and s.active_model == "gm"
    s.llm_provider = "gemini"
    assert not s.has_api_key and s.active_model == "mm"
