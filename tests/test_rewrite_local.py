"""Tests for the local-only Layer B path: paid-host gate, model preflight, chunking."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rewrite_text  # noqa: E402
from rewrite_text import (  # noqa: E402
    _check_remote,
    _model_matches,
    default_base_url,
    resolve_model,
    split_chunks,
)


# --- paid-inference host gate --------------------------------------------


PAID_URLS = [
    "https://api.openai.com",
    "https://api.anthropic.com",
    "https://generativelanguage.googleapis.com",
    "https://api.mistral.ai",
    "https://openrouter.ai",
    "https://api.groq.com",
    "https://api-inference.huggingface.co",
    "https://my-resource.openai.azure.com",
]


@pytest.mark.parametrize("url", PAID_URLS)
def test_paid_hosts_refused_even_with_allow_remote(url):
    """--allow-remote is for your own LAN box, not for a metered vendor."""
    with pytest.raises(SystemExit) as exc:
        _check_remote(url, allow_remote=True, allow_paid=False)
    assert "paid" in str(exc.value).lower() or "metered" in str(exc.value).lower()


@pytest.mark.parametrize("url", PAID_URLS)
def test_paid_hosts_allowed_with_explicit_opt_in(url):
    _check_remote(url, allow_remote=True, allow_paid=True)


def test_paid_host_without_allow_remote_still_refused():
    with pytest.raises(SystemExit):
        _check_remote("https://api.openai.com", allow_remote=False, allow_paid=False)


def test_paid_gate_does_not_block_a_self_hosted_lan_box():
    """A local vLLM on the LAN is exactly what we want people using."""
    _check_remote("http://192.168.1.50:8000", allow_remote=True, allow_paid=False)


def test_loopback_is_never_treated_as_paid():
    _check_remote("http://127.0.0.1:11434", allow_remote=False, allow_paid=False)
    _check_remote("http://localhost:8080", allow_remote=False, allow_paid=False)


def test_paid_gate_matches_subdomains_not_lookalikes():
    """openai.com.evil.test must not be allowed through, and must not be blocked
    as if it were the real vendor either -- it is just a non-loopback host."""
    with pytest.raises(SystemExit) as exc:
        _check_remote("https://api.openai.com.evil.test", allow_remote=False, allow_paid=False)
    assert "loopback" in str(exc.value).lower()


def test_cli_refuses_paid_host_without_contacting_it(monkeypatch, tmp_path):
    """The gate must fire before the model preflight opens a connection."""
    import subprocess

    src = tmp_path / "doc.txt"
    src.write_text("Some prose to rewrite.\n")

    calls: list[str] = []

    def boom(*args, **kwargs):
        calls.append("contacted")
        raise AssertionError("must not contact a refused host")

    monkeypatch.setattr(rewrite_text, "list_served_models", boom)
    monkeypatch.setattr(
        sys, "argv",
        ["rewrite_text.py", str(src), "--backend", "openai-compatible", "--model", "gpt-4o",
         "--base-url", "https://api.openai.com", "--allow-remote"],
    )
    with pytest.raises(SystemExit) as exc:
        rewrite_text.main()
    assert "metered" in str(exc.value).lower()
    assert not calls


def test_allow_paid_env_flag(monkeypatch):
    monkeypatch.setenv("WATERMARKS_ALLOW_PAID_API", "1")
    assert rewrite_text._flag_env("WATERMARKS_ALLOW_PAID_API") is True
    monkeypatch.setenv("WATERMARKS_ALLOW_PAID_API", "0")
    assert rewrite_text._flag_env("WATERMARKS_ALLOW_PAID_API") is False


# --- per-backend defaults -------------------------------------------------


def test_default_base_url_per_backend():
    assert "11434" in default_base_url("ollama")
    assert "8080" in default_base_url("llamacpp")


def test_default_base_urls_are_loopback():
    for backend in ("ollama", "llamacpp", "openai-compatible"):
        url = default_base_url(backend)
        assert "127.0.0.1" in url or "localhost" in url


# --- model matching / preflight ------------------------------------------


@pytest.mark.parametrize(
    "requested,served,expected",
    [
        ("llama3.2", "llama3.2:latest", True),
        ("llama3.2:latest", "llama3.2:latest", True),
        ("llama3.2:3b", "llama3.2:3b", True),
        ("qwen2.5:7b", "llama3.2:latest", False),
        ("llama3.2:3b", "llama3.2:1b", False),
        ("llama3.2", "llama3.2:1b", True),
    ],
)
def test_model_matches(requested, served, expected):
    assert _model_matches(requested, served) is expected


def test_resolve_model_auto_picks_a_served_recommendation():
    picked = resolve_model("auto", served=["qwen2.5:7b", "nomic-embed-text:latest"], tier="tier1-low")
    assert picked == "qwen2.5:7b"


def test_resolve_model_auto_falls_back_to_whatever_is_served():
    """A model we do not recommend still beats failing outright."""
    picked = resolve_model("auto", served=["some-local-gguf:latest"], tier="tier1-low")
    assert picked == "some-local-gguf:latest"


def test_resolve_model_auto_with_nothing_served_names_the_pull_command():
    with pytest.raises(SystemExit) as exc:
        resolve_model("auto", served=[], tier="tier1-low")
    assert "pull" in str(exc.value)


def test_resolve_model_explicit_missing_names_the_pull_command():
    with pytest.raises(SystemExit) as exc:
        resolve_model("qwen2.5:7b", served=["llama3.2:latest"], tier="tier1-low")
    assert "qwen2.5:7b" in str(exc.value)
    assert "pull" in str(exc.value)


def test_resolve_model_explicit_present_passes_through():
    assert resolve_model("llama3.2", served=["llama3.2:latest"], tier="tier1-low") == "llama3.2"


def test_resolve_model_skips_preflight_when_served_is_unknown():
    """If we could not list models, do not block the run on a guess."""
    assert resolve_model("llama3.2", served=None, tier="tier1-low") == "llama3.2"


def test_resolve_model_auto_without_listing_still_returns_a_recommendation():
    picked = resolve_model("auto", served=None, tier="tier1-low")
    assert picked and picked != "auto"


# --- chunking -------------------------------------------------------------


def test_short_text_is_one_chunk():
    text = "One short paragraph."
    assert split_chunks(text, 1000) == [text]


def test_paragraphs_group_under_the_budget():
    text = "\n\n".join([f"Paragraph number {i} with some words in it." for i in range(10)])
    chunks = split_chunks(text, 120)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 120 or "\n\n" not in c


def test_chunking_preserves_all_paragraph_content_in_order():
    paras = [f"Paragraph {i} carries fact {i * 7} and name Alpha{i}." for i in range(12)]
    text = "\n\n".join(paras)
    rejoined = "\n\n".join(split_chunks(text, 100))
    for p in paras:
        assert p in rejoined
    # order preserved
    positions = [rejoined.index(p) for p in paras]
    assert positions == sorted(positions)


def test_oversized_paragraph_splits_on_sentences():
    para = " ".join([f"Sentence number {i} is here." for i in range(60)])
    chunks = split_chunks(para, 200)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks), [len(c) for c in chunks]


def test_single_unbreakable_run_is_hard_split_not_dropped():
    """A 5k-character run with no sentence breaks must still be chunked."""
    blob = "x" * 5000
    chunks = split_chunks(blob, 500)
    assert len(chunks) >= 10
    assert "".join(chunks) == blob


def test_split_chunks_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        split_chunks("text", 0)


def test_empty_text_yields_no_chunks():
    assert split_chunks("", 100) == []
    assert split_chunks("   \n\n  ", 100) == []


def test_chunks_never_exceed_budget_for_normal_prose():
    paras = [f"This is paragraph {i}. It has two sentences. " * 3 for i in range(20)]
    text = "\n\n".join(paras)
    for chunk in split_chunks(text, 400):
        assert len(chunk) <= 400 * 1.3


# --- end-to-end rewrite with a fake local backend ------------------------


def _fake_kwargs(**overrides):
    kwargs = dict(
        backend="ollama",
        model="llama3.2",
        base_url="http://127.0.0.1:11434",
        api_key=None,
        strength="paraphrase",
        lang="French",
        original_lang="English",
        timeout=5.0,
        layer_a_after=True,
        temperature=0.9,
        candidates=1,
    )
    kwargs.update(overrides)
    return kwargs


def test_rewrite_chunks_a_long_document(monkeypatch):
    calls: list[str] = []

    def fake_call(base_url, model, prompt, timeout, temperature):
        calls.append(prompt)
        return f"REWRITTEN-{len(calls)}"

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_call)

    paras = [f"Paragraph {i} with enough words to matter here." for i in range(20)]
    text = "\n\n".join(paras)

    out, info = rewrite_text.rewrite(text, max_chunk_chars=150, **_fake_kwargs())

    assert info["chunks"] > 1
    assert len(calls) == info["chunks"]
    assert "REWRITTEN-1" in out


def test_rewrite_without_chunking_makes_one_call(monkeypatch):
    calls: list[str] = []

    def fake_call(base_url, model, prompt, timeout, temperature):
        calls.append(prompt)
        return "REWRITTEN"

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_call)
    out, info = rewrite_text.rewrite("Short text.", max_chunk_chars=10_000, **_fake_kwargs())
    assert len(calls) == 1
    assert info.get("chunks", 1) == 1


def test_llamacpp_backend_routes_through_openai_transport(monkeypatch):
    seen = {}

    def fake_call(base_url, model, prompt, api_key, timeout, temperature):
        seen["base_url"] = base_url
        seen["api_key"] = api_key
        return "OUT"

    monkeypatch.setattr(rewrite_text, "call_openai_compatible", fake_call)
    rewrite_text.rewrite(
        "Text.", **_fake_kwargs(backend="llamacpp", base_url="http://127.0.0.1:8080")
    )
    assert seen["base_url"] == "http://127.0.0.1:8080"
    assert seen["api_key"] is None, "llamacpp must never send an API key"
