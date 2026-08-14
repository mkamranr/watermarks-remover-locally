#!/usr/bin/env python3
"""Layer B rewrite hook for statistical (token-sampling) watermarks.

Runs against a self-hosted open-weight model. No metered inference API is
required, and reaching one takes two explicit opt-ins.

Backends:
  print-prompt       — emit prompt only (CI-safe, no model)
  ollama             — POST to a local Ollama /api/chat
  llamacpp           — POST to a local llama-server /v1/chat/completions
  openai-compatible  — any OpenAI-style /v1/chat/completions (vLLM, LM Studio, …)

Env (optional):
  WATERMARKS_REWRITE_BACKEND
  WATERMARKS_REWRITE_BASE_URL
  WATERMARKS_REWRITE_MODEL
  WATERMARKS_REWRITE_API_KEY      (env-only; never pass keys on argv)
  WATERMARKS_REWRITE_ALLOW_REMOTE (set to 1 to allow non-loopback endpoints)
  WATERMARKS_ALLOW_PAID_API       (set to 1 to allow metered vendor endpoints)

Security and cost notes:
  - Only http(s) endpoints are accepted; redirects are refused outright so an
    Authorization header (API key) can never be re-sent to an unvalidated host.
  - Non-loopback endpoints are denied unless WATERMARKS_REWRITE_ALLOW_REMOTE=1
    (or --allow-remote) is set explicitly.
  - Known metered inference vendors are denied on top of that, and need
    WATERMARKS_ALLOW_PAID_API=1 as a second, separate opt-in. Self-hosting on
    another box on your LAN only needs the first one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import cleaned_path, eprint, read_text_input, write_text_output  # noqa: E402
from local_models import (  # noqa: E402
    TIER_LOW,
    detect_hardware,
    family_for_origin,
    max_chunk_chars,
    pick_tier,
    recommend_models,
)
from text_unicode import clean_text  # noqa: E402

# Loopback defaults per backend. Ollama and llama-server listen on different
# ports, so a single shared default would silently point one of them at the
# other's socket.
_BACKEND_BASE_URLS = {
    "ollama": "http://127.0.0.1:11434",
    "llamacpp": "http://127.0.0.1:8080",
    "openai-compatible": "http://127.0.0.1:8000",
}

BACKENDS = ("print-prompt", "ollama", "llamacpp", "openai-compatible")

PROMPTS = {
    "paraphrase": (
        "Rewrite the following text so that it uses substantially different wording at "
        "the token level. Change clause order, connectors, and transition words; vary "
        "sentence boundaries and length; and replace both content words and function "
        "words where meaning allows. Preserve all facts, numbers, names, and technical "
        "identifiers. Do not add or remove claims. Output only the rewritten text.\n\n---\n{TEXT}"
    ),
    "humanize": (
        "Rewrite the following text so it reads as if a human wrote it from scratch. "
        "Vary sentence rhythm and length, replace formulaic AI-style transitions and "
        "filler with concrete natural phrasing, and use plain, varied wording. Preserve "
        "all facts, numbers, names, and technical identifiers. Do not add or remove "
        "claims. Output only the rewritten text.\n\n---\n{TEXT}"
    ),
    "code": (
        "Rewrite the natural-language parts of this code — comments, docstrings, and "
        "string literals — using different wording. Rename local variables, function "
        "parameters, and private helper names to semantically equivalent names. Preserve "
        "program behavior, public API names, and all values that affect output. Output "
        "only the rewritten code.\n\n---\n{TEXT}"
    ),
    "backtranslate_out": (
        "Translate the following text to {LANG}. Output only the translation.\n\n---\n{TEXT}"
    ),
    "backtranslate_back": (
        "Translate the following text to {ORIGINAL_LANG}. Preserve meaning; use natural "
        "phrasing. Output only the translation.\n\n---\n{TEXT}"
    ),
    "structural_outline": (
        "Extract a bullet outline of all claims and structure from the text "
        "(no full sentences). Output only the outline.\n\n---\n{TEXT}"
    ),
    "structural_write": (
        "Write a complete document from this outline in natural, varied human prose. "
        "Avoid formulaic transitions. Do not omit any bullet. Output only the document."
        "\n\n---\n{TEXT}"
    ),
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def _lexical_divergence(original: str, candidate: str) -> float:
    """Bigram Jaccard distance: 0.0 identical, 1.0 fully different."""
    a = _tokens(original)
    b = _tokens(candidate)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    ba = _bigrams(a)
    bb = _bigrams(b)
    union = ba | bb
    if not union:
        return 0.0
    return 1.0 - len(ba & bb) / len(union)


def _select_candidate(original: str, candidates: list[str]) -> tuple[str, list[float]]:
    """Pick the most lexically diverged rewrite, gently guarding extreme length drift."""
    scores: list[float] = []
    for cand in candidates:
        score = _lexical_divergence(original, cand)
        if original:
            ratio = len(cand) / len(original)
            if ratio > 2.0 or ratio < 0.5:
                score -= 0.15
        scores.append(score)
    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best_idx], scores


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _flag_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# Registrable domains of hosted, metered inference vendors. The point of this
# project is that a self-hosted open-weight model does the rewriting, so
# reaching one of these is both a per-token cost and a re-stamping risk: the
# vendor's own watermark can land in the "cleaned" output.
#
# Matched as exact host or dotted suffix, so api.openai.com and
# my-resource.openai.azure.com both hit, while openai.com.evil.test does not.
_PAID_INFERENCE_DOMAINS = frozenset(
    {
        "openai.com",
        "openai.azure.com",
        "anthropic.com",
        "mistral.ai",
        "cohere.ai",
        "cohere.com",
        "together.ai",
        "together.xyz",
        "groq.com",
        "deepseek.com",
        "perplexity.ai",
        "fireworks.ai",
        "openrouter.ai",
        "x.ai",
        "replicate.com",
        "anyscale.com",
        "generativelanguage.googleapis.com",
        "aiplatform.googleapis.com",
        "api-inference.huggingface.co",
    }
)


def _is_paid_host(host: str) -> bool:
    h = host.strip().lower().rstrip(".")
    return any(h == d or h.endswith("." + d) for d in _PAID_INFERENCE_DOMAINS)


def _check_remote(base_url: str, allow_remote: bool, allow_paid: bool = False) -> None:
    """Enforce the rewrite-endpoint allowlist.

    Two independent gates, because they guard different things:

    - *off-machine* (--allow-remote): sending content to another host at all.
      A self-hosted vLLM elsewhere on your LAN needs only this one.
    - *metered vendor* (WATERMARKS_ALLOW_PAID_API): sending content to a hosted
      inference API that bills per token and may re-stamp the output with its
      own watermark. Checked after the loopback test and before the remote one,
      so pointing at api.openai.com reports the cost problem rather than the
      generic "not loopback" message.

    Non-http(s) schemes (e.g. file://) are always refused.
    """
    u = urlparse(base_url)
    if u.scheme not in ("http", "https"):
        raise SystemExit(
            f"error: rewrite base URL must be http(s), got scheme '{u.scheme}': {base_url}"
        )
    host = u.hostname or ""
    if host in _LOOPBACK_HOSTS:
        return
    if _is_paid_host(host) and not allow_paid:
        raise SystemExit(
            f"error: '{host}' is a metered inference API. This tool is built to run on "
            "self-hosted open weights, and a vendor model can re-stamp the text with its "
            "own watermark.\n"
            "Run a local model instead (see doctor.py), or set "
            "WATERMARKS_ALLOW_PAID_API=1 to accept the cost and the re-stamping risk."
        )
    if not allow_remote:
        raise SystemExit(
            "error: rewrite base URL host is not loopback "
            f"('{host}'); refusing to send content off-machine. "
            "Set WATERMARKS_REWRITE_ALLOW_REMOTE=1 or pass --allow-remote to override."
        )
    if _is_paid_host(host):
        eprint(f"warning: '{host}' is a metered inference API; this run will cost money")
    eprint(
        f"warning: rewrite base URL host is '{host}' (not localhost); "
        "content will leave this machine"
    )


def default_base_url(backend: str) -> str:
    """Loopback endpoint for *backend*, honoring an explicit env override."""
    override = _env("WATERMARKS_REWRITE_BASE_URL")
    if override:
        return override
    return _BACKEND_BASE_URLS.get(backend, _BACKEND_BASE_URLS["ollama"])


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects.

    urllib's default handler re-sends the request headers on 301/302/303,
    which would forward the Authorization header (API key) to an unvalidated
    host behind the localhost allowlist. Any 3xx now surfaces as HTTPError.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


def _model_matches(requested: str, served: str) -> bool:
    """Does a served model satisfy the requested name?

    Ollama reports fully-qualified tags ("llama3.2:latest") while operators type
    the bare name ("llama3.2"). A bare request matches any tag of that model; an
    explicitly tagged request must match the tag, so asking for :3b never
    silently runs :1b.
    """
    req = requested.strip().lower()
    srv = served.strip().lower()
    if req == srv:
        return True
    if ":" in req:
        return False
    return srv.split(":", 1)[0] == req


def list_served_models(
    backend: str, base_url: str, timeout: float = 5.0
) -> list[str] | None:
    """Models the local runtime currently has loaded, or None if unreachable.

    None means "could not tell" and must not be confused with "none pulled" —
    the first is not grounds to block a run, the second is.
    """
    try:
        if backend == "ollama":
            data = _http_get_json(base_url.rstrip("/") + "/api/tags", timeout)
            return [m.get("name", "") for m in (data.get("models") or []) if m.get("name")]
        data = _http_get_json(base_url.rstrip("/") + "/v1/models", timeout)
        return [m.get("id", "") for m in (data.get("data") or []) if m.get("id")]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def resolve_model(
    model: str | None,
    *,
    served: list[str] | None,
    tier: str,
    exclude_families: tuple[str, ...] = (),
) -> str:
    """Resolve --model (possibly "auto") against what the runtime actually serves.

    Catching a missing model here turns a mid-run backend error into an
    actionable `ollama pull …` before any work starts.
    """
    picks = recommend_models(tier, exclude_families=exclude_families)
    fallback = picks[0].name if picks else "qwen2.5:7b"
    pull = picks[0].pull_cmd if picks else "ollama pull qwen2.5:7b"

    if model in (None, "", "auto"):
        if served is None:
            return fallback
        for rec in picks:
            for name in served:
                if _model_matches(rec.name, name):
                    return rec.name
        if served:
            # Not one we recommend, but the operator has it loaded and ready.
            return served[0]
        raise SystemExit(
            f"error: no model is available on the local runtime.\n  {pull}\n"
            "Then re-run. (doctor.py lists every model that fits this machine.)"
        )

    if served is None:
        return model
    if any(_model_matches(model, name) for name in served):
        return model
    available = ", ".join(served) if served else "none"
    raise SystemExit(
        f"error: model '{model}' is not available on the local runtime "
        f"(served: {available}).\n  ollama pull {model}"
    )


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def _hard_split(text: str, max_chars: int) -> list[str]:
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _split_paragraph(para: str, max_chars: int) -> list[str]:
    """Break one oversized paragraph on sentence boundaries, then bluntly."""
    out: list[str] = []
    buf = ""
    for sentence in _SENTENCE_SPLIT.split(para):
        if len(sentence) > max_chars:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_hard_split(sentence, max_chars))
            continue
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if len(candidate) > max_chars and buf:
            out.append(buf)
            buf = sentence
        else:
            buf = candidate
    if buf:
        out.append(buf)
    return out


def split_chunks(text: str, max_chars: int) -> list[str]:
    """Split *text* into rewrite-sized pieces on paragraph boundaries.

    Layer B against a self-hosted model is context-bound in a way a hosted
    frontier model is not: an 8k-context 7B silently truncates a long document,
    which would drop content from the "cleaned" output without saying so.
    Splitting on paragraphs keeps each request inside the window and preserves
    document order.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_split_paragraph(para, max_chars))
            continue
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) > max_chars and buf:
            chunks.append(buf)
            buf = para
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


def build_prompt(strength: str, text: str, *, lang: str, original_lang: str) -> str:
    if strength == "paraphrase":
        return PROMPTS["paraphrase"].format(TEXT=text)
    if strength == "humanize":
        return PROMPTS["humanize"].format(TEXT=text)
    if strength == "code":
        return PROMPTS["code"].format(TEXT=text)
    if strength == "backtranslate":
        # single combined instruction for print-prompt / one-shot backends
        return (
            f"Translate the text to {lang}, then translate that result back to "
            f"{original_lang}. Preserve all facts, numbers, and names. "
            f"Output only the final {original_lang} text.\n\n---\n{text}"
        )
    if strength == "structural":
        return (
            "First extract a bullet outline of all claims (no full sentences). "
            "Then write a complete document from that outline in natural, varied human "
            "prose without omitting any bullet. Output only the final document.\n\n---\n"
            f"{text}"
        )
    raise ValueError(f"unknown strength: {strength}")


def _http_get_json(url: str, timeout: float) -> dict:
    """GET JSON from a local runtime's discovery endpoint (no body, no auth)."""
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"refusing non-http(s) endpoint: {url}")
    req = urllib.request.Request(url, method="GET")
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"refusing non-http(s) rewrite endpoint: {url}")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_ollama(
    base_url: str, model: str, prompt: str, timeout: float, temperature: float
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    data = _http_json(
        url,
        {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature},
        },
        {},
        timeout,
    )
    msg = data.get("message") or {}
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"ollama empty response: {data!r}"[:500])
    return str(content).strip()


def call_openai_compatible(
    base_url: str,
    model: str,
    prompt: str,
    api_key: str | None,
    timeout: float,
    temperature: float,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = _http_json(
        url,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        },
        headers,
        timeout,
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"openai-compatible empty choices: {data!r}"[:500])
    content = (choices[0].get("message") or {}).get("content")
    if not content:
        raise RuntimeError(f"openai-compatible empty content: {data!r}"[:500])
    return str(content).strip()


def rewrite(
    text: str,
    *,
    backend: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    strength: str,
    lang: str,
    original_lang: str,
    timeout: float,
    layer_a_after: bool,
    temperature: float,
    candidates: int,
    allow_remote: bool = False,
    allow_paid: bool = False,
    max_chunk_chars: int | None = None,
) -> tuple[str, dict]:
    prompt = build_prompt(strength, text, lang=lang, original_lang=original_lang)
    info: dict = {
        "backend": backend,
        "strength": strength,
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "prompt_chars": len(prompt),
        "input_chars": len(text),
    }

    if backend == "print-prompt":
        info["mode"] = "print-prompt"
        if candidates > 1:
            eprint("note: --candidates ignored in print-prompt mode")
        return prompt, info

    if not model:
        raise SystemExit(f"error: --model required for the {backend} backend")
    if not base_url:
        raise SystemExit(f"error: --base-url required for the {backend} backend")

    _check_remote(base_url, allow_remote, allow_paid)

    # Chunk only when the document actually exceeds the window. Back-translation
    # and the structural outline->regen pass reason over the whole document, so
    # splitting them would change what they do, not just how much fits.
    chunkable = strength in ("paraphrase", "humanize", "code")
    if max_chunk_chars and chunkable and len(text) > max_chunk_chars:
        pieces = split_chunks(text, max_chunk_chars)
    else:
        pieces = [text]
    info["chunks"] = len(pieces)
    if len(pieces) > 1:
        eprint(f"chunking into {len(pieces)} pieces (context budget {max_chunk_chars} chars)")

    n = max(1, candidates)
    rewritten: list[str] = []
    for idx, piece in enumerate(pieces, start=1):
        if len(pieces) > 1:
            eprint(f"  chunk {idx}/{len(pieces)} ({len(piece)} chars)")
        piece_prompt = build_prompt(strength, piece, lang=lang, original_lang=original_lang)
        outs = [
            _call_backend(backend, base_url, model, piece_prompt, api_key, timeout, temperature)
            for _ in range(n)
        ]
        if len(outs) == 1:
            rewritten.append(outs[0])
        else:
            best, scores = _select_candidate(piece, outs)
            info.setdefault("candidate_scores", []).append(scores)
            rewritten.append(best)

    out = "\n\n".join(rewritten)
    if n > 1:
        info["candidates"] = n

    if layer_a_after:
        out, stats = clean_text(out)
        info["layer_a_after"] = stats

    info["output_chars"] = len(out)
    info["mode"] = "rewritten"
    info["note"] = (
        "Layer B is best-effort against statistical token-sampling watermarks; "
        "cannot certify removal against a vendor detector."
    )
    return out, info


def _call_backend(
    backend: str,
    base_url: str,
    model: str,
    prompt: str,
    api_key: str | None,
    timeout: float,
    temperature: float,
) -> str:
    if backend == "ollama":
        return call_ollama(base_url, model, prompt, timeout, temperature)
    if backend == "llamacpp":
        # llama-server speaks the OpenAI chat API, so the transport is shared.
        # The key is dropped rather than forwarded: a local server needs none,
        # and sending one would leak it to whatever is listening on that port.
        return call_openai_compatible(base_url, model, prompt, None, timeout, temperature)
    if backend == "openai-compatible":
        return call_openai_compatible(base_url, model, prompt, api_key, timeout, temperature)
    raise SystemExit(f"unknown backend: {backend}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default="-", help="Input text file, or - for stdin")
    p.add_argument("-o", "--output", help="Output path (default: stdout or *.rewritten.*)")
    p.add_argument(
        "--backend",
        choices=BACKENDS,
        default=_env("WATERMARKS_REWRITE_BACKEND", "print-prompt"),
    )
    p.add_argument(
        "--model",
        default=_env("WATERMARKS_REWRITE_MODEL"),
        help="Model name, or 'auto' to pick one that fits this machine and is pulled",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="Runtime endpoint (default: this backend's loopback address)",
    )
    p.add_argument(
        "--allow-remote",
        action="store_true",
        default=None,
        help="Allow non-loopback rewrite endpoints (default: deny; "
        "WATERMARKS_REWRITE_ALLOW_REMOTE=1 has the same effect)",
    )
    p.add_argument(
        "--allow-paid-api",
        action="store_true",
        default=None,
        help="Allow metered vendor endpoints, accepting the per-token cost and the "
        "risk of re-stamping the text (default: deny)",
    )
    p.add_argument(
        "--origin",
        default=None,
        help="Suspected origin vendor (claude|gemini|openai|…) so --model auto "
        "never picks that vendor's own family and re-stamps the text",
    )
    p.add_argument(
        "--max-chunk-chars",
        type=int,
        default=None,
        help="Split long input into pieces this size (default: from the detected tier; "
        "0 disables chunking)",
    )
    # NOTE: no --api-key flag on purpose — keys on argv are visible in `ps`
    # and shell history. Set WATERMARKS_REWRITE_API_KEY instead.
    p.add_argument(
        "--strength",
        choices=("paraphrase", "backtranslate", "structural", "humanize", "code"),
        default="paraphrase",
    )
    p.add_argument("--lang", default="French", help="Pivot language for backtranslate")
    p.add_argument("--original-lang", default="English")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for the rewrite backend",
    )
    p.add_argument(
        "--candidates",
        type=int,
        default=1,
        help="Number of rewrite candidates to generate and score",
    )
    p.add_argument(
        "--no-layer-a-after",
        action="store_true",
        help="Skip Layer A scrub on model output",
    )
    p.add_argument("--json-stats", action="store_true", help="Stats JSON on stderr")
    p.add_argument(
        "--force-text",
        action="store_true",
        help="Rewrite even when the input looks like a binary container",
    )
    args = p.parse_args()

    text = read_text_input(args.path, allow_binary=args.force_text)
    allow_remote = (
        args.allow_remote
        if args.allow_remote is not None
        else _flag_env("WATERMARKS_REWRITE_ALLOW_REMOTE")
    )
    allow_paid = (
        args.allow_paid_api
        if args.allow_paid_api is not None
        else _flag_env("WATERMARKS_ALLOW_PAID_API")
    )

    base_url = args.base_url or default_base_url(args.backend)
    model = args.model
    chunk_budget = args.max_chunk_chars

    # Model resolution and the chunk budget both depend on the machine, so the
    # hardware probe only runs for backends that actually call a model.
    if args.backend != "print-prompt":
        # Gate the endpoint *before* the preflight. list_served_models() would
        # otherwise open a connection to a host we are about to refuse, which
        # both defeats the point of the refusal and tips off the vendor.
        _check_remote(base_url, allow_remote, allow_paid)
        tier = pick_tier(detect_hardware())
        if chunk_budget is None:
            chunk_budget = max_chunk_chars(tier)
        exclude = tuple(f for f in (family_for_origin(args.origin),) if f)
        served = list_served_models(args.backend, base_url, timeout=5.0)
        if served is None:
            eprint(
                f"warning: could not reach the {args.backend} runtime at {base_url} "
                "to list models; continuing without a preflight check"
            )
        model = resolve_model(model, served=served, tier=tier, exclude_families=exclude)
        if model != args.model:
            eprint(f"using local model: {model} (tier {tier})")

    try:
        result, info = rewrite(
            text,
            backend=args.backend,
            model=model,
            base_url=base_url,
            api_key=_env("WATERMARKS_REWRITE_API_KEY"),
            strength=args.strength,
            lang=args.lang,
            original_lang=args.original_lang,
            timeout=args.timeout,
            layer_a_after=not args.no_layer_a_after,
            temperature=args.temperature,
            candidates=args.candidates,
            allow_remote=allow_remote,
            allow_paid=allow_paid,
            max_chunk_chars=chunk_budget or None,
        )
    except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
        eprint(f"rewrite failed: {e}")
        return 1

    out = args.output
    if out is None and args.path not in (None, "-") and args.backend != "print-prompt":
        out = str(cleaned_path(Path(args.path), suffix=".rewritten"))
    elif out is None and args.backend == "print-prompt":
        out = "-"

    write_text_output(result, out)
    if args.json_stats:
        eprint(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        eprint(
            f"backend={info['backend']} strength={info['strength']} "
            f"mode={info.get('mode')} chars {info['input_chars']}->{info.get('output_chars', len(result))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
