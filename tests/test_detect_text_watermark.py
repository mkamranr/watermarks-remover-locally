"""Tests for the local Kirchenbauer green-list z-score detector."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

np = pytest.importorskip("numpy")

import detect_text_watermark as dtw  # noqa: E402
from detect_text_watermark import (  # noqa: E402
    DEFAULT_GAMMA,
    DEFAULT_HASH_KEY,
    green_ids,
    score_tokens,
    z_score,
)

VOCAB = 4096


# --- z-score math ---------------------------------------------------------


def test_z_score_at_the_expected_rate_is_zero():
    """Unwatermarked text hits the green list at exactly gamma."""
    assert abs(z_score(green=250, total=1000, gamma=0.25)) < 1e-9


def test_z_score_positive_when_green_is_over_represented():
    assert z_score(green=500, total=1000, gamma=0.25) > 0


def test_z_score_negative_when_green_is_under_represented():
    assert z_score(green=100, total=1000, gamma=0.25) < 0


def test_z_score_grows_with_sample_size_at_fixed_rate():
    """The same bias over more tokens is stronger evidence."""
    small = z_score(green=60, total=100, gamma=0.5)
    large = z_score(green=600, total=1000, gamma=0.5)
    assert large > small


def test_z_score_known_value():
    # 600/1000 green at gamma=0.5: (600 - 500) / sqrt(1000*0.25) = 100/15.811 = 6.32
    assert z_score(green=600, total=1000, gamma=0.5) == pytest.approx(6.3246, abs=1e-3)


def test_z_score_zero_total_is_zero():
    assert z_score(green=0, total=0, gamma=0.25) == 0.0


@pytest.mark.parametrize("gamma", [0.0, 1.0, -0.1, 1.5])
def test_z_score_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError):
        z_score(green=1, total=2, gamma=gamma)


# --- green list -----------------------------------------------------------


def test_green_list_is_deterministic():
    a = green_ids(prev_id=7, vocab_size=VOCAB, gamma=DEFAULT_GAMMA, hash_key=DEFAULT_HASH_KEY)
    b = green_ids(prev_id=7, vocab_size=VOCAB, gamma=DEFAULT_GAMMA, hash_key=DEFAULT_HASH_KEY)
    assert np.array_equal(a, b)


def test_green_list_differs_by_prefix_token():
    a = green_ids(prev_id=7, vocab_size=VOCAB, gamma=DEFAULT_GAMMA, hash_key=DEFAULT_HASH_KEY)
    b = green_ids(prev_id=8, vocab_size=VOCAB, gamma=DEFAULT_GAMMA, hash_key=DEFAULT_HASH_KEY)
    assert not np.array_equal(a, b)


def test_green_list_differs_by_key():
    """A different key is a different watermark; lists must not collide."""
    a = green_ids(prev_id=7, vocab_size=VOCAB, gamma=DEFAULT_GAMMA, hash_key=1)
    b = green_ids(prev_id=7, vocab_size=VOCAB, gamma=DEFAULT_GAMMA, hash_key=2)
    assert not np.array_equal(a, b)


def test_green_list_has_the_right_size():
    ids = green_ids(prev_id=3, vocab_size=VOCAB, gamma=0.25, hash_key=DEFAULT_HASH_KEY)
    assert len(ids) == int(VOCAB * 0.25)


# --- detection on synthetic sequences ------------------------------------


def _unmarked(n: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, VOCAB, size=n)]


def _marked(n: int, seed: int = 0, gamma: float = DEFAULT_GAMMA) -> list[int]:
    """Build a sequence where every token is drawn from its green list."""
    rng = np.random.default_rng(seed)
    tokens = [int(rng.integers(0, VOCAB))]
    for _ in range(n - 1):
        allowed = green_ids(tokens[-1], VOCAB, gamma, DEFAULT_HASH_KEY)
        tokens.append(int(allowed[rng.integers(0, len(allowed))]))
    return tokens


def test_unmarked_text_scores_near_zero():
    result = score_tokens(_unmarked(2000), vocab_size=VOCAB)
    assert abs(result["z_score"]) < 4.0, result


def test_marked_text_scores_high():
    result = score_tokens(_marked(2000), vocab_size=VOCAB)
    assert result["z_score"] > 10.0, result
    assert result["watermark_detected"] is True


def test_unmarked_text_is_not_flagged():
    assert score_tokens(_unmarked(2000), vocab_size=VOCAB)["watermark_detected"] is False


def test_wrong_key_does_not_detect_a_mark():
    """Detection requires the right key -- which is why vendor marks are opaque."""
    result = score_tokens(_marked(2000), vocab_size=VOCAB, hash_key=999)
    assert result["watermark_detected"] is False


def test_disrupting_a_marked_sequence_lowers_the_score():
    """The core claim: breaking token choices reduces the statistical signal."""
    marked = _marked(2000)
    before = score_tokens(marked, vocab_size=VOCAB)["z_score"]

    # Replace half the tokens with random ones, as a rewrite would.
    rng = np.random.default_rng(11)
    disrupted = [
        int(rng.integers(0, VOCAB)) if rng.random() < 0.5 else t for t in marked
    ]
    after = score_tokens(disrupted, vocab_size=VOCAB)["z_score"]

    assert after < before / 2, (before, after)


def test_score_reports_counts_and_rate():
    result = score_tokens(_marked(500), vocab_size=VOCAB)
    assert result["tokens_scored"] == 499  # first token has no prefix
    assert 0 <= result["green_tokens"] <= result["tokens_scored"]
    assert 0.0 <= result["green_rate"] <= 1.0


def test_short_text_is_reported_as_inconclusive():
    """A handful of tokens cannot support any claim either way."""
    result = score_tokens(_marked(12), vocab_size=VOCAB)
    assert result["inconclusive"] is True


def test_long_text_is_not_inconclusive():
    assert score_tokens(_marked(500), vocab_size=VOCAB)["inconclusive"] is False


def test_empty_token_list():
    result = score_tokens([], vocab_size=VOCAB)
    assert result["tokens_scored"] == 0
    assert result["watermark_detected"] is False


# --- CLI ------------------------------------------------------------------


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "detect_text_watermark.py"), *args],
        capture_output=True, text=True, timeout=120,
    )


def test_cli_reports_unavailable_without_a_tokenizer(tmp_path):
    """No tokenizer means no honest answer -- exit 3, not a fabricated score."""
    src = tmp_path / "doc.txt"
    src.write_text("Some ordinary prose that nobody watermarked.\n" * 20)
    r = _run(str(src), "--tokenizer", "definitely-not-a-real-tokenizer-xyz",
             "--llamacpp-url", "http://127.0.0.1:59999")
    assert r.returncode == 3
    assert "tokenizer" in (r.stdout + r.stderr).lower()


def test_cli_scores_pretokenized_json(tmp_path):
    """--token-ids lets the detector run with no tokenizer at all."""
    ids = _marked(600)
    src = tmp_path / "ids.json"
    src.write_text(json.dumps(ids))
    r = _run(str(src), "--token-ids", "--vocab-size", str(VOCAB), "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["z_score"] > 10
    assert payload["watermark_detected"] is True


def test_cli_unmarked_pretokenized(tmp_path):
    src = tmp_path / "ids.json"
    src.write_text(json.dumps(_unmarked(600)))
    r = _run(str(src), "--token-ids", "--vocab-size", str(VOCAB), "--json")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["watermark_detected"] is False


def test_cli_missing_file_exits_2(tmp_path):
    r = _run(str(tmp_path / "nope.txt"), "--token-ids")
    assert r.returncode == 2


def test_cli_states_its_scope_limits(tmp_path):
    """The report must say what it cannot detect, every time."""
    src = tmp_path / "ids.json"
    src.write_text(json.dumps(_unmarked(600)))
    r = _run(str(src), "--token-ids", "--vocab-size", str(VOCAB), "--json")
    payload = json.loads(r.stdout)
    note = payload["scope"].lower()
    assert "claude" in note or "vendor" in note
    assert "cannot" in note
