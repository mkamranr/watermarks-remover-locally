"""Tests for local hardware detection and self-hosted model tier policy."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import local_models  # noqa: E402
from local_models import (  # noqa: E402
    TIER_CPU,
    TIER_HIGH,
    TIER_LOW,
    TIER_MID,
    Hardware,
    detect_hardware,
    max_chunk_chars,
    pick_tier,
    recommend_models,
    resolve_device,
    resolve_dtype,
    tier_policy,
)


def _hw(**overrides) -> Hardware:
    base = dict(
        backend="cuda",
        vram_gb=8.0,
        ram_gb=32.0,
        free_disk_gb=100.0,
        torch_available=True,
        device_name="test-gpu",
    )
    base.update(overrides)
    return Hardware(**base)


# --- tier selection -------------------------------------------------------


@pytest.mark.parametrize(
    "vram,expected",
    [
        (0.0, TIER_CPU),
        (2.0, TIER_CPU),
        (3.9, TIER_CPU),
        (4.0, TIER_LOW),
        (6.0, TIER_LOW),
        (7.9, TIER_LOW),
        (8.0, TIER_MID),
        (12.0, TIER_MID),
        (15.9, TIER_MID),
        (16.0, TIER_HIGH),
        (24.0, TIER_HIGH),
    ],
)
def test_pick_tier_vram_boundaries(vram, expected):
    assert pick_tier(_hw(vram_gb=vram)) == expected


def test_pick_tier_cpu_backend_is_always_cpu_tier():
    """No GPU means the diffusion path is out no matter how much RAM there is."""
    assert pick_tier(_hw(backend="cpu", vram_gb=0.0, ram_gb=128.0)) == TIER_CPU


def test_pick_tier_without_torch_is_cpu_tier():
    assert pick_tier(_hw(backend="cuda", vram_gb=24.0, torch_available=False)) == TIER_CPU


def test_pick_tier_mps_uses_vram_budget():
    assert pick_tier(_hw(backend="mps", vram_gb=10.0)) == TIER_MID


# --- tier policy ----------------------------------------------------------


def test_cpu_tier_offers_no_diffusion_path():
    policy = tier_policy(TIER_CPU)
    assert policy.image_backends == ("lite",)
    assert "ctrlregen" not in policy.image_backends


def test_every_tier_offers_the_lite_image_path():
    """clean_pixel_lite has no GPU requirement, so it is always available."""
    for tier in (TIER_CPU, TIER_LOW, TIER_MID, TIER_HIGH):
        assert "lite" in tier_policy(tier).image_backends


def test_gpu_tiers_offer_ctrlregen():
    for tier in (TIER_LOW, TIER_MID, TIER_HIGH):
        assert "ctrlregen" in tier_policy(tier).image_backends


def test_low_tier_uses_sequential_offload_and_high_tier_uses_none():
    assert tier_policy(TIER_LOW).ctrlregen_offload == "sequential"
    assert tier_policy(TIER_MID).ctrlregen_offload == "model"
    assert tier_policy(TIER_HIGH).ctrlregen_offload is None


def test_tier_policy_rejects_unknown_tier():
    with pytest.raises(ValueError):
        tier_policy("tier9-imaginary")


# --- model recommendations ------------------------------------------------


def test_every_tier_recommends_at_least_one_model():
    for tier in (TIER_CPU, TIER_LOW, TIER_MID, TIER_HIGH):
        assert recommend_models(tier), f"no models for {tier}"


def test_recommended_models_fit_the_tier_budget():
    """A recommendation that cannot load on the tier's hardware is a bug."""
    for tier in (TIER_CPU, TIER_LOW, TIER_MID, TIER_HIGH):
        budget = tier_policy(tier).text_budget_gb
        for rec in recommend_models(tier):
            assert rec.approx_gb <= budget, f"{rec.name} ({rec.approx_gb}GB) > {budget}GB on {tier}"


def test_cpu_tier_models_are_small():
    for rec in recommend_models(TIER_CPU):
        assert rec.approx_gb <= 4.0


def test_recommendations_carry_a_pull_command():
    for rec in recommend_models(TIER_LOW):
        assert rec.pull_cmd
        assert rec.name in rec.pull_cmd


def test_exclude_families_filters_out_the_origin_vendor():
    """Model hygiene: never recommend rewriting Gemini text with a Google model."""
    all_recs = recommend_models(TIER_LOW)
    families = {r.family for r in all_recs}
    assert len(families) > 1, "need multiple families for this test to mean anything"

    victim = sorted(families)[0]
    filtered = recommend_models(TIER_LOW, exclude_families=(victim,))
    assert filtered, "filtering one family must not empty the list"
    assert all(r.family != victim for r in filtered)


def test_exclude_families_is_case_insensitive():
    filtered = recommend_models(TIER_LOW, exclude_families=("GOOGLE",))
    assert all(r.family != "google" for r in filtered)


def test_origin_vendor_maps_to_a_family_to_exclude():
    assert local_models.family_for_origin("gemini") == "google"
    assert local_models.family_for_origin("claude") == "anthropic"
    assert local_models.family_for_origin("openai") == "openai"
    assert local_models.family_for_origin("something-unknown") is None


# --- dtype / device resolution -------------------------------------------


def test_resolve_dtype_defaults_to_fp16_on_gpu():
    assert resolve_dtype("cuda", "auto") == "fp16"
    assert resolve_dtype("mps", "auto") == "fp16"


def test_resolve_dtype_defaults_to_fp32_on_cpu():
    """fp16 on CPU is slower than fp32 and unsupported by many kernels."""
    assert resolve_dtype("cpu", "auto") == "fp32"


def test_resolve_dtype_respects_explicit_choice():
    assert resolve_dtype("cpu", "bf16") == "bf16"
    assert resolve_dtype("cuda", "fp32") == "fp32"


def test_resolve_dtype_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_dtype("cuda", "int3")


def test_resolve_device_without_torch_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(local_models, "_probe_torch", lambda: None)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_honors_explicit_value():
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cpu") == "cpu"


# --- chunking budget ------------------------------------------------------


def test_max_chunk_chars_grows_with_tier():
    sizes = [max_chunk_chars(t) for t in (TIER_CPU, TIER_LOW, TIER_MID, TIER_HIGH)]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_max_chunk_chars_is_positive_everywhere():
    for tier in (TIER_CPU, TIER_LOW, TIER_MID, TIER_HIGH):
        assert max_chunk_chars(tier) > 0


# --- hardware detection ---------------------------------------------------


def test_detect_hardware_without_torch_reports_cpu(monkeypatch):
    """Detection must degrade, never raise, on a machine with no torch."""
    monkeypatch.setattr(local_models, "_probe_torch", lambda: None)
    hw = detect_hardware()
    assert hw.backend == "cpu"
    assert hw.vram_gb == 0.0
    assert hw.torch_available is False


def test_detect_hardware_survives_a_broken_torch(monkeypatch):
    """A torch that imports but explodes on probing must not crash the tool."""

    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("driver mismatch")

    monkeypatch.setattr(local_models, "_probe_torch", lambda: Boom())
    hw = detect_hardware()
    assert hw.backend == "cpu"


def test_detect_hardware_reports_real_numbers():
    """On the dev machine the probe must still produce a usable, sane record."""
    hw = detect_hardware()
    assert hw.backend in ("cuda", "mps", "cpu")
    assert hw.ram_gb > 0
    assert hw.free_disk_gb >= 0
    assert pick_tier(hw) in (TIER_CPU, TIER_LOW, TIER_MID, TIER_HIGH)


def test_hardware_to_dict_is_json_safe():
    import json

    json.dumps(detect_hardware().to_dict())
