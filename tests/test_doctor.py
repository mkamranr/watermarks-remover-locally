"""Tests for the doctor.py preflight report."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import doctor  # noqa: E402
from local_models import TIER_CPU, TIER_LOW, Hardware  # noqa: E402


def _hw(**overrides) -> Hardware:
    base = dict(
        backend="cpu",
        vram_gb=0.0,
        ram_gb=16.0,
        free_disk_gb=50.0,
        torch_available=False,
        device_name="cpu",
    )
    base.update(overrides)
    return Hardware(**base)


def _probes(**overrides) -> dict:
    """A machine with nothing optional installed."""
    base = dict(
        tools={"exiftool": None, "c2patool": None, "ollama": None, "llama-server": None},
        runtimes={"ollama": {"reachable": False, "models": []},
                  "llamacpp": {"reachable": False, "models": []}},
        checkouts={"reverse_synthid": None, "noai_watermark": None},
        modules={"PIL": False, "numpy": False, "torch": False},
    )
    base.update(overrides)
    return base


def _report(hw=None, **probe_overrides) -> dict:
    return doctor.build_report(hw or _hw(), **_probes(**probe_overrides))


def _caps(report: dict) -> dict[str, dict]:
    return {c["key"]: c for c in report["capabilities"]}


# --- baseline -------------------------------------------------------------


def test_layer_a_is_always_ready():
    """Layer A is pure stdlib; it must work on the barest possible machine."""
    assert _caps(_report())["layer_a"]["status"] == "ready"


def test_report_is_json_serializable():
    json.dumps(_report())


def test_report_includes_hardware_and_tier():
    report = _report()
    assert report["hardware"]["backend"] == "cpu"
    assert report["tier"] == TIER_CPU


def test_every_non_ready_capability_names_a_fix():
    """A preflight that says 'unavailable' without saying how to fix it is noise."""
    for cap in _report()["capabilities"]:
        if cap["status"] != "ready":
            assert cap["fix"], f"{cap['key']} is {cap['status']} but offers no fix"


# --- Layer B local rewrite -----------------------------------------------


def test_layer_b_unavailable_without_a_local_runtime():
    cap = _caps(_report())["layer_b_local"]
    assert cap["status"] == "unavailable"
    assert "ollama" in cap["fix"].lower()


def test_layer_b_ready_when_ollama_serves_a_model():
    report = _report(
        tools={"exiftool": None, "c2patool": None, "ollama": "/usr/local/bin/ollama",
               "llama-server": None},
        runtimes={"ollama": {"reachable": True, "models": ["qwen2.5:7b"]},
                  "llamacpp": {"reachable": False, "models": []}},
    )
    assert _caps(report)["layer_b_local"]["status"] == "ready"


def test_layer_b_degraded_when_runtime_up_but_no_model_pulled():
    """The classic failure: `ollama serve` running with nothing pulled."""
    report = _report(
        runtimes={"ollama": {"reachable": True, "models": []},
                  "llamacpp": {"reachable": False, "models": []}},
    )
    cap = _caps(report)["layer_b_local"]
    assert cap["status"] == "degraded"
    assert "pull" in cap["fix"]


def test_layer_b_ready_via_llamacpp_alone():
    report = _report(
        runtimes={"ollama": {"reachable": False, "models": []},
                  "llamacpp": {"reachable": True, "models": ["local-gguf"]}},
    )
    assert _caps(report)["layer_b_local"]["status"] == "ready"


# --- file cleaners --------------------------------------------------------


def test_pdf_clean_is_degraded_without_exiftool():
    cap = _caps(_report())["pdf_clean"]
    assert cap["status"] == "degraded"
    assert "exiftool" in cap["fix"]


def test_pdf_clean_ready_with_exiftool():
    report = _report(
        tools={"exiftool": "/usr/bin/exiftool", "c2patool": None, "ollama": None,
               "llama-server": None},
    )
    assert _caps(report)["pdf_clean"]["status"] == "ready"


# --- pixel paths ----------------------------------------------------------


def test_pixel_lite_needs_pillow_and_numpy():
    cap = _caps(_report())["pixel_lite"]
    assert cap["status"] == "unavailable"
    assert "pillow" in cap["fix"].lower() or "numpy" in cap["fix"].lower()


def test_pixel_lite_ready_with_modules():
    report = _report(modules={"PIL": True, "numpy": True, "torch": False})
    assert _caps(report)["pixel_lite"]["status"] == "ready"


def test_ctrlregen_unavailable_on_cpu_tier_even_with_checkout(tmp_path):
    """No GPU means the diffusion path is not offered, checkout or not."""
    report = _report(
        checkouts={"reverse_synthid": None, "noai_watermark": str(tmp_path)},
        modules={"PIL": True, "numpy": True, "torch": True},
    )
    cap = _caps(report)["ctrlregen"]
    assert cap["status"] == "unavailable"
    assert "lite" in cap["fix"].lower() or "gpu" in cap["fix"].lower()


def test_ctrlregen_ready_on_gpu_tier_with_checkout(tmp_path):
    report = doctor.build_report(
        _hw(backend="cuda", vram_gb=6.0, torch_available=True, device_name="RTX 3060"),
        **_probes(
            checkouts={"reverse_synthid": None, "noai_watermark": str(tmp_path)},
            modules={"PIL": True, "numpy": True, "torch": True},
        ),
    )
    assert report["tier"] == TIER_LOW
    cap = _caps(report)["ctrlregen"]
    assert cap["status"] == "ready"


def test_ctrlregen_warns_when_disk_cannot_hold_the_weights(tmp_path):
    """~10 GB of weights on a nearly-full disk fails halfway through the download."""
    report = doctor.build_report(
        _hw(backend="cuda", vram_gb=12.0, torch_available=True, free_disk_gb=6.0),
        **_probes(
            checkouts={"reverse_synthid": None, "noai_watermark": str(tmp_path)},
            modules={"PIL": True, "numpy": True, "torch": True},
        ),
    )
    cap = _caps(report)["ctrlregen"]
    assert cap["status"] == "degraded"
    assert "disk" in cap["detail"].lower()


def test_ctrlregen_ready_when_disk_is_ample(tmp_path):
    report = doctor.build_report(
        _hw(backend="cuda", vram_gb=12.0, torch_available=True, free_disk_gb=80.0),
        **_probes(
            checkouts={"reverse_synthid": None, "noai_watermark": str(tmp_path)},
            modules={"PIL": True, "numpy": True, "torch": True},
        ),
    )
    assert _caps(report)["ctrlregen"]["status"] == "ready"


def test_ctrlregen_missing_checkout_names_the_bootstrap_script():
    report = doctor.build_report(
        _hw(backend="cuda", vram_gb=12.0, torch_available=True),
        **_probes(modules={"PIL": True, "numpy": True, "torch": True}),
    )
    assert "setup_ctrlregen.sh" in _caps(report)["ctrlregen"]["fix"]


# --- recommendations ------------------------------------------------------


def test_report_recommends_models_for_the_tier():
    report = _report()
    assert report["recommended_models"]
    for rec in report["recommended_models"]:
        assert rec["pull_cmd"]


def test_gpu_tier_reports_a_ctrlregen_vram_estimate():
    report = doctor.build_report(
        _hw(backend="cuda", vram_gb=6.0, torch_available=True),
        **_probes(),
    )
    assert report["ctrlregen_vram_estimate_gb"] > 0


# --- CLI ------------------------------------------------------------------


def test_cli_json_runs_on_this_machine():
    """The real end-to-end check: doctor must survive whatever this box is."""
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "doctor.py"), "--json"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["tier"]
    assert payload["capabilities"]


def test_cli_human_output_runs():
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "doctor.py")],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "Layer A" in r.stdout or "layer_a" in r.stdout
