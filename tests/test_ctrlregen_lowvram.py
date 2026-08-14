"""Tests for CtrlRegen dtype control, low-VRAM knobs, and the VRAM preflight.

Mock-based throughout: CI has no torch, no GPU, and no model weights.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import clean_ctrlregen  # noqa: E402

CLEAN_SCRIPT = SCRIPTS / "clean_ctrlregen.py"

FAKE_PIL = '''\
class Image:
    def __init__(self, size=(10, 20)):
        self.size = size
    @staticmethod
    def open(path):
        return Image()
    def convert(self, mode):
        return self
    def load(self):
        return self
    def save(self, fp, format=None, **kwargs):
        fp.write(b"FAKEIMAGE")
'''

# A fake engine that records the low-VRAM calls made against its pipeline, so a
# test can assert what the adapter actually configured.
FAKE_ENGINE = '''\
import json, os

class _Pipe:
    def __init__(self):
        self.calls = []
    def enable_attention_slicing(self, *a, **k):
        self.calls.append("attention_slicing")
    def enable_vae_slicing(self, *a, **k):
        self.calls.append("vae_slicing")
    def enable_vae_tiling(self, *a, **k):
        self.calls.append("vae_tiling")
    def enable_model_cpu_offload(self, *a, **k):
        self.calls.append("model_cpu_offload")
    def enable_sequential_cpu_offload(self, *a, **k):
        self.calls.append("sequential_cpu_offload")

class CtrlRegenEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._pipeline = None
        self._loaded = False
    def load(self):
        self._loaded = True
        self._pipeline = _Pipe()
    def run(self, image, strength=0.5, num_inference_steps=50,
            guidance_scale=2.0, seed=None):
        sink = os.environ.get("FAKE_ENGINE_SINK")
        if sink:
            dtype = self.kwargs.get("torch_dtype")
            with open(sink, "w") as f:
                json.dump({
                    "loaded": self._loaded,
                    "calls": self._pipeline.calls if self._pipeline else [],
                    "dtype": str(dtype),
                }, f)
        return image

def is_ctrlregen_available():
    return True
'''


def _make_fake_upstream(tmp_path: Path) -> Path:
    upstream = tmp_path / "noai-watermark"
    ctrlregen = upstream / "src" / "ctrlregen"
    pil = upstream / "src" / "PIL"
    ctrlregen.mkdir(parents=True)
    pil.mkdir(parents=True)
    (ctrlregen / "__init__.py").write_text("")
    (pil / "__init__.py").write_text(FAKE_PIL)
    (ctrlregen / "engine.py").write_text(FAKE_ENGINE)
    return upstream


def _minimal_png() -> bytes:
    def chunk(ctype: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(ctype)
        crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00")
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _run(*args: str, sink: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("NOAI_WATERMARK_DIR", None)
    if sink:
        env["FAKE_ENGINE_SINK"] = str(sink)
    return subprocess.run(
        [sys.executable, str(CLEAN_SCRIPT), *args],
        capture_output=True, text=True, env=env, timeout=120,
    )


@pytest.fixture()
def scene(tmp_path):
    upstream = _make_fake_upstream(tmp_path)
    src = tmp_path / "in.png"
    src.write_bytes(_minimal_png())
    return upstream, src, tmp_path


# --- low-VRAM knobs -------------------------------------------------------


def test_low_vram_applies_slicing_and_tiling(scene):
    upstream, src, tmp = scene
    sink = tmp / "sink.json"
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--device", "cuda", "--low-vram", "--json", sink=sink,
    )
    assert r.returncode == 0, r.stderr
    rec = json.loads(sink.read_text())
    assert rec["loaded"] is True, "pipeline must be loaded before knobs are applied"
    assert "attention_slicing" in rec["calls"]
    assert "vae_tiling" in rec["calls"]


def test_low_vram_offload_only_on_cuda(scene):
    """CPU offload is a CUDA/accelerate path; on MPS it must not be attempted."""
    upstream, src, tmp = scene
    sink = tmp / "sink.json"
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--device", "mps", "--low-vram", "--offload", "sequential", "--json", sink=sink,
    )
    assert r.returncode == 0, r.stderr
    rec = json.loads(sink.read_text())
    assert "sequential_cpu_offload" not in rec["calls"]
    assert "model_cpu_offload" not in rec["calls"]
    # slicing still applies and is the useful knob on MPS
    assert "attention_slicing" in rec["calls"]


def test_sequential_offload_on_cuda(scene):
    upstream, src, tmp = scene
    sink = tmp / "sink.json"
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--device", "cuda", "--offload", "sequential", "--json", sink=sink,
    )
    assert r.returncode == 0, r.stderr
    assert "sequential_cpu_offload" in json.loads(sink.read_text())["calls"]


def test_model_offload_on_cuda(scene):
    upstream, src, tmp = scene
    sink = tmp / "sink.json"
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--device", "cuda", "--offload", "model", "--json", sink=sink,
    )
    assert r.returncode == 0, r.stderr
    assert "model_cpu_offload" in json.loads(sink.read_text())["calls"]


def test_no_low_vram_by_default(scene):
    """Default must stay fast; low-VRAM knobs cost speed."""
    upstream, src, tmp = scene
    sink = tmp / "sink.json"
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--device", "cuda", "--offload", "none", "--json", sink=sink,
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(sink.read_text())["calls"] == []


def test_json_reports_applied_knobs(scene):
    upstream, src, tmp = scene
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--device", "cuda", "--low-vram", "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert "attention_slicing" in payload["low_vram_applied"]
    assert payload["dtype"]


# --- dtype ----------------------------------------------------------------


def test_dtype_flag_is_passed_to_the_engine(scene):
    upstream, src, tmp = scene
    sink = tmp / "sink.json"
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--device", "cuda", "--dtype", "fp32", "--json", sink=sink,
    )
    assert r.returncode == 0, r.stderr
    assert "float32" in json.loads(sink.read_text())["dtype"]


def test_dtype_auto_reports_the_resolved_choice(scene):
    upstream, src, tmp = scene
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--device", "cpu", "--dtype", "auto", "--json",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["dtype"] == "fp32"


def test_bad_dtype_rejected(scene):
    upstream, src, tmp = scene
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--dtype", "int3",
    )
    assert r.returncode != 0


# --- VRAM preflight -------------------------------------------------------


def test_preflight_refuses_when_vram_is_short(scene, monkeypatch):
    """Refuse *before* the ~10 GB download, not after the OOM."""
    upstream, src, tmp = scene
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--device", "cuda", "--dtype", "fp32", "--offload", "none",
        "--assume-vram-gb", "2.0",
    )
    assert r.returncode == 3
    combined = r.stdout + r.stderr
    assert "vram" in combined.lower()
    assert "--low-vram" in combined or "--offload" in combined


def test_preflight_passes_with_enough_vram(scene):
    upstream, src, tmp = scene
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--device", "cuda", "--assume-vram-gb", "24.0", "--json",
    )
    assert r.returncode == 0, r.stderr


def test_preflight_satisfied_by_offload(scene):
    """Sequential offload is exactly the escape hatch for a small card."""
    upstream, src, tmp = scene
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--device", "cuda", "--offload", "sequential", "--assume-vram-gb", "2.0", "--json",
    )
    assert r.returncode == 0, r.stderr


def test_preflight_skipped_on_cpu(scene):
    """CPU runs use system RAM; the VRAM estimate does not apply."""
    upstream, src, tmp = scene
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--device", "cpu", "--assume-vram-gb", "0.1", "--json",
    )
    assert r.returncode == 0, r.stderr


def test_preflight_override(scene):
    upstream, src, tmp = scene
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--device", "cuda", "--dtype", "fp32", "--offload", "none",
        "--assume-vram-gb", "2.0", "--skip-vram-check", "--json",
    )
    assert r.returncode == 0, r.stderr


# --- tile-count guard -----------------------------------------------------


def test_tile_estimate_helper():
    # 512 tiles with 192 overlap => stride 320
    assert clean_ctrlregen.estimate_tiles(512, 512) == 1
    assert clean_ctrlregen.estimate_tiles(1024, 1024) >= 4
    assert clean_ctrlregen.estimate_tiles(3840, 2160) > 50


def test_max_pixels_guard_refuses_a_huge_image(scene, monkeypatch):
    upstream, src, tmp = scene
    # The fake PIL reports 10x20; force the guard by setting a tiny ceiling.
    r = _run(
        str(src), "-o", str(tmp / "out.png"), "--upstream-dir", str(upstream),
        "--max-pixels", "10",
    )
    assert r.returncode == 2
    assert "pixel" in (r.stdout + r.stderr).lower()
