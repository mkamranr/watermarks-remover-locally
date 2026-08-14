"""Tests for the zero-GPU pixel-attack tier."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

# The lite tier is optional-dependency territory; CI without them must skip,
# not fail.
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")

import clean_pixel_lite  # noqa: E402
from clean_pixel_lite import PRESETS, apply_preset, psnr  # noqa: E402
from PIL import Image  # noqa: E402


def _test_image(size=(128, 128), seed=7, grain=3.0) -> Image.Image:
    """Photo-like content: smooth structure plus light grain.

    Deliberately NOT per-pixel white noise. Noise is the worst case for every
    lowpass stage, so calibrating presets against it would make the tool look
    far more destructive than it is on the photographs people actually clean.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0 : size[1], 0 : size[0]].astype(np.float64)
    base = np.sin(x / 23) * 40 + np.cos(y / 17) * 40 + (x + y) * 0.3
    base = (base - base.min()) / (base.max() - base.min()) * 220 + 15
    arr = np.stack([base, base * 0.8 + 20, 240 - base], axis=-1)
    arr = arr + rng.normal(0, grain, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _write(tmp_path: Path, name="in.png") -> Path:
    p = tmp_path / name
    _test_image().save(p)
    return p


# --- psnr -----------------------------------------------------------------


def test_psnr_identical_is_infinite():
    img = _test_image()
    assert psnr(img, img) == float("inf")


def test_psnr_of_different_images_is_finite_and_positive():
    a = _test_image(seed=1)
    b = _test_image(seed=2)
    score = psnr(a, b)
    assert 0 < score < float("inf")


def test_psnr_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        psnr(_test_image((64, 64)), _test_image((128, 128)))


# --- presets --------------------------------------------------------------


def test_expected_presets_exist():
    assert set(PRESETS) == {"light", "medium", "heavy"}


def test_preset_output_keeps_size_and_mode():
    img = _test_image()
    for name in PRESETS:
        out = apply_preset(img, name, seed=1)
        assert out.size == img.size
        assert out.mode == "RGB"


def test_preset_actually_changes_pixels():
    """A 'removal' that leaves the image bit-identical removes nothing."""
    img = _test_image()
    for name in PRESETS:
        out = apply_preset(img, name, seed=1)
        assert np.asarray(out).tobytes() != np.asarray(img).tobytes(), name


def test_heavier_presets_damage_more():
    img = _test_image()
    scores = [psnr(img, apply_preset(img, n, seed=1)) for n in ("light", "medium", "heavy")]
    assert scores[0] > scores[1] > scores[2], scores


def test_light_preset_stays_visually_close():
    img = _test_image()
    assert psnr(img, apply_preset(img, "light", seed=1)) > 25


def test_preset_is_deterministic_for_a_seed():
    img = _test_image()
    a = apply_preset(img, "medium", seed=42)
    b = apply_preset(img, "medium", seed=42)
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_different_seeds_give_different_results():
    """The frequency-domain stage is seeded, so seeds must change the output.

    This also guards the pywt fallback: if the DWT stage silently no-opped when
    PyWavelets is absent, medium would become seed-independent and this fails.
    """
    img = _test_image()
    a = apply_preset(img, "medium", seed=1)
    b = apply_preset(img, "medium", seed=2)
    assert np.asarray(a).tobytes() != np.asarray(b).tobytes()


def test_presets_declare_a_psnr_floor_below_their_typical_output():
    """A preset whose floor its own normal output cannot clear is unusable."""
    img = _test_image()
    for name, preset in PRESETS.items():
        score = psnr(img, apply_preset(img, name, seed=1))
        assert score >= preset.min_psnr, f"{name}: {score:.2f} dB < floor {preset.min_psnr}"


def test_grainy_source_can_trip_the_guard():
    """Heavy filtering of very noisy content is exactly what the floor is for."""
    noisy = _test_image(grain=40.0)
    score = psnr(noisy, apply_preset(noisy, "heavy", seed=1))
    assert score < PRESETS["light"].min_psnr


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        apply_preset(_test_image(), "nuclear", seed=1)


def test_handles_non_square_and_tiny_images():
    for size in ((64, 33), (17, 64), (8, 8)):
        img = _test_image(size)
        out = apply_preset(img, "medium", seed=1)
        assert out.size == img.size


# --- CLI ------------------------------------------------------------------


def _run(*args, **kwargs):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "clean_pixel_lite.py"), *args],
        capture_output=True, text=True, timeout=120, **kwargs,
    )


def test_cli_writes_output(tmp_path):
    src = _write(tmp_path)
    out = tmp_path / "out.png"
    r = _run(str(src), "-o", str(out))
    assert r.returncode == 0, r.stderr
    assert out.is_file()
    assert Image.open(out).size == Image.open(src).size


def test_cli_json_reports_psnr_and_preset(tmp_path):
    src = _write(tmp_path)
    out = tmp_path / "out.png"
    r = _run(str(src), "-o", str(out), "--preset", "medium", "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["preset"] == "medium"
    assert payload["psnr"] > 0
    assert payload["available"] is True


def test_cli_default_output_path(tmp_path):
    src = _write(tmp_path)
    r = _run(str(src))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "in.lite.png").is_file()


def test_cli_missing_file_exits_2(tmp_path):
    r = _run(str(tmp_path / "nope.png"))
    assert r.returncode == 2


def test_cli_bad_preset_is_rejected(tmp_path):
    src = _write(tmp_path)
    r = _run(str(src), "--preset", "nuclear")
    assert r.returncode != 0


def test_psnr_floor_refuses_and_writes_nothing(tmp_path):
    """An impossible floor must refuse rather than quietly wreck the image."""
    src = _write(tmp_path)
    out = tmp_path / "out.png"
    r = _run(str(src), "-o", str(out), "--preset", "heavy", "--keep-psnr", "99")
    assert r.returncode == 1
    assert not out.exists()
    assert "psnr" in (r.stderr + r.stdout).lower()


def test_psnr_floor_zero_disables_the_guard(tmp_path):
    src = _write(tmp_path)
    out = tmp_path / "out.png"
    r = _run(str(src), "-o", str(out), "--preset", "heavy", "--keep-psnr", "0")
    assert r.returncode == 0, r.stderr
    assert out.is_file()


def test_cli_jpeg_output_suffix(tmp_path):
    src = _write(tmp_path)
    out = tmp_path / "out.jpg"
    r = _run(str(src), "-o", str(out))
    assert r.returncode == 0, r.stderr
    assert Image.open(out).format == "JPEG"


def test_cli_refuses_symlink_output(tmp_path):
    """Matches the safe-write posture of every other cleaner in this repo."""
    src = _write(tmp_path)
    victim = tmp_path / "victim.png"
    victim.write_bytes(b"original")
    link = tmp_path / "link.png"
    link.symlink_to(victim)
    r = _run(str(src), "-o", str(link))
    assert r.returncode != 0
    assert victim.read_bytes() == b"original"


def test_cli_seed_makes_runs_reproducible(tmp_path):
    src = _write(tmp_path)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    assert _run(str(src), "-o", str(a), "--seed", "5").returncode == 0
    assert _run(str(src), "-o", str(b), "--seed", "5").returncode == 0
    assert a.read_bytes() == b.read_bytes()
