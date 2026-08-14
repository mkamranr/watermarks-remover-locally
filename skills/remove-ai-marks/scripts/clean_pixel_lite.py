#!/usr/bin/env python3
"""Zero-GPU pixel-watermark attack tier (Pillow + numpy only).

This is the path for machines that cannot run a 512x512 diffusion pipeline. It
applies the classic cheap signal-processing attacks — resampling, mild blur plus
resharpen, median filtering, sub-degree rotation, optional wavelet-band noise,
quantization, and JPEG recompression cycles — in one pass, in under a second, on
any laptop.

HONEST LIMITS — read before trusting this:

  These attacks degrade *fragile* pixel watermarks. Robust modern schemes
  (SynthID, StableSignature, Tree-Ring) are trained specifically to survive
  exactly this class of transformation, and they usually do. This tier is a free
  first pass and a "something rather than nothing" option, NOT a substitute for
  CtrlRegen regeneration (`clean_image.py --remove-pixel ctrlregen`).

  There is no local detector that can tell you whether it worked. When a
  reverse-SynthID checkout is configured, `clean_image.py` reports the score
  before and after so you can see for yourself how little it typically moves.

Exit codes:
  0  cleaned successfully
  1  runtime error, or the result fell below --keep-psnr
  2  bad input (missing/unreadable image, bad args)
  3  unavailable (Pillow/numpy not installed)
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import cleaned_path, eprint, emit_json, safe_write_bytes  # noqa: E402

try:
    import numpy as np
    from PIL import Image, ImageFilter
except ImportError as _e:  # pragma: no cover - exercised via the CLI exit code
    eprint(f"pixel-lite dependencies missing: {_e}")
    eprint("pip install Pillow numpy")
    raise SystemExit(3) from None


@dataclass(frozen=True)
class Preset:
    """One point on the damage/removal curve.

    Every field is a knob on a published watermark attack. They compose: each
    stage attacks a different domain (geometric, spatial-frequency, colour,
    block-DCT), so a mark tuned to survive one may still be hit by another.
    """

    name: str
    crop_fraction: float  # crop then resize back — breaks pixel alignment
    rotate_degrees: float  # rotate out and back — pure resampling loss
    blur_radius: float  # attacks high-frequency carriers
    unsharp: tuple[float, int, int]  # restore apparent sharpness after the blur
    median_size: int  # nonlinear filter; 0 disables
    dwt_sigma: float  # noise into wavelet detail bands; 0 disables
    quantize_colors: int  # palette reduction + dither; 0 disables
    jpeg_cycles: tuple[int, ...]  # successive JPEG qualities
    min_psnr: float  # default damage floor for this preset, in dB


# Floors are per-preset because a single global default is wrong at both ends:
# it either rejects `heavy` (which the operator explicitly asked for) or fails
# to catch a `light` run that wrecked the image. Typical photographic content
# lands around 37 dB on light, 32 on medium and 26 on heavy; the floors sit
# below those so the guard fires on genuine damage, not on grainy source.
PRESETS: dict[str, Preset] = {
    "light": Preset(
        name="light",
        crop_fraction=0.005,
        rotate_degrees=0.15,
        blur_radius=0.3,
        unsharp=(1.0, 40, 3),
        median_size=0,
        dwt_sigma=0.0,
        quantize_colors=0,
        jpeg_cycles=(92,),
        min_psnr=30.0,
    ),
    "medium": Preset(
        name="medium",
        crop_fraction=0.012,
        rotate_degrees=0.35,
        blur_radius=0.3,
        unsharp=(1.5, 60, 3),
        median_size=3,
        dwt_sigma=1.2,
        quantize_colors=0,
        jpeg_cycles=(88, 92),
        min_psnr=26.0,
    ),
    "heavy": Preset(
        name="heavy",
        crop_fraction=0.03,
        rotate_degrees=0.8,
        blur_radius=1.0,
        unsharp=(2.0, 100, 3),
        median_size=3,
        dwt_sigma=3.0,
        quantize_colors=192,
        jpeg_cycles=(70, 82, 90),
        min_psnr=22.0,
    ),
}

# Pillow moved the resampling constants onto Image.Resampling in 9.1 and
# deprecated the old aliases. Resolve once so the module works across the range
# of Pillow versions this project supports without emitting warnings.
_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
_DITHER_FS = getattr(getattr(Image, "Dither", Image), "FLOYDSTEINBERG")


def psnr(a: "Image.Image", b: "Image.Image") -> float:
    """Peak signal-to-noise ratio in dB between two same-sized RGB images.

    Used as the damage guard: above ~35 dB differences are hard to see, below
    ~25 dB they are obvious. Identical images return infinity.
    """
    if a.size != b.size:
        raise ValueError(f"size mismatch: {a.size} vs {b.size}")
    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


def _crop_resize(img: "Image.Image", fraction: float) -> "Image.Image":
    """Crop a border off and scale back up, destroying pixel-grid alignment."""
    if fraction <= 0:
        return img
    w, h = img.size
    dx, dy = int(w * fraction), int(h * fraction)
    if dx < 1 or dy < 1 or w - 2 * dx < 4 or h - 2 * dy < 4:
        return img
    return img.crop((dx, dy, w - dx, h - dy)).resize((w, h), _BICUBIC)


def _rotate_roundtrip(img: "Image.Image", degrees: float) -> "Image.Image":
    """Rotate out and back: geometry is restored, resampling loss is not."""
    if degrees <= 0:
        return img
    out = img.rotate(degrees, resample=_BICUBIC, expand=False)
    return out.rotate(-degrees, resample=_BICUBIC, expand=False)


def _dwt_noise(img: "Image.Image", sigma: float, rng) -> "Image.Image":
    """Add noise to wavelet detail bands, where frequency-domain marks live.

    PyWavelets is optional. When it is missing the stage degrades to mild
    spatial noise rather than vanishing: silently skipping it would make the
    preset weaker than its name claims on any machine without pywt, which is
    exactly the kind of quiet capability gap this project should not have. The
    fallback sigma is halved because spatial noise costs more PSNR than the
    same perturbation confined to detail bands.
    """
    if sigma <= 0:
        return img

    arr = np.asarray(img.convert("RGB"), dtype=np.float64)

    try:
        import pywt
    except ImportError:
        noisy = arr + rng.normal(0, sigma * 0.5, arr.shape)
        return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="RGB")

    out = np.empty_like(arr)
    for c in range(3):
        cA, (cH, cV, cD) = pywt.dwt2(arr[:, :, c], "haar")
        cH = cH + rng.normal(0, sigma, cH.shape)
        cV = cV + rng.normal(0, sigma, cV.shape)
        cD = cD + rng.normal(0, sigma, cD.shape)
        rec = pywt.idwt2((cA, (cH, cV, cD)), "haar")
        out[:, :, c] = rec[: arr.shape[0], : arr.shape[1]]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _jpeg_cycle(img: "Image.Image", quality: int) -> "Image.Image":
    """One encode/decode round trip: block-DCT quantization loss."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_preset(img: "Image.Image", preset: str, *, seed: int = 0) -> "Image.Image":
    """Run the full attack chain for *preset* over *img*.

    Deterministic for a given seed so runs are reproducible and testable.
    """
    try:
        cfg = PRESETS[preset]
    except KeyError:
        raise ValueError(
            f"unknown preset: {preset!r} (expected one of {', '.join(PRESETS)})"
        ) from None

    rng = np.random.default_rng(seed)
    out = img.convert("RGB")

    out = _crop_resize(out, cfg.crop_fraction)
    out = _rotate_roundtrip(out, cfg.rotate_degrees)
    if cfg.blur_radius > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=cfg.blur_radius))
    if cfg.unsharp[1] > 0:
        radius, percent, threshold = cfg.unsharp
        out = out.filter(
            ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
        )
    # MedianFilter needs an odd window and enough pixels to slide over.
    if cfg.median_size >= 3 and min(out.size) > cfg.median_size:
        out = out.filter(ImageFilter.MedianFilter(size=cfg.median_size | 1))
    out = _dwt_noise(out, cfg.dwt_sigma, rng)
    if cfg.quantize_colors > 0:
        out = out.quantize(colors=cfg.quantize_colors, dither=_DITHER_FS)
        out = out.convert("RGB")
    for quality in cfg.jpeg_cycles:
        out = _jpeg_cycle(out, quality)

    return out.convert("RGB")


def encode(img: "Image.Image", output: Path) -> bytes:
    ext = output.suffix.lower()
    buf = io.BytesIO()
    if ext in (".jpg", ".jpeg", ".jpe", ".jfif"):
        img.convert("RGB").save(buf, format="JPEG", quality=95)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("path", type=Path, help="Input image")
    p.add_argument("-o", "--output", type=Path, help="Output path (default: *.lite.*)")
    p.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="medium",
        help="Attack strength (default: medium)",
    )
    p.add_argument(
        "--keep-psnr",
        type=float,
        default=None,
        help="Refuse to write if the result falls below this PSNR in dB "
        "(default: the preset's own floor — 30/26/22 for light/medium/heavy; "
        "0 disables the guard)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible runs")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2

    try:
        src = Image.open(args.path)
        src.load()
        src = src.convert("RGB")
    except Exception as e:
        eprint(f"could not load image: {e}")
        return 2

    try:
        out_img = apply_preset(src, args.preset, seed=args.seed)
    except ValueError as e:
        eprint(f"error: {e}")
        return 2
    except Exception as e:
        eprint(f"pixel-lite error: {e}")
        return 1

    score = psnr(src, out_img)
    output = args.output or cleaned_path(args.path, ".lite")
    floor = PRESETS[args.preset].min_psnr if args.keep_psnr is None else args.keep_psnr

    if floor > 0 and score < floor:
        eprint(
            f"refusing to write: PSNR {score:.2f} dB is below the floor "
            f"of {floor:.2f} dB for preset '{args.preset}'."
        )
        eprint("Use a lighter --preset, or lower/disable the floor with --keep-psnr 0.")
        return 1

    try:
        data = encode(out_img, output)
        safe_write_bytes(output, data)
    except (OSError, ValueError) as e:
        eprint(f"cannot write output: {e}")
        return 1

    payload = {
        "available": True,
        "backend": "lite",
        "output": str(output),
        "preset": args.preset,
        "seed": args.seed,
        "psnr": round(score, 3),
        "psnr_floor": floor,
        "input_size": list(src.size),
        "output_size": list(out_img.size),
        "bytes_out": len(data),
        "note": (
            "Signal-processing attack only. Degrades fragile pixel marks; robust "
            "schemes (SynthID, StableSignature, Tree-Ring) are designed to survive "
            "these transforms and usually do. Not a substitute for CtrlRegen."
        ),
    }

    if args.json:
        emit_json(payload)
    else:
        eprint(
            f"pixel-lite cleaned: {args.path} -> {output} "
            f"(preset {args.preset}, PSNR {score:.2f} dB)"
        )
        eprint(
            "note: fragile marks only — robust watermarks likely survive; "
            "see --remove-pixel ctrlregen for regeneration-based removal"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
