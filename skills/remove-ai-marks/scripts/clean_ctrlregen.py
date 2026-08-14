#!/usr/bin/env python3
"""Optional CtrlRegen pixel-watermark remover backed by an external noai-watermark checkout.

This script does NOT vendor upstream code. It imports ``CtrlRegenEngine`` from
a user-provided checkout of ``mertizci/noai-watermark`` at runtime, using that
environment's optional dependencies (torch, diffusers, transformers, etc.).

Exit codes:
  0  removed successfully
  1  remover runtime error
  2  bad input (missing/unreadable image, bad args)
  3  remover unavailable (not configured / missing checkout / missing deps)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import cleaned_path, safe_write_bytes  # noqa: E402
from local_models import (  # noqa: E402
    detect_hardware,
    estimate_ctrlregen_vram_gb,
    resolve_device,
    resolve_dtype,
)

# Backend default guidance scale (CtrlRegenEngine.run() default). Kept
# internal: strength is the user-facing knob, not the CFG scale.
DEFAULT_GUIDANCE_SCALE = 2.0

# Upstream's tiling geometry (src/ctrlregen/engine.py). Mirrored here only to
# estimate tile counts for the --max-pixels warning; the backend still owns the
# actual tiling.
TILE_SIZE = 512
TILE_OVERLAP = 192


def resolve_upstream(raw: str | None) -> Path | None:
    if not raw:
        return None
    upstream = Path(raw).expanduser().resolve()
    if not upstream.is_dir():
        return None
    return upstream


def estimate_tiles(width: int, height: int) -> int:
    """How many 512px tiles the backend will process for this image.

    Runs scale with tile count, so a 4K input is ~100 sequential diffusion
    passes. Worth telling the user before they start, not after.
    """
    stride = TILE_SIZE - TILE_OVERLAP
    if width <= TILE_SIZE and height <= TILE_SIZE:
        return 1
    cols = max(1, -(-max(0, width - TILE_SIZE) // stride) + 1)
    rows = max(1, -(-max(0, height - TILE_SIZE) // stride) + 1)
    return cols * rows


def _torch_dtype(name: str):
    """Map our dtype name onto a torch dtype object.

    Returns None when torch is not importable, which hands the choice back to
    the backend's own default. The adapter must stay runnable (and testable)
    without torch present, since the real dependency lives in the backend venv.
    """
    try:
        import torch
    except ImportError:
        return None

    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def apply_low_vram(pipeline, *, device: str, offload: str | None, low_vram: bool) -> list[str]:
    """Apply diffusers memory knobs to an already-loaded pipeline.

    Upstream's CtrlRegenEngine builds the pipeline and attempts xformers, but
    exposes none of these, so they are applied here against ``engine._pipeline``
    after an explicit ``engine.load()``.

    Every call is guarded: the pipeline is a third-party object at a pinned
    commit, and a missing or renamed method must degrade to a warning rather
    than kill a run that would otherwise have worked.

    CPU offload is CUDA/accelerate territory. On MPS it either errors or
    silently thrashes unified memory, so it is skipped there and the slicing
    knobs carry the load instead.
    """
    applied: list[str] = []
    wanted: list[str] = []

    if low_vram:
        wanted += ["enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"]

    if offload and offload != "none":
        if device == "cuda":
            wanted.append(
                "enable_sequential_cpu_offload"
                if offload == "sequential"
                else "enable_model_cpu_offload"
            )
        else:
            print(
                f"[ctrlregen] skipping {offload} CPU offload: not supported on {device}",
                file=sys.stderr,
            )

    for name in wanted:
        fn = getattr(pipeline, name, None)
        if fn is None:
            print(f"[ctrlregen] pipeline has no {name}; skipping", file=sys.stderr)
            continue
        try:
            fn()
            applied.append(name.replace("enable_", ""))
        except Exception as e:
            print(f"[ctrlregen] {name} failed ({e}); continuing", file=sys.stderr)

    return applied


def save_image_bytes(image, output: Path) -> bytes:
    """Encode the regenerated image, honoring a JPEG output suffix."""
    ext = output.suffix.lower()
    buf = io.BytesIO()
    if ext in (".jpg", ".jpeg", ".jpe", ".jfif"):
        image = image.convert("RGB")
        image.save(buf, format="JPEG", quality=95)
    else:
        image.save(buf, format="PNG")
    return buf.getvalue()


def _progress(message: str) -> None:
    print(f"[ctrlregen] {message}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image (PNG/JPEG/etc.)")
    p.add_argument("-o", "--output", type=Path, help="Output path (default: *.ctrlregen.*)")
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=None,
        help="noai-watermark checkout root (default: $NOAI_WATERMARK_DIR)",
    )
    p.add_argument(
        "--strength",
        type=float,
        default=0.25,
        help="Regeneration strength in (0, 1]; lower preserves more detail (default: 0.25)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Diffusion inference steps (default: 50; effective steps ~= steps * strength)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto|cpu|cuda|mps (default: auto)",
    )
    p.add_argument(
        "--dtype",
        choices=("auto", "fp16", "bf16", "fp32"),
        default="auto",
        help="Weight precision (default: auto — fp16 on GPU, fp32 on CPU)",
    )
    p.add_argument(
        "--low-vram",
        action="store_true",
        help="Enable attention slicing and VAE slicing/tiling. Slower, much less VRAM",
    )
    p.add_argument(
        "--offload",
        choices=("none", "model", "sequential"),
        default="none",
        help="CUDA-only CPU offload: 'model' moves submodules between steps, "
        "'sequential' offloads per layer (slowest, smallest footprint)",
    )
    p.add_argument(
        "--assume-vram-gb",
        type=float,
        default=None,
        help="Override detected VRAM for the preflight (testing/CI)",
    )
    p.add_argument(
        "--skip-vram-check",
        action="store_true",
        help="Run even when the VRAM preflight says it will not fit",
    )
    p.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Refuse images larger than this many pixels (guards against a 4K "
        "input becoming ~100 sequential diffusion passes)",
    )
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        print(f"not a file: {args.path}", file=sys.stderr)
        return 2
    if not 0 < args.strength <= 1:
        print(f"strength must be in (0, 1]: {args.strength}", file=sys.stderr)
        return 2
    if args.steps < 1:
        print(f"steps must be >= 1: {args.steps}", file=sys.stderr)
        return 2

    raw_upstream = args.upstream_dir or os.environ.get("NOAI_WATERMARK_DIR")
    upstream = resolve_upstream(str(raw_upstream) if raw_upstream else None)
    if upstream is None:
        print(
            "CtrlRegen not configured: set NOAI_WATERMARK_DIR or pass --upstream-dir",
            file=sys.stderr,
        )
        return 3

    src_dir = upstream / "src"
    if not src_dir.is_dir():
        print(f"upstream src dir not found: {src_dir}", file=sys.stderr)
        return 3

    sys.path.insert(0, str(src_dir))
    try:
        from PIL import Image  # noqa: E402
        from ctrlregen.engine import CtrlRegenEngine, is_ctrlregen_available  # noqa: E402
    except ImportError as e:
        print(f"CtrlRegen dependencies missing: {e}", file=sys.stderr)
        print("run setup_ctrlregen.sh first", file=sys.stderr)
        return 3

    if not is_ctrlregen_available():
        print(
            "CtrlRegen dependencies not installed; run setup_ctrlregen.sh first",
            file=sys.stderr,
        )
        return 3

    try:
        image = Image.open(args.path).convert("RGB")
        image.load()
    except Exception as e:
        print(f"could not load image: {e}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    try:
        dtype_name = resolve_dtype(device, args.dtype)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    offload = None if args.offload == "none" else args.offload
    width, height = image.size
    tiles = estimate_tiles(width, height)

    if args.max_pixels is not None and width * height > args.max_pixels:
        print(
            f"refusing image of {width}x{height} = {width * height} pixels "
            f"(--max-pixels {args.max_pixels}); it would need ~{tiles} diffusion "
            "passes. Downscale it first, or raise --max-pixels.",
            file=sys.stderr,
        )
        return 2

    if tiles > 16:
        _progress(
            f"warning: {width}x{height} needs ~{tiles} sequential 512px passes; "
            "consider downscaling first"
        )

    # VRAM preflight. The whole point is to fail here rather than after the
    # ~10 GB of weights have been pulled and the first denoising step OOMs.
    # CPU runs draw on system RAM, so the GPU estimate does not apply.
    need_gb = estimate_ctrlregen_vram_gb(dtype_name, offload)
    have_gb = args.assume_vram_gb
    if have_gb is None and device != "cpu":
        have_gb = detect_hardware().vram_gb
    if device != "cpu" and not args.skip_vram_check and have_gb is not None and have_gb > 0:
        if have_gb < need_gb:
            print(
                f"insufficient VRAM: this configuration needs ~{need_gb} GB "
                f"({dtype_name}, offload={offload or 'none'}) but only {have_gb} GB "
                "is available.",
                file=sys.stderr,
            )
            print(
                "Try --low-vram, or --offload sequential, or --dtype fp16; "
                "use --skip-vram-check to run anyway, or --remove-pixel lite "
                "from clean_image.py for a no-GPU path.",
                file=sys.stderr,
            )
            return 3

    output = args.output or cleaned_path(args.path, ".ctrlregen")

    engine = CtrlRegenEngine(
        base_model_id=None,
        device=device,
        torch_dtype=_torch_dtype(dtype_name),
        hf_token=os.environ.get("HF_TOKEN"),
        progress_callback=_progress,
    )

    # Load explicitly so the memory knobs can be applied to the real pipeline
    # before inference. engine.run() calls load() itself and it is idempotent,
    # so this only moves the work earlier.
    low_vram_applied: list[str] = []
    try:
        loader = getattr(engine, "load", None)
        if loader is not None and (args.low_vram or offload):
            loader()
            pipeline = getattr(engine, "_pipeline", None)
            if pipeline is None:
                _progress("warning: backend exposed no pipeline; skipping low-VRAM knobs")
            else:
                low_vram_applied = apply_low_vram(
                    pipeline, device=device, offload=offload, low_vram=args.low_vram
                )
    except Exception as e:
        print(f"CtrlRegen error during load: {e}", file=sys.stderr)
        return 1

    try:
        result = engine.run(
            image,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            seed=args.seed,
        )
    except Exception as e:
        print(f"CtrlRegen error: {e}", file=sys.stderr)
        return 1

    try:
        data = save_image_bytes(result, output)
        safe_write_bytes(output, data)
    except (OSError, ValueError) as e:
        print(f"cannot write output: {e}", file=sys.stderr)
        return 1

    payload = {
        "available": True,
        "upstream_dir": str(upstream),
        "output": str(output),
        "strength": args.strength,
        "steps": args.steps,
        "device": device,
        "dtype": dtype_name,
        "offload": offload,
        "low_vram_applied": low_vram_applied,
        "vram_estimate_gb": need_gb,
        "vram_available_gb": have_gb,
        "tiles": tiles,
        "seed": args.seed,
        "input_size": list(image.size),
        "output_size": list(result.size),
        "bytes_out": len(data),
    }

    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(
            f"CtrlRegen removed: {args.path} -> {output} "
            f"({payload['input_size']} -> {payload['output_size']}, "
            f"strength {args.strength}, device {device}, {dtype_name}"
            + (f", offload={offload}" if offload else "")
            + (f", {tiles} tiles" if tiles > 1 else "")
            + ")"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
