"""Hardware detection and self-hosted model policy.

Single source of truth for "what can this machine actually run?". Imported by
the Layer B text path, the image pixel-removal path, and ``doctor.py`` so the
three never disagree about the device, the dtype, or which model to suggest.

The project's goal here is cost, not secrecy: every model named below is
open-weight and runs locally, so no metered inference API is ever required.
Downloading those weights over the network is fine and expected.

Nothing in this module imports torch at module scope. A machine with no GPU
stack must still be able to run ``doctor.py`` and the deterministic Layer A
cleaners, so every probe degrades to "cpu" rather than raising.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Iterable

# Tier identifiers. Ordered weakest to strongest; the ordering is load-bearing
# for max_chunk_chars() and for doctor.py's reporting.
TIER_CPU = "tier0-cpu"
TIER_LOW = "tier1-low"
TIER_MID = "tier2-mid"
TIER_HIGH = "tier3-high"

TIERS: tuple[str, ...] = (TIER_CPU, TIER_LOW, TIER_MID, TIER_HIGH)

# VRAM floor for each GPU tier, in GB.
_TIER_LOW_FLOOR = 4.0
_TIER_MID_FLOOR = 8.0
_TIER_HIGH_FLOOR = 16.0

# Apple Silicon shares one pool between CPU and GPU. Metal's recommended
# working set is well under total RAM, and leaving headroom for the OS is what
# keeps a large model from swapping the machine to a standstill.
_MPS_VRAM_FRACTION = 0.7

DTYPES: tuple[str, ...] = ("fp16", "bf16", "fp32")


@dataclass(frozen=True)
class Hardware:
    """A snapshot of what this machine offers for local inference."""

    backend: str  # cuda | mps | cpu
    vram_gb: float
    ram_gb: float
    free_disk_gb: float
    torch_available: bool
    device_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TierPolicy:
    """What a tier is allowed to attempt, and how it must be configured."""

    name: str
    text_budget_gb: float
    context_tokens: int
    image_backends: tuple[str, ...]
    ctrlregen_dtype: str
    # None = keep the whole pipeline resident (fastest). "model" = offload
    # whole submodules between steps. "sequential" = offload per layer; much
    # slower but fits SD1.5 + ControlNet + IP-Adapter into ~4 GB.
    ctrlregen_offload: str | None


@dataclass(frozen=True)
class ModelRec:
    """A concrete open-weight model recommendation."""

    name: str
    family: str
    params: str
    quant: str
    approx_gb: float
    context_tokens: int
    runtime: str  # ollama | llamacpp
    pull_cmd: str


_POLICIES: dict[str, TierPolicy] = {
    TIER_CPU: TierPolicy(
        name=TIER_CPU,
        text_budget_gb=4.0,
        context_tokens=8192,
        image_backends=("lite",),
        ctrlregen_dtype="fp32",
        ctrlregen_offload="sequential",
    ),
    TIER_LOW: TierPolicy(
        name=TIER_LOW,
        text_budget_gb=6.0,
        context_tokens=8192,
        image_backends=("lite", "ctrlregen"),
        ctrlregen_dtype="fp16",
        ctrlregen_offload="sequential",
    ),
    TIER_MID: TierPolicy(
        name=TIER_MID,
        text_budget_gb=12.0,
        context_tokens=16384,
        image_backends=("lite", "ctrlregen"),
        ctrlregen_dtype="fp16",
        ctrlregen_offload="model",
    ),
    TIER_HIGH: TierPolicy(
        name=TIER_HIGH,
        text_budget_gb=24.0,
        context_tokens=32768,
        image_backends=("lite", "ctrlregen"),
        ctrlregen_dtype="fp16",
        ctrlregen_offload=None,
    ),
}


# Open-weight instruct models that paraphrase well at their size. Sizes are the
# Q4_K_M GGUF footprint as served by Ollama, which is the quantization these
# tags resolve to by default.
#
# `family` exists for model hygiene, not marketing: rewriting text with a model
# from the same vendor that generated it risks re-stamping the output with that
# vendor's own watermark. recommend_models(exclude_families=...) enforces it.
_CATALOG: tuple[ModelRec, ...] = (
    # --- small enough for CPU-only or a 4 GB card ---
    ModelRec("llama3.2:3b", "meta", "3B", "Q4_K_M", 2.0, 8192, "ollama", "ollama pull llama3.2:3b"),
    ModelRec("qwen2.5:3b", "alibaba", "3B", "Q4_K_M", 1.9, 32768, "ollama", "ollama pull qwen2.5:3b"),
    ModelRec("gemma2:2b", "google", "2B", "Q4_K_M", 1.6, 8192, "ollama", "ollama pull gemma2:2b"),
    # --- 4-8 GB ---
    ModelRec("qwen2.5:7b", "alibaba", "7B", "Q4_K_M", 4.7, 32768, "ollama", "ollama pull qwen2.5:7b"),
    ModelRec("llama3.1:8b", "meta", "8B", "Q4_K_M", 4.9, 32768, "ollama", "ollama pull llama3.1:8b"),
    ModelRec("mistral:7b", "mistralai", "7B", "Q4_K_M", 4.4, 32768, "ollama", "ollama pull mistral:7b"),
    # --- 8-16 GB ---
    ModelRec("qwen2.5:14b", "alibaba", "14B", "Q4_K_M", 9.0, 32768, "ollama", "ollama pull qwen2.5:14b"),
    ModelRec("phi4:14b", "microsoft", "14B", "Q4_K_M", 9.1, 16384, "ollama", "ollama pull phi4:14b"),
    ModelRec(
        "mistral-nemo:12b", "mistralai", "12B", "Q4_K_M", 7.1, 32768, "ollama",
        "ollama pull mistral-nemo:12b",
    ),
    # --- 16 GB+ ---
    ModelRec("qwen2.5:32b", "alibaba", "32B", "Q4_K_M", 20.0, 32768, "ollama", "ollama pull qwen2.5:32b"),
    ModelRec("gemma2:27b", "google", "27B", "Q4_K_M", 16.7, 8192, "ollama", "ollama pull gemma2:27b"),
)


# Suspected-origin vendor -> the model family to keep away from. Used so the
# README's "prefer a non-origin model" advice becomes mechanically enforceable
# instead of prose the operator has to remember.
_ORIGIN_FAMILIES: dict[str, str] = {
    "claude": "anthropic",
    "anthropic": "anthropic",
    "gemini": "google",
    "google": "google",
    "gemma": "google",
    "bard": "google",
    "openai": "openai",
    "gpt": "openai",
    "chatgpt": "openai",
    "mistral": "mistralai",
    "llama": "meta",
    "meta": "meta",
    "qwen": "alibaba",
}


def family_for_origin(origin: str | None) -> str | None:
    """Map a suspected origin vendor to the model family to exclude."""
    if not origin:
        return None
    return _ORIGIN_FAMILIES.get(origin.strip().lower())


def _probe_torch() -> Any:
    """Import torch, or return None when it is unavailable.

    Isolated into its own function so tests can simulate a machine with no
    torch, or one whose torch imports but fails on probing (driver mismatch).
    """
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:
        return None


def _total_ram_gb() -> float:
    """Physical RAM in GB, best-effort across platforms."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1 << 30), 2)
    except (AttributeError, ValueError, OSError):
        pass
    try:  # Windows
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
        return round(status.ullTotalPhys / (1 << 30), 2)
    except Exception:
        return 0.0


def _free_disk_gb(path: str | None = None) -> float:
    """Free space where model weights will land (~10 GB for CtrlRegen)."""
    target = path or os.path.expanduser("~")
    try:
        return round(shutil.disk_usage(target).free / (1 << 30), 2)
    except OSError:
        return 0.0


def detect_hardware(disk_path: str | None = None) -> Hardware:
    """Probe the machine. Never raises: an unusable GPU stack reports as cpu."""
    ram_gb = _total_ram_gb()
    free_disk_gb = _free_disk_gb(disk_path)

    torch = _probe_torch()
    if torch is None:
        return Hardware("cpu", 0.0, ram_gb, free_disk_gb, False, "cpu")

    # Everything below touches driver-backed attributes, any of which can throw
    # on a broken CUDA install. A detection helper that crashes the tool it is
    # meant to make safe would defeat the purpose, so one guard covers them all.
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1 << 30), 2)
            return Hardware("cuda", vram_gb, ram_gb, free_disk_gb, True, props.name)

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return Hardware(
                "mps",
                round(ram_gb * _MPS_VRAM_FRACTION, 2),
                ram_gb,
                free_disk_gb,
                True,
                "apple-silicon-unified",
            )
    except Exception:
        return Hardware("cpu", 0.0, ram_gb, free_disk_gb, False, "cpu")

    return Hardware("cpu", 0.0, ram_gb, free_disk_gb, True, "cpu")


def pick_tier(hw: Hardware) -> str:
    """Select the capability tier for this machine."""
    # No torch means no diffusion path at all, whatever the card reports.
    if not hw.torch_available or hw.backend == "cpu":
        return TIER_CPU
    if hw.vram_gb >= _TIER_HIGH_FLOOR:
        return TIER_HIGH
    if hw.vram_gb >= _TIER_MID_FLOOR:
        return TIER_MID
    if hw.vram_gb >= _TIER_LOW_FLOOR:
        return TIER_LOW
    return TIER_CPU


def tier_policy(tier: str) -> TierPolicy:
    try:
        return _POLICIES[tier]
    except KeyError:
        raise ValueError(f"unknown tier: {tier!r} (expected one of {', '.join(TIERS)})") from None


def recommend_models(
    tier: str,
    *,
    exclude_families: Iterable[str] | None = None,
) -> list[ModelRec]:
    """Open-weight models that fit *tier*, largest first.

    ``exclude_families`` drops a vendor entirely — pass the suspected origin's
    family so a rewrite never re-stamps the text with the same vendor's mark.
    """
    budget = tier_policy(tier).text_budget_gb
    blocked = {f.strip().lower() for f in (exclude_families or ()) if f and f.strip()}
    fitting = [
        rec for rec in _CATALOG if rec.approx_gb <= budget and rec.family.lower() not in blocked
    ]
    return sorted(fitting, key=lambda r: r.approx_gb, reverse=True)


def resolve_device(raw: str | None) -> str:
    """Resolve an ``auto`` device hint to a concrete torch device."""
    if raw and raw != "auto":
        return raw
    torch = _probe_torch()
    if torch is None:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def resolve_dtype(device: str, raw: str | None) -> str:
    """Resolve an ``auto`` dtype hint for *device*.

    fp16 halves weight memory and is what makes the 512x512 CtrlRegen pipeline
    fit on a small card. On CPU it is a pessimization — most CPU kernels have no
    fp16 path and fall back to slow emulation — so CPU stays fp32.
    """
    if raw and raw != "auto":
        if raw not in DTYPES:
            raise ValueError(f"unknown dtype: {raw!r} (expected one of {', '.join(DTYPES)})")
        return raw
    return "fp32" if device == "cpu" else "fp16"


def max_chunk_chars(tier: str) -> int:
    """Rewrite chunk size in characters for *tier*.

    Layer B sends a document to a local model that has a far smaller context
    than a hosted frontier model. The prompt, the source text and the rewrite
    all share that window, so budget roughly a third of it for the source and
    assume ~3.5 characters per token.
    """
    ctx = tier_policy(tier).context_tokens
    return int(ctx / 3 * 3.5)


def estimate_ctrlregen_vram_gb(dtype: str, offload: str | None) -> float:
    """Rough VRAM need for the CtrlRegen pipeline under a given configuration.

    SD1.5 UNet + ControlNet + DINOv2 IP-Adapter + VAE, plus activations for one
    512x512 tile. Deliberately an over-estimate: refusing a run that would have
    just fit is a far better failure than an OOM after a ~10 GB download.
    """
    base = 5.4 if dtype == "fp32" else 2.9
    if offload == "sequential":
        return round(base * 0.45, 2)
    if offload == "model":
        return round(base * 0.7, 2)
    return round(base, 2)
