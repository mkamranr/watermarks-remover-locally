#!/usr/bin/env python3
"""Preflight: what can this machine actually do, and how do I fix what it can't?

Run this before anything else. It answers the two questions that otherwise get
answered the expensive way — by a CUDA OOM after a ~10 GB model download, or by
a Layer B run that silently falls back to a metered cloud API.

Every capability is reported as ready / degraded / unavailable, and anything
that is not ready names the exact command that fixes it.

Exit code is always 0: this is a report, not a gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import emit_json, which  # noqa: E402
from local_models import (  # noqa: E402
    TIER_CPU,
    Hardware,
    detect_hardware,
    estimate_ctrlregen_vram_gb,
    max_chunk_chars,
    pick_tier,
    recommend_models,
    tier_policy,
)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_LLAMACPP_URL = "http://127.0.0.1:8080"

# SD1.5 + ControlNet + DINOv2 IP-Adapter + VAE, as pulled on first CtrlRegen run.
CTRLREGEN_WEIGHTS_GB = 10.0

READY = "ready"
DEGRADED = "degraded"
UNAVAILABLE = "unavailable"

_STATUS_MARK = {READY: "OK  ", DEGRADED: "WARN", UNAVAILABLE: "MISS"}


# --- probes ---------------------------------------------------------------


def probe_tools() -> dict[str, str | None]:
    return {name: which(name) for name in ("exiftool", "c2patool", "ollama", "llama-server")}


def probe_modules() -> dict[str, bool]:
    """Check optional imports without paying their import cost twice."""
    import importlib.util

    found = {}
    for name in ("PIL", "numpy", "torch"):
        try:
            found[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found[name] = False
    return found


def _http_get_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def probe_ollama(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 2.0) -> dict[str, Any]:
    """Ask a local Ollama which models are actually pulled."""
    try:
        data = _http_get_json(base_url.rstrip("/") + "/api/tags", timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return {"reachable": False, "models": []}
    models = [m.get("name", "") for m in (data.get("models") or []) if m.get("name")]
    return {"reachable": True, "models": models}


def probe_llamacpp(base_url: str = DEFAULT_LLAMACPP_URL, timeout: float = 2.0) -> dict[str, Any]:
    """Ask a local llama-server which model it has loaded."""
    try:
        data = _http_get_json(base_url.rstrip("/") + "/v1/models", timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return {"reachable": False, "models": []}
    models = [m.get("id", "") for m in (data.get("data") or []) if m.get("id")]
    return {"reachable": True, "models": models}


def probe_checkouts() -> dict[str, str | None]:
    """Locate the two optional external backends."""
    out: dict[str, str | None] = {}
    for key, env in (
        ("reverse_synthid", "REVERSE_SYNTHID_DIR"),
        ("noai_watermark", "NOAI_WATERMARK_DIR"),
    ):
        raw = os.environ.get(env)
        path = Path(raw).expanduser() if raw else None
        out[key] = str(path.resolve()) if path and path.is_dir() else None
    return out


# --- report ---------------------------------------------------------------


def _cap(key: str, label: str, status: str, detail: str, fix: str = "") -> dict[str, str]:
    return {"key": key, "label": label, "status": status, "detail": detail, "fix": fix}


def build_report(
    hw: Hardware,
    *,
    tools: dict[str, str | None],
    runtimes: dict[str, dict[str, Any]],
    checkouts: dict[str, str | None],
    modules: dict[str, bool],
) -> dict[str, Any]:
    """Assemble the capability report. Pure function of the probe results."""
    tier = pick_tier(hw)
    policy = tier_policy(tier)
    caps: list[dict[str, str]] = []

    # Layer A and the container cleaners are stdlib-only by design.
    caps.append(
        _cap("layer_a", "Layer A — invisible Unicode scrub", READY, "stdlib only, no model needed")
    )
    caps.append(
        _cap(
            "container_clean",
            "Container metadata strip (PNG/JPEG/SVG/DOCX/ODT/HTML/MD)",
            READY,
            "stdlib only",
        )
    )

    # PDF is the one container that genuinely needs a helper binary.
    if tools.get("exiftool"):
        caps.append(_cap("pdf_clean", "PDF metadata strip", READY, tools["exiftool"] or ""))
    else:
        caps.append(
            _cap(
                "pdf_clean",
                "PDF metadata strip",
                DEGRADED,
                "byte-level strip only; residual metadata likely",
                "install exiftool (brew install exiftool / apt install libimage-exiftool-perl)",
            )
        )

    if tools.get("c2patool"):
        caps.append(_cap("c2pa_inspect", "C2PA manifest inspection", READY, tools["c2patool"] or ""))
    else:
        caps.append(
            _cap(
                "c2pa_inspect",
                "C2PA manifest inspection",
                DEGRADED,
                "falling back to byte-scan heuristics",
                "install c2patool (cargo install c2patool)",
            )
        )

    # Layer B: the capability that otherwise silently costs money.
    ollama = runtimes.get("ollama") or {"reachable": False, "models": []}
    llamacpp = runtimes.get("llamacpp") or {"reachable": False, "models": []}
    served = list(ollama.get("models") or []) + list(llamacpp.get("models") or [])
    any_reachable = ollama.get("reachable") or llamacpp.get("reachable")
    picks = recommend_models(tier)
    suggestion = picks[0].name if picks else "qwen2.5:7b"

    if any_reachable and served:
        caps.append(
            _cap(
                "layer_b_local",
                "Layer B — local rewrite (no API cost)",
                READY,
                f"{len(served)} model(s) served: {', '.join(served[:4])}",
            )
        )
    elif any_reachable:
        caps.append(
            _cap(
                "layer_b_local",
                "Layer B — local rewrite (no API cost)",
                DEGRADED,
                "runtime is up but no model is pulled",
                f"ollama pull {suggestion}",
            )
        )
    else:
        caps.append(
            _cap(
                "layer_b_local",
                "Layer B — local rewrite (no API cost)",
                UNAVAILABLE,
                "no local inference runtime reachable",
                f"install and start Ollama, then: ollama pull {suggestion}",
            )
        )

    # Zero-GPU pixel tier.
    if modules.get("PIL") and modules.get("numpy"):
        caps.append(
            _cap("pixel_lite", "Pixel removal — lite (no GPU)", READY, "Pillow + numpy present")
        )
    else:
        missing = [m for m in ("PIL", "numpy") if not modules.get(m)]
        caps.append(
            _cap(
                "pixel_lite",
                "Pixel removal — lite (no GPU)",
                UNAVAILABLE,
                f"missing: {', '.join(missing)}",
                "pip install Pillow numpy",
            )
        )

    # Diffusion pixel tier.
    ctrlregen_ok = "ctrlregen" in policy.image_backends
    if not ctrlregen_ok:
        caps.append(
            _cap(
                "ctrlregen",
                "Pixel removal — CtrlRegen (diffusion)",
                UNAVAILABLE,
                f"{tier} has no usable GPU for a 512x512 diffusion pipeline",
                "use --remove-pixel lite instead, or run on a machine with >=4 GB VRAM",
            )
        )
    elif not checkouts.get("noai_watermark"):
        caps.append(
            _cap(
                "ctrlregen",
                "Pixel removal — CtrlRegen (diffusion)",
                UNAVAILABLE,
                "backend checkout not found",
                "scripts/setup_ctrlregen.sh, then export NOAI_WATERMARK_DIR",
            )
        )
    elif not modules.get("torch"):
        caps.append(
            _cap(
                "ctrlregen",
                "Pixel removal — CtrlRegen (diffusion)",
                UNAVAILABLE,
                "torch not importable in this interpreter",
                "run via the backend venv: $NOAI_WATERMARK_DIR/.venv/bin/python",
            )
        )
    elif hw.free_disk_gb < CTRLREGEN_WEIGHTS_GB * 1.2:
        caps.append(
            _cap(
                "ctrlregen",
                "Pixel removal — CtrlRegen (diffusion)",
                DEGRADED,
                f"only {hw.free_disk_gb} GB free disk; weights need ~{CTRLREGEN_WEIGHTS_GB} GB",
                "free up disk space, or set HF_HOME to a volume with room",
            )
        )
    else:
        caps.append(
            _cap(
                "ctrlregen",
                "Pixel removal — CtrlRegen (diffusion)",
                READY,
                f"{policy.ctrlregen_dtype}, offload={policy.ctrlregen_offload or 'none'}",
            )
        )

    # SynthID scoring — the only local signal for pixel marks.
    if checkouts.get("reverse_synthid"):
        caps.append(
            _cap(
                "synthid_score",
                "SynthID pixel scoring (local, best-effort)",
                READY,
                checkouts["reverse_synthid"] or "",
            )
        )
    else:
        caps.append(
            _cap(
                "synthid_score",
                "SynthID pixel scoring (local, best-effort)",
                UNAVAILABLE,
                "scorer checkout not found",
                "scripts/setup_synthid.sh, then export REVERSE_SYNTHID_DIR",
            )
        )

    return {
        "python": sys.version.split()[0],
        "hardware": hw.to_dict(),
        "tier": tier,
        "tier_detail": {
            "text_budget_gb": policy.text_budget_gb,
            "context_tokens": policy.context_tokens,
            "max_chunk_chars": max_chunk_chars(tier),
            "image_backends": list(policy.image_backends),
            "ctrlregen_dtype": policy.ctrlregen_dtype,
            "ctrlregen_offload": policy.ctrlregen_offload,
        },
        "ctrlregen_vram_estimate_gb": estimate_ctrlregen_vram_gb(
            policy.ctrlregen_dtype, policy.ctrlregen_offload
        ),
        "tools": tools,
        "runtimes": runtimes,
        "checkouts": checkouts,
        "modules": modules,
        "recommended_models": [
            {
                "name": r.name,
                "family": r.family,
                "params": r.params,
                "quant": r.quant,
                "approx_gb": r.approx_gb,
                "context_tokens": r.context_tokens,
                "pull_cmd": r.pull_cmd,
            }
            for r in recommend_models(tier)
        ],
        "capabilities": caps,
    }


def human_report(report: dict[str, Any]) -> str:
    hw = report["hardware"]
    lines: list[str] = []
    lines.append("watermarks-remover doctor")
    lines.append("=" * 60)
    lines.append(f"Python        {report['python']}")
    lines.append(
        f"Device        {hw['backend']} ({hw['device_name']})"
        + (f", {hw['vram_gb']} GB VRAM" if hw["vram_gb"] else "")
    )
    lines.append(f"Memory        {hw['ram_gb']} GB RAM, {hw['free_disk_gb']} GB free disk")
    lines.append(f"Tier          {report['tier']}")

    detail = report["tier_detail"]
    lines.append(
        f"Image paths   {', '.join(detail['image_backends'])}"
        + (
            f"  (CtrlRegen {detail['ctrlregen_dtype']},"
            f" offload={detail['ctrlregen_offload'] or 'none'},"
            f" ~{report['ctrlregen_vram_estimate_gb']} GB)"
            if "ctrlregen" in detail["image_backends"]
            else ""
        )
    )
    lines.append(f"Rewrite chunk {detail['max_chunk_chars']} chars")
    lines.append("")
    lines.append("Capabilities")
    lines.append("-" * 60)
    for cap in report["capabilities"]:
        lines.append(f"[{_STATUS_MARK[cap['status']]}] {cap['label']}")
        if cap["detail"]:
            lines.append(f"       {cap['detail']}")
        if cap["fix"]:
            lines.append(f"       fix: {cap['fix']}")

    if report["recommended_models"]:
        lines.append("")
        lines.append(f"Suggested local models for {report['tier']}")
        lines.append("-" * 60)
        for rec in report["recommended_models"][:4]:
            lines.append(
                f"  {rec['name']:<20} {rec['params']:>4} {rec['quant']:<8}"
                f" ~{rec['approx_gb']} GB   {rec['pull_cmd']}"
            )
        lines.append("")
        lines.append(
            "  Model hygiene: pick a family different from the text's suspected"
            " origin, so the rewrite does not re-stamp it."
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="Emit the full report as JSON")
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--llamacpp-url", default=DEFAULT_LLAMACPP_URL)
    p.add_argument(
        "--probe-timeout",
        type=float,
        default=2.0,
        help="Per-runtime HTTP probe timeout in seconds (default: 2)",
    )
    args = p.parse_args()

    report = build_report(
        detect_hardware(),
        tools=probe_tools(),
        runtimes={
            "ollama": probe_ollama(args.ollama_url, args.probe_timeout),
            "llamacpp": probe_llamacpp(args.llamacpp_url, args.probe_timeout),
        },
        checkouts=probe_checkouts(),
        modules=probe_modules(),
    )

    if args.json:
        emit_json(report)
    else:
        print(human_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
