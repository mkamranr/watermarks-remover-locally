# Removal matrix

| Target | Method | Script / action | Side effects | Verifiable today? |
| --- | --- | --- | --- | --- |
| Invisible Unicode / exotic spaces / bidi / tags | Strip / normalize | `inspect_text.py`, `clean_text.py`, `clean_file.py` | Minimal | Yes (codepoint report) |
| Statistical text watermark (SynthID-class / Kirchenbauer) | Multi-pass paraphrase / humanize / back-translate / structural | `rewrite_text.py` against a **local** model (Ollama / llama.cpp) | Meaning/style drift | Open-LLM Kirchenbauer class: **yes**, locally via `detect_text_watermark.py`. Vendor marks: no without key/detector |
| C2PA on PNG/JPEG | Drop APP11 / text chunks / exiftool | `clean_image.py` | Loses provenance metadata | Yes |
| SVG metadata / XMP | Drop `<metadata>`, xmpmeta | `clean_file.py` | Loses SVG metadata | Yes (re-inspect) |
| PDF XMP / info | exiftool `-all=` preferred | `clean_file.py` | Loses PDF metadata; degraded without exiftool | Partial |
| DOCX props / customXml | Rewrite OOXML zip | `clean_file.py` | Loses doc properties | Yes |
| ODT meta:generator | Scrub `meta.xml` | `clean_file.py` | Loses generator tag | Yes |
| HTML generator / JSON-LD provenance | Strip tags | `clean_file.py` | Loses meta | Yes |
| Markdown AI frontmatter keys | Drop keys | `clean_file.py` | Loses YAML keys | Yes |
| Pixel image watermark — fragile marks | Signal-processing attack (resample, blur+resharpen, median, DWT noise, JPEG cycles) | `clean_pixel_lite.py` / `clean_image.py --remove-pixel lite` | Mild quality loss (PSNR-guarded); **robust marks survive** | No; reverse-SynthID score is the only local surrogate |
| Pixel image watermark (SynthID-media / StegaStamp / Tree-Ring / StableSignature) | CtrlRegen regeneration (external backend) | `clean_ctrlregen.py` / `clean_image.py --remove-pixel ctrlregen` | Regenerates pixels; heavy compute; detail drift at higher strength | No without official detector; reverse-SynthID score is a local surrogate |
| Audio / video watermarks (SynthID-media) | — | Out of scope | — | — |
| C2PA soft binding (in-content link to manifest) | — | Out of scope (survives our metadata strip) | — | Vendor detector only |
| Data-driven model backdoors | — | Out of scope | — | — |

## Default pipeline

0. **Preflight** (`doctor.py --json`) — capability tier, what is ready, what to install.
1. **Inspect** (`inspect_file.py` or specific inspect_*).
2. **Deterministic clean** — Layer A text and/or container/image metadata; for images, optionally add pixel removal (`--remove-pixel lite` on any machine, `--remove-pixel ctrlregen` with a GPU) after the metadata strip.
3. **Always offer Layer B** rewrite for prose (paraphrase → optional strong pass: `humanize` / back-translate / structural), run against a **local** model.
4. Prefer a **non-origin, open-weight** rewrite model (avoid re-stamping). `--origin <vendor>` enforces this when picking `--model auto`.
5. Layer A again after rewrite.
6. Report: Layer B is best-effort; residual risk remains.

## How much rewriting a statistical mark actually needs

Measured with `detect_text_watermark.py` on a synthetic Kirchenbauer-marked
sequence (2000 tokens, gamma 0.25, detection threshold z=4):

| Fraction of tokens rewritten | z-score | Still detected? |
| --- | --- | --- |
| 0% (original) | 77.4 | Yes |
| 25% | 43.0 | Yes |
| 50% | 20.2 | Yes |
| 75% | 4.1 | Yes (marginal) |
| 100% | −1.6 | No |

This is the quantitative version of the README's disclaimer: the signal is
spread across nearly every token, so light editing does almost nothing. A pass
that leaves half the wording intact still scores as firmly watermarked.

The same text scored with the **wrong key** gives z = 0.7 (undetected) — which
is exactly why a vendor's secret-key mark cannot be checked locally, and why a
low local score is never evidence that a vendor detector would fail.

## Code vs prose

- **Prose / Markdown / HTML body:** full A + B.
- **Code:** Layer A + formatter; statistical marks are weak; offer `code` rewrite (comments/docstrings/string-literal wording + local identifier renames) with user OK.

## Layer B strengths

| Strength | When |
| --- | --- |
| `paraphrase` | Default; explicit word-choice + syntax churn |
| `humanize` | Zero-shot "write like a human" token reshuffle |
| `backtranslate` | Stronger token reshuffle via pivot language |
| `structural` | Strongest; most drift (outline → human prose) |
| `code` | Comments/docstrings/string-literal wording + local identifier renames |

Frontier production watermarks are currently **token-by-token** (streaming
constraint); paragraph-level robust methods (SemStamp / PostMark) are not yet
deployed, so paraphrase-class attacks remain effective today.
