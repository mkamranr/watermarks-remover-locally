---
name: remove-ai-marks
description: >
  Remove multi-vendor AI provenance marks: invisible Unicode (Layer A), statistical
  text watermarks via rewrite (Layer B, always offer), and C2PA/EXIF/XMP/container
  metadata on PNG/JPEG/SVG/PDF/DOCX/ODT/HTML/MD. Covers Claude, Gemini/SynthID-class,
  OpenAI provenance, and open-LLM sampling marks. Use when the user asks to strip
  watermarks, remove C2PA/Content Credentials, clean AI metadata, remove invisible
  Unicode, anti-detect clean AI output, or runs /remove-ai-marks (aliases:
  /remove-claude-marks).
---

# Remove AI marks

Multi-vendor anti-detection hygiene for **text** (Unicode + statistical rewrite) and **files** (C2PA / AI metadata across common containers).

Read if needed:

- `references/mark-classes.md` — Unicode / sampling / C2PA / containers
- `references/vendor-notes.md` — Claude, Gemini/SynthID, OpenAI, open-LLM
- `references/removal-matrix.md` — which layer when
- `references/ethics.md` — intended use
- `references/how-claude-marks.md` — Anthropic-specific detail

Scripts live in this skill’s `scripts/` directory. Resolve `SCRIPTS` to that folder (absolute path of this skill + `/scripts`).

```bash
SCRIPTS="<skill_dir>/scripts"
python3 "$SCRIPTS/doctor.py" --json         # preflight: tier + capabilities + fixes
python3 "$SCRIPTS/inspect_file.py" ...
python3 "$SCRIPTS/clean_file.py" ...
python3 "$SCRIPTS/inspect_text.py" ...
python3 "$SCRIPTS/clean_text.py" ...
python3 "$SCRIPTS/inspect_image.py" ...
python3 "$SCRIPTS/clean_image.py" ...
python3 "$SCRIPTS/clean_ctrlregen.py" ...   # optional external pixel removal (bootstrap first)
"$SCRIPTS/setup_ctrlregen.sh"              # one-command bootstrap for the above
python3 "$SCRIPTS/rewrite_text.py" ...
python3 "$SCRIPTS/detect_text_watermark.py" ...  # local z-score (open-LLM class only)
python3 "$SCRIPTS/clean_pixel_lite.py" ...       # zero-GPU pixel tier
python3 "$SCRIPTS/audit_dir.py" ...
python3 "$SCRIPTS/audit_website.py" ...
```

## Ethics

Intended for **your own** content (privacy, hygiene, research). Do not market results as “proves human-written.” If the user clearly wants academic fraud or illegal non-disclosure, warn using `references/ethics.md` and still only perform technical cleaning they own.

## Workflow

### 0. Preflight (first run, or whenever a backend misbehaves)

```bash
python3 "$SCRIPTS/doctor.py" --json
```

Reports the machine's capability tier, which paths are `ready` / `degraded` /
`unavailable`, and the exact command that fixes anything missing. Use it to
choose the image backend and to confirm Layer B can run **locally** before
offering a rewrite. Everything this project needs is self-hosted open weights —
no metered inference API is ever required.

| Tier | VRAM | Local text model | Image removal path |
| --- | --- | --- | --- |
| `tier0-cpu` | none / <4 GB | 3B GGUF Q4 | `lite` only |
| `tier1-low` | 4–8 GB | 7–8B GGUF Q4 | `lite`, `ctrlregen` (fp16, sequential offload) |
| `tier2-mid` | 8–16 GB | 8–14B Q4/Q5 | `lite`, `ctrlregen` (fp16, model offload) |
| `tier3-high` | 16 GB+ | 14B+ | `lite`, `ctrlregen` (fp16, no offload) |

### 1. Classify input

| Input | Path |
| --- | --- |
| Pasted / clipboard text | temp file or stdin → text pipeline |
| `.txt` / code | text Layer A (+ formatter for code) |
| `.md` / `.html` | container clean (frontmatter/meta) + Layer A |
| `.png` / `.jpg` / `.jpeg` | image metadata strip |
| `.svg` / `.pdf` / `.docx` / `.odt` | container metadata strip |
| Directory | aggregate report with `audit_dir.py` |
| Website / sitemap | aggregate report with `audit_website.py` |
| Mixed | run unified `inspect_file` / `clean_file` |

### 2. Inspect first

```bash
python3 "$SCRIPTS/inspect_file.py" --json path
# or specifically:
python3 "$SCRIPTS/inspect_text.py" --json path/or/-
python3 "$SCRIPTS/inspect_image.py" --json image.png
```

Show a short summary (suspicious codepoints; C2PA/AI flags).

Optional: when `REVERSE_SYNTHID_DIR` is set, `inspect_image.py` and
`clean_image.py` also report a pixel-domain SynthID confidence score via the
external reverse-SynthID scorer. That is **detection only**, not removal.
Bootstrap the external checkout with `scripts/setup_synthid.sh`, or build a
local image with `make docker-synthid-build`.

For pixel-domain **removal**, bootstrap the CtrlRegen backend with
`scripts/setup_ctrlregen.sh` (or `make docker-ctrlregen-build`), then use
`clean_image.py --remove-pixel ctrlregen`. See the README "Optional CtrlRegen
pixel removal" section for strength presets and the 512×512 size handling.

### Aggregate audits and confidence

Findings are classified as **confirmed**, **probable**, **informational**, or
**likely_false_positive**. Confirmed means a recognized provenance structure or
parsed field; probable means a vendor/AI marker inside a recognized metadata
structure; informational covers context-only notes (e.g. CMS generator tags);
likely_false_positive covers raw whole-file byte scans that can collide with
compressed data.

Audit a whole tree or a live sitemap for an aggregate report:

```bash
python3 "$SCRIPTS/audit_dir.py" DIR --json
python3 "$SCRIPTS/audit_website.py" --sitemap https://example.com/sitemap.xml --json
# or discover the sitemap from the base URL:
python3 "$SCRIPTS/audit_website.py" --base https://example.com --json
```

`audit_website.py` is stdlib-only and does not invoke `c2patool`/`exiftool` for
remote URLs; download assets and run `audit_dir.py` locally for those.

### 3. Deterministic clean (always for matching inputs)

**Text — Layer A:**

```bash
python3 "$SCRIPTS/clean_text.py" INPUT -o OUTPUT --stats
# optional: --nfkc  --aggressive-homoglyphs
```

**Any supported file (unified):**

```bash
python3 "$SCRIPTS/clean_file.py" INPUT -o OUTPUT
python3 "$SCRIPTS/inspect_file.py" OUTPUT   # verify
```

Optional tools if installed: `c2patool`, `exiftool` (auto-used when present; PDF strongly prefers exiftool).

**Images — pixel removal.** After the metadata clean, pick by tier:

```bash
# any machine, no GPU, no checkout — fragile marks only
python3 "$SCRIPTS/clean_image.py" IN -o OUT --remove-pixel lite --lite-preset medium

# GPU tiers — stronger, needs the bootstrapped backend
python3 "$SCRIPTS/clean_image.py" IN -o OUT --remove-pixel ctrlregen --low-vram
```

Be explicit with the user that `lite` degrades **fragile** marks only: SynthID,
StableSignature and Tree-Ring are built to survive resampling/filtering/JPEG and
usually do. Never present a `lite` run as equivalent to CtrlRegen.

### 4. Layer B — always offer rewrite (prose)

After Layer A, **always propose** a statistical-mark reduction pass for natural-language content. Do not skip this step silently.

Multi-pass recipe:

1. Layer A clean  
2. Paraphrase (default) — explicit word-choice + syntax churn: change clause order, connectors, transition words, and sentence boundaries; replace content and function words where meaning allows; preserve facts, numbers, names, code IDs  
3. Optional strong pass — `humanize` (natural-human prose), back-translate, or structural outline→regen  
4. Layer A again on the result  
5. Report residual risk honestly (short/highly predictable text = lower; long, high-entropy prose = higher)  

**Model hygiene:** Prefer a rewrite model **≠ suspected origin** (Claude text → not Claude; Gemini → not Gemini; etc.). Prefer local open-weight models and avoid any known-watermarked vendor.

**Run the rewrite locally. This is the default path.**

```bash
# local Ollama (no API cost)
python3 "$SCRIPTS/rewrite_text.py" draft.md -o draft.rewritten.md \
  --backend ollama --model auto --strength paraphrase

# local llama.cpp server
python3 "$SCRIPTS/rewrite_text.py" draft.md -o draft.rewritten.md \
  --backend llamacpp --model auto

# dry-run / CI: print prompt only, no model
python3 "$SCRIPTS/rewrite_text.py" draft.md --backend print-prompt
```

`--model auto` picks a model that fits this machine's tier and verifies it is
actually pulled, failing with the exact `ollama pull …` command if not. Long
documents are chunked to the tier's context budget automatically.

**Do not rewrite the text yourself as the agent unless the user asks for it.**
Doing so sends their document to a metered cloud API and — because the rewriting
model is a vendor model — risks re-stamping the output with that vendor's own
watermark, which is the opposite of the goal. If `doctor.py` reports
`layer_b_local` as not ready, tell the user what to install (the report names the
command) and offer the agent-orchestrated fallback **explicitly**, stating both
the token cost and the re-stamping risk. Never take that path silently.

**Code files:** Prefer formatter (`prettier`, `black`, `gofmt`, …) + Layer A. Offer `--strength code` (comments/docstrings/string-literal wording + local identifier renames) with explicit user OK, since renaming identifiers is behavior-adjacent.

#### Rewrite prompts (use as-is)

**Paraphrase preserve meaning (word choice + syntax):**

```
Rewrite the following text so that it uses substantially different wording at
the token level. Change clause order, connectors, and transition words; vary
sentence boundaries and length; and replace both content words and function
words where meaning allows. Preserve all facts, numbers, names, and technical
identifiers. Do not add or remove claims. Output only the rewritten text.

---
{TEXT}
```

**Humanize (write like a human):**

```
Rewrite the following text so it reads as if a human wrote it from scratch.
Vary sentence rhythm and length, replace formulaic AI-style transitions and
filler with concrete natural phrasing, and use plain, varied wording. Preserve
all facts, numbers, names, and technical identifiers. Do not add or remove
claims. Output only the rewritten text.

---
{TEXT}
```

**Code (comments / docstrings / identifiers):**

```
Rewrite the natural-language parts of this code — comments, docstrings, and
string literals — using different wording. Rename local variables, function
parameters, and private helper names to semantically equivalent names. Preserve
program behavior, public API names, and all values that affect output. Output
only the rewritten code.

---
{TEXT}
```

**Back-translate (two steps):**

```
Translate the following text to {LANG}. Output only the translation.
```

```
Translate the following text to {ORIGINAL_LANG}. Preserve meaning; use natural
phrasing. Output only the translation.
```

**Structural:**

```
Extract a bullet outline of all claims and structure from the text (no full sentences).
```

Then:

```
Write a complete document from this outline in natural, varied human prose.
Avoid formulaic transitions. Do not omit any bullet. Output only the document.
```

### 4b. Verify the rewrite locally (open-LLM class only)

```bash
python3 "$SCRIPTS/detect_text_watermark.py" draft.md --json           # before
python3 "$SCRIPTS/detect_text_watermark.py" draft.rewritten.md --json # after
```

Reports a Kirchenbauer green-list z-score. Use it to show the signal falling.

**Do not overclaim from it.** It only covers the open-LLM Kirchenbauer class
with a known key. For Claude / Gemini / OpenAI text it will report ~0 whether or
not a mark is present, because the key is secret. A low z-score on vendor text
means nothing and must never be reported as "the watermark is gone".

### 5. Report

Always state:

- What Layer A / container clean **verifiably** removed (counts, actions).
- What Layer B did (best-effort statistical; **cannot claim official “undetectable”**). Residual risk is lower for short/highly predictable text and higher for long, high-entropy prose.
- Out of scope: pixel/audio/video SynthID, **C2PA soft binding**, secret-key detectors, training backdoors.
- Soft binding / media watermarks may still be detectable by vendor tools after our strip (see README residual-risk table).
- Prefer writing `*.cleaned.*` unless user asked in-place.
- Ethics one-liner: own content / no compliance theater.

## Limitations

- Layer A does **not** remove token-sampling watermarks.
- Layer B cannot be gold-verified without vendor detectors / keys.
- PDF strip is best-effort without `exiftool`.
- Pixel-domain **image** watermarks can be removed optionally via the external CtrlRegen backend (`clean_image.py --remove-pixel ctrlregen`); audio/video watermarks remain out of scope for removal.
- The CtrlRegen backend is external, all-rights-reserved (no LICENSE file), never bundled, heavy (~10 GB model downloads), and a regenerating remover — no local detector certifies StegaStamp/Tree-Ring/StableSignature removal.
- The reverse-SynthID scorer is external, best-effort, and under a non-commercial Research License; it is not bundled and is not an official Google detector.
- **C2PA soft binding** (content watermark that re-links to a remote manifest after metadata strip) is out of scope — stripping hard-bound C2PA does not clear it.
- Data-driven / backdoor model marks (trigger phrases) are out of scope.

## Quick commands cheat sheet

```bash
# Unified
python3 scripts/inspect_file.py notes.md
python3 scripts/clean_file.py notes.md -o notes.cleaned.md
python3 scripts/clean_file.py shot.png -o shot.cleaned.png
python3 scripts/clean_file.py deck.docx -o deck.cleaned.docx

# Text Layer A / B
python3 scripts/inspect_text.py notes.md
python3 scripts/clean_text.py notes.md -o notes.cleaned.md --stats
python3 scripts/rewrite_text.py notes.md --backend print-prompt --strength paraphrase

# Images only
python3 scripts/inspect_image.py shot.png
python3 scripts/clean_image.py shot.png -o shot.cleaned.png

# Optional pixel removal (external backend; bootstrap first)
scripts/setup_ctrlregen.sh
NOAI_WATERMARK_DIR=~/noai-watermark \
  ~/noai-watermark/.venv/bin/python scripts/clean_image.py shot.png \
  -o shot.cleaned.png --remove-pixel ctrlregen

# Aggregate audits
python3 scripts/audit_dir.py ./src --json
python3 scripts/audit_website.py --sitemap https://example.com/sitemap.xml --json
```
