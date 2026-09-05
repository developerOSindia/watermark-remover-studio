# Watermark Studio · Gemini, Veo, NotebookLM & SynthID Cleaner

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://watermark-remover-studio.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C2PA Sanitizer](https://img.shields.io/badge/C2PA-Sanitizer-emerald.svg)](https://c2pa.org/)
[![SynthID Disrupter](https://img.shields.io/badge/SynthID-Disrupted-orange.svg)](https://deepmind.google/technologies/synthid/)
[![NotebookLM Cleaner](https://img.shields.io/badge/NotebookLM-Deck%20%26%20Video%20Cleaner-9333ea.svg)](https://github.com/developerOSindia/watermark-remover-studio)

> **Free, air-gapped, and open-source studio for eradicating Google Gemini and Google Veo visible watermarks, scrubbing NotebookLM slide deck logos from PDFs and presentation slides, removing Gemini Notebook logos from videos, disrupting DeepMind SynthID latent watermarks, and stripping C2PA Content Credentials from photos and videos with lossless audio preservation.**

---

## Key Highlights

Most generic watermark tools rely on heavy neural diffusion inpainting that smears background artwork, leaves visible blur halos, strips video audio tracks, and ignores imperceptible AI provenance tracking.

**Watermark Studio** solves this with a cleanroom, mathematically calibrated pipeline:

1. **Zero-Artifact Watermark Eradication**:
   - **Discrete Resolution Catalogs:** Exact canonical pixel priors for Google Gemini images (0.5k, 1k, 2k, 4k) and Google Veo videos (1080p, 720p portrait and landscape).
   - **Veo Text Logo Templates:** Multi-scale binary template matching for rectangular `"Veo"` marks (23x10, 68x30, 99x43).
   - **Safety Dilation Inpainting:** Morphological `(7, 7)` dilation plus a 6px safety perimeter completely eliminates watermark star tips and anti-aliasing glow without edge bleeding.

2. **NotebookLM Presentation & Video Cleaner (PDFs, Slides & Video)**:
   - **Calibrated Geometry Engine:** Precision bottom-right detection targeting NotebookLM exported slides and portrait overview videos (`widthRatio: 0.0825` for landscape slides, `0.28` for portrait videos, with configurable scan regions).
   - **Boundary-Feathered Gradient Patching:** Mathematical source patch sampling with smooth gradient blending — completely eliminates blurring and preserves crisp slide borders, cards, and solid gradients.
   - **Triple Execution Workbenches:** In-browser client-side presentation cleaner powered by `pdf.js` and `jspdf` for zero-upload multi-page PDF processing, alongside a forensic Python laboratory for slide images and audio overview MP4 videos.

3. **Google DeepMind SynthID Disruption & Phase Scrambling**:
   - SynthID embeds imperceptible pseudo-random frequency perturbations into generated pixels.
   - **Nuclear Mode:** Applies 0.997 asymmetric Lanczos rescaling, 2px border uncoupling crop, subtle channel bias, and spatial-frequency micro-dithering to de-synchronize the sub-LSB phase trees required by SynthID verification models while maintaining visually pristine fidelity.

4. **Lossless C2PA & Provenance Sanitization**:
   - Low-level byte chunk walkers for PNG (`iter_png_chunks`) and JPEG (`iter_jpeg_segments`).
   - Audits and scrubs 12 container fingerprint categories: C2PA manifests (`caBX` chunks, APP11 JUMBF boxes), AI prompts (Stable Diffusion Automatic1111, ComfyUI node graphs, Midjourney flags, DALL-E/OpenAI tags), IPTC `DigitalSourceType = trainedAlgorithmicMedia`, camera EXIF/GPS, Photoshop IRB blocks, and trailing payloads.
   - **Safe Mode:** 100% lossless container sanitation — input and output pixels are **bit-identical** (pixel delta = 0, verified by SHA-256).

5. **Lossless Audio Passthrough**:
   - Uses FFmpeg stream copying (`-c:v libx264 -c:a copy`) to preserve original AAC, MP3, and PCM audio tracks without re-encoding, volume loss, or sync drift. Tested and verified on NotebookLM overview videos like `How_to_Scale_PSI_for_Data_Drift.mp4`.

6. **100% Air-Gapped & Private**:
   - Zero telemetry, zero external APIs, zero database storage. All processing occurs in temporary, ephemeral memory on your local machine.

---

## Cleaning & Sanitization Tiers

| Tier | Name | Target Vectors | Visual Impact | Pixel Guarantee |
| :--- | :--- | :--- | :--- | :--- |
| **`none`** | Off | Visible watermark only | Imperceptible | Inpainting restricted strictly to logo bounding box |
| **`safe`** | Lossless C2PA & Metadata Strip | C2PA manifests, AI prompts (A1111, ComfyUI, MJ), EXIF, GPS | None | **100% Bit-Identical Pixels** ($\Delta = 0$) |
| **`paranoid`** | Quantization & Sensor Reset | `safe` + JPEG quantization table fingerprints, camera PRNU | Imperceptible | Standardized sRGB re-encode + Gaussian micro-dither ($\sigma=0.5$) |
| **`nuclear`** | SynthID & Latent Frequency Disruption | `paranoid` + Google DeepMind SynthID sub-LSB phase trees | Imperceptible | 0.997 Lanczos scale + 2px border uncoupling crop + spatial frequency dither |

---

## Technical Comparison Matrix

| Capability | Generic Online Removers | Reference Catalogs | Watermark Studio |
| :--- | :--- | :--- | :--- |
| **Watermark Matching** | Continuous float heuristics | Discrete size catalogs | Discrete 0.5k-4k catalogs + continuous fallbacks |
| **Veo Video Presets** | Static fixed offsets | Discrete video catalog | Veo Inset (144), Standard (108), Compact (40), Corner, Text Logo |
| **Veo Text Watermark** | Unsupported | Binary base64 templates | Multi-scale template matching (23x10, 68x30, 99x43) |
| **NotebookLM Slide & Video** | Unsupported | External third-party scripts | Triple engine: In-browser PDF cleaner + Python image patcher + Video cleaner |
| **Star Tip Clipping** | Halo / smudged tips remain | Alpha unblending | Morphological `(7, 7)` dilation + 6px safety padding (100% clean) |
| **Audio Pipeline** | Stripped or lossy re-encode | OpenCV standard | Lossless FFmpeg stream copying (`-c:a copy`) |
| **Google SynthID Disruption** | Not supported | Concept only | 0.997 Lanczos phase scrambling + frequency dithering |
| **C2PA / Provenance Stripping** | Incomplete / Re-compressed | Chunk walker | Low-level byte parser + 12 finding categories + Bit-identical Safe tier |
| **Privacy & Telemetry** | Cloud upload / data retention | Local script | 100% local, ephemeral RAM execution, zero tracking |

---

## Quick Start (Local Run)

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/developerOSindia/watermark-remover-studio.git
cd watermark-remover-studio/developeros-watermark-studio

# Install Python requirements
python3 -m pip install -r requirements.txt
```

### 2. Launch Streamlit Studio

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

Use the **Workspace Selector** in the sidebar to toggle between:
* **`✦ Gemini & Veo (Images & Video)`**: Eradicate 4-pointed stars, Veo text badges, disrupt SynthID, and sanitize C2PA manifests with lossless audio passthrough.
* **`📑 NotebookLM Studio (PDFs, Slides & Video)`**: Clean entire multi-page PDF slide decks directly in your browser with zero upload, test slide images, or clean full NotebookLM audio overview videos with lossless audio.

> [!TIP]
> The studio uses system `ffmpeg` when available and falls back to bundled `imageio-ffmpeg`. Verify FFmpeg is active with `ffmpeg -version`.

---

## Command-Line Interface (CLI)

The underlying processing engine (`remove_watermark.py`) can be executed directly from your terminal or integrated into automated shell pipelines.

### 1. Basic Watermark Removal

```bash
# Clean an image
python3 remove_watermark.py input.png output.png

# Clean a video with lossless audio preservation
python3 remove_watermark.py input.mp4 output.mp4
```

### 2. Deep Container Inspection (`--inspect`)

Audit any image for C2PA manifests, AI generation prompts, and EXIF leaks:

```bash
python3 remove_watermark.py sample.png --inspect
```

Example Output:
```text
======================================================================
[INSPECTION REPORT]
File: sample.png (Format: PNG)
Findings: 3
  • [exif] EXIF metadata segment present (size: 142 bytes)
  • [software] Creation tool signature detected: Software
  • [date_time] Timestamp tag present: DateTime
Clean status: UNCLEAN
======================================================================
```

### 3. Google SynthID Disruption & Metadata Scrubbing (`--synthid-mode`)

```bash
# Safe mode: Bit-identical pixels with C2PA and prompts stripped
python3 remove_watermark.py input.png cleaned_safe.png --synthid-mode safe

# Nuclear mode: Disrupt DeepMind SynthID latent watermarks
python3 remove_watermark.py input.png cleaned_nuclear.png --synthid-mode nuclear --inspect
```

### 4. Advanced Video Options

```bash
python3 remove_watermark.py input.mp4 cleaned.mp4 \
    --preset veo_inset \
    --method inpaint \
    --gain 0.60 \
    --size-scale 1.05
```

**CLI Flags Reference:**
**Gemini & Veo CLI Flags Reference:**
* `--method {inpaint, reconstruct, math}`: Inpainting algorithm (default: `inpaint`).
* `--preset {auto, veo_inset, veo_standard, veo_compact, corner, veo_text}`: Watermark geometry preset.
* `--synthid-mode {none, safe, paranoid, nuclear}`: Provenance & invisible watermark cleaning tier.
* `--strip-metadata`: Enable lossless metadata stripping.
* `--inspect`: Display detailed JSON/table container audit report.
* `--gain FLOAT`: Watermark opacity sensitivity (default: `0.60`).
* `--size-scale FLOAT`: Bounding box dilation factor (default: `1.00`).

### 5. NotebookLM Slide & Presentation Deck Cleaning (`notebooklm_cleaner.py`)

Scrub NotebookLM logos from exported slides, slide captures, and presentation decks:

```bash
# Clean a single presentation slide image using boundary-feathered gradient patching
python3 notebooklm_cleaner.py slide.png cleaned_slide.png

# Clean using OpenCV Telea inpainting on complex or textured presentation backgrounds
python3 notebooklm_cleaner.py slide.png cleaned_slide.png --method inpaint

# Batch process an entire directory of presentation slides
python3 notebooklm_cleaner.py ./exported_slides/ ./cleaned_deck/ --method patch

# Fine-tune the bottom-right corner scan region
python3 notebooklm_cleaner.py slide.png cleaned_slide.png --scan-w 0.22 --scan-h 0.08
```

**NotebookLM Flags Reference:**
* `input`: Path to input image file or directory containing slides.
* `output`: Optional output image path or output directory (defaults to `_cleaned` suffix).
* `--method {patch, inpaint, both}`: Removal technique (`patch` for smooth gradients, `inpaint` for textures, `both` to generate comparison outputs).
* `--scan-w FLOAT`: Bottom-right width scan ratio (default: `0.22`).
* `--scan-h FLOAT`: Bottom-right height scan ratio (default: `0.08`).

---

## Python API Integration

Integrate Watermark Studio directly into your Python scripts or data pipelines:

```python
from PIL import Image
from remove_watermark import remove_watermark, remove_watermark_from_video
from fingerprint_cleaner import inspect_image_fingerprints, clean_image_fingerprints
from notebooklm_cleaner import clean_notebooklm_image

# 1. Audit image for C2PA and AI provenance
with open("sample.png", "rb") as f:
    report = inspect_image_fingerprints(f.read(), "sample.png")
print(f"Detected {report['finding_count']} metadata items. Has C2PA: {report['has_c2pa']}")

# 2. Remove visible watermark and disrupt SynthID in one call
image = Image.open("sample.png")
cleaned = remove_watermark(
    image,
    gain=0.60,
    size_scale=1.00,
    method="inpaint",
    synthid_mode="nuclear",  # Disrupts SynthID frequency trees
)
cleaned.save("sample_sanitized.png")

# 3. Clean video with lossless audio passthrough
remove_watermark_from_video(
    "input.mp4",
    "output.mp4",
    preset="veo_inset",
    method="inpaint",
)

# 4. Clean NotebookLM presentation slide with mathematical gradient patch
slide = Image.open("notebooklm_slide.png")
clean_slide = clean_notebooklm_image(slide, method="gradient_patch")
clean_slide.save("cleaned_slide.png")
```

---

## Project Structure

```text
developeros-watermark-studio/
├── app.py                         Streamlit Studio web interface with Workspace switcher
├── remove_watermark.py            Core Gemini & Veo removal pipeline + CLI
├── fingerprint_cleaner.py         SynthID disrupter & 12-point C2PA container walker
├── notebooklm_cleaner.py          NotebookLM bottom-right slide & presentation cleaner CLI
├── DESIGN.md                      Semantic Design System (Darkroom Workbench)
├── requirements.txt               Python dependencies
├── packages.txt                   Streamlit cloud debian packages (ffmpeg)
├── assets/
│   ├── bg_48.png                  Small watermark mask template
│   └── bg_96.png                  Large watermark mask template
└── static/
    ├── robots.txt                 Crawler directives (GPTBot, PerplexityBot, ClaudeBot)
    ├── sitemap.xml                Search engine sitemap
    └── notebooklm_embedded.html   In-browser client-side NotebookLM Studio app (pdf.js + jspdf)
```

---

## Privacy & Security

Watermark Studio is engineered with strict privacy principles:
* **No Telemetry or Tracking:** Zero Google Analytics, cookies, or logging beacons.
* **Air-Gapped Processing:** No media files are sent to external cloud APIs or third-party servers.
* **Ephemeral Storage:** Processing executes in RAM and temporary system files which are deleted immediately upon session termination.

---

## License & Credits

MIT License. Built with precision by [DeveloperOS](https://github.com/developerOSindia).


