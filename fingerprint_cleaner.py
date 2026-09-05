"""Cleanroom — Image Fingerprint, Provenance, and SynthID Disruption Engine.

Ported and adapted from reference libraries (gemini-watermark-and-synthid-remover & image-fingerprint-remover).
Provides container-level chunk/marker inspection and three-tier sanitization:
  - safe: Lossless metadata & C2PA manifest strip (100% bit-identical pixel bytes)
  - paranoid: Metadata strip + stock sRGB re-encode + Gaussian micro-dither (sigma=0.5)
  - nuclear: Robust spatial-frequency disruption (0.997 scale, 2px crop, channel bias, dither)
    specifically targeting Google DeepMind SynthID and invisible latent watermarks.
"""

from __future__ import annotations

import io
import re
import struct
import zlib
from typing import Iterator

import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# Binary Headers & Chunk Signatures
# -----------------------------------------------------------------------------
PNG_HEADER = b"\x89PNG\r\n\x1a\n"
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"

PNG_CRITICAL_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}
PNG_SAFE_ANCILLARY = {
    b"tRNS",  # transparency
    b"gAMA",  # gamma
    b"cHRM",  # chromaticities
    b"sBIT",  # significant bits
    b"bKGD",  # background color
    b"hIST",  # histogram
    b"sRGB",  # standard sRGB rendering intent (1 byte)
    b"acTL", b"fcTL", b"fdAT",  # APNG animation frames
}

# C2PA / Content Credentials JUMBF box magic
JUMBF_MAGIC = b"jumb"
C2PA_LABELS = (b"c2pa", b"c2ma", b"c2as", b"c2cl")

# Known AI-generation text keys inside PNG tEXt / zTXt / iTXt
PNG_AI_TEXT_KEYS = {
    "parameters": ("Stable Diffusion (A1111)", "Stores full prompt, negative prompt, seed, model hash, sampler."),
    "prompt": ("ComfyUI / AI Prompt", "ComfyUI or API generation prompt JSON."),
    "workflow": ("ComfyUI Workflow", "Full ComfyUI node graph JSON."),
    "comfy": ("ComfyUI", "Marker key written by ComfyUI exporters."),
    "sd-metadata": ("InvokeAI", "InvokeAI generation metadata block."),
    "invokeai": ("InvokeAI", "InvokeAI generation marker."),
    "novelai": ("NovelAI", "NovelAI generation marker."),
    "dream": ("InvokeAI Dream", "InvokeAI dream parameters."),
    "openai": ("OpenAI / ChatGPT", "OpenAI image-generation marker."),
    "dall-e": ("DALL-E", "DALL-E generation marker."),
    "midjourney": ("Midjourney", "Midjourney prompt and version parameters."),
    "firefly": ("Adobe Firefly", "Adobe Firefly AI provenance marker."),
}

PNG_AI_VALUE_PATTERNS = [
    (re.compile(r"\bSteps:\s*\d+", re.I), "Stable Diffusion A1111 parameter block"),
    (re.compile(r"\bSampler:\s*[A-Za-z]", re.I), "Stable Diffusion sampler block"),
    (re.compile(r"\bModel hash:\s*[0-9a-f]{6,}", re.I), "Stable Diffusion model hash"),
    (re.compile(r"\bCFG scale:\s*[\d.]+", re.I), "Stable Diffusion CFG scale"),
    (re.compile(r'"class_type"\s*:', re.I), "ComfyUI node graph JSON"),
    (re.compile(r"midjourney|--ar\s+\d+:\d+|--v\s+\d", re.I), "Midjourney prompt block"),
    (re.compile(r"dall[\s\-]?e|gpt[\s\-]?image|openai", re.I), "OpenAI/DALL-E marker"),
    (re.compile(r"stable[\s\-]?diffusion|automatic1111|a1111", re.I), "Stable Diffusion marker"),
    (re.compile(r"firefly|adobe stock", re.I), "Adobe Firefly marker"),
    (re.compile(r"trainedalgorithmicmedia|compositewithtrainedalgorithmicmedia", re.I), "IPTC DigitalSourceType = AI-generated"),
]

XMP_AI_PATTERNS = [
    (re.compile(r"Iptc4xmpExt:DigitalSourceType[^<]*trainedAlgorithmicMedia", re.I), "IPTC DigitalSourceType = trainedAlgorithmicMedia"),
    (re.compile(r"Iptc4xmpExt:DigitalSourceType[^<]*compositeWithTrainedAlgorithmicMedia", re.I), "IPTC DigitalSourceType = compositeWithTrainedAlgorithmicMedia"),
    (re.compile(r"<xmpMM:History>.*?</xmpMM:History>", re.S | re.I), "Adobe XMP edit history (leaks editor actions and file paths)"),
    (re.compile(r"firefly|adobestock", re.I), "Adobe Firefly/Stock marker"),
    (re.compile(r"openai|chatgpt|dall[\s\-]?e", re.I), "OpenAI/ChatGPT provenance marker"),
    (re.compile(r"midjourney", re.I), "Midjourney marker"),
    (re.compile(r"stable[\s\-]?diffusion|stability\.ai", re.I), "Stable Diffusion marker"),
    (re.compile(r"generativeAI|generative-ai", re.I), "Generic generative-AI marker"),
]

EXIF_LEAK_TAGS = {
    "Make", "Model", "Software", "BodySerialNumber", "SerialNumber",
    "LensSerialNumber", "OwnerName", "Artist", "Copyright", "UserComment",
    "ImageUniqueID", "DateTimeOriginal", "DateTimeDigitized", "ImageDescription",
    "HostComputer", "CameraOwnerName",
}


# -----------------------------------------------------------------------------
# Binary Parsers (PNG Chunks & JPEG Segments)
# -----------------------------------------------------------------------------
def iter_png_chunks(data: bytes) -> Iterator[tuple[int, bytes, bytes]]:
    """Yield (offset, type_bytes, body_bytes) for each PNG chunk."""
    if not data.startswith(PNG_HEADER):
        return
    off = len(PNG_HEADER)
    n = len(data)
    while off + 8 <= n:
        (length,) = struct.unpack(">I", data[off : off + 4])
        ctype = data[off + 4 : off + 8]
        body = data[off + 8 : off + 8 + length]
        yield off, ctype, body
        off += 12 + length
        if ctype == b"IEND":
            break


def iter_jpeg_segments(data: bytes) -> Iterator[tuple[int, int, bytes]]:
    """Yield (offset, marker_byte, body_bytes) for each JPEG segment before SOS."""
    if len(data) < 2 or data[0] != 0xFF or data[1] != 0xD8:
        return
    off = 2
    n = len(data)
    while off < n:
        if data[off] != 0xFF:
            break
        while off < n and data[off] == 0xFF:
            off += 1
        if off >= n:
            break
        marker = data[off]
        off += 1
        # Standalone markers with no payload
        if marker in (0xD8, 0xD9, 0x00) or (0xD0 <= marker <= 0xD7):
            yield off - 2, marker, b""
            continue
        if marker == 0xDA:  # Start of Scan (SOS)
            if off + 2 <= n:
                (length,) = struct.unpack(">H", data[off : off + 2])
                body = data[off + 2 : off + length]
                yield off - 2, marker, body
            else:
                yield off - 2, marker, b""
            break
        if off + 2 > n:
            break
        (length,) = struct.unpack(">H", data[off : off + 2])
        if length < 2 or off + length > n:
            break
        body = data[off + 2 : off + length]
        yield off - 2, marker, body
        off += length


def _crc(chunk_type: bytes, body: bytes) -> bytes:
    return struct.pack(">I", zlib.crc32(chunk_type + body) & 0xFFFFFFFF)


def _encode_png_chunk(chunk_type: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + chunk_type + body + _crc(chunk_type, body)


# -----------------------------------------------------------------------------
# Deep Container Inspection
# -----------------------------------------------------------------------------
def inspect_image_fingerprints(data: bytes, filename: str = "") -> dict:
    """Inspect an image container for C2PA manifests, AI prompts, EXIF, and tracking metadata."""
    findings: list[dict] = []
    fmt = "UNKNOWN"

    if data.startswith(PNG_HEADER):
        fmt = "PNG"
        last_end = len(PNG_HEADER)
        saw_iend = False

        for off, ctype, body in iter_png_chunks(data):
            last_end = off + 12 + len(body)
            ctype_s = ctype.decode("latin1", "replace")

            # C2PA Manifest chunk (caBX)
            if ctype == b"caBX" or (JUMBF_MAGIC in body and any(lbl in body for lbl in C2PA_LABELS)):
                findings.append({
                    "category": "c2pa_manifest",
                    "severity": "critical",
                    "name": "C2PA / Content Credentials Manifest",
                    "detail": f"Signed provenance manifest detected in PNG chunk '{ctype_s}'. Links image to generator/service.",
                    "location": f"PNG {ctype_s} @ {off}",
                    "size_bytes": len(body),
                    "value_preview": "C2PA JUMBF Manifest Block",
                })
                continue

            if ctype in PNG_CRITICAL_CHUNKS or ctype in PNG_SAFE_ANCILLARY:
                if ctype == b"IEND":
                    saw_iend = True
                continue

            # Embedded EXIF chunk (eXIf)
            if ctype == b"eXIf":
                findings.append({
                    "category": "exif",
                    "severity": "high",
                    "name": "Embedded EXIF Chunk",
                    "detail": "PNG carries raw EXIF metadata (may include camera model, timestamps, software, and serials).",
                    "location": f"PNG eXIf @ {off}",
                    "size_bytes": len(body),
                    "value_preview": "Raw EXIF block",
                })
                continue

            # Text Chunks (tEXt, zTXt, iTXt)
            if ctype in (b"tEXt", b"zTXt", b"iTXt"):
                key, val = _decode_png_text(body, ctype)
                label, reason = _match_ai_signature(key, val)
                severity = "high" if label else "medium"
                cat = "ai_prompt" if label else "png_text"
                findings.append({
                    "category": cat,
                    "severity": severity,
                    "name": label or f"PNG Text Key: {key}",
                    "detail": f"PNG text key '{key}' ({len(val)} chars)" + (f" — {reason}" if reason else ""),
                    "location": f"PNG {ctype_s} @ {off}",
                    "size_bytes": len(body),
                    "value_preview": val[:180].replace("\n", " "),
                })
                continue

            # Ancillary/Unknown chunk
            findings.append({
                "category": "png_chunk_unknown",
                "severity": "low",
                "name": f"Non-standard PNG Chunk ({ctype_s})",
                "detail": f"Custom ancillary chunk '{ctype_s}' ({len(body)} bytes).",
                "location": f"PNG {ctype_s} @ {off}",
                "size_bytes": len(body),
                "value_preview": f"{len(body)} bytes payload",
            })

        if saw_iend and len(data) > last_end:
            trailing = len(data) - last_end
            findings.append({
                "category": "trailing_bytes",
                "severity": "high",
                "name": "Appended Trailing Data",
                "detail": f"Found {trailing} bytes appended after the IEND chunk (potential steganographic payload).",
                "location": f"After IEND @ {last_end}",
                "size_bytes": trailing,
                "value_preview": f"{trailing} extra bytes",
            })

    elif data.startswith(JPEG_SOI):
        fmt = "JPEG"
        for off, marker, body in iter_jpeg_segments(data):
            # APP11: C2PA / JUMBF segment
            if marker == 0xEB:
                has_c2pa = JUMBF_MAGIC in body or any(lbl in body for lbl in C2PA_LABELS)
                findings.append({
                    "category": "c2pa_manifest" if has_c2pa else "jpeg_app_segment",
                    "severity": "critical" if has_c2pa else "medium",
                    "name": "C2PA / Content Credentials (APP11)" if has_c2pa else "JPEG APP11 Segment",
                    "detail": "Signed C2PA provenance manifest in JPEG APP11." if has_c2pa else f"JPEG APP11 segment ({len(body)} bytes).",
                    "location": f"JPEG APP11 @ {off}",
                    "size_bytes": len(body),
                    "value_preview": "C2PA JUMBF Manifest" if has_c2pa else "APP11 Data",
                })
            # APP1: EXIF / XMP
            elif marker == 0xE1:
                is_exif = body.startswith(b"Exif\x00\x00")
                is_xmp = body.startswith(b"http://ns.adobe.com/xap/1.0/\x00")
                if is_xmp:
                    xmp_str = body.decode("utf-8", "replace")
                    ai_flag = any(pat.search(xmp_str) for pat, _ in XMP_AI_PATTERNS)
                    findings.append({
                        "category": "ai_tag" if ai_flag else "xmp",
                        "severity": "high" if ai_flag else "medium",
                        "name": "XMP AI Provenance Metadata" if ai_flag else "XMP Metadata Block",
                        "detail": "XMP data tags image as AI-generated (IPTC/Adobe/OpenAI)." if ai_flag else "Standard XMP packet.",
                        "location": f"JPEG APP1 (XMP) @ {off}",
                        "size_bytes": len(body),
                        "value_preview": xmp_str[:180].replace("\n", " "),
                    })
                elif is_exif:
                    findings.append({
                        "category": "exif",
                        "severity": "high",
                        "name": "EXIF Metadata Segment",
                        "detail": "JPEG APP1 carries EXIF tags (camera hardware, timestamps, serials).",
                        "location": f"JPEG APP1 (EXIF) @ {off}",
                        "size_bytes": len(body),
                        "value_preview": "Raw EXIF segment",
                    })
                else:
                    findings.append({
                        "category": "jpeg_app_segment",
                        "severity": "medium",
                        "name": "JPEG APP1 Segment",
                        "detail": f"APP1 segment ({len(body)} bytes).",
                        "location": f"JPEG APP1 @ {off}",
                        "size_bytes": len(body),
                        "value_preview": body[:80].decode("latin1", "replace"),
                    })
            # APP13: Photoshop IRB
            elif marker == 0xED:
                findings.append({
                    "category": "photoshop",
                    "severity": "medium",
                    "name": "Photoshop IRB / 8BIM Block",
                    "detail": "Carries Photoshop document history, slices, and original filenames.",
                    "location": f"JPEG APP13 @ {off}",
                    "size_bytes": len(body),
                    "value_preview": "Photoshop 8BIM segment",
                })
            # COM: Comments
            elif marker == 0xFE:
                cmt = body.decode("latin1", "replace")
                findings.append({
                    "category": "jpeg_comment",
                    "severity": "low",
                    "name": "JPEG Comment Segment",
                    "detail": f"Plaintext comment: '{cmt[:60]}'",
                    "location": f"JPEG COM @ {off}",
                    "size_bytes": len(body),
                    "value_preview": cmt[:120],
                })

    # High-level category flags
    has_c2pa = any(f["category"] == "c2pa_manifest" for f in findings)
    has_ai_prompt = any(f["category"] in ("ai_prompt", "ai_tag") for f in findings)
    has_exif = any(f["category"] == "exif" for f in findings)
    has_gps = any("gps" in f["detail"].lower() or "gps" in f["name"].lower() for f in findings)

    return {
        "format": fmt,
        "filename": filename,
        "file_size": len(data),
        "is_clean": len(findings) == 0,
        "finding_count": len(findings),
        "has_c2pa": has_c2pa,
        "has_ai_prompt": has_ai_prompt,
        "has_exif": has_exif,
        "has_gps": has_gps,
        "findings": findings,
    }


def _decode_png_text(body: bytes, chunk_type: bytes) -> tuple[str, str]:
    try:
        if chunk_type == b"tEXt":
            k, _, v = body.partition(b"\x00")
            return k.decode("latin1", "replace"), v.decode("latin1", "replace")
        if chunk_type == b"zTXt":
            k, _, rest = body.partition(b"\x00")
            if not rest:
                return k.decode("latin1", "replace"), ""
            decomp = zlib.decompress(rest[1:])
            return k.decode("latin1", "replace"), decomp.decode("utf-8", "replace")
        if chunk_type == b"iTXt":
            k, rest = body.split(b"\x00", 1)
            comp_flag = rest[0]
            _, rest2 = rest[2:].split(b"\x00", 1)
            _, text = rest2.split(b"\x00", 1)
            if comp_flag:
                text = zlib.decompress(text)
            return k.decode("utf-8", "replace"), text.decode("utf-8", "replace")
    except Exception:
        pass
    return "unknown", ""


def _match_ai_signature(key: str, val: str) -> tuple[str | None, str | None]:
    kl = key.lower().strip()
    if kl in PNG_AI_TEXT_KEYS:
        return PNG_AI_TEXT_KEYS[kl]
    for pat, reason in PNG_AI_VALUE_PATTERNS:
        if pat.search(val):
            return "AI Generation Marker", reason
    for pat, reason in XMP_AI_PATTERNS:
        if pat.search(val):
            return "AI Provenance Marker", reason
    return None, None


# -----------------------------------------------------------------------------
# Three-Tier Cleaning & SynthID Disruption
# -----------------------------------------------------------------------------
def strip_png_metadata(data: bytes) -> bytes:
    """Safe Mode: Re-encode PNG copying IDAT pixel chunks byte-for-byte.
    Strips all C2PA, EXIF, AI text chunks, and trailing bytes.
    """
    out = io.BytesIO()
    out.write(PNG_HEADER)
    saw_iend = False
    for _off, ctype, body in iter_png_chunks(data):
        if ctype in PNG_CRITICAL_CHUNKS or ctype in PNG_SAFE_ANCILLARY:
            out.write(_encode_png_chunk(ctype, body))
            if ctype == b"IEND":
                saw_iend = True
    if not saw_iend:
        out.write(_encode_png_chunk(b"IEND", b""))
    return out.getvalue()


def strip_jpeg_metadata(data: bytes) -> bytes:
    """Safe Mode: Re-encode JPEG preserving DCT scan data byte-for-byte.
    Strips APP1..APP15 (EXIF/XMP/C2PA) and COM segments.
    """
    out = io.BytesIO()
    out.write(JPEG_SOI)
    sos_offset = None

    for off, marker, body in iter_jpeg_segments(data):
        if marker == 0xDA:  # SOS
            sos_offset = off
            out.write(bytes([0xFF, marker]))
            out.write(struct.pack(">H", len(body) + 2))
            out.write(body)
            break
        if marker == 0xFE:  # COM
            continue
        if 0xE1 <= marker <= 0xEF:  # APP1..APP15
            continue
        out.write(bytes([0xFF, marker]))
        if body:
            out.write(struct.pack(">H", len(body) + 2))
            out.write(body)

    if sos_offset is None:
        return data

    scan_start = None
    for off, marker, body in iter_jpeg_segments(data):
        if marker == 0xDA:
            scan_start = off + 2 + 2 + len(body)
            break
    if scan_start is None:
        return data

    eoi = data.rfind(JPEG_EOI)
    if eoi == -1:
        return data
    out.write(data[scan_start:eoi])
    out.write(JPEG_EOI)
    return out.getvalue()


def reencode_with_dither(data: bytes, fmt: str, noise_sigma: float = 0.5) -> bytes:
    """Paranoid Mode: Decodes pixels, adds micro-Gaussian dither to scrub quantization
    artifacts, and writes a pristine sRGB stream.
    """
    im = Image.open(io.BytesIO(data))
    im.load()
    mode = im.mode

    if fmt == "JPEG" and mode != "RGB":
        im = im.convert("RGB")
        mode = "RGB"

    arr = np.asarray(im, dtype=np.int16)
    if noise_sigma > 0:
        rng = np.random.default_rng()
        noise = rng.normal(0.0, noise_sigma, arr.shape)
        arr = np.clip(arr + noise.round().astype(np.int16), 0, 255)
    arr = arr.astype(np.uint8)

    out_im = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()

    if fmt == "PNG":
        out_im.save(buf, format="PNG", optimize=True)
        return strip_png_metadata(buf.getvalue())
    else:
        out_im.save(buf, format="JPEG", quality=92, subsampling=2, optimize=True)
        return strip_jpeg_metadata(buf.getvalue())


def nuclear_synthid_disruption(data: bytes, fmt: str) -> bytes:
    """Nuclear Mode: Specifically designed to disrupt Google DeepMind SynthID
    and robust spatial-frequency watermarks.
    
    SynthID embeds imperceptible spatial-frequency perturbations into the latent
    sampling space. This function breaks the phase-lock and sub-LSB alignment by:
      1. Converting to RGB and transmuting through a high-frequency DCT cycle
      2. Non-uniform 0.997 Lanczos resizing to scramble spatial carrier frequencies
      3. 2px edge cropping to destroy peripheral coordinate anchors
      4. Subtle Gaussian noise (sigma=0.7) + per-channel integer bias
      5. Re-encoding cleanly with stripped metadata
    """
    im = Image.open(io.BytesIO(data))
    im.load()

    has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    if not has_alpha and im.mode != "RGB":
        im = im.convert("RGB")

    # If no alpha, apply a lossless JPEG DCT cycle to disrupt frequency phase
    if not has_alpha:
        dct_buf = io.BytesIO()
        im.save(dct_buf, format="JPEG", quality=92, subsampling=2, optimize=True)
        im = Image.open(dct_buf)
        im.load()

    w, h = im.size
    # Step 1: Subtle aspect scale (0.997)
    nw = max(8, int(round(w * 0.997)))
    nh = max(8, int(round(h * 0.997)))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)

    # Step 2: 2px edge crop
    if nw > 8 and nh > 8:
        im = im.crop((2, 2, nw - 2, nh - 2))

    # Step 3: Subtle spatial noise + per-channel bias
    arr = np.asarray(im, dtype=np.int16)
    rng = np.random.default_rng()
    noise = rng.normal(0.0, 0.7, arr.shape).round().astype(np.int16)
    arr = arr + noise

    if arr.shape[-1] >= 3:
        bias = rng.integers(-1, 2, size=arr.shape[-1]).astype(np.int16)
        arr = arr + bias

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    final_im = Image.fromarray(arr, mode=im.mode)

    buf = io.BytesIO()
    if fmt == "PNG":
        final_im.save(buf, format="PNG", optimize=True)
        return strip_png_metadata(buf.getvalue())
    else:
        final_im.save(buf, format="JPEG", quality=88, subsampling=2, optimize=True)
        return strip_jpeg_metadata(buf.getvalue())


def clean_image_fingerprints(data: bytes, mode: str = "safe", fmt: str | None = None) -> bytes:
    """Clean image fingerprints according to the chosen security tier.
    
    Modes:
      - 'safe': Lossless C2PA & metadata strip (pixel bytes untouched)
      - 'paranoid': Strips metadata + resets quantization tables + Gaussian micro-dither
      - 'nuclear': Disrupts Google SynthID & robust invisible watermarks
    """
    if not fmt:
        if data.startswith(PNG_HEADER):
            fmt = "PNG"
        elif data.startswith(JPEG_SOI):
            fmt = "JPEG"
        else:
            fmt = "PNG"

    if mode == "safe":
        if fmt == "PNG":
            return strip_png_metadata(data)
        elif fmt == "JPEG":
            return strip_jpeg_metadata(data)
        return data

    elif mode == "paranoid":
        return reencode_with_dither(data, fmt, noise_sigma=0.5)

    elif mode == "nuclear":
        return nuclear_synthid_disruption(data, fmt)

    return data

