"""Remove the Gemini sparkle watermark from an image."""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from fingerprint_cleaner import inspect_image_fingerprints, clean_image_fingerprints
except ImportError:
    try:
        from .fingerprint_cleaner import inspect_image_fingerprints, clean_image_fingerprints
    except ImportError:
        inspect_image_fingerprints = None
        clean_image_fingerprints = None


def get_ffmpeg_binary() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return None


ALPHA_THRESHOLD = 0.002
MAX_ALPHA = 0.99
LOGO_VALUE = 255.0


def watermark_info(width: int, height: int) -> dict[str, int]:
    min_dimension = min(width, height)
    ratio = min_dimension / 1536
    size = max(16, round(96 * ratio))
    margin = max(8, round(64 * ratio))
    return {
        "size": size,
        "x": max(0, width - margin - size),
        "y": max(0, height - margin - size),
        "width": size,
        "height": size,
    }


def resolve_box(
    base: dict[str, int],
    width: int,
    height: int,
    size_scale: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
) -> dict[str, int]:
    size = max(8, min(round(base["size"] * size_scale), width, height))
    res = {
        "size": size,
        "x": max(0, min(base["x"] + offset_x, width - size)),
        "y": max(0, min(base["y"] + offset_y, height - size)),
        "width": size,
        "height": size,
    }
    if "preset" in base:
        res["preset"] = base["preset"]
    if "score" in base:
        res["score"] = base["score"]
    if "type" in base:
        res["type"] = base["type"]
    return res


def alpha_map(mask: Image.Image, size: int, gain: float) -> list[float]:
    resized = mask.resize((size, size), Image.Resampling.BICUBIC).convert("RGB")
    return [min(max(pixel) / 255.0 * gain, MAX_ALPHA) for pixel in resized.getdata()]


import base64

# Official Gemini image generation resolution catalog (geminiSizeCatalog.js)
# Discrete resolution mapping provides exact Bayesian priors over generic aspect ratio equations
OFFICIAL_GEMINI_IMAGE_CATALOG: dict[tuple[int, int], dict[str, int]] = {
    # 0.5K Tier (size: 48, margin: 32)
    (512, 512): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    (256, 1024): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    (192, 1536): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    (424, 632): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    (632, 424): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    (448, 600): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    (1024, 256): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    (600, 448): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    (464, 576): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    (576, 464): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    (1536, 192): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    (384, 688): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    (688, 384): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    (792, 168): {"size": 48, "margin_right": 32, "margin_bottom": 32, "tier": "0.5k"},
    # 1K Tier (size: 96, margin: 64)
    (1024, 1024): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (512, 2048): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (384, 3072): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (848, 1264): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (1264, 848): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (896, 1200): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (2048, 512): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (1200, 896): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (928, 1152): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (1152, 928): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (3072, 384): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (768, 1376): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (1376, 768): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (1408, 768): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    (1584, 672): {"size": 96, "margin_right": 64, "margin_bottom": 64, "tier": "1k"},
    # 2K Tier (size: 96, margin: 192 or 64)
    (2048, 2048): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (1024, 4096): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (768, 6144): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (1696, 2528): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (2528, 1696): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (1792, 2400): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (4096, 1024): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (2400, 1792): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (1856, 2304): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (2304, 1856): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (6144, 768): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (1536, 2752): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (2752, 1536): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (3168, 1344): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    (2816, 1536): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "2k"},
    # 4K Tier
    (4096, 4096): {"size": 96, "margin_right": 192, "margin_bottom": 192, "tier": "4k"},
}

# Official Google Veo text watermark templates (veoTextWatermarkTemplates.js)
RAW_VEO_TEXT_BASE64: dict[str, tuple[int, int, str]] = {
    "veo-text-23x10": (
        23,
        10,
        "AQoHAAAAAgsEAAAAAAAAAAAAAAAAAAAKV0AHAAEYXCMDAAAAAAAAAAAAAAAAAAY+YQ4ABjpdDwIHCQYAAAABBgkHAgAAAyVmJwMNXEkLGD9MOA4BARQ7SkIaBQAADFFCCiJnKBxYST5bTwsOWlxASV8nAwAGO18WSF0VQ1okHC9jIS5iIQgQSk0JAAASZTVfOBBUZ05KS1AfOVMLAAUzWgsAAApWY2sYCUNWGg8UGwotYBYGD0hRCQAABTRrVgoCHlxINU5QDxRcXDpMYisEAAAADEAmBAAEHEdOPhoDAxlET0wjBgA=",
    ),
    "veo-text-68x30": (
        68,
        30,
        "AgEABgQEBgIBAQICAQICAQECAgIABgQFAwACAgEBAgMBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQECAAl1en15IwIBAgMCAQEBAQECAgx7foF7KAMCAQECAgEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQIAAUZ/fn9fAgECAgIBAgICAgIDNH2AgG4FAwIBAQEBAQEBAQEBAQEBAQABAQEBAQEBAQEBAQICAgICAgICAQEBAQEBAQECHYCAgHcFAwIDAgICAwMDAgNhfX+BTwMEAgEBAQECAgEBAgEBAQEBAQEBAQEBAQEBAQEBAgICAgICAgIBAQEBAQEBAQECdoCBgDECAwMCAgICAwEBEYGAgn8YAgICAQEBAQICAgEBAQEBAQEBAQEBAQEBAQEBAQEBAQICAgICAgEBAQIBAQEBAQFLfoB+XgIDAwICAgICAgMvgYF/bwQBAgEBAQECAgIBAQEBAQEBAQEBAQEBAQAAAQEBAQECAgIBAgEBAQEBAgEBAQEBASJ/fn91DQICAgICAgICBVyAf344AgIBAQEBAQIDAwMCAgEBAQEBAQEBAQEBAQEBAQEBAgICAQEAAAABAQEBAQIBAQEBBWx/f4AvAwICAgIDAwMQeH9/exQBAgECAQEBAggMCwgCAQEBAQEBAQIBAQEBAQECAgMHCAgGAgEBAQEBAQECAgEBAQEDQIGBgV8EAQEBAgICA0F/gIBeAwECAgICDEBvf39/gXFSFgUCAgICAwIBAQIBBRREb4GCgIJwVBQEAQEBAQECAQEBAQMUfoGAfg0CAQECAgIFbX9/fy8DAgIDBDF1fX+BgYCBgIB7QAUDAwMDAgEBAgM4dX+Bf4KDgYKBe0EFAgEBAgIBAQECAwZggIGBOAQBAgICBRx/foB9DAECAQdAf4CBf4B/f3+AgIGAQAQCAwMCAgMFUH6CgoGAgH+AgIGBf2gNAwICAwEBAQEBATh/gIFeAgEBAgEETX+BflAFAwMDKXt/gH1hMhgZMV9/gYF+IwMCAwICA0B+gYB/eFM7OlFwgIB/fVEDAgICAQEBAQMDC3yBg38RAgIDAwduf4B8KgUDAwVtgH9+QQQDAgMDBUB9gH9uBgIDAwMUfIB+fkwMBQMCBQ0+gH+CfikDAgIBAQEBAwIJX4KCgDoCAwMEInuCgG0IAgIBMH5/f1IEAgECAgIBAlF/gX0jAwMDBVt9gIBSBAEBAQAEAwhMf4GAZQcDAgEBAQECAgcwgoJ/YgMCAgNRgYCATwQEAwNef398FAMEAwQEAwMDJn6Af0cDAwMTfH6AahUCAwIDAwICAgx0gYB/IgECAQEBAQICAg17gYJ+BAMDBHeCgIEaBAQDB32BgHZAPz0+Pj8+Pj5GfoOBVgMDAyl/f35RAwIBAwMCAwEDBUCAgn44BAIBAQEBAgECA1p/gIA3AgMzgIGAbQICBAIaf4GAgoCAgoGBgYGBgIGBgX9wBAICPX9/fzEDAgECAgIDAgICKYCCfk4EAgEBAAACAgICJX+CgVwEBFeCgn8zAQEDAh5+gYGAf35/fn9/f39+fn9+fm0GAwNBfoCAMQIBAQICAgICAQEjgIF+UQMCAQEAAQEBAgIGdIKDfg4MfIGBehUBAQICGX6AgWAzMTExMTExMjEwMS4wKgQCAz5/gYE1AwEBAgICAgICAiaAgX5OAgIBAQEBAgICAwVJf4F/NTB9gYBgBAICAwIOfYCCXQYDBAICAgIDAgECAwQEAgICNICCg0cEAQECAgICAgIENIGCf0YCAgEBAQEBAQICAyKAgoFgXX2BfTECAgICAwRugIF8DgMDAQECAwQDAwIDAgMCAgESfoOAdgYBAQIDAgEBAgdvgoJ8IAMEAQEBAQEBAQIBBXd/gH5+fn99CwECAgICBEB/f3xIBQMCAgICAgMTXTkVAwICAwZcg4N9QwIBAQIBAwECOHuAgWsJAwIBAQEBAQEBAQEEQn6Bf4B/f08EAQICAgMCFHWBgX1DBgMCBAUGFW5/fG0hAwICAyh+gYB9PQQFAwMEBC9+g4B+PwQCAgEBAQEBAQICAQIkgIKBgIB8JgICAgIDAwIFOn+CgH5sLxsQFkJ2gH6AbgQCAgIDBGGDgYB7bDogIjFdfIB/gm4EAQIBAQEBAQEBAQEBAgdcgoCCgGwOAgECAgICAwIJR4CAf4GBgYGBgYB+fmcVAQIBAQICDm19gIB/gX6AgX+AgH1rEwMCAgEBAQAAAQEBAQIBAz2Af4B+TQoCAgICAgICAQEJQX1/gIGAgIGBf3xnFgQDAgICAgMFC1B8f4CBgYGBgYB9XhQDAgICAQEBAAABAQEBAQIBDV9fX14UAgIBAgICAgEBAQIFElB7fH+Af3xuPwsCAgICAgICAgUDBzJifn9+foB9YzAFAgICAgIBAQEAAAEBAQECAgIBAwMDAwEBAQECAQICAgECAgIDBAYlKy4rEwUEAgICAgICAgEBAgIDAwMNIysyIAsEAgIDAwICAgIBAQABAQEBAQEBAQABAQEBAQEBAQICAQICAgECAgECAQIDAwMCAgEBAgICAgICAgICAgEBAgEBAQEBAQEBAQEBAQEBAgEBAQEBAQEBAQEBAQEBAQEBAQEBAgICAgICAgICAgMCAgICAgIBAQEBAgEBAQEBAgEBAQEBAgICAgICAgIBAQICAQEBAQEB",
    ),
    "veo-text-99x43": (
        99,
        43,
        "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",
    ),
}


def adaptive_video_offset(width: int, height: int) -> int:
    scale_ratio = max(0.3, min(1.5, min(width, height) / 720))
    return round(-24 * scale_ratio)


def veo_watermark_info(width: int, height: int) -> dict[str, int]:
    base = min(width, height)
    size = max(24, min(round(base / 15), base))
    margin = round(base / 10)
    return {
        "size": size,
        "x": max(0, width - margin - size),
        "y": max(0, height - margin - size),
        "width": size,
        "height": size,
    }


def resolve_video_watermark_candidates(width: int, height: int) -> list[dict]:
    """Resolve discrete candidates from the official Google Veo video catalog (videoWatermarkCatalog.js)."""
    raw_candidates = []

    if width == 1080 and height == 1920:
        raw_candidates.extend(
            [
                {
                    "id": "veo-1080x1920-portrait-inset-72",
                    "name": "Veo 1080p Portrait Inset (Margin 144)",
                    "size": 72,
                    "marginRight": 144,
                    "marginBottom": 144,
                    "prior": 1.30,
                },
                {
                    "id": "veo-1080x1920-portrait-standard-72",
                    "name": "Veo 1080p Portrait Standard (Margin 108)",
                    "size": 72,
                    "marginRight": 108,
                    "marginBottom": 108,
                    "prior": 1.15,
                },
            ]
        )
    elif width == 1920 and height == 1080:
        raw_candidates.extend(
            [
                {
                    "id": "veo-1920x1080-landscape-standard-72",
                    "name": "Veo 1080p Landscape Standard (Margin 108)",
                    "size": 72,
                    "marginRight": 108,
                    "marginBottom": 108,
                    "prior": 1.30,
                },
                {
                    "id": "veo-1920x1080-landscape-inset-72",
                    "name": "Veo 1080p Landscape Inset (Margin 144)",
                    "size": 72,
                    "marginRight": 144,
                    "marginBottom": 144,
                    "prior": 1.15,
                },
            ]
        )
    elif width == 1280 and height == 720:
        raw_candidates.extend(
            [
                {
                    "id": "veo-720p-inset-48",
                    "name": "Veo 720p Inset (Margin 96)",
                    "size": 48,
                    "marginRight": 96,
                    "marginBottom": 96,
                    "prior": 1.30,
                },
                {
                    "id": "veo-720p-standard-48",
                    "name": "Veo 720p Standard (Margin 72)",
                    "size": 48,
                    "marginRight": 72,
                    "marginBottom": 72,
                    "prior": 1.20,
                },
                {
                    "id": "veo-720p-compact-44",
                    "name": "Veo 720p Compact (Margin 29/40)",
                    "size": 44,
                    "marginRight": 29,
                    "marginBottom": 40,
                    "prior": 1.10,
                },
            ]
        )
    elif width == 720 and height == 1280:
        raw_candidates.extend(
            [
                {
                    "id": "veo-720x1280-portrait-inset-48",
                    "name": "Veo 720x1280 Portrait Inset (Margin 96)",
                    "size": 48,
                    "marginRight": 96,
                    "marginBottom": 96,
                    "prior": 1.30,
                },
                {
                    "id": "veo-720x1280-portrait-standard-48",
                    "name": "Veo 720x1280 Portrait Standard (Margin 72)",
                    "size": 48,
                    "marginRight": 72,
                    "marginBottom": 72,
                    "prior": 1.20,
                },
                {
                    "id": "veo-720x1280-compact-44",
                    "name": "Veo 720x1280 Compact (Margin 29/40)",
                    "size": 44,
                    "marginRight": 29,
                    "marginBottom": 40,
                    "prior": 1.10,
                },
                {
                    "id": "veo-720x1280-animated-24",
                    "name": "Veo 720x1280 Animated (Margin 48)",
                    "size": 24,
                    "marginRight": 48,
                    "marginBottom": 48,
                    "prior": 1.05,
                },
            ]
        )
    else:
        # Projected candidates from 1920x1080 reference
        scale = min(width / 1920.0, height / 1080.0)
        s = max(24, min(round(72 * scale), min(width, height) - 8))
        m_std = max(8, round(108 * scale))
        m_ins = max(8, round(144 * scale))
        raw_candidates.extend(
            [
                {
                    "id": "veo-projected-standard",
                    "name": f"Veo Projected Standard ({s}px, m={m_std})",
                    "size": s,
                    "marginRight": m_std,
                    "marginBottom": m_std,
                    "prior": 1.15,
                },
                {
                    "id": "veo-projected-inset",
                    "name": f"Veo Projected Inset ({s}px, m={m_ins})",
                    "size": s,
                    "marginRight": m_ins,
                    "marginBottom": m_ins,
                    "prior": 1.10,
                },
            ]
        )

    formatted = []
    for c in raw_candidates:
        sz = c["size"]
        mr = c["marginRight"]
        mb = c["marginBottom"]
        cx = max(0, width - mr - sz)
        cy = max(0, height - mb - sz)
        formatted.append(
            {
                "presetKey": c["id"],
                "name": c["name"],
                "baseSize": sz,
                "calcPos": lambda s, _x=cx, _y=cy, _sz=sz: (
                    max(0, min(width - s, _x + (_sz - s) // 2)),
                    max(0, min(height - s, _y + (_sz - s) // 2)),
                ),
                "gain": 1.0,
                "prior": c["prior"],
            }
        )
    return formatted


def detect_veo_text_watermark(image: Image.Image | np.ndarray) -> dict | None:
    """Detect the Google Veo text watermark ('Veo' logo) in bottom right corner."""
    if isinstance(image, Image.Image):
        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    best_match = None
    best_score = -1.0

    for name, (tw, th, b64) in RAW_VEO_TEXT_BASE64.items():
        raw = base64.b64decode(b64)
        tpl = np.frombuffer(raw, dtype=np.uint8).reshape((th, tw))
        if name == "veo-text-23x10":
            mr, mb = 15, 16
        elif name == "veo-text-68x30":
            mr, mb = 44, 48
        else:
            mr, mb = 64, 68

        cx = max(0, width - mr - tw)
        cy = max(0, height - mb - th)

        win_x0 = max(0, cx - 20)
        win_y0 = max(0, cy - 20)
        win_x1 = min(width, cx + tw + 20)
        win_y1 = min(height, cy + th + 20)

        if win_x1 - win_x0 < tw or win_y1 - win_y0 < th:
            continue

        patch = gray[win_y0:win_y1, win_x0:win_x1]
        res = cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_l = cv2.minMaxLoc(res)

        if max_v > best_score and max_v >= 0.50:
            best_score = float(max_v)
            best_match = {
                "x": win_x0 + max_l[0],
                "y": win_y0 + max_l[1],
                "width": tw,
                "height": th,
                "size": max(tw, th),
                "score": float(max_v),
                "preset": name,
                "type": "text",
            }

    return best_match


def get_image_layout_candidates(width: int, height: int) -> list[dict]:
    """Candidate layout families including official Gemini image catalog, Veo video catalog, and adaptive fallbacks."""
    min_dim = min(width, height)
    base_ratio = min_dim / 1536.0
    base_size = max(16, round(96 * base_ratio))
    candidates = []

    # 1. Official Gemini Image Generation Catalog Prior
    cat_entry = OFFICIAL_GEMINI_IMAGE_CATALOG.get((width, height))
    if cat_entry:
        cat_sz = cat_entry["size"]
        cat_mr = cat_entry["margin_right"]
        cat_mb = cat_entry["margin_bottom"]
        cat_x = max(0, width - cat_mr - cat_sz)
        cat_y = max(0, height - cat_mb - cat_sz)
        candidates.append(
            {
                "presetKey": f"gemini-catalog-{cat_entry['tier']}",
                "name": f"Gemini Catalog ({cat_entry['tier'].upper()} {cat_sz}px)",
                "baseSize": cat_sz,
                "calcPos": lambda s, _x=cat_x, _y=cat_y, _sz=cat_sz: (
                    max(0, min(width - s, _x + (_sz - s) // 2)),
                    max(0, min(height - s, _y + (_sz - s) // 2)),
                ),
                "gain": 0.6 if cat_entry["tier"] in ("1k", "2k", "4k") else 1.0,
                "prior": 1.35,
            }
        )

    # 2. Discrete Video Catalog
    candidates.extend(resolve_video_watermark_candidates(width, height))

    # 3. Fallback Adaptive and Fixed Gemini Image Layouts
    candidates.extend(
        [
            {
                "presetKey": "new",
                "name": "New Gemini (Adaptive)",
                "baseSize": base_size,
                "calcPos": lambda s, _w=width, _h=height, _br=base_ratio: (
                    max(0, _w - max(8, round(192 * _br)) - s),
                    max(0, _h - max(8, round(192 * _br)) - s),
                ),
                "gain": 0.6,
                "prior": 1.08,
            },
            {
                "presetKey": "classic",
                "name": "Classic Corner (Adaptive)",
                "baseSize": base_size,
                "calcPos": lambda s, _w=width, _h=height, _br=base_ratio: (
                    max(0, _w - max(8, round(64 * _br)) - s),
                    max(0, _h - max(8, round(64 * _br)) - s),
                ),
                "gain": 1.0,
                "prior": 1.04,
            },
            {
                "presetKey": "new",
                "name": "Gemini (Fixed 96px Inset)",
                "baseSize": 96,
                "calcPos": lambda s, _w=width, _h=height, _md=min_dim: (
                    max(0, _w - (192 if _md >= 1400 else round(128 * max(0.5, _md / 1024.0))) - s),
                    max(0, _h - (192 if _md >= 1400 else round(128 * max(0.5, _md / 1024.0))) - s),
                ),
                "gain": 0.6,
                "prior": 1.02,
            },
            {
                "presetKey": "classic",
                "name": "Classic Corner (Fixed 96px)",
                "baseSize": 96,
                "calcPos": lambda s, _w=width, _h=height, _md=min_dim: (
                    max(0, _w - (64 if _md >= 1024 else 32) - s),
                    max(0, _h - (64 if _md >= 1024 else 32) - s),
                ),
                "gain": 1.0,
                "prior": 1.01,
            },
        ]
    )
    return candidates


def detect_watermark_box(
    image: Image.Image | np.ndarray,
    mask_image: Image.Image | None = None,
    base: dict[str, int] | None = None,
) -> dict[str, int]:
    """Find the exact location and scale of the watermark using candidate layout search and multi-scale template matching."""
    if isinstance(image, Image.Image):
        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    assets_dir = Path(__file__).parent / "assets"
    m96 = Image.open(assets_dir / "bg_96.png")
    m48 = Image.open(assets_dir / "bg_48.png")
    m96_gray = np.array(m96.convert("L"))
    m48_gray = np.array(m48.convert("L"))

    layout_families = get_image_layout_candidates(width, height)
    if base and "x" in base and "y" in base and "size" in base:
        bx = int(base["x"])
        by = int(base["y"])
        bs = int(base["size"])
        layout_families.insert(
            0,
            {
                "presetKey": base.get("preset", "base_prior"),
                "name": "Base Prior Hint",
                "baseSize": bs,
                "calcPos": lambda s, _bx=bx, _by=by, _bs=bs: (
                    max(0, min(width - s, _bx + (_bs - s) // 2)),
                    max(0, min(height - s, _by + (_bs - s) // 2)),
                ),
                "gain": 1.0,
                "prior": 1.30,
            },
        )

    scale_pyramid = (0.75, 0.85, 0.95, 1.00, 1.05, 1.15, 1.25)

    best_match = None
    best_score = -1.0

    # 1. Test candidate layout families
    for layout in layout_families:
        for scale in scale_pyramid:
            s = max(16, min(round(layout["baseSize"] * scale), min(width, height) - 8))
            x, y = layout["calcPos"](s)
            if x < 0 or y < 0 or x + s > width or y + s > height:
                continue
            patch = gray[y : y + s, x : x + s]
            tpl = cv2.resize(m48_gray if s <= 48 else m96_gray, (s, s), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED)
            weighted = float(res[0, 0]) * layout["prior"]
            if weighted > best_score:
                best_score = weighted
                best_match = {
                    "x": x,
                    "y": y,
                    "size": s,
                    "width": s,
                    "height": s,
                    "score": weighted,
                    "preset": layout["presetKey"],
                    "gain": layout["gain"],
                }

    # 2. Fine-tuning (±24px position, ±10% scale)
    if best_match and best_match["score"] > 0.10:
        fine_sizes = {
            max(16, round(best_match["size"] * 0.90)),
            max(16, round(best_match["size"] * 0.95)),
            best_match["size"],
            min(min(width, height) - 8, round(best_match["size"] * 1.05)),
            min(min(width, height) - 8, round(best_match["size"] * 1.10)),
        }
        refined_x = best_match["x"]
        refined_y = best_match["y"]
        refined_s = best_match["size"]
        refined_score = best_match["score"]

        for ts in fine_sizes:
            tpl = cv2.resize(m48_gray if ts <= 48 else m96_gray, (ts, ts), interpolation=cv2.INTER_AREA)
            win_y0 = max(0, best_match["y"] - 24)
            win_y1 = min(height, best_match["y"] + 24 + ts)
            win_x0 = max(0, best_match["x"] - 24)
            win_x1 = min(width, best_match["x"] + 24 + ts)
            if win_y1 - win_y0 >= ts and win_x1 - win_x0 >= ts:
                sub_patch = gray[win_y0:win_y1, win_x0:win_x1]
                res = cv2.matchTemplate(sub_patch, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, max_l = cv2.minMaxLoc(res)
                if max_v > refined_score:
                    refined_score = float(max_v)
                    refined_x = win_x0 + max_l[0]
                    refined_y = win_y0 + max_l[1]
                    refined_s = ts

        best_match["x"] = refined_x
        best_match["y"] = refined_y
        best_match["size"] = refined_s
        best_match["width"] = refined_s
        best_match["height"] = refined_s
        best_match["score"] = refined_score

    # 3. If layout candidates had low confidence, search the broader lower-right quadrant (for cropped images)
    if best_score < 0.35:
        roi = gray[int(height * 0.5) :, int(width * 0.5) :]
        rx0, ry0 = int(width * 0.5), int(height * 0.5)
        for mask_tpl, bsz in ((m48_gray, 48), (m96_gray, 96)):
            for sc in (0.75, 1.0, 1.25):
                sz = int(round(bsz * sc))
                if sz < 16 or sz >= roi.shape[0] - 2 or sz >= roi.shape[1] - 2:
                    continue
                tpl = cv2.resize(mask_tpl, (sz, sz), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(res)
                if mv > best_score:
                    best_score = float(mv)
                    best_match = {
                        "x": rx0 + ml[0],
                        "y": ry0 + ml[1],
                        "size": sz,
                        "width": sz,
                        "height": sz,
                        "score": float(mv),
                        "preset": "custom",
                        "gain": 0.6,
                    }

    if best_match and best_match["score"] >= 0.35:
        return best_match

    # Fallback to first layout candidate (or base prior)
    fallback = layout_families[0]
    s = fallback["baseSize"]
    x, y = fallback["calcPos"](s)
    return {
        "x": x,
        "y": y,
        "size": s,
        "width": s,
        "height": s,
        "score": 0.0,
        "preset": fallback.get("presetKey", "new"),
        "gain": fallback["gain"],
    }


def detect_video_box(
    frame: np.ndarray,
    base: dict[str, int],
    mask_image: Image.Image,
) -> dict[str, int]:
    return detect_watermark_box(frame, mask_image, base)


def remove_watermark(
    image: Image.Image,
    mask: Image.Image | None = None,
    gain: float = 0.6,
    size_scale: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
    method: str = "inpaint",
    box: dict[str, int] | None = None,
    synthid_mode: str = "none",
    strip_metadata: bool = False,
) -> Image.Image:
    width, height = image.size

    if mask is None:
        mask_path = Path(__file__).parent / "assets" / "bg_96.png"
        mask = Image.open(mask_path)

    if box is None:
        detected = detect_watermark_box(image, mask)
        box = resolve_box(detected, width, height, size_scale, offset_x, offset_y)
    else:
        box = resolve_box(box, width, height, size_scale, offset_x, offset_y)

    x, y, s = box["x"], box["y"], box["size"]

    if method == "inpaint":
        frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        pad = 6
        bx = max(0, x - pad)
        by = max(0, y - pad)
        bs = s + pad * 2
        inpaint_mask = sparkle_inpaint_mask(bs, mask)
        pad_patch = 16
        py0 = max(0, by - pad_patch)
        py1 = min(height, by + bs + pad_patch)
        px0 = max(0, bx - pad_patch)
        px1 = min(width, bx + bs + pad_patch)
        patch = frame[py0:py1, px0:px1].copy()
        patch_mask = np.zeros(patch.shape[:2], dtype=np.uint8)
        patch_mask[by - py0 : by - py0 + bs, bx - px0 : bx - px0 + bs] = inpaint_mask
        cleaned_patch = cv2.inpaint(patch, patch_mask, 5, cv2.INPAINT_TELEA)
        frame[py0:py1, px0:px1] = cleaned_patch
        cleaned_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    elif method == "reconstruct":
        frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        inpaint_mask = sparkle_inpaint_mask(s, mask)
        cleaned_frame = reconstruct_rows(frame, box, inpaint_mask)
        cleaned_image = Image.fromarray(cv2.cvtColor(cleaned_frame, cv2.COLOR_BGR2RGB))

    else:  # math (alpha unblending)
        output = image.convert("RGBA")
        alphas = alpha_map(mask, s, gain)
        pixels = output.load()
        for row in range(s):
            for col in range(s):
                alpha = alphas[row * s + col]
                if alpha < ALPHA_THRESHOLD:
                    continue
                px = x + col
                py = y + row
                if px >= width or py >= height:
                    continue
                r, g, b, a = pixels[px, py]
                restored = tuple(
                    max(0, min(255, round((channel - alpha * LOGO_VALUE) / (1.0 - alpha))))
                    for channel in (r, g, b)
                )
                pixels[px, py] = (*restored, a)
        cleaned_image = output.convert("RGB")

    active_mode = synthid_mode if synthid_mode in ("safe", "paranoid", "nuclear") else ("safe" if strip_metadata else None)
    if active_mode and clean_image_fingerprints is not None:
        buf = io.BytesIO()
        cleaned_image.save(buf, format="PNG")
        cleaned_bytes = clean_image_fingerprints(buf.getvalue(), mode=active_mode, fmt="PNG")
        return Image.open(io.BytesIO(cleaned_bytes))

    return cleaned_image


def sparkle_inpaint_mask(size: int, mask_image: Image.Image | None = None) -> np.ndarray:
    """Generate a high-fidelity inpainting mask for the Gemini / Veo sparkle watermark.
    Uses the canonical asset alpha channel with elliptical dilation to ensure 100% of the
    star extremities, subtle glow, and anti-aliasing boundary are cleanly erased.
    """
    raw_mask = None
    if mask_image is not None:
        raw_mask = np.asarray(mask_image.convert("L"))
    else:
        candidates = [
            Path(__file__).parent / "assets" / "bg_96.png",
            Path(__file__).parent / "bg_96.png",
            Path.cwd() / "assets" / "bg_96.png",
            Path.cwd() / "developeros-watermark-studio" / "assets" / "bg_96.png",
        ]
        found = next((c for c in candidates if c.exists()), None)
        if found:
            raw_mask = np.asarray(Image.open(found).convert("L"))

    if raw_mask is not None:
        resized = cv2.resize(raw_mask, (size, size), interpolation=cv2.INTER_LINEAR)
        binary = (resized > 4).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.dilate(binary, kernel, iterations=1)

    # Fallback analytical polygon
    mask = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    inner = max(4, round(size * 0.28))
    points = np.array(
        [
            [center, 0],
            [center + inner, center - inner],
            [size - 1, center],
            [center + inner, center + inner],
            [center, size - 1],
            [center - inner, center + inner],
            [0, center],
            [center - inner, center - inner],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [points], 255)
    return cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)


def reconstruct_rows(frame: np.ndarray, box: dict[str, int], mask: np.ndarray) -> np.ndarray:
    output = frame.copy()
    x0, y0 = box["x"], box["y"]
    for row in range(box["height"]):
        masked = np.flatnonzero(mask[row])
        if masked.size == 0:
            continue
        left = x0 + int(masked[0])
        right = x0 + int(masked[-1])
        sample_left = max(0, left - 8)
        sample_right = min(frame.shape[1] - 1, right + 8)
        left_pixel = frame[y0 + row, sample_left].astype(np.float32)
        right_pixel = frame[y0 + row, sample_right].astype(np.float32)
        for x in range(left, right + 1):
            ratio = (x - left) / max(1, right - left)
            output[y0 + row, x] = np.clip(
                left_pixel * (1 - ratio) + right_pixel * ratio,
                0,
                255,
            ).astype(np.uint8)
    return output


def remove_watermark_from_video(
    input_path: Path,
    output_path: Path,
    mask_path: Path | None = None,
    gain: float = 1.0,
    size_scale: float = 1.0,
    offset_x: int | None = None,
    offset_y: int | None = None,
    preset: str = "auto",
    method: str = "inpaint",
    progress_callback: callable | None = None,
    trim_end_seconds: float | None = None,
    crf: int = 12,
    preset_speed: str = "slow",
) -> dict:
    if mask_path is None or not Path(mask_path).exists():
        candidates = [
            Path(__file__).parent / "assets" / "bg_96.png",
            Path(__file__).parent / "assets" / "bg_48.png",
            Path(__file__).parent / "bg_96.png",
            Path(__file__).parent / "bg_48.png",
            Path.cwd() / "assets" / "bg_96.png",
            Path.cwd() / "gemini-watermark-remover" / "assets" / "bg_96.png",
        ]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise FileNotFoundError("Could not find watermark asset bg_96.png")
        mask_path = found

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    effective_trim = trim_end_seconds
    if effective_trim is None:
        effective_trim = 3.0 if preset == "notebooklm" else 0.0

    total_duration = frame_count / fps if fps > 0 else 0
    if effective_trim > 0 and total_duration > effective_trim:
        target_duration = max(0.1, total_duration - effective_trim)
        frames_to_process = max(1, int(round(target_duration * fps)))
    else:
        target_duration = total_duration
        frames_to_process = frame_count

    sample_indices = [0]
    if frame_count > 1:
        step = max(1, min(int(fps * 0.75), frame_count // 6))
        sample_indices = sorted(list({0, step, step * 2, step * 3, step * 4, min(frame_count - 1, int(fps * 2))}))

    is_text_watermark = False
    text_detected = None

    if preset == "veo_text":
        for s_idx in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, s_idx)
            ret, s_frame = capture.read()
            if not ret or s_frame is None:
                continue
            cand_text = detect_veo_text_watermark(s_frame)
            if cand_text and cand_text.get("score", 0.0) >= 0.45:
                text_detected = cand_text
                break
        if text_detected:
            is_text_watermark = True

    elif preset == "auto":
        # Check if first frame has a strong Veo text watermark
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, s_frame = capture.read()
        if ret and s_frame is not None:
            cand_text = detect_veo_text_watermark(s_frame)
            if cand_text and cand_text.get("score", 0.0) >= 0.75:
                text_detected = cand_text
                is_text_watermark = True

    if is_text_watermark and text_detected:
        pad = 6
        tx = text_detected["x"]
        ty = text_detected["y"]
        tw = text_detected["width"]
        th = text_detected["height"]
        box = {
            "x": max(0, tx - pad),
            "y": max(0, ty - pad),
            "width": min(width - tx, tw + pad * 2),
            "height": min(height - ty, th + pad * 2),
            "size": max(tw, th) + pad * 2,
            "score": text_detected["score"],
            "preset": text_detected["preset"],
            "type": "text",
        }
        inpaint_mask = np.ones((box["height"], box["width"]), dtype=np.uint8) * 255
        alpha = None
        active = None
    elif preset == "notebooklm":
        is_portrait = height > width
        if is_portrait:
            wm_w = int(round(width * 0.2722))
            wm_h = int(round(height * 0.0328))
            mr = int(round(width * 0.0167))
            mb = int(round(height * 0.0109))
            feather = 10
            nlm_offset_y = 6
        else:
            wm_w = int(round(width * 0.1220))
            wm_h = int(round(height * 0.0400))
            mr = int(round(width * 0.0270))
            mb = int(round(height * 0.0460))
            feather = 6
            nlm_offset_y = 4
        bx = max(0, width - wm_w - mr)
        by = max(0, height - wm_h - mb)
        box = {
            "x": bx,
            "y": by,
            "width": wm_w,
            "height": wm_h,
            "size": max(wm_w, wm_h),
            "score": 1.0,
            "preset": "notebooklm",
            "type": "box",
        }
        nlm_alpha = np.ones((wm_h, wm_w, 1), dtype=np.float32)
        for i in range(wm_h):
            for j in range(wm_w):
                a = 1.0
                if i < feather:
                    a = min(a, float(i) / float(feather))
                if i > wm_h - feather:
                    a = min(a, float(wm_h - i) / float(feather))
                if j < feather:
                    a = min(a, float(j) / float(feather))
                if j > wm_w - feather:
                    a = min(a, float(wm_w - j) / float(feather))
                nlm_alpha[i, j, 0] = a
        inpaint_mask = np.ones((box["height"], box["width"]), dtype=np.uint8) * 255
        alpha = None
        active = None
    else:
        # Standard or Inset Veo Sparkle watermark
        base = veo_watermark_info(width, height)
        if preset in ("auto", "veo", "veo_inset"):
            default_offset = adaptive_video_offset(width, height)
            base["x"] = max(0, base["x"] + default_offset)
            base["y"] = max(0, base["y"] + default_offset)
            base["preset"] = "veo-inset"
        elif preset == "veo_standard":
            base["preset"] = "veo-standard"
        elif preset == "veo_compact":
            scale = min(width / 1280.0, height / 720.0)
            sz = max(24, round(44 * scale))
            mr = max(8, round(29 * scale))
            mb = max(8, round(40 * scale))
            base = {
                "size": sz,
                "x": max(0, width - mr - sz),
                "y": max(0, height - mb - sz),
                "width": sz,
                "height": sz,
                "preset": "veo-compact",
            }
        elif preset == "corner":
            base["preset"] = "corner"

        with Image.open(mask_path) as mask_image:
            best_detected = None
            best_score = -1.0

            for s_idx in sample_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, s_idx)
                ret, s_frame = capture.read()
                if not ret or s_frame is None:
                    continue
                cand = detect_watermark_box(s_frame, mask_image, base)
                cand_score = cand.get("score", 0.0)
                if cand_score > best_score:
                    best_score = cand_score
                    best_detected = cand
                if best_score >= 0.85:
                    break

            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if best_detected is None or best_detected.get("score", 0.0) < 0.35:
                detected = {
                    "x": base["x"],
                    "y": base["y"],
                    "size": base["size"],
                    "score": 0.0,
                    "preset": base.get("preset", "fallback"),
                }
            else:
                detected = best_detected

            # Generous safety padding (6px) around the detected box guarantees all 4 star tips,
            # subtle glow, and anti-aliasing edges are 100% encompassed inside the reconstruction zone.
            pad = 6
            exp_detected = {
                "x": max(0, detected["x"] - pad),
                "y": max(0, detected["y"] - pad),
                "size": detected["size"] + pad * 2,
                "width": detected.get("width", detected["size"]) + pad * 2,
                "height": detected.get("height", detected["size"]) + pad * 2,
                "preset": detected.get("preset", base.get("preset", "veo")),
                "score": detected.get("score", 0.0),
            }

            box = resolve_box(
                exp_detected,
                width,
                height,
                size_scale,
                0 if offset_x is None else offset_x,
                0 if offset_y is None else offset_y,
            )
            box["preset"] = detected.get("preset", base.get("preset", "veo"))
            box["score"] = detected.get("score", 0.0)
            mask = np.asarray(
                mask_image.resize((box["size"], box["size"]), Image.Resampling.BICUBIC).convert("RGB"),
                dtype=np.float32,
            )
            alpha = np.minimum(mask.max(axis=2) / 255.0 * gain, MAX_ALPHA)
            active = alpha >= ALPHA_THRESHOLD
            inpaint_mask = sparkle_inpaint_mask(box["size"], mask_image)

    ffmpeg_bin = get_ffmpeg_binary()

    # Strategy 1: High-Fidelity Direct FFmpeg Pipe (Zero intermediate compression, CRF 17 master quality)
    if ffmpeg_bin is not None:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-t", f"{target_duration:.3f}",
            "-i", str(input_path),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset_speed,
            "-pix_fmt", "yuv420p",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-color_range", "tv",
            "-c:a", "copy",
            "-movflags", "+faststart",
            "-t", f"{target_duration:.3f}",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        processed = 0
        pipe_failed = False
        try:
            while processed < frames_to_process:
                success, frame = capture.read()
                if not success:
                    break

                if preset == "notebooklm":
                    bx, by = box["x"], box["y"]
                    bw, bh = box["width"], box["height"]
                    src_y = max(0, by - bh - nlm_offset_y)
                    src_patch = frame[src_y : src_y + bh, bx : bx + bw].astype(np.float32)
                    dst_patch = frame[by : by + bh, bx : bx + bw].astype(np.float32)
                    blended = (dst_patch * (1.0 - nlm_alpha) + src_patch * nlm_alpha).clip(0, 255).astype(np.uint8)
                    frame[by : by + bh, bx : bx + bw] = blended
                elif is_text_watermark or method == "inpaint":
                    pad = 16
                    py0 = max(0, box["y"] - pad)
                    py1 = min(height, box["y"] + box["height"] + pad)
                    px0 = max(0, box["x"] - pad)
                    px1 = min(width, box["x"] + box["width"] + pad)

                    patch = frame[py0:py1, px0:px1].copy()
                    patch_mask = np.zeros(patch.shape[:2], dtype=np.uint8)
                    my0 = box["y"] - py0
                    mx0 = box["x"] - px0
                    patch_mask[my0 : my0 + box["height"], mx0 : mx0 + box["width"]] = inpaint_mask

                    cleaned_patch = cv2.inpaint(patch, patch_mask, 5, cv2.INPAINT_TELEA)
                    frame[py0:py1, px0:px1] = cleaned_patch
                elif method == "reconstruct":
                    frame = reconstruct_rows(frame, box, inpaint_mask)
                else:
                    region = frame[box["y"] : box["y"] + box["height"], box["x"] : box["x"] + box["width"]].astype(np.float32)
                    restored = (region - alpha[:, :, None] * LOGO_VALUE) / (1.0 - alpha[:, :, None])
                    region[active] = np.clip(restored[active], 0, 255)
                    frame[box["y"] : box["y"] + box["height"], box["x"] : box["x"] + box["width"]] = region.astype(np.uint8)

                try:
                    proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError):
                    pipe_failed = True
                    break

                processed += 1
                if progress_callback:
                    progress_callback(processed, frames_to_process)
                elif frames_to_process and processed % 30 == 0:
                    print(f"Processed {processed}/{frames_to_process} frames", end="\r")

            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            proc.wait()

            if not pipe_failed and proc.returncode == 0:
                capture.release()
                print(f"Processed {processed} frames. Master-quality pipeline complete.")
                return box
        except Exception:
            pipe_failed = True
            try:
                proc.kill()
            except Exception:
                pass

        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Strategy 2: Fallback via temporary file
    temporary = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temporary_path = Path(temporary.name)
    temporary.close()
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Could not create temporary video output")

    try:
        processed = 0
        while processed < frames_to_process:
            success, frame = capture.read()
            if not success:
                break

            if preset == "notebooklm":
                bx, by = box["x"], box["y"]
                bw, bh = box["width"], box["height"]
                src_y = max(0, by - bh - nlm_offset_y)
                src_patch = frame[src_y : src_y + bh, bx : bx + bw].astype(np.float32)
                dst_patch = frame[by : by + bh, bx : bx + bw].astype(np.float32)
                blended = (dst_patch * (1.0 - nlm_alpha) + src_patch * nlm_alpha).clip(0, 255).astype(np.uint8)
                frame[by : by + bh, bx : bx + bw] = blended
            elif is_text_watermark or method == "inpaint":
                pad = 16
                py0 = max(0, box["y"] - pad)
                py1 = min(height, box["y"] + box["height"] + pad)
                px0 = max(0, box["x"] - pad)
                px1 = min(width, box["x"] + box["width"] + pad)

                patch = frame[py0:py1, px0:px1].copy()
                patch_mask = np.zeros(patch.shape[:2], dtype=np.uint8)
                my0 = box["y"] - py0
                mx0 = box["x"] - px0
                patch_mask[my0 : my0 + box["height"], mx0 : mx0 + box["width"]] = inpaint_mask

                cleaned_patch = cv2.inpaint(patch, patch_mask, 5, cv2.INPAINT_TELEA)
                frame[py0:py1, px0:px1] = cleaned_patch
            elif method == "reconstruct":
                frame = reconstruct_rows(frame, box, inpaint_mask)
            else:
                region = frame[box["y"] : box["y"] + box["height"], box["x"] : box["x"] + box["width"]].astype(np.float32)
                restored = (region - alpha[:, :, None] * LOGO_VALUE) / (1.0 - alpha[:, :, None])
                region[active] = np.clip(restored[active], 0, 255)
                frame[box["y"] : box["y"] + box["height"], box["x"] : box["x"] + box["width"]] = region.astype(np.uint8)
            writer.write(frame)
            processed += 1
            if progress_callback:
                progress_callback(processed, frames_to_process)
            elif frames_to_process and processed % 30 == 0:
                print(f"Processed {processed}/{frames_to_process} frames", end="\r")
    finally:
        capture.release()
        writer.release()

    print(f"Processed {processed} frames. Muxing audio...       ")
    ffmpeg_bin = get_ffmpeg_binary()
    if ffmpeg_bin is not None:
        try:
            try:
                subprocess.run(
                    [
                        ffmpeg_bin, "-y",
                        "-i", str(temporary_path),
                        "-t", f"{target_duration:.3f}",
                        "-i", str(input_path),
                        "-map", "0:v:0?", "-map", "1:a?",
                        "-c:v", "libx264", "-crf", str(crf),
                        "-preset", preset_speed,
                        "-pix_fmt", "yuv420p",
                        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                        "-c:a", "copy",
                        "-movflags", "+faststart",
                        "-t", f"{target_duration:.3f}",
                        str(output_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except subprocess.CalledProcessError:
                subprocess.run(
                    [
                        ffmpeg_bin, "-y", "-i", str(temporary_path),
                        "-c:v", "libx264", "-crf", str(crf), "-preset", preset_speed,
                        "-pix_fmt", "yuv420p",
                        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                        "-movflags", "+faststart",
                        "-t", f"{target_duration:.3f}",
                        str(output_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
        except Exception:
            shutil.copyfile(temporary_path, output_path)
    else:
        shutil.copyfile(temporary_path, output_path)

    temporary_path.unlink(missing_ok=True)
    return box


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove a Gemini sparkle watermark and/or SynthID and provenance fingerprints from an image or video."
    )
    parser.add_argument("input", type=Path, help="Input image or video")
    parser.add_argument("output", type=Path, nargs="?", default=None, help="Output PNG or MP4 path")
    parser.add_argument("--gain", type=float, default=0.6, help="Watermark strength")
    parser.add_argument("--size-scale", type=float, default=1.0)
    parser.add_argument("--offset-x", type=int, default=None)
    parser.add_argument("--offset-y", type=int, default=None)
    parser.add_argument(
        "--preset",
        choices=("auto", "veo", "veo_inset", "veo_standard", "veo_compact", "corner", "veo_text", "notebooklm"),
        default="auto",
    )
    parser.add_argument("--method", choices=("math", "inpaint", "reconstruct"), default="inpaint")
    parser.add_argument("--mask", type=Path, help="Optional custom logo mask image")
    parser.add_argument(
        "--synthid-mode",
        choices=("none", "safe", "paranoid", "nuclear"),
        default="none",
        help="SynthID & fingerprint cleaning tier (safe=lossless C2PA strip, paranoid=quantization reset, nuclear=frequency disruption)",
    )
    parser.add_argument("--strip-metadata", action="store_true", help="Strip C2PA and non-essential container metadata")
    parser.add_argument("--trim-end", type=float, default=None, help="Seconds to trim from end of video (default: 3.0 for notebooklm, 0.0 otherwise)")
    parser.add_argument("--no-trim", dest="trim_end", action="store_const", const=0.0, help="Do not trim the end of the video")
    parser.add_argument("--crf", type=int, default=12, help="H.264 Constant Rate Factor: 0=lossless, 12=ultra-master (default: 12)")
    parser.add_argument("--preset-speed", choices=["veryslow", "slow", "medium", "fast"], default="slow", help="H.264 motion search preset (default: slow for max fidelity)")
    parser.add_argument("--lossless", action="store_true", help="Enable 100% mathematically lossless x264 encoding (CRF 0)")
    parser.add_argument("--inspect", action="store_true", help="Inspect file for C2PA manifests, AI prompts, and EXIF fingerprints")
    args = parser.parse_args()

    effective_crf = 0 if args.lossless else args.crf

    if args.inspect:
        if inspect_image_fingerprints is None:
            print("Fingerprint inspector module not available.")
            return
        data = args.input.read_bytes()
        rep = inspect_image_fingerprints(data, args.input.name)
        print(f"File: {args.input.name} | Format: {rep['format']} | Findings: {rep['finding_count']} | Clean: {rep['is_clean']}")
        for f in rep["findings"]:
            print(f" • [{f['severity'].upper()}] {f['name']} ({f['category']}): {f['detail']}")
        return

    if not args.output:
        parser.error("Output path is required when not using --inspect")

    if args.gain <= 0 or args.size_scale <= 0:
        parser.error("--gain and --size-scale must be greater than zero")

    video_extensions = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    if args.input.suffix.lower() in video_extensions:
        remove_watermark_from_video(
            args.input,
            args.output,
            args.mask or Path(__file__).parent / "assets" / "bg_96.png",
            gain=args.gain,
            size_scale=args.size_scale,
            offset_x=args.offset_x,
            offset_y=args.offset_y,
            preset=args.preset,
            method=args.method,
            trim_end_seconds=args.trim_end,
            crf=effective_crf,
            preset_speed=args.preset_speed,
        )
        return

    with Image.open(args.input) as image:
        base = watermark_info(*image.size)
        default_mask_name = "bg_48.png" if base["size"] <= 48 else "bg_96.png"
        mask_path = args.mask or Path(__file__).parent / "assets" / default_mask_name
        with Image.open(mask_path) as mask:
            cleaned = remove_watermark(
                image,
                mask,
                gain=args.gain,
                size_scale=args.size_scale,
                offset_x=args.offset_x if args.offset_x is not None else 0,
                offset_y=args.offset_y if args.offset_y is not None else 0,
                method=args.method,
                synthid_mode=args.synthid_mode,
                strip_metadata=args.strip_metadata,
            )
        cleaned.save(args.output, "PNG")


if __name__ == "__main__":
    main()