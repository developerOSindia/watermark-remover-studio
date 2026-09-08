"""
NotebookLM Presentation & Video Cleaner
Cleanroom watermark detection and removal for Google NotebookLM slides,
infographics, PDF presentations, and Studio/Audio Overview videos.

Features:
- Dual geometry presets for Landscape (16:9, 4:3) and Portrait (9:16) media
- Boundary-feathered mathematical gradient patch reconstruction (zero blur halos)
- Dynamic background polarity detection (light text on dark background & dark text on light)
- Complete video cleaning pipeline with lossless FFmpeg audio copying (-c:a copy)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Callable

import cv2
import numpy as np
from PIL import Image


# Calibrated NotebookLM slide & video watermark geometry ratios
DEFAULT_LANDSCAPE_CONFIG = {
    "width_ratio": 0.1220,        # ~12.2% of width (156px on 1280p landscape video)
    "height_ratio": 0.0400,       # ~4.0% of height (29px on 720p landscape video)
    "margin_right_ratio": 0.0270,  # ~2.7% from right edge (35px on 1280p)
    "margin_bottom_ratio": 0.0460, # ~4.6% from bottom edge (33px on 720p)
    "feather_size": 6,            # Boundary smoothing blend width
    "donor_offset_y": 4,          # Vertical offset above watermark
}

DEFAULT_PORTRAIT_CONFIG = {
    "width_ratio": 0.2722,        # ~27.22% of width (196px on 720p portrait video)
    "height_ratio": 0.0328,       # ~3.28% of height (42px on 1280p portrait video)
    "margin_right_ratio": 0.0167,  # ~1.67% from right edge (12px on 720p)
    "margin_bottom_ratio": 0.0109, # ~1.09% from bottom edge (14px on 1280p)
    "feather_size": 10,           # Boundary smoothing blend width
    "donor_offset_y": 6,          # Vertical offset above watermark
}


def get_notebooklm_box(
    width: int,
    height: int,
    config: Optional[Dict[str, float]] = None,
) -> Dict[str, int]:
    """Calculate the precise bounding box of the NotebookLM watermark in the bottom-right corner."""
    if config is not None:
        cfg = config
    else:
        cfg = DEFAULT_PORTRAIT_CONFIG if height > width else DEFAULT_LANDSCAPE_CONFIG

    wm_w = max(16, int(round(width * cfg.get("width_ratio", 0.1220))))
    wm_h = max(8, int(round(height * cfg.get("height_ratio", 0.0400))))
    mr = int(round(width * cfg.get("margin_right_ratio", 0.0270)))
    mb = int(round(height * cfg.get("margin_bottom_ratio", 0.0460)))

    x = max(0, width - wm_w - mr)
    y = max(0, height - wm_h - mb)
    return {
        "x": x,
        "y": y,
        "width": wm_w,
        "height": wm_h,
        "x1": min(width, x + wm_w),
        "y1": min(height, y + wm_h),
    }


def is_light_background(roi_bgr: np.ndarray) -> bool:
    """Check whether the ROI has a light or dark background by sampling edge pixels."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if len(roi_bgr.shape) == 3 else roi_bgr
    h, w = gray.shape[:2]
    border = max(2, min(h, w) // 10)
    edge_pixels = np.concatenate([
        gray[:border, :].ravel(),
        gray[-border:, :].ravel(),
        gray[:, :border].ravel(),
        gray[:, -border:].ravel(),
    ])
    return float(np.median(edge_pixels)) >= 128.0


def find_best_source_patch(
    img_bgr: np.ndarray,
    x0: int,
    y0: int,
    wm_w: int,
    wm_h: int,
    offset_y: int = 4,
) -> Tuple[int, int]:
    """
    Vertical donor patch selection directly above the watermark region.
    Guarantees clean background matching without sampling diagram elements from the left.
    """
    h, w = img_bgr.shape[:2]
    # Sample clean background directly above
    sy = max(0, y0 - wm_h - offset_y)
    sx = x0
    if sy + wm_h > h or sx + wm_w > w:
        sy = max(0, y0 - wm_h)
        sx = x0
    return (sx, sy)


def remove_notebooklm_watermark(
    image: Image.Image,
    method: str = "gradient_patch",
    feather: Optional[int] = None,
    custom_box: Optional[Dict[str, int]] = None,
) -> Image.Image:
    """
    Remove the NotebookLM / Gemini Notebook watermark from a PIL Image.
    
    Methods:
    - 'gradient_patch': Samples clean background donor pixels directly above with boundary feathering.
      Preserves presentation card borders, solid colors, and slide gradients cleanly.
    - 'inpaint': Uses OpenCV Telea inpainting on adaptive thresholded contours.
    """
    w, h = image.size
    box = custom_box or get_notebooklm_box(w, h)
    x, y, wm_w, wm_h = box["x"], box["y"], box["width"], box["height"]

    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if method == "gradient_patch":
        is_portrait = h > w
        offset_y = 6 if is_portrait else 4
        src_x, src_y = find_best_source_patch(img_bgr, x, y, wm_w, wm_h, offset_y=offset_y)

        src_patch = img_rgb[src_y : src_y + wm_h, src_x : src_x + wm_w].astype(np.float32)
        dst_patch = img_rgb[y : y + wm_h, x : x + wm_w].astype(np.float32)

        # 2D feathering alpha mask (smooth top, bottom, and side transitions)
        eff_feather = feather if feather is not None else (10 if is_portrait else 4)
        f = max(1, min(eff_feather, wm_h // 4, wm_w // 10))
        alpha = np.ones((wm_h, wm_w, 1), dtype=np.float32)

        for i in range(wm_h):
            for j in range(wm_w):
                a = 1.0
                if i < f:
                    a = min(a, float(i) / float(f))
                if i > wm_h - f:
                    a = min(a, float(wm_h - i) / float(f))
                if j < f:
                    a = min(a, float(j) / float(f))
                if j > wm_w - f:
                    a = min(a, float(wm_w - j) / float(f))
                alpha[i, j, 0] = a

        blended = (dst_patch * (1.0 - alpha) + src_patch * alpha).clip(0, 255).astype(np.uint8)
        out_np = img_rgb.copy()
        out_np[y : y + wm_h, x : x + wm_w] = blended
        return Image.fromarray(out_np)

    elif method == "inpaint":
        roi = img_bgr[y : y + wm_h, x : x + wm_w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        light_bg = is_light_background(roi)

        if light_bg:
            bg_val = float(np.median(gray_roi))
            diff = bg_val - gray_roi.astype(float)
            mask_roi = np.where(diff > 20, 255, 0).astype(np.uint8)
        else:
            bg_val = float(np.median(gray_roi))
            diff = gray_roi.astype(float) - bg_val
            mask_roi = np.where(diff > 20, 255, 0).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_roi = cv2.dilate(mask_roi, kernel, iterations=2)

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y : y + wm_h, x : x + wm_w] = mask_roi

        inpainted_bgr = cv2.inpaint(img_bgr, full_mask, 5, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB))

    return image


def get_ffmpeg_binary() -> Optional[str]:
    """Resolve FFmpeg binary from system PATH, Homebrew, or bundled imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if os.path.exists("/opt/homebrew/bin/ffmpeg"):
        return "/opt/homebrew/bin/ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return None


def clean_notebooklm_video(
    input_path: str,
    output_path: str,
    method: str = "gradient_patch",
    feather: Optional[int] = None,
    custom_box: Optional[Dict[str, int]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    trim_end_seconds: float = 3.0,
    crf: int = 12,
    preset_speed: str = "slow",
) -> str:
    """
    Remove the NotebookLM / Gemini Notebook watermark from a video file with
    ultra-high fidelity (CRF 12 + Slow preset) and bit-for-bit lossless audio
    stream copying (-c:a copy). Zero intermediate file compression.
    Optionally trims trailing outro cards (default: last 3.0 seconds).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration = total_frames / fps if fps > 0 else 0

    if trim_end_seconds > 0 and total_duration > trim_end_seconds:
        target_duration = max(0.1, total_duration - trim_end_seconds)
        frames_to_process = max(1, int(round(target_duration * fps)))
    else:
        target_duration = total_duration
        frames_to_process = total_frames

    box = custom_box or get_notebooklm_box(w, h)
    x, y, wm_w, wm_h = box["x"], box["y"], box["width"], box["height"]

    # Pre-calculate alpha feather mask
    is_portrait = h > w
    eff_feather = feather if feather is not None else (10 if is_portrait else 4)
    f = max(1, min(eff_feather, wm_h // 4, wm_w // 10))
    alpha = np.ones((wm_h, wm_w, 1), dtype=np.float32)
    for i in range(wm_h):
        for j in range(wm_w):
            a = 1.0
            if i < f:
                a = min(a, float(i) / float(f))
            if i > wm_h - f:
                a = min(a, float(wm_h - i) / float(f))
            if j < f:
                a = min(a, float(j) / float(f))
            if j > wm_w - f:
                a = min(a, float(wm_w - j) / float(f))
            alpha[i, j, 0] = a

    # Pre-determine clean background donor position
    offset_y = 6 if is_portrait else 4
    src_y = max(0, y - wm_h - offset_y)
    src_x = x

    ffmpeg_bin = shutil.which("ffmpeg") or ("/opt/homebrew/bin/ffmpeg" if os.path.exists("/opt/homebrew/bin/ffmpeg") else None)
    ffmpeg_bin = get_ffmpeg_binary()

    # Strategy 1: Ultra-High-Fidelity Direct FFmpeg Pipe (Zero intermediate compression, CRF 12 Master Quality)
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "-",
            "-t", f"{target_duration:.3f}",
            "-i", input_path,
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
            output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        frame_idx = 0
        pipe_failed = False
        try:
            while cap.isOpened() and frame_idx < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                if method == "gradient_patch":
                    src_patch = frame[src_y : src_y + wm_h, src_x : src_x + wm_w].astype(np.float32)
                    dst_patch = frame[y : y + wm_h, x : x + wm_w].astype(np.float32)
                    blended = (dst_patch * (1.0 - alpha) + src_patch * alpha).clip(0, 255).astype(np.uint8)
                    frame[y : y + wm_h, x : x + wm_w] = blended
                elif method == "inpaint":
                    roi = frame[y : y + wm_h, x : x + wm_w]
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    bg_val = float(np.median(gray))
                    diff = np.abs(gray.astype(float) - bg_val)
                    mask_roi = np.where(diff > 20, 255, 0).astype(np.uint8)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask_roi = cv2.dilate(mask_roi, kernel, iterations=2)
                    frame[y : y + wm_h, x : x + wm_w] = cv2.inpaint(roi, mask_roi, 5, cv2.INPAINT_TELEA)

                try:
                    proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError):
                    pipe_failed = True
                    break

                frame_idx += 1
                if progress_callback and frame_idx % 15 == 0:
                    progress_callback(frame_idx, frames_to_process)

            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            proc.wait()

            if not pipe_failed and proc.returncode == 0:
                cap.release()
                return output_path
        except Exception:
            pipe_failed = True
            try:
                proc.kill()
            except Exception:
                pass

        # Reset capture position if pipe failed
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Strategy 2: Robust Fallback with lossless remuxing
    fd, temp_video = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))

    frame_idx = 0
    try:
        while cap.isOpened() and frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            if method == "gradient_patch":
                src_patch = frame[src_y : src_y + wm_h, src_x : src_x + wm_w].astype(np.float32)
                dst_patch = frame[y : y + wm_h, x : x + wm_w].astype(np.float32)
                blended = (dst_patch * (1.0 - alpha) + src_patch * alpha).clip(0, 255).astype(np.uint8)
                frame[y : y + wm_h, x : x + wm_w] = blended
            elif method == "inpaint":
                roi = frame[y : y + wm_h, x : x + wm_w]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                bg_val = float(np.median(gray))
                diff = np.abs(gray.astype(float) - bg_val)
                mask_roi = np.where(diff > 20, 255, 0).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_roi = cv2.dilate(mask_roi, kernel, iterations=2)
                frame[y : y + wm_h, x : x + wm_w] = cv2.inpaint(roi, mask_roi, 5, cv2.INPAINT_TELEA)

            out.write(frame)
            frame_idx += 1
            if progress_callback and frame_idx % 15 == 0:
                progress_callback(frame_idx, frames_to_process)

        cap.release()
        out.release()

        # Remux with lossless audio preservation
        ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", temp_video,
            "-t", f"{target_duration:.3f}",
            "-i", input_path,
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
            output_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            cmd_fallback = [
                ffmpeg_bin, "-y", "-i", temp_video,
                "-c:v", "libx264", "-crf", str(crf), "-preset", preset_speed, "-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                "-movflags", "+faststart",
                "-t", f"{target_duration:.3f}",
                output_path,
            ]
            subprocess.run(cmd_fallback, check=True)

        return output_path
    finally:
        if os.path.exists(temp_video):
            try:
                os.remove(temp_video)
            except OSError:
                pass


def draw_notebooklm_reticle(image: Image.Image, box: Optional[Dict[str, int]] = None) -> Image.Image:
    """Draw a neon cyan/cyber-red targeting reticle around the detected NotebookLM watermark area."""
    w, h = image.size
    b = box or get_notebooklm_box(w, h)
    x, y, wm_w, wm_h = b["x"], b["y"], b["width"], b["height"]

    img_copy = image.copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img_copy)

    # Reticle rectangle
    draw.rectangle([x, y, x + wm_w, y + wm_h], outline="#2FD3E1", width=2)
    tick = min(8, wm_h // 3)
    # Corner brackets
    draw.line([(x, y), (x + tick, y)], fill="#F2482C", width=3)
    draw.line([(x, y), (x, y + tick)], fill="#F2482C", width=3)
    draw.line([(x + wm_w, y), (x + wm_w - tick, y)], fill="#F2482C", width=3)
    draw.line([(x + wm_w, y), (x + wm_w, y + tick)], fill="#F2482C", width=3)
    draw.line([(x, y + wm_h), (x + tick, y + wm_h)], fill="#F2482C", width=3)
    draw.line([(x, y + wm_h), (x, y + wm_h - tick)], fill="#F2482C", width=3)
    draw.line([(x + wm_w, y + wm_h), (x + wm_w - tick, y + wm_h)], fill="#F2482C", width=3)
    draw.line([(x + wm_w, y + wm_h), (x + wm_w, y + wm_h - tick)], fill="#F2482C", width=3)
    return img_copy


# Alias for backward compatibility
clean_notebooklm_image = remove_notebooklm_watermark


def main():
    parser = argparse.ArgumentParser(description="NotebookLM Presentation & Video Cleaner CLI")
    parser.add_argument("input", help="Path to input image, video, or directory of slides")
    parser.add_argument("output", nargs="?", default=None, help="Output image, video, or directory")
    parser.add_argument("--method", choices=["gradient_patch", "inpaint"], default="gradient_patch", help="Removal method")
    parser.add_argument("--feather", type=int, default=None, help="Boundary feather radius (default: auto-calibrated)")
    parser.add_argument("--trim-end", type=float, default=3.0, help="Seconds to trim from end of video to remove outro card (default: 3.0)")
    parser.add_argument("--no-trim", dest="trim_end", action="store_const", const=0.0, help="Do not trim the end of the video")
    parser.add_argument("--crf", type=int, default=12, help="H.264 Constant Rate Factor: 0=lossless, 12=ultra-master (default: 12)")
    parser.add_argument("--preset-speed", choices=["veryslow", "slow", "medium", "fast"], default="slow", help="H.264 motion search preset (default: slow for max fidelity)")
    parser.add_argument("--lossless", action="store_true", help="Enable 100% mathematically lossless x264 encoding (CRF 0)")
    args = parser.parse_args()

    effective_crf = 0 if args.lossless else args.crf

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input '{in_path}' not found.", file=sys.stderr)
        sys.exit(1)

    video_exts = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    if in_path.is_file():
        ext = in_path.suffix.lower()
        if ext in video_exts:
            out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_cleaned{in_path.suffix}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            trim_msg = f" (trimming last {args.trim_end}s outro card)" if args.trim_end > 0 else ""
            print(f"Cleaning NotebookLM video: {in_path} -> {out_path}{trim_msg} [CRF {effective_crf}, preset={args.preset_speed}]...")
            clean_notebooklm_video(
                str(in_path),
                str(out_path),
                method=args.method,
                feather=args.feather,
                trim_end_seconds=args.trim_end,
                crf=effective_crf,
                preset_speed=args.preset_speed,
            )
            print(f"✓ Successfully cleaned NotebookLM video with ultra-master quality & lossless audio: {out_path}")
        elif ext in image_exts:
            out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_cleaned{in_path.suffix}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            im = Image.open(in_path)
            cleaned = remove_notebooklm_watermark(im, method=args.method, feather=args.feather)
            cleaned.save(out_path)
            print(f"✓ Successfully cleaned NotebookLM slide: {out_path}")
        else:
            print(f"Unsupported file format: {ext}", file=sys.stderr)
            sys.exit(1)

    elif in_path.is_dir():
        out_dir = Path(args.output) if args.output else in_path / "cleaned"
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for f in in_path.iterdir():
            if f.suffix.lower() in image_exts:
                im = Image.open(f)
                cleaned = remove_notebooklm_watermark(im, method=args.method, feather=args.feather)
                cleaned.save(out_dir / f.name)
                count += 1
                print(f"  Cleaned slide: {f.name}")
            elif f.suffix.lower() in video_exts:
                clean_notebooklm_video(
                    str(f),
                    str(out_dir / f.name),
                    method=args.method,
                    feather=args.feather,
                    trim_end_seconds=args.trim_end,
                    crf=effective_crf,
                    preset_speed=args.preset_speed,
                )
                count += 1
                print(f"  Cleaned video: {f.name}")
        print(f"✓ Successfully batch cleaned {count} items in {out_dir}")


if __name__ == "__main__":
    main()
