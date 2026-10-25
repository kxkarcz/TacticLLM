"""Pobieranie fragmentu z YouTube, czytanie klatek i wycinanie klipów."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def parse_time(value: str | float | None) -> float | None:
    """'1:12:30' / '12:30' / '90' / 90.0 -> sekundy."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    seconds = 0.0
    for part in str(value).strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def format_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("Brak ffmpeg w PATH – zainstaluj go (np. `winget install ffmpeg`).")
    return exe


def download(source: str, out_path: Path, start: str | None = None, end: str | None = None,
             max_height: int = 1080) -> Path:
    """Pobiera film (albo tylko fragment start–end) do out_path (mp4).

    `source` może być linkiem (YouTube i inne strony obsługiwane przez yt-dlp)
    albo ścieżką do lokalnego pliku.
    """
    out_path = Path(out_path)
    start_s, end_s = parse_time(start), parse_time(end)

    if Path(source).exists():
        return cut(Path(source), out_path, start_s, end_s)

    _require_ffmpeg()
    import yt_dlp
    from yt_dlp.utils import download_range_func

    opts = {
        "format": f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/b[height<={max_height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_path.with_suffix("")) + ".%(ext)s",
        "quiet": False,
        "noplaylist": True,
    }
    if start_s is not None or end_s is not None:
        opts["download_ranges"] = download_range_func(None, [(start_s or 0.0, end_s or float("inf"))])
        opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([source])

    if not out_path.exists():
        candidates = sorted(out_path.parent.glob(out_path.stem + ".*"))
        if not candidates:
            raise RuntimeError("yt-dlp nie zapisał pliku wideo")
        candidates[0].rename(out_path)
    return out_path


def cut(src: Path, dst: Path, start_s: float | None, end_s: float | None, reencode: bool = True,
        max_height: int | None = None) -> Path:
    """Wycina fragment pliku wideo ffmpegiem (opcjonalnie zmniejszając rozdzielczość)."""
    ffmpeg = _require_ffmpeg()
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    if start_s:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(src)]
    if end_s is not None:
        cmd += ["-t", f"{end_s - (start_s or 0.0):.3f}"]
    if reencode:
        if max_height:
            cmd += ["-vf", f"scale=-2:'min({max_height},ih)'"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an"]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(dst))
    subprocess.run(cmd, check=True)
    return Path(dst)


def read_frame(video: Path, t: float) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Nie udało się odczytać klatki w {t:.1f}s")
    return frame


def iter_frames(video: Path, fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Klatki próbkowane co 1/fps sekundy: (czas_s, obraz BGR)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(native_fps / fps))
    index = 0
    try:
        while True:
            if index % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield index / native_fps, frame
            elif not cap.grab():
                break
            index += 1
    finally:
        cap.release()


def video_duration(video: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video))
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return frames / fps
