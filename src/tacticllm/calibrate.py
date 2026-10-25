"""Kalibracja: 4 narożniki kortu na klatce z głównej kamery transmisyjnej.

Kamera transmisyjna w padlu prawie się nie rusza, więc wystarczy zrobić to raz
na film. Ta sama klatka służy potem do rozpoznawania, czy dane ujęcie to główna
kamera (a nie powtórka / zbliżenie / publiczność).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .court import CORNER_NAMES
from .models import Calibration


def parse_corners(text: str) -> list[list[float]]:
    """'x1,y1;x2,y2;x3,y3;x4,y4' -> [[x1,y1], ...]."""
    corners = [[float(v) for v in pair.split(",")] for pair in text.split(";") if pair.strip()]
    if len(corners) != 4 or any(len(c) != 2 for c in corners):
        raise ValueError("Podaj dokładnie 4 punkty w formacie x,y;x,y;x,y;x,y")
    return corners


def click_corners(frame: np.ndarray, kind: str = "corners") -> list[list[float]]:
    """Okno OpenCV: kliknij punkty w kolejności CORNER_NAMES. Backspace = cofnij, Enter = OK."""
    import cv2

    points: list[list[float]] = []
    scale = min(1.0, 1600 / frame.shape[1])
    shown = cv2.resize(frame, None, fx=scale, fy=scale)
    window = "Kalibracja kortu"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x / scale, y / scale])

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = shown.copy()
        for i, (px, py) in enumerate(points):
            cv2.circle(canvas, (int(px * scale), int(py * scale)), 6, (0, 255, 255), -1)
            cv2.putText(canvas, CORNER_NAMES[i], (int(px * scale) + 8, int(py * scale) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(points) > 1:
            poly = (np.array(points) * scale).astype(np.int32)
            cv2.polylines(canvas, [poly], len(points) == 4, (0, 255, 0), 2)
        what = "narożnik podłogi kortu" if kind == "corners" else "koniec linii serwisowej przy ścianie"
        hint = (f"Kliknij: {CORNER_NAMES[len(points)]} ({what})" if len(points) < 4
                else "Enter = zapisz, Backspace = cofnij")
        cv2.putText(canvas, hint, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 8 and points:
            points.pop()
        elif key in (13, 10) and len(points) == 4:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Kalibracja przerwana")
    cv2.destroyAllWindows()
    return points


def calibrate(video: Path, out_json: Path, out_frame: Path, at_time: float,
              corners: str | None = None, kind: str = "corners") -> Calibration:
    import cv2

    from .video import read_frame

    frame = read_frame(video, at_time)
    cv2.imwrite(str(out_frame), frame)
    corner_list = parse_corners(corners) if corners else click_corners(frame, kind)
    calib = Calibration(corners_px=corner_list, kind=kind, reference_time_s=at_time)
    out_json.write_text(calib.model_dump_json(indent=2), encoding="utf-8")
    return calib


def draw_overlay(frame: np.ndarray, calib: Calibration) -> np.ndarray:
    """Podgląd: linie kortu wyliczone z kalibracji (do sprawdzenia, czy jest OK)."""
    import cv2

    from .court import COURT_LENGTH, COURT_WIDTH, NET_Y, SERVICE_LINE_FROM_NET

    court = calib.court()
    lines_m = [
        [(0, 0), (COURT_WIDTH, 0)], [(0, COURT_LENGTH), (COURT_WIDTH, COURT_LENGTH)],
        [(0, 0), (0, COURT_LENGTH)], [(COURT_WIDTH, 0), (COURT_WIDTH, COURT_LENGTH)],
        [(0, NET_Y), (COURT_WIDTH, NET_Y)],
        [(0, NET_Y - SERVICE_LINE_FROM_NET), (COURT_WIDTH, NET_Y - SERVICE_LINE_FROM_NET)],
        [(0, NET_Y + SERVICE_LINE_FROM_NET), (COURT_WIDTH, NET_Y + SERVICE_LINE_FROM_NET)],
        [(COURT_WIDTH / 2, NET_Y - SERVICE_LINE_FROM_NET), (COURT_WIDTH / 2, NET_Y + SERVICE_LINE_FROM_NET)],
    ]
    out = frame.copy()
    for a, b in lines_m:
        pa, pb = court.to_image(np.array([a, b]))
        cv2.line(out, tuple(map(int, pa)), tuple(map(int, pb)), (0, 255, 0), 2)
    return out
