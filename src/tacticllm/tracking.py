"""Detekcja zawodników (YOLO) na ujęciach z głównej kamery i podział na punkty."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.progress import Progress

from .config import settings
from .court import COURT_WIDTH, inside_court, lane_of, side_of
from .models import Calibration, PlayerObs, Segment
from .teams import assign_teams


class MainViewDetector:
    """Czy klatka pochodzi z głównej kamery? Porównanie z klatką referencyjną z kalibracji."""

    size = (96, 54)

    def __init__(self, reference: np.ndarray, threshold: float):
        self.ref = self._signature(reference)
        self.threshold = threshold

    def _signature(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        gray = cv2.cvtColor(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
        return (gray - gray.mean()) / (gray.std() + 1e-6)

    def similarity(self, frame: np.ndarray) -> float:
        return float((self._signature(frame) * self.ref).mean())

    def __call__(self, frame: np.ndarray) -> bool:
        return self.similarity(frame) >= self.threshold


def torso_color(frame: np.ndarray, box: np.ndarray) -> list[float]:
    """Średni kolor koszulki w przestrzeni Lab (do rozróżniania par)."""
    import cv2

    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    crop = frame[int(y1 + 0.2 * h):int(y1 + 0.5 * h), int(x1 + 0.3 * w):int(x1 + 0.7 * w)]
    if crop.size == 0:
        return [0.0, 0.0, 0.0]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return [float(v) for v in np.median(lab, axis=0)]


def frame_observations(t: float, frame: np.ndarray, boxes: np.ndarray, confs: np.ndarray,
                       court, margin: float) -> list[PlayerObs]:
    """Zamienia detekcje osób na maks. 2 zawodników na każdej połowie kortu."""
    if len(boxes) == 0:
        return []
    feet_px = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
    feet_m = court.to_court(feet_px)
    keep = inside_court(feet_m, margin)

    by_side: dict[str, list[int]] = {"near": [], "far": []}
    for i in np.flatnonzero(keep):
        by_side[side_of(feet_m[i, 1])].append(int(i))

    result = []
    for side, idx in by_side.items():
        idx = sorted(idx, key=lambda i: -confs[i])[:2]  # sędzia/ballboy zwykle mają niższą pewność
        idx = sorted(idx, key=lambda i: feet_m[i, 0])
        if len(idx) == 2:
            # para: lewy na obrazie i prawy na obrazie
            lanes = ["reves", "drive"] if side == "near" else ["drive", "reves"]
        else:
            lanes = [lane_of(feet_m[i, 0], side) for i in idx]
        for i, lane in zip(idx, lanes):
            x, y = feet_m[i]
            result.append(PlayerObs(
                t=round(t, 3), side=side, lane=lane,
                x=round(float(np.clip(x, 0, COURT_WIDTH)), 3), y=round(float(y), 3),
                conf=round(float(confs[i]), 3), color=torso_color(frame, boxes[i]),
            ))
    return result


def build_segments(main_flags: list[tuple[float, bool]], max_gap_s: float) -> list[tuple[float, float]]:
    """Ciągi klatek głównej kamery -> przedziały (start, koniec). Krótkie przerwy są sklejane."""
    spans: list[list[float]] = []
    for t, is_main in main_flags:
        if not is_main:
            continue
        if spans and t - spans[-1][1] <= max_gap_s:
            spans[-1][1] = t
        else:
            spans.append([t, t])
    return [(a, b) for a, b in spans]


def track(video: Path, calib: Calibration, reference_frame: np.ndarray) -> list[Segment]:
    from ultralytics import YOLO

    from .video import iter_frames, video_duration

    court = calib.court()
    detector = MainViewDetector(reference_frame, settings.main_view_threshold)
    model = YOLO(settings.yolo_weights)

    flags: list[tuple[float, bool]] = []
    observations: list[PlayerObs] = []
    total = video_duration(video)

    with Progress() as progress:
        task = progress.add_task("Analiza klatek", total=total)
        for t, frame in iter_frames(video, settings.analysis_fps):
            progress.update(task, completed=t)
            is_main = detector(frame)
            flags.append((t, is_main))
            if not is_main:
                continue
            res = model.predict(frame, classes=[0], imgsz=settings.yolo_imgsz, conf=settings.yolo_conf,
                                device=settings.device, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
            confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.empty(0)
            observations.extend(frame_observations(t, frame, boxes, confs, court, settings.court_margin_m))

    spans = build_segments(flags, max_gap_s=3.0 / settings.analysis_fps)
    spans = [s for s in spans if s[1] - s[0] >= settings.min_point_seconds]

    segments = []
    obs_iter = iter(sorted(observations, key=lambda o: o.t))
    pending = next(obs_iter, None)
    for i, (start, end) in enumerate(spans):
        seg = Segment(index=i, start_s=round(start, 3), end_s=round(end, 3))
        while pending is not None and pending.t <= end:
            if pending.t >= start:
                seg.observations.append(pending)
            pending = next(obs_iter, None)
        segments.append(seg)

    assign_teams(segments)
    return segments
