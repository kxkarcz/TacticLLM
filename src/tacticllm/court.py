"""Geometria kortu do padla i przeliczanie pikseli na metry.

Układ współrzędnych kortu (metry), widziany z kamery transmisyjnej:
    x: 0 (lewa ściana z perspektywy kamery) .. 10 (prawa ściana)
    y: 0 (dalsza linia końcowa) .. 20 (bliższa linia końcowa), siatka na y = 10
"""

from __future__ import annotations

import numpy as np

COURT_WIDTH = 10.0
COURT_LENGTH = 20.0
NET_Y = 10.0
SERVICE_LINE_FROM_NET = 6.95

# Kolejność klikania punktów przy kalibracji.
CORNER_NAMES = ("dalszy-lewy", "dalszy-prawy", "bliższy-prawy", "bliższy-lewy")

_FAR_SERVICE = NET_Y - SERVICE_LINE_FROM_NET
_NEAR_SERVICE = NET_Y + SERVICE_LINE_FROM_NET
# Zestawy punktów referencyjnych (metry). "corners" = narożniki podłogi kortu,
# "service" = końce linii serwisowych przy ścianach bocznych (gdy narożniki
# są poza kadrem albo zasłonięte).
REFERENCE_POINTS_M = {
    "corners": np.array([[0.0, 0.0], [COURT_WIDTH, 0.0], [COURT_WIDTH, COURT_LENGTH], [0.0, COURT_LENGTH]]),
    "service": np.array([[0.0, _FAR_SERVICE], [COURT_WIDTH, _FAR_SERVICE],
                         [COURT_WIDTH, _NEAR_SERVICE], [0.0, _NEAR_SERVICE]]),
}


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Macierz 3x3 H taka, że dst ~ H @ src (DLT dla 4+ punktów)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 4:
        raise ValueError("Potrzeba co najmniej 4 par punktów")
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    ones = np.ones((pts.shape[0], 1))
    proj = np.hstack([pts, ones]) @ h.T
    return proj[:, :2] / proj[:, 2:3]


class Court:
    """Kalibracja jednej kamery: 4 punkty referencyjne w pikselach -> metry."""

    def __init__(self, corners_px: np.ndarray, kind: str = "corners"):
        self.corners_px = np.asarray(corners_px, dtype=np.float64)
        self.h = homography(self.corners_px, REFERENCE_POINTS_M[kind])
        self.h_inv = np.linalg.inv(self.h)

    def to_court(self, points_px: np.ndarray) -> np.ndarray:
        return apply_homography(self.h, points_px)

    def to_image(self, points_m: np.ndarray) -> np.ndarray:
        return apply_homography(self.h_inv, points_m)


def inside_court(xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
    xy = np.atleast_2d(xy)
    return (
        (xy[:, 0] >= -margin)
        & (xy[:, 0] <= COURT_WIDTH + margin)
        & (xy[:, 1] >= -margin)
        & (xy[:, 1] <= COURT_LENGTH + margin)
    )


def distance_to_net(y: float | np.ndarray) -> float | np.ndarray:
    return np.abs(np.asarray(y) - NET_Y)


def side_of(y: float) -> str:
    """'near' = połowa bliżej kamery, 'far' = dalsza połowa."""
    return "near" if y >= NET_Y else "far"


def lane_of(x: float, side: str) -> str:
    """Strona gry z perspektywy zawodnika: 'drive' (prawa) albo 'reves' (lewa).

    Bliższa para stoi plecami do kamery, więc jej prawa = prawa obrazu.
    Dalsza para stoi twarzą do kamery, więc jej prawa = lewa obrazu.
    """
    right_of_image = x >= COURT_WIDTH / 2
    if side == "near":
        return "drive" if right_of_image else "reves"
    return "reves" if right_of_image else "drive"
