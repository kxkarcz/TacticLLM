"""Która para stoi po której stronie w każdym punkcie.

Pary zmieniają strony co nieparzysty gem, więc sama strona kortu nie mówi,
kto jest kim. W obrębie jednego punktu pary się nie zamieniają, więc
porównujemy średni kolor koszulek bliższej i dalszej pary z kolorami drużyn
A i B (A = para, która w pierwszym punkcie stoi bliżej kamery).
"""

from __future__ import annotations

import numpy as np

from .models import Segment


def _side_color(seg: Segment, side: str) -> np.ndarray | None:
    colors = [o.color for o in seg.observations if o.side == side]
    return np.median(np.array(colors), axis=0) if colors else None


def assign_teams(segments: list[Segment], momentum: float = 0.9) -> None:
    """Ustawia seg.near_team ('A' albo 'B') w miejscu."""
    color_a = color_b = None
    for seg in segments:
        near, far = _side_color(seg, "near"), _side_color(seg, "far")
        if near is None or far is None:
            continue
        if color_a is None:
            color_a, color_b = near, far
            seg.near_team = "A"
            continue
        keep = np.linalg.norm(near - color_a) + np.linalg.norm(far - color_b)
        swap = np.linalg.norm(near - color_b) + np.linalg.norm(far - color_a)
        seg.near_team = "A" if keep <= swap else "B"
        a_now, b_now = (near, far) if seg.near_team == "A" else (far, near)
        color_a = momentum * color_a + (1 - momentum) * a_now
        color_b = momentum * color_b + (1 - momentum) * b_now

    # punkty bez obu par w kadrze: weź stronę z sąsiedniego punktu
    last = "A"
    for seg in segments:
        if seg.near_team is None:
            seg.near_team = last
        last = seg.near_team


def team_of_side(seg: Segment, side: str) -> str:
    near = seg.near_team or "A"
    return near if side == "near" else ("B" if near == "A" else "A")
