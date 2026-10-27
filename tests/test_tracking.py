import numpy as np
import pytest

from tacticllm import tracking
from tacticllm.court import Court
from tacticllm.tracking import build_segments, frame_observations

CORNERS_PX = np.array([[700, 200], [1220, 200], [1600, 950], [320, 950]], dtype=float)


@pytest.fixture(autouse=True)
def no_cv2_colors(monkeypatch):
    # kolor koszulki wymaga OpenCV i prawdziwego obrazu – tu nie jest istotny
    monkeypatch.setattr(tracking, "torso_color", lambda frame, box: [0.0, 0.0, 0.0])


def boxes_at(court, feet_m):
    """Prostokąty osób, których stopy stoją w podanych punktach kortu (metry)."""
    feet_px = court.to_image(np.array(feet_m, dtype=float))
    return np.array([[x - 20, y - 100, x + 20, y] for x, y in feet_px])


def observe(feet_m, confs):
    court = Court(CORNERS_PX)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    return frame_observations(1.0, frame, boxes_at(court, feet_m), np.array(confs, dtype=float), court, margin=1.5)


def test_no_detections():
    court = Court(CORNERS_PX)
    assert frame_observations(0.0, None, np.empty((0, 4)), np.empty(0), court, 1.5) == []


def test_pairs_get_lanes_by_image_position():
    obs = observe([[2, 15], [8, 16], [3, 5], [7, 4]], [0.9, 0.9, 0.9, 0.9])
    by = {(o.side, o.lane): o for o in obs}
    assert len(obs) == 4
    # bliższa para plecami do kamery: lewy na obrazie = revés; dalsza twarzą: lewy = drive
    assert by[("near", "reves")].x == pytest.approx(2, abs=1e-3)
    assert by[("near", "drive")].x == pytest.approx(8, abs=1e-3)
    assert by[("far", "drive")].x == pytest.approx(3, abs=1e-3)
    assert by[("far", "reves")].x == pytest.approx(7, abs=1e-3)
    assert by[("near", "drive")].y == pytest.approx(16, abs=1e-3)


def test_keeps_two_most_confident_per_side():
    # trzecia osoba po bliższej stronie (np. sędzia) ma najniższą pewność
    obs = observe([[2, 15], [8, 16], [5, 18]], [0.9, 0.8, 0.4])
    assert len(obs) == 2
    assert sorted(o.x for o in obs) == pytest.approx([2, 8], abs=1e-3)


def test_people_outside_court_are_dropped_and_x_clipped():
    # 1 m za ścianą boczną mieści się w marginesie (x przycięty do 0); 5 m za kortem – nie
    obs = observe([[-1, 15], [5, 27]], [0.9, 0.9])
    assert len(obs) == 1
    assert obs[0].x == 0.0 and obs[0].side == "near"


def test_single_player_lane_from_position():
    obs = observe([[8, 15], [8, 5]], [0.9, 0.9])
    lanes = {o.side: o.lane for o in obs}
    assert lanes == {"near": "drive", "far": "reves"}


def test_build_segments_merges_short_gaps():
    flags = [(t / 10, True) for t in range(0, 30)]  # 0.0 – 2.9 s
    flags[12] = (1.2, False)  # jedna zgubiona klatka – nadal ten sam punkt
    flags += [(t / 10, False) for t in range(30, 60)]  # 3 s powtórki
    flags += [(t / 10, True) for t in range(60, 80)]
    spans = build_segments(flags, max_gap_s=0.3)
    assert spans == [(0.0, 2.9), (6.0, 7.9)]


def test_build_segments_empty():
    assert build_segments([(0.0, False), (0.1, False)], max_gap_s=0.3) == []


def test_main_view_detector():
    pytest.importorskip("cv2")
    from tacticllm.tracking import MainViewDetector

    rng = np.random.default_rng(0)
    reference = (rng.random((540, 960, 3)) * 255).astype(np.uint8)
    reference[:, :480] //= 4  # wyraźna struktura obrazu
    other = (rng.random((540, 960, 3)) * 255).astype(np.uint8)
    detector = MainViewDetector(reference, threshold=0.55)
    assert detector(reference)
    assert detector.similarity(reference) == pytest.approx(1.0, abs=1e-3)
    assert not detector(other)
