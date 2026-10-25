import numpy as np

from tacticllm.court import Court, lane_of, side_of

# perspektywa jak z kamery transmisyjnej: dalsza linia węższa i wyżej
CORNERS_PX = np.array([[700, 200], [1220, 200], [1600, 950], [320, 950]], dtype=float)


def test_corners_map_to_court():
    court = Court(CORNERS_PX)
    np.testing.assert_allclose(court.to_court(CORNERS_PX), [[0, 0], [10, 0], [10, 20], [0, 20]], atol=1e-6)


def test_roundtrip():
    court = Court(CORNERS_PX)
    pts = np.array([[5, 10], [2.5, 16], [8, 3]])
    np.testing.assert_allclose(court.to_court(court.to_image(pts)), pts, atol=1e-6)


def test_service_calibration_matches_corners():
    corners = Court(CORNERS_PX)
    service_px = corners.to_image(np.array([[0, 3.05], [10, 3.05], [10, 16.95], [0, 16.95]]))
    service = Court(service_px, kind="service")
    np.testing.assert_allclose(service.to_court(CORNERS_PX), corners.to_court(CORNERS_PX), atol=1e-6)


def test_sides_and_lanes():
    assert side_of(15) == "near" and side_of(4) == "far"
    # bliższa para: prawa strona obrazu = drive; dalsza: odwrotnie
    assert lane_of(8, "near") == "drive" and lane_of(2, "near") == "reves"
    assert lane_of(8, "far") == "reves" and lane_of(2, "far") == "drive"
