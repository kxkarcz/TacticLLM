from tacticllm.models import MatchInfo, PlayerObs, PointAnnotation, Segment, TeamProfile
from tacticllm.stats import compute_stats, formation, point_features
from tacticllm.teams import assign_teams

RED, BLUE = [150.0, 180.0, 160.0], [80.0, 130.0, 90.0]


def obs(t, side, lane, x, y, color):
    return PlayerObs(t=t, side=side, lane=lane, x=x, y=y, conf=0.9, color=color)


def make_segment(index, near_color, far_color, near_y, far_y, start=0.0):
    """Punkt 4 s: bliższa para na głębokości near_y(t), dalsza far_y(t)."""
    o = []
    for k in range(40):
        t = start + k * 0.1
        o += [obs(t, "near", "reves", 3, near_y(k), near_color), obs(t, "near", "drive", 7, near_y(k), near_color),
              obs(t, "far", "drive", 3, far_y(k), far_color), obs(t, "far", "reves", 7, far_y(k), far_color)]
    return Segment(index=index, start_s=start, end_s=start + 3.9, observations=o)


def test_formation():
    assert formation([2, 3]) == "net"
    assert formation([8, 9]) == "back"
    assert formation([2, 9]) == "split"
    assert formation([5, 6]) == "transition"
    assert formation([3]) == "unknown"


def test_teams_follow_shirt_colors_across_end_change():
    seg1 = make_segment(0, RED, BLUE, lambda k: 12, lambda k: 8)
    seg2 = make_segment(1, BLUE, RED, lambda k: 12, lambda k: 8, start=10)  # zmiana stron
    assign_teams([seg1, seg2])
    assert seg1.near_team == "A" and seg2.near_team == "B"


def test_net_gain_after_moving_forward():
    # bliższa para zaczyna z tyłu (głębokość 9 m) i w 2 s biegnie do siatki (2 m)
    seg = make_segment(0, RED, BLUE, lambda k: 19 - 7 * min(max(k - 10, 0), 20) / 20, lambda k: 1)
    seg.near_team = "A"
    f = point_features(seg)
    assert f["A"]["net_gains"] == 1 and f["A"]["end_formation"] == "net"
    assert f["B"]["end_formation"] == "back"  # dalsza para przy linii końcowej
    assert f["A"]["players"]["drive"]["distance_m"] == 7.0


def test_detection_jumps_are_not_counted_as_running():
    seg = make_segment(0, RED, BLUE, lambda k: 19 if k < 20 else 12, lambda k: 1)  # 7 m w 0.1 s
    seg.near_team = "A"
    assert point_features(seg)["A"]["players"]["drive"]["distance_m"] == 0.0


def test_compute_stats_with_annotations():
    seg = make_segment(0, RED, BLUE, lambda k: 12, lambda k: 1)
    seg2 = make_segment(1, BLUE, RED, lambda k: 12, lambda k: 1, start=10)
    assign_teams([seg, seg2])
    anns = {
        0: PointAnnotation(is_rally=True, winner_side="near", ending="winner", last_shot="volley", server_side="near",
                           tactics_near=["middle_ball", "chiquita"], tactics_far=["lob_to_take_net"]),
        1: PointAnnotation(is_rally=True, winner_side="far", ending="unforced_error", last_shot="lob",
                           tactics_near=["lob_to_take_net"], tactics_far=["middle_ball"]),
    }
    info = MatchInfo(teams=[TeamProfile(name="Czerwoni", start_side="near"), TeamProfile(name="Niebiescy", start_side="far")])
    stats = compute_stats([seg, seg2], anns, info)
    a = stats["overall"]["A"]
    # punkt 0: A przy siatce wygrywa; punkt 1: A (teraz daleko, z tyłu) wygrywa po błędzie B
    assert stats["teams"]["A"] == "Czerwoni"
    assert a["points_won"] == 2 and a["win_rate"] == 1.0
    assert a["endings"] == {"winners": 1}
    assert stats["overall"]["B"]["endings"] == {"unforced_errors": 1}
    assert a["win_rate_by_end_formation"]["net"]["points"] == 1
    # A grała środkiem w obu punktach (raz jako near, raz jako far) i oba wygrała
    assert a["tactics"]["middle_ball"] == {"points": 2, "share_of_points": 1.0, "win_rate": 1.0}
    assert stats["overall"]["B"]["tactics"]["lob_to_take_net"]["win_rate"] == 0.0


def test_frame_times_cover_whole_point():
    from tacticllm.annotate import frame_times

    times = frame_times(10.0, 20.0, fps=3, max_frames=64)
    assert times[0] == 10.0 and abs(times[-1] - 20.0) < 1e-9 and len(times) == 31
    long_rally = frame_times(0.0, 60.0, fps=3, max_frames=64)
    assert len(long_rally) == 64 and abs(long_rally[-1] - 60.0) < 1e-9
