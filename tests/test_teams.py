from tacticllm.models import PlayerObs, Segment
from tacticllm.teams import assign_teams, team_of_side

RED, BLUE = [150.0, 180.0, 160.0], [80.0, 130.0, 90.0]


def segment(index, near_color=None, far_color=None):
    o = []
    for k in range(5):
        if near_color:
            o.append(PlayerObs(t=k, side="near", lane="drive", x=7, y=15, conf=0.9, color=near_color))
        if far_color:
            o.append(PlayerObs(t=k, side="far", lane="drive", x=3, y=5, conf=0.9, color=far_color))
    return Segment(index=index, start_s=index * 10, end_s=index * 10 + 5, observations=o)


def test_first_full_segment_defines_team_a():
    segs = [segment(0, RED, BLUE), segment(1, RED, BLUE), segment(2, BLUE, RED), segment(3, BLUE, RED)]
    assign_teams(segs)
    assert [s.near_team for s in segs] == ["A", "A", "B", "B"]


def test_segments_without_both_pairs_take_previous_side():
    segs = [segment(0), segment(1, RED, BLUE), segment(2, near_color=BLUE), segment(3, BLUE, RED), segment(4)]
    assign_teams(segs)
    # brak danych na początku -> A; potem strona z poprzedniego punktu
    assert [s.near_team for s in segs] == ["A", "A", "A", "B", "B"]


def test_slowly_changing_colors_are_tracked():
    # oświetlenie zmienia się stopniowo – kolory dryfują, ale pary nie zamieniają się stronami
    segs = []
    for i in range(10):
        shift = 4.0 * i
        segs.append(segment(i, [c + shift for c in RED], [c + shift for c in BLUE]))
    assign_teams(segs)
    assert all(s.near_team == "A" for s in segs)


def test_team_of_side():
    seg = segment(0)
    seg.near_team = "B"
    assert team_of_side(seg, "near") == "B" and team_of_side(seg, "far") == "A"
    seg.near_team = None
    assert team_of_side(seg, "near") == "A" and team_of_side(seg, "far") == "B"
