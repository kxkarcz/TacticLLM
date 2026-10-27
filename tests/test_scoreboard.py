import pytest

from tacticllm.models import MatchInfo, PlayerObs, PointAnnotation, Segment, TeamProfile
from tacticllm.scoreboard import (RawScoreboard, ScoreReading, ScoreRow, games_columns, match_rows, parse_all,
                                  parse_board, point_winner, rally_start, split_row)
from tacticllm.stats import build_points


def raw(top, bottom):
    return RawScoreboard(visible=True, top=top, bottom=bottom)


def board(top, bottom):
    """'2 15' -> gemy [2], punkty '15'; '2' -> sam gem."""
    def row(text):
        *games, last = text.split()
        if len(text.split()) == 1:
            return ScoreRow(name="X", games=[int(text)])
        return ScoreRow(name="X", games=[int(g) for g in games], points=last)
    return ScoreReading(visible=True, top=row(top), bottom=row(bottom))


def test_split_row():
    assert split_row("GONZALEZ / CAMPAGNOLO 2 15") == ("GONZALEZ / CAMPAGNOLO", ["2", "15"])
    assert split_row("BAUTISTA / BERGAMINI 6 3 AD") == ("BAUTISTA / BERGAMINI", ["6", "3", "AD"])


def test_parse_board_tells_points_from_games():
    # w tej transmisji kolumna punktów znika na początku gema -> 1 kolumna gemów
    raws = [raw("A / B 2", "C / D 2"), raw("A / B 2 0", "C / D 2 0"), raw("A / B 2 15", "C / D 2 0")]
    assert games_columns(raws) == 1
    first, zero_zero, fifteen = parse_all(raws)
    assert first.top.games == [2] and first.top.points is None
    assert zero_zero.top.games == [2] and zero_zero.top.points == "0"
    assert fifteen.top.points == "15" and fifteen.bottom.points == "0"
    # drugi set: "6 2" to dwa sety (bez punktów), a "6 2 40" – dwa sety i 40
    second = parse_board(raw("A / B 6 2 40", "C / D 3 1 15"), games_columns=2)
    assert second.top.games == [6, 2] and second.top.points == "40"
    assert parse_board(RawScoreboard(visible=False), 1).visible is False


@pytest.mark.parametrize("before, after, expected", [
    (("2 15", "2 0"), ("2 30", "2 0"), ("top", 1)),
    (("2 15", "2 0"), ("2 15", "2 15"), ("bottom", 1)),
    (("2 40", "2 15"), ("3", "2"), ("top", 1)),  # gem: kolumna punktów znika
    (("3 0", "2 40"), ("3", "3"), ("bottom", 1)),
    (("3 40", "3 40"), ("3 AD", "3 40"), ("top", 1)),  # przewaga
    (("3 40", "3 AD"), ("3 40", "3 40"), ("top", 1)),  # odrobiona przewaga
    (("3 0", "2 15"), ("3 0", "2 40"), ("bottom", 2)),  # pominięty punkt
    (("3 0", "2 15"), ("3 0", "2 15"), (None, 0)),  # wynik się nie zmienił
    (("3 15", "2 15"), ("3 30", "2 30"), (None, 0)),  # sprzeczny odczyt
])
def test_point_winner(before, after, expected):
    assert point_winner(board(*before), board(*after)) == expected


def test_point_winner_needs_both_readings():
    assert point_winner(ScoreReading(visible=False), board("2 15", "2 0")) == (None, 0)


def test_match_rows_by_player_names():
    info = MatchInfo(teams=[TeamProfile(name="González / Campagnolo", start_side="near"),
                            TeamProfile(name="Bautista / Bergamini", start_side="far")])
    reading = parse_board(raw("GONZALEZ / CAMPAGNOLO 2 0", "BAUTISTA / BERGAMINI 1 30"), 1)
    assert match_rows(reading, info) == {"top": "A", "bottom": "B"}
    swapped = parse_board(raw("BAUTISTA / BERGAMINI 1 30", "GONZALEZ / CAMPAGNOLO 2 0"), 1)
    assert match_rows(swapped, info) == {"top": "B", "bottom": "A"}
    assert match_rows(reading, MatchInfo()) is None  # bez pliku meczu nie zgadujemy


def test_rally_start_after_serve_setup():
    obs = []
    for k in range(100):  # 10 s: 4 s stania, potem bieg 3 m/s
        t = 20 + k * 0.1
        x = 5 + (0.3 * (k - 40) if k >= 40 else 0)
        obs.append(PlayerObs(t=t, side="near", lane="drive", x=x, y=15, conf=0.9, color=[0, 0, 0]))
    seg = Segment(index=0, start_s=20, end_s=29.9, observations=obs)
    assert rally_start(seg) == pytest.approx(23.5)


def test_scoreboard_winner_beats_model_guess():
    seg = Segment(index=0, start_s=0, end_s=5, near_team="A")
    ann = {0: PointAnnotation(is_rally=False, winner_side="near", ending="winner", last_shot="smash")}
    # tablica mówi, że punkt był i wygrała go para B – to wygrywa z oceną modelu
    points = build_points([seg], ann, MatchInfo(), scores={0: "B"})
    assert len(points) == 1 and points[0]["winner"] == "B"
    assert build_points([seg], ann, MatchInfo()) == []  # bez tablicy: model mówi "nie punkt"
