"""Zwycięzca punktu z tablicy wyników transmisji.

Model wizyjny źle ocenia, kto wygrał wymianę, ale bardzo dobrze odczytuje grafikę z wynikiem.
Odczytujemy więc wynik tuż przed każdym punktem; zwycięzca punktu to para, która między tym
odczytem a odczytem przed następnym punktem zyskała punkt (albo gema).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel, Field
from rich.progress import track

from .config import settings
from .models import MatchInfo, Segment, TeamId
from .tracking import QUIET_SPEED_MS, speed_profile

POINT_ORDER = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4}
POINT_TOKENS = {"15", "30", "40", "A", "AD"}


class RawScoreboard(BaseModel):
    """Co odpowiada model: same przepisane wiersze. Złożony schemat (listy gemów, punkty)
    myli małe modele – rozbiór wiersza robi kod (parse_row)."""

    visible: bool = Field(description="Czy na obrazie widać tablicę z wynikiem")
    top: str | None = Field(default=None, description="Górny wiersz przepisany dokładnie: nazwiska i wszystkie liczby")
    bottom: str | None = Field(default=None, description="Dolny wiersz przepisany dokładnie: nazwiska i wszystkie liczby")


class ScoreRow(BaseModel):
    name: str | None = None
    games: list[int] = Field(default_factory=list)  # gemy w kolejnych setach
    points: str | None = None  # punkty w bieżącym gemie; None = kolumny nie ma (początek gema)


class ScoreReading(BaseModel):
    visible: bool
    top: ScoreRow | None = None
    bottom: ScoreRow | None = None


PROMPT = """Na obrazie jest grafika z wynikiem meczu padla: dwie pary, jedna nad drugą.
Przepisz dokładnie górny (top) i dolny (bottom) wiersz: nazwiska i wszystkie liczby od lewej
do prawej, oddzielone spacjami (np. "KOWALSKI / NOWAK 6 3 40"). Pomiń symbole i ikony.
Jeśli tablicy nie widać, visible = false. Odpowiedz wyłącznie JSON-em."""


# "star point" (Premier Padel): 40-40 -> AD1 -> D2 -> AD2 -> SP (punkt decydujący). Numer przy
# AD / D to tylko licznik równowag – D2 i SP to po prostu 40-40.
_DEUCE = re.compile(r"\bAD\s*\d\b|\bD\d\b|\bSP\b")


def _deuce_token(match: re.Match) -> str:
    return " AD " if match.group().startswith("AD") else " 40 "


def split_row(text: str) -> tuple[str, list[str]]:
    """'GONZALEZ / CAMPAGNOLO 2 15' -> ('GONZALEZ / CAMPAGNOLO', ['2', '15'])."""
    text = _DEUCE.sub(_deuce_token, text.upper())
    tokens = re.findall(r"\bAD\b|\bA\b(?=\s*$)|\d+", text)
    name = re.sub(r"\bAD\b|\d+", " ", text).strip(" -|·*")
    return re.sub(r"\s+", " ", name).strip(), tokens


def parse_board(raw: RawScoreboard, games_columns: int) -> ScoreReading:
    """Rozbiór przepisanych wierszy. games_columns = liczba kolumn z gemami (setów); jeśli liczb
    jest o jedną więcej, ostatnia to punkty w gemie (dzięki temu '2 0' nie myli się z dwoma setami)."""
    if not (raw.visible and raw.top and raw.bottom):
        return ScoreReading(visible=False)
    (top_name, top), (bottom_name, bottom) = split_row(raw.top), split_row(raw.bottom)
    if not top or not bottom:
        return ScoreReading(visible=False)
    # przy przewadze rywal ma pustą komórkę punktów – czyli 40
    if len(top) == len(bottom) + 1 and top[-1] in ("AD", "A"):
        bottom = bottom + ["40"]
    elif len(bottom) == len(top) + 1 and bottom[-1] in ("AD", "A"):
        top = top + ["40"]
    with_points = (len(top) == len(bottom) == games_columns + 1
                   or (len(top) == len(bottom) and (top[-1] in POINT_TOKENS or bottom[-1] in POINT_TOKENS)))

    def row(name: str, tokens: list[str]) -> ScoreRow:
        games, points = (tokens[:-1], tokens[-1]) if with_points else (tokens, None)
        return ScoreRow(name=name, games=[int(g) for g in games if g.isdigit()], points=points)

    return ScoreReading(visible=True, top=row(top_name, top), bottom=row(bottom_name, bottom))


def games_columns(raws: list[RawScoreboard]) -> int:
    """Liczba setów na grafice = najmniejsza liczba liczb w wierszu (gdy kolumny punktów nie ma)."""
    counts = [min(len(split_row(r.top)[1]), len(split_row(r.bottom)[1]))
              for r in raws if r.visible and r.top and r.bottom]
    counts = [c for c in counts if c]
    return min(counts) if counts else 1


def _norm(text: str) -> set[str]:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return {w for w in re.findall(r"[a-z]+", text) if len(w) > 2}


def match_rows(reading: ScoreReading, info: MatchInfo) -> dict[str, TeamId] | None:
    """Który wiersz tablicy to która para – po nazwiskach z pliku meczu."""
    if not (reading.top and reading.bottom and reading.top.name and reading.bottom.name):
        return None
    words = {}
    for tid in ("A", "B"):
        team = info.team(tid)
        if team:
            words[tid] = _norm(team.name + " " + " ".join(p.name for p in team.players))
    if len(words) < 2:
        return None
    top, bottom = _norm(reading.top.name), _norm(reading.bottom.name)
    keep = len(top & words["A"]) + len(bottom & words["B"])
    swap = len(top & words["B"]) + len(bottom & words["A"])
    if keep == swap:
        return None
    return {"top": "A", "bottom": "B"} if keep > swap else {"top": "B", "bottom": "A"}


def _state(row: ScoreRow) -> tuple[int, int] | None:
    """(gemy w bieżącym secie, punkty w gemie); brak kolumny punktów = początek gema."""
    if not row.games:
        return None
    points = POINT_ORDER.get((row.points or "0").strip().upper())
    return (row.games[-1], points) if points is not None else None


def point_winner(before: ScoreReading, after: ScoreReading) -> tuple[str | None, int]:
    """('top' / 'bottom' / None, ile punktów zostało rozegranych między odczytami)."""
    if not (before.visible and after.visible and before.top and before.bottom and after.top and after.bottom):
        return None, 0
    b = (_state(before.top), _state(before.bottom))
    a = (_state(after.top), _state(after.bottom))
    if None in b or None in a:
        return None, 0
    (bg_t, bp_t), (bg_b, bp_b) = b
    (ag_t, ap_t), (ag_b, ap_b) = a
    if len(after.top.games) != len(before.top.games):
        return None, 0  # nowy set – rzadkie, nie zgadujemy
    if (ag_t, ag_b) != (bg_t, bg_b):  # ktoś wygrał gema
        if ag_t == bg_t + 1 and ag_b == bg_b:
            return "top", 1
        if ag_b == bg_b + 1 and ag_t == bg_t:
            return "bottom", 1
        return None, 0
    # równowaga: 40-AD -> 40-40 to punkt dla strony, która wróciła do 40
    if bp_t == 3 and bp_b == 4 and ap_t == ap_b == 3:
        return "top", 1
    if bp_b == 3 and bp_t == 4 and ap_t == ap_b == 3:
        return "bottom", 1
    gained_t, gained_b = ap_t - bp_t, ap_b - bp_b
    if gained_t > 0 and gained_b == 0:
        return "top", gained_t
    if gained_b > 0 and gained_t == 0:
        return "bottom", gained_b
    return None, 0


def rally_start(seg: Segment) -> float:
    """Pierwsza sekunda wymiany (po ustawieniu do serwisu) – tuż przed nią wynik jest aktualny."""
    speed = speed_profile(seg.observations)
    for t in range(int(seg.start_s), int(seg.end_s)):
        if speed.get(t, 0) >= QUIET_SPEED_MS and speed.get(t + 1, 0) >= QUIET_SPEED_MS:
            return max(seg.start_s, t - 0.5)
    return seg.start_s + 0.5


class ScoreboardReader:
    """Odczyt tablicy lokalnym modelem wizyjnym (Ollama) z wycinka klatki."""

    def __init__(self, video: Path, box: list[float] | None):
        import ollama

        self.client = ollama.Client(host=settings.ollama_host)
        self.video, self.box = video, box

    def _image(self, t: float) -> bytes | None:
        import cv2

        from .video import read_frame

        try:
            frame = read_frame(self.video, t)
        except RuntimeError:
            return None
        if self.box:
            x1, y1, x2, y2 = (int(v) for v in self.box)
            frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return cv2.imencode(".png", frame)[1].tobytes()

    def read(self, t: float) -> RawScoreboard:
        image = self._image(t)
        if image is None:
            return RawScoreboard(visible=False)
        response = self.client.chat(
            model=settings.scoreboard_model,
            messages=[{"role": "user", "content": PROMPT, "images": [image]}],
            format=RawScoreboard.model_json_schema(),
            options={"temperature": 0, "num_ctx": 8192, "num_predict": 200},
        )
        return RawScoreboard.model_validate_json(response.message.content)

    def read_near(self, t: float, lo: float, hi: float) -> tuple[float, RawScoreboard]:
        """Pierwszy czytelny odczyt w pobliżu t (grafika bywa chwilowo schowana)."""
        raw = RawScoreboard(visible=False)
        for dt in (0, -1, 1, -2, 2, -3, 3):
            when = min(max(t + dt, lo), hi)
            raw = self.read(when)
            if raw.visible and raw.top and raw.bottom and split_row(raw.top)[1] and split_row(raw.bottom)[1]:
                return when, raw
        return t, raw


def parse_all(raws: list[RawScoreboard], window: int = 6) -> list[ScoreReading]:
    """Liczbę setów na grafice ustalamy z sąsiednich odczytów (zmienia się po secie)."""
    return [parse_board(raw, games_columns(raws[max(0, i - window):i + window + 1]))
            for i, raw in enumerate(raws)]


def read_scores(video: Path, segments: list[Segment], box: list[float] | None, info: MatchInfo,
                duration: float) -> dict[int, dict]:
    """Wynik przed każdym punktem i zwycięzca każdego punktu (A/B)."""
    reader = ScoreboardReader(video, box)
    times, raws = [], []
    for seg in track(segments, description="Odczyt tablicy wyników"):
        t, raw = reader.read_near(rally_start(seg), seg.start_s, seg.end_s)
        times.append(t)
        raws.append(raw)
    # po ostatnim punkcie: koniec nagrania (wynik mógł się jeszcze nie zmienić)
    t, raw = reader.read_near(duration - 0.5, max(0.0, duration - 8), duration - 0.1)
    times.append(t)
    raws.append(raw)
    return score_points(segments, times, raws, info)


def score_points(segments: list[Segment], times: list[float], raws: list[RawScoreboard],
                 info: MatchInfo) -> dict[int, dict]:
    """Zwycięzcy punktów z przepisanych tablic: raws[i] przed punktem i, ostatni – po ostatnim punkcie."""
    readings = list(zip(times, parse_all(raws)))

    rows = next((m for _, r in readings if (m := match_rows(r, info))), None)
    result = {}
    for seg, raw_before, (t, before), (_, after) in zip(segments, raws, readings, readings[1:]):
        side, played = point_winner(before, after)
        result[seg.index] = {
            "read_at_s": round(t, 1),
            "raw": raw_before.model_dump(),
            "before": before.model_dump(),
            "points_played": played,
            # dwa punkty między odczytami (np. pominięty punkt) – nie wiemy, który był w tym segmencie
            "winner": rows[side] if rows and side and played == 1 else None,
            "winner_row": side,
        }
    if rows is None:
        print("Nie udało się dopasować wierszy tablicy do par – uzupełnij nazwiska w pliku meczu (--match).")
    return result
