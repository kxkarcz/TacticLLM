"""Modele danych zapisywane w katalogu analizy (runs/<nazwa>/)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Side = Literal["near", "far"]
Lane = Literal["drive", "reves"]
TeamId = Literal["A", "B"]


class Calibration(BaseModel):
    corners_px: list[list[float]]  # 4 punkty, kolejność jak court.CORNER_NAMES
    kind: Literal["corners", "service"] = "corners"  # patrz court.REFERENCE_POINTS_M
    reference_time_s: float  # klatka z widokiem głównej kamery
    scoreboard_px: list[float] | None = None  # x1, y1, x2, y2 grafiki z wynikiem (brak = cała klatka)

    def court(self):
        import numpy as np

        from .court import Court

        return Court(np.array(self.corners_px), self.kind)


class PlayerObs(BaseModel):
    """Jedna obserwacja zawodnika w jednej klatce."""

    t: float
    side: Side
    lane: Lane
    x: float
    y: float
    conf: float
    color: list[float]  # średni kolor koszulki (Lab)


class Segment(BaseModel):
    """Ciągłe ujęcie głównej kamery – kandydat na jeden punkt."""

    index: int
    start_s: float
    end_s: float
    near_team: TeamId | None = None
    observations: list[PlayerObs] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


# Taktyki rozpoznawane w każdym punkcie (opisy dla modelu: annotate.TACTIC_GUIDE).
TacticTag = Literal[
    "lob_to_take_net",
    "chiquita",
    "middle_ball",
    "target_drive_player",
    "target_reves_player",
    "bandeja_vibora_control",
    "smash_winner_attempt",
    "smash_x3_x4",
    "wall_defense",
    "counter_attack",
    "volley_exchange",
    "serve_and_volley",
    "passing_shot",
]


class PointAnnotation(BaseModel):
    """Opis punktu z modelu wizyjnego (opcjonalny krok `annotate`)."""

    is_rally: bool = Field(description="Czy klip pokazuje rozegrany punkt (a nie przerwę/powtórkę)")
    winner_side: Literal["near", "far", "unknown"] = Field(
        description="Która para wygrała punkt: near = bliżej kamery, far = dalej"
    )
    ending: Literal["winner", "forced_error", "unforced_error", "unknown"]
    last_shot: Literal[
        "smash", "bandeja", "vibora", "volley", "lob", "groundstroke", "wall_exit", "serve", "other", "unknown"
    ]
    shots: int | None = Field(default=None, description="Liczba uderzeń w wymianie")
    lobs_near: int | None = Field(default=None, description="Ile lobów zagrała para bliżej kamery")
    lobs_far: int | None = Field(default=None, description="Ile lobów zagrała para dalej od kamery")
    server_side: Literal["near", "far", "unknown"] = "unknown"
    scoreboard: str | None = Field(default=None, description="Wynik z grafiki na ekranie, jeśli widoczny")
    set_number: int | None = Field(default=None, description="Numer seta z tablicy wyników, jeśli widoczny")
    tactics_near: list[TacticTag] = Field(
        default_factory=list, description="Taktyki zastosowane w tym punkcie przez parę bliżej kamery"
    )
    tactics_far: list[TacticTag] = Field(
        default_factory=list, description="Taktyki zastosowane w tym punkcie przez parę dalej od kamery"
    )
    decisive_moment: str | None = Field(
        default=None, description="Co zadecydowało o punkcie (1 zdanie)"
    )
    notes: str | None = Field(default=None, description="Przebieg punktu krok po kroku (2-4 zdania)")


class PlayerProfile(BaseModel):
    name: str
    hand: Literal["right", "left", "unknown"] = "unknown"
    lane: Lane | None = None  # zwykła strona gry
    notes: str | None = None


class TeamProfile(BaseModel):
    name: str
    start_side: Side  # po której stronie zaczyna w analizowanym fragmencie
    players: list[PlayerProfile] = Field(default_factory=list)


class MatchInfo(BaseModel):
    title: str | None = None
    event: str | None = None
    focus_team: str | None = None  # dla kogo wskazówki (nazwa drużyny)
    set_starts_s: list[float] = Field(default_factory=list)  # początki setów w sekundach fragmentu
    teams: list[TeamProfile] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None) -> "MatchInfo":
        if path is None:
            return cls()
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def team(self, team_id: TeamId) -> TeamProfile | None:
        """Drużyna A = ta, która w pierwszym punkcie stoi bliżej kamery."""
        wanted = "near" if team_id == "A" else "far"
        return next((t for t in self.teams if t.start_side == wanted), None)

    def team_name(self, team_id: TeamId) -> str:
        team = self.team(team_id)
        return team.name if team else f"Para {team_id}"


class RunPaths:
    """Wszystkie pliki jednej analizy w jednym katalogu."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    video = property(lambda self: self.root / "video.mp4")
    calibration = property(lambda self: self.root / "calibration.json")
    reference_frame = property(lambda self: self.root / "reference.jpg")
    segments = property(lambda self: self.root / "segments.json")
    annotations = property(lambda self: self.root / "annotations.json")
    scores = property(lambda self: self.root / "scores.json")
    stats = property(lambda self: self.root / "stats.json")
    heatmaps = property(lambda self: self.root / "heatmaps")
    clips = property(lambda self: self.root / "clips")
    report = property(lambda self: self.root / "report.md")
    match = property(lambda self: self.root / "match.yaml")


def save_json(path: Path, data) -> None:
    if isinstance(data, BaseModel):
        text = data.model_dump_json(indent=2)
    elif isinstance(data, list) and data and isinstance(data[0], BaseModel):
        text = json.dumps([d.model_dump() for d in data], indent=2, ensure_ascii=False)
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False)
    Path(path).write_text(text, encoding="utf-8")


def load_segments(path: Path) -> list[Segment]:
    return [Segment.model_validate(s) for s in json.loads(Path(path).read_text(encoding="utf-8"))]


def load_scores(path: Path) -> dict[int, dict]:
    """Z tablicy wyników: {indeks: {"winner": "A"/"B"/None, "set": numer seta albo None}}.

    Numer seta = liczba kolumn z gemami na tablicy przed punktem.
    """
    if not Path(path).exists():
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for k, v in raw.items():
        top = (v.get("before") or {}).get("top") or {}
        result[int(k)] = {"winner": v.get("winner"), "set": len(top["games"]) if top.get("games") else None}
    return result


def load_annotations(path: Path) -> dict[int, PointAnnotation]:
    if not Path(path).exists():
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): PointAnnotation.model_validate(v) for k, v in raw.items()}
