"""Raport taktyczny i czat z agentem (Agno + Claude)."""

from __future__ import annotations

import json
from pathlib import Path

from .config import settings
from .models import MatchInfo, RunPaths

INSTRUCTIONS = [
    "Jesteś doświadczonym trenerem i analitykiem padla. Piszesz dla trenera, nie dla programisty.",
    "Dostajesz statystyki wyliczone automatycznie z wideo: pozycje zawodników (śledzenie) "
    "i – jeśli są – opisy punktów z modelu wideo.",
    "Słownik danych: net_share = % czasu, gdy OBAJ zawodnicy pary są przy siatce (<4.5 m); "
    "back_share = obaj z tyłu (>7 m); split_share = jeden przy siatce, drugi z tyłu; "
    "net_gains = przejęcia siatki (tył -> siatka); net_losses = utrata siatki (siatka -> tył, "
    "zwykle po lobie rywala); end_formation = ustawienie pary w chwili końca punktu; "
    "drive/reves = zawodnik grający po prawej / lewej stronie; avg_depth_m = średnia "
    "odległość od siatki; heatmap: wiersze = szerokość od strony revésu do drive'u, "
    "kolumny = odległość od siatki (od siatki do tyłu), wartości = udział czasu; "
    "tactics = jak często para stosowała daną taktykę (share_of_points) i jaki % tych punktów "
    "wygrała (win_rate). Tagi: lob_to_take_net (lob i przejęcie siatki), chiquita (miękka piłka "
    "w nogi), middle_ball (gra środkiem), target_drive_player / target_reves_player (gra w "
    "zawodnika z prawej / lewej strony), bandeja_vibora_control, smash_winner_attempt, "
    "smash_x3_x4 (smecz za kort), wall_defense (obrona po szybie), counter_attack, "
    "volley_exchange, serve_and_volley, passing_shot. "
    "points[].notes i decisive_moment = opis przebiegu każdego punktu.",
    "Interpretuj liczby taktycznie (np. kto dominował przy siatce, czy loby działały, "
    "w którego zawodnika grano, jak zmieniała się gra między setami). Podawaj konkretne liczby.",
    "Bądź uczciwy co do jakości danych: to automatyczna analiza, bez śledzenia piłki. "
    "Jeśli czegoś nie ma w danych (np. brak adnotacji punktów = brak informacji kto wygrał), "
    "nie wymyślaj – napisz, czego brakuje. Gdy próbka jest mała (<10 punktów), zaznacz to.",
    "Uwzględnij profile zawodników (ręka, strona gry), jeśli są podane – np. leworęczny na revésie.",
]

REPORT_TEMPLATE = """Napisz raport taktyczny po {language} w Markdown z sekcjami:

1. **Podsumowanie** (3-5 zdań: kto kontrolował mecz i dlaczego)
2. **Taktyka ogólna** – osobno dla każdej pary: plan gry (najczęstsze taktyki i ich skuteczność),
   ustawienie, gra przy siatce, obrona, loby, w którego rywala grano, podział ról drive/revés
3. **Przebieg meczu** – jak taktyka zmieniała się w czasie: fazy meczu, kto i kiedy zmienił plan
   gry, czy rywal się dostosował (opieraj się na kolejnych punktach, podawaj czasy)
4. **Analiza setów** – co zmieniło się między setami (jeśli jest więcej niż jeden)
5. **Kluczowe wzorce** – 3-6 punktów z liczbami (np. taktyka działająca / niedziałająca)
6. **Wskazówki na przyszłe mecze** – {focus}; konkretne, wykonalne zalecenia treningowe i meczowe
7. **Ograniczenia analizy** – krótko, czego dane nie pokazują

## Informacje o meczu
{match}

## Statystyki
```json
{stats}
```
"""


def _model():
    if settings.report_provider == "ollama":
        from agno.models.ollama import Ollama

        return Ollama(id=settings.ollama_report_model, host=settings.ollama_host,
                      options={"num_ctx": settings.ollama_report_ctx, "temperature": 0.3})
    from agno.models.anthropic import Claude

    return Claude(id=settings.report_model)


def _match_text(info: MatchInfo) -> str:
    lines = []
    if info.title:
        lines.append(f"Mecz: {info.title}")
    if info.event:
        lines.append(f"Turniej: {info.event}")
    for tid in ("A", "B"):
        team = info.team(tid)
        if not team:
            lines.append(f"Para {tid}: brak danych o zawodnikach")
            continue
        players = "; ".join(
            f"{p.name} (ręka: {p.hand}, strona: {p.lane or '?'}{', ' + p.notes if p.notes else ''})"
            for p in team.players
        )
        lines.append(f"Para {tid} = {team.name}: {players}")
    return "\n".join(lines)


def _compact(stats: dict) -> dict:
    """Wszystkie punkty meczu zostają (potrzebne do opisu przebiegu). Model lokalny ma mniejszy
    kontekst, więc dostaje tylko decisive_moment zamiast pełnego opisu każdego punktu."""
    if settings.report_provider != "ollama":
        return stats
    data = dict(stats)
    data["points"] = [{k: v for k, v in p.items() if k != "notes"} for p in stats.get("points", [])]
    return data


def generate_report(run: RunPaths, info: MatchInfo) -> str:
    from agno.agent import Agent

    stats = json.loads(run.stats.read_text(encoding="utf-8"))
    focus = (f"głównie dla pary {info.focus_team}" if info.focus_team
             else "dla obu par (jak wygrać z rywalem)")
    prompt = REPORT_TEMPLATE.format(
        language=settings.report_language,
        focus=focus,
        match=_match_text(info),
        stats=json.dumps(_compact(stats), ensure_ascii=False, separators=(",", ":")),
    )
    agent = Agent(model=_model(), instructions=INSTRUCTIONS, markdown=True)
    report = agent.run(prompt).content
    run.report.write_text(report, encoding="utf-8")
    return report


def chat(run: RunPaths, info: MatchInfo) -> None:
    """Interaktywny czat o meczu; agent sam sięga po dane narzędziami."""
    from agno.agent import Agent

    stats = json.loads(run.stats.read_text(encoding="utf-8"))

    def get_summary() -> str:
        """Zwraca statystyki całego meczu i każdego seta dla obu par."""
        return json.dumps({k: stats[k] for k in ("teams", "overall", "by_set", "points_count")}, ensure_ascii=False)

    def list_points(set_number: int | None = None) -> str:
        """Lista punktów (czas, zwycięzca, zakończenie, ustawienie). Opcjonalnie tylko z danego seta."""
        pts = [p for p in stats["points"] if set_number is None or p["set"] == set_number]
        return json.dumps(pts, ensure_ascii=False)

    def get_report() -> str:
        """Zwraca wygenerowany wcześniej raport (jeśli istnieje)."""
        return run.report.read_text(encoding="utf-8") if run.report.exists() else "Brak raportu."

    agent = Agent(
        model=_model(),
        instructions=INSTRUCTIONS + [
            f"Odpowiadaj po {settings.report_language}. Informacje o meczu:\n{_match_text(info)}",
            "Przy pytaniach o konkretne momenty podawaj czas punktu (start_s, w sekundach od początku fragmentu).",
        ],
        tools=[get_summary, list_points, get_report],
        markdown=True,
    )
    agent.cli_app(markdown=True)
