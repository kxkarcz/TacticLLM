"""Komendy: tacticllm analyze <link> --start 1:02:00 --end 1:45:00 --match mecz.yaml"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .models import Calibration, MatchInfo, RunPaths, load_annotations, load_segments, save_json
from .video import format_time, parse_time

app = typer.Typer(help="TacticLLM – analiza taktyczna meczów padla z wideo.", no_args_is_help=True)

RunOpt = typer.Option(Path("runs/mecz"), "--run", "-r", help="Katalog analizy (wszystkie pliki wynikowe)")
MatchOpt = typer.Option(None, "--match", "-m", help="Plik YAML z danymi meczu i zawodników")


def _match(run: RunPaths, match: Optional[Path]) -> MatchInfo:
    if match and Path(match).resolve() != run.match.resolve():
        shutil.copy(match, run.match)
    return MatchInfo.load(run.match if run.match.exists() else None)


@app.command()
def download(source: str = typer.Argument(..., help="Link (YouTube itp.) albo ścieżka do pliku"),
             start: Optional[str] = typer.Option(None, help="Początek fragmentu, np. 1:02:30"),
             end: Optional[str] = typer.Option(None, help="Koniec fragmentu, np. 1:45:00"),
             run: Path = RunOpt):
    """Pobiera film albo tylko wybrany fragment."""
    paths = RunPaths(run)
    download_impl(source, start, end, paths)


def download_impl(source, start, end, paths: RunPaths):
    from .video import download as dl

    dl(source, paths.video, start, end)
    print(f"[green]Wideo:[/green] {paths.video}")


@app.command()
def calibrate(run: Path = RunOpt,
              at: str = typer.Option("0:10", help="Czas klatki z główną kamerą (względem fragmentu)"),
              corners: Optional[str] = typer.Option(None, help="Punkty bez okna: 'x,y;x,y;x,y;x,y'"),
              kind: str = typer.Option("corners", help="'corners' (narożniki) albo 'service' (linie serwisowe)")):
    """Zaznacz 4 punkty kortu na klatce z główną kamerą."""
    import cv2

    from .calibrate import calibrate as calib_fn, draw_overlay

    paths = RunPaths(run)
    calib = calib_fn(paths.video, paths.calibration, paths.reference_frame, parse_time(at), corners, kind)
    preview = paths.root / "calibration_preview.jpg"
    cv2.imwrite(str(preview), draw_overlay(cv2.imread(str(paths.reference_frame)), calib))
    print(f"[green]Kalibracja zapisana.[/green] Sprawdź linie kortu na podglądzie: {preview}")


@app.command(name="track")
def track_cmd(run: Path = RunOpt):
    """Wykrywa zawodników (YOLO, GPU) i dzieli film na punkty."""
    import cv2

    from .tracking import track

    paths = RunPaths(run)
    calib = Calibration.model_validate_json(paths.calibration.read_text(encoding="utf-8"))
    segments = track(paths.video, calib, cv2.imread(str(paths.reference_frame)))
    save_json(paths.segments, segments)
    print(f"[green]Znaleziono {len(segments)} punktów[/green] -> {paths.segments}")


BackendOpt = typer.Option(None, "--backend", "-b",
                          help="'ollama' (lokalnie, za darmo) albo 'gemini'; domyślnie TACTIC_ANNOTATOR")


@app.command()
def annotate(run: Path = RunOpt, backend: Optional[str] = BackendOpt,
             model: Optional[str] = typer.Option(None, help="Inny model niż w .env, np. qwen3-vl:8b-instruct"),
             limit: Optional[int] = typer.Option(None, help="Tylko pierwsze N punktów (do porównania modeli)"),
             out: Optional[str] = typer.Option(None, help="Inna nazwa pliku wynikowego w katalogu analizy")):
    """Opisuje każdy punkt modelem wizyjnym: przebieg, wynik, taktyki obu par."""
    from .annotate import annotate as annotate_fn
    from .config import settings

    backend = backend or settings.annotator
    if model:
        if backend == "gemini":
            settings.video_model = model
        else:
            settings.ollama_model = model
    paths = RunPaths(run)
    target = paths.root / out if out else paths.annotations
    segments = load_segments(paths.segments)[:limit]
    anns = annotate_fn(paths.video, segments, paths.clips, target, backend)
    print(f"[green]Adnotacje: {len(anns)} punktów[/green] -> {target}")


@app.command()
def scores(run: Path = RunOpt, match: Optional[Path] = MatchOpt,
           box: Optional[str] = typer.Option(None, help="Prostokąt z wynikiem 'x1,y1,x2,y2' (zapamiętywany)"),
           model: Optional[str] = typer.Option(None, help="Model Ollamy; domyślnie TACTIC_SCOREBOARD_MODEL")):
    """Odczytuje tablicę wyników przed każdym punktem i ustala zwycięzcę punktu (lokalnie)."""
    from .config import settings
    from .scoreboard import read_scores
    from .video import video_duration

    if model:
        settings.scoreboard_model = model
    paths = RunPaths(run)
    info = _match(paths, match)
    calib = Calibration.model_validate_json(paths.calibration.read_text(encoding="utf-8"))
    if box:
        calib.scoreboard_px = [float(v) for v in box.split(",")]
        paths.calibration.write_text(calib.model_dump_json(indent=2), encoding="utf-8")
    result = read_scores(paths.video, load_segments(paths.segments), calib.scoreboard_px, info,
                         video_duration(paths.video))
    save_json(paths.scores, {str(k): v for k, v in result.items()})
    known = sum(1 for v in result.values() if v["winner"])
    print(f"[green]Zwycięzca z tablicy: {known}/{len(result)} punktów[/green] -> {paths.scores}")


@app.command()
def stats(run: Path = RunOpt, match: Optional[Path] = MatchOpt):
    """Liczy statystyki taktyczne i mapy cieplne."""
    from .models import load_scores
    from .stats import compute_stats, save_heatmaps

    paths = RunPaths(run)
    info = _match(paths, match)
    result = compute_stats(load_segments(paths.segments), load_annotations(paths.annotations), info,
                           load_scores(paths.scores))
    save_json(paths.stats, result)
    try:
        save_heatmaps(result, paths.heatmaps)
    except ImportError:
        print("[yellow]Brak matplotlib – pomijam mapy cieplne[/yellow]")
    for tid, name in result["teams"].items():
        o = result["overall"][tid]
        print(f"{name}: siatka {o['avg_net_share']}, wygrane {o['points_won']}/{o['points_with_result']}")
    print(f"[green]Statystyki:[/green] {paths.stats}")


@app.command()
def report(run: Path = RunOpt, match: Optional[Path] = MatchOpt):
    """Generuje raport taktyczny dla trenera (TACTIC_REPORT_PROVIDER: anthropic / gemini / ollama)."""
    from .report import generate_report

    paths = RunPaths(run)
    text = generate_report(paths, _match(paths, match))
    print(text)
    print(f"\n[green]Raport:[/green] {paths.report}")


@app.command()
def chat(run: Path = RunOpt):
    """Rozmowa z agentem o meczu, np. 'dlaczego przegrali 2. seta?'."""
    from .report import chat as chat_fn

    paths = RunPaths(run)
    chat_fn(paths, _match(paths, None))


@app.command()
def analyze(source: str = typer.Argument(..., help="Link (YouTube itp.) albo ścieżka do pliku"),
            start: Optional[str] = typer.Option(None, help="Początek fragmentu, np. 1:02:30"),
            end: Optional[str] = typer.Option(None, help="Koniec fragmentu, np. 1:45:00"),
            run: Path = RunOpt, match: Optional[Path] = MatchOpt,
            calib_at: str = typer.Option("0:10", help="Czas klatki do kalibracji (względem fragmentu)"),
            corners: Optional[str] = typer.Option(None, help="Punkty kalibracji bez okna"),
            kind: str = typer.Option("corners", help="'corners' albo 'service'"),
            with_annotations: bool = typer.Option(True, "--annotate/--no-annotate",
                                                  help="Adnotacje punktów modelem wizyjnym"),
            with_scores: bool = typer.Option(True, "--scores/--no-scores",
                                             help="Zwycięzca punktu z tablicy wyników (Ollama)"),
            scoreboard: Optional[str] = typer.Option(None, help="Prostokąt z wynikiem 'x1,y1,x2,y2'"),
            backend: Optional[str] = BackendOpt):
    """Cały proces: pobranie -> kalibracja -> śledzenie -> tablica -> (adnotacje) -> statystyki -> raport."""
    paths = RunPaths(run)
    if not paths.video.exists():
        download_impl(source, start, end, paths)
    if not paths.calibration.exists():
        calibrate(run=run, at=calib_at, corners=corners, kind=kind)
    if not paths.segments.exists():
        track_cmd(run=run)
    if with_scores and not paths.scores.exists():
        scores(run=run, match=match, box=scoreboard, model=None)
    if with_annotations:
        annotate(run=run, backend=backend, model=None, limit=None, out=None)
    stats(run=run, match=match)
    report(run=run, match=None)


@app.command()
def points(run: Path = RunOpt):
    """Lista wykrytych punktów (czas w fragmencie) – do szybkiej weryfikacji."""
    from .models import load_scores

    paths = RunPaths(run)
    anns = load_annotations(paths.annotations)
    board = load_scores(paths.scores)
    for seg in load_segments(paths.segments):
        ann = anns.get(seg.index)
        winner = board.get(seg.index, {}).get("winner")
        extra = f" | wygrała para {winner}" if winner else ""
        extra += f" | {ann.winner_side} {ann.ending} {ann.last_shot}" if ann else ""
        print(f"#{seg.index:3d}  {format_time(seg.start_s)}–{format_time(seg.end_s)}  "
              f"bliżej: para {seg.near_team}  obs: {len(seg.observations)}{extra}")


if __name__ == "__main__":
    app()
