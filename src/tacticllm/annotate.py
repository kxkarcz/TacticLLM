"""Opis każdego punktu przez model wizyjny: przebieg, wynik i zastosowane taktyki.

Śledzenie zawodników mówi, GDZIE stali. Ten krok dodaje, CO grali: kto wygrał,
jak skończył się punkt, loby i taktyki każdej pary (tagi z models.TacticTag).

Dwa backendy (TACTIC_ANNOTATOR):
- "ollama" (domyślny, darmowy, lokalnie na GPU): model wizyjny typu Qwen-VL
  ogląda CAŁY punkt jako gęstą sekwencję klatek (domyślnie 3 klatki/s),
- "gemini": klip wideo wysyłany do Google Gemini (wymaga GOOGLE_API_KEY).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from rich.progress import track

from .config import settings
from .models import PointAnnotation, Segment
from .video import cut

TACTIC_GUIDE = """Taktyki (dla każdej pary osobno, tylko te, które naprawdę widać):
- lob_to_take_net: lob, po którym para przechodzi do siatki / wypycha rywali do tyłu
- chiquita: miękka, niska piłka w nogi rywali przy siatce
- middle_ball: gra środkiem, między zawodników rywala
- target_drive_player / target_reves_player: większość piłek kierowana do zawodnika
  rywala grającego po prawej (drive) / lewej (revés) stronie
- bandeja_vibora_control: utrzymywanie siatki bandejami / víborami zamiast ryzyka
- smash_winner_attempt: smecz na skończenie (płaski, w kąt)
- smash_x3_x4: smecz wyrzucający piłkę za kort (por tres / por cuatro)
- wall_defense: obrona z tyłu po odbiciu od szyby
- counter_attack: przejście z obrony do ataku (np. kontra po słabej piłce rywala)
- volley_exchange: szybka wymiana wolejów, obie pary przy siatce
- serve_and_volley: serwis i od razu wejście do siatki
- passing_shot: mijający po ścianie bocznej / wzdłuż szyby"""

PROMPT = """Jesteś analitykiem padla. {source}
Kamera transmisyjna stoi za kortem.

- "near" = para po bliższej stronie siatki (plecami do kamery),
  "far" = para po dalszej stronie (twarzą do kamery).
- Prześledź cały punkt od serwisu do końca, uderzenie po uderzeniu, zanim odpowiesz.
- Jeśli to nie jest rozgrywany punkt (przerwa, zmiana stron, powtórka), ustaw is_rally=false.
- winner_side: która para wygrała punkt; jeśli nie widać końca – "unknown".
- ending: "winner" (piłka nie do odebrania), "forced_error" (błąd wymuszony
  dobrym zagraniem rywala), "unforced_error" (błąd niewymuszony).
- last_shot: ostatnie uderzenie wygrywającej pary (smash, bandeja, vibora,
  volley, lob, groundstroke, wall_exit = wyjście za ścianę / zagranie po szybie,
  serve, other).
- shots: liczba uderzeń łącznie z serwisem.
- lobs_near / lobs_far: liczba lobów każdej pary.
- server_side: która para serwowała.
- scoreboard i set_number: przepisz wynik z grafiki, jeśli jest widoczna.

{tactics}

- tactics_near / tactics_far: lista taktyk każdej pary w tym punkcie.
- decisive_moment: co zadecydowało o punkcie (1 zdanie).
- notes: przebieg punktu krok po kroku (2-4 zdania).
Nie zgaduj – gdy nie widać, użyj "unknown", null albo pustej listy. Odpowiedz wyłącznie JSON-em."""


class GeminiAnnotator:
    def __init__(self, video: Path, clips_dir: Path, pad_s: float):
        from google import genai

        self.client = genai.Client()  # klucz z GOOGLE_API_KEY / GEMINI_API_KEY
        self.video, self.clips_dir, self.pad_s = video, clips_dir, pad_s
        clips_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, seg: Segment) -> PointAnnotation:
        from google.genai import types

        clip = self.clips_dir / f"point_{seg.index:04d}.mp4"
        if not clip.exists():
            cut(self.video, clip, max(0.0, seg.start_s - self.pad_s), seg.end_s + self.pad_s, max_height=720)
        response = self.client.models.generate_content(
            model=settings.video_model,
            contents=types.Content(parts=[
                types.Part(
                    inline_data=types.Blob(data=clip.read_bytes(), mime_type="video/mp4"),
                    video_metadata=types.VideoMetadata(fps=5),
                ),
                types.Part(text=PROMPT.format(source="Klip pokazuje (prawdopodobnie) jeden punkt meczu.",
                                              tactics=TACTIC_GUIDE)),
            ]),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PointAnnotation,
                temperature=0.1,
            ),
        )
        if isinstance(response.parsed, PointAnnotation):
            return response.parsed
        return PointAnnotation.model_validate_json(response.text)


def frame_times(start: float, end: float, fps: float, max_frames: int) -> list[float]:
    """Równe odstępy przez CAŁY punkt; przy bardzo długiej wymianie rzadziej, ale nadal do końca."""
    n = min(max_frames, max(4, math.ceil((end - start) * fps) + 1))
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]


class OllamaAnnotator:
    """Lokalny model wizyjny przez Ollamę – bez klucza API i bez kosztów."""

    tokens_per_frame_px = 28 * 28  # Qwen-VL: 1 token na łatkę 28x28 px

    def __init__(self, video: Path, pad_s: float):
        import ollama

        self.client = ollama.Client(host=settings.ollama_host)
        self.video, self.pad_s = video, pad_s

    def _frames(self, times: list[float], seg_start: float) -> list[bytes]:
        import cv2

        frames = []
        cap = cv2.VideoCapture(str(self.video))
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            scale = settings.ollama_frame_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            # znacznik czasu w rogu – model wie, ile minęło między klatkami
            cv2.putText(frame, f"{t - seg_start:+.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, f"{t - seg_start:+.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frames.append(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        cap.release()
        return frames

    def _context_size(self, n_frames: int) -> int:
        w = settings.ollama_frame_width
        per_frame = (w * w * 9 / 16) / self.tokens_per_frame_px
        needed = int(n_frames * per_frame) + 3000  # + prompt i odpowiedź
        return max(8192, math.ceil(needed / 4096) * 4096)

    def __call__(self, seg: Segment) -> PointAnnotation:
        start, end = max(0.0, seg.start_s - self.pad_s), seg.end_s + self.pad_s
        times = frame_times(start, end, settings.ollama_fps, settings.ollama_max_frames)
        frames = self._frames(times, seg.start_s)
        interval = (end - start) / max(1, len(frames) - 1)
        source = (f"Dostajesz {len(frames)} klatek w kolejności czasowej, co ~{interval:.2f} s, "
                  f"obejmujących cały punkt ({end - start:.0f} s, czas w lewym górnym rogu).")
        response = self.client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": PROMPT.format(source=source, tactics=TACTIC_GUIDE),
                       "images": frames}],
            format=PointAnnotation.model_json_schema(),
            options={"temperature": 0.1, "num_ctx": self._context_size(len(frames))},
        )
        return PointAnnotation.model_validate_json(response.message.content)


def make_annotator(backend: str, video: Path, clips_dir: Path, pad_s: float = 1.0):
    if backend == "ollama":
        return OllamaAnnotator(video, pad_s)
    if backend == "gemini":
        return GeminiAnnotator(video, clips_dir, pad_s)
    raise ValueError(f"Nieznany backend adnotacji: {backend} (dostępne: ollama, gemini)")


def annotate(video: Path, segments: list[Segment], clips_dir: Path, out_json: Path,
             backend: str | None = None) -> dict[int, PointAnnotation]:
    """Adnotuje wszystkie punkty; wynik zapisuje po każdym punkcie (można wznowić)."""
    annotator = make_annotator(backend or settings.annotator, video, clips_dir)
    done: dict[str, dict] = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}

    for seg in track(segments, description="Adnotacje punktów"):
        if str(seg.index) in done:
            continue
        try:
            ann = annotator(seg)
        except Exception as exc:  # pojedynczy punkt nie powinien zatrzymać całości
            print(f"Punkt {seg.index}: błąd adnotacji ({exc}) – pomijam")
            continue
        done[str(seg.index)] = ann.model_dump()
        out_json.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")

    return {int(k): PointAnnotation.model_validate(v) for k, v in done.items()}
