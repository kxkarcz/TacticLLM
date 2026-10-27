# TacticLLM

Analiza taktyczna meczów padla z wideo (np. transmisji z YouTube) i raport dla trenera generowany przez LLM.

Przebieg: **pobranie → kalibracja kortu → wykrywanie zawodników (YOLO) → wynik z tablicy → opis punktów modelem wizyjnym → statystyki → raport / czat**.

Program pokazuje, gdzie stali zawodnicy (siatka / tył, przejęcia siatki, mapy cieplne) – po krokach `scores` i `annotate` – kto wygrał punkt (z tablicy wyników) i co grali: jak się skończył, jakie taktyki stosowała każda para i z jakim skutkiem.

## Wymagania

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) w `PATH` (Windows: `winget install ffmpeg`)
- karta NVIDIA z CUDA – do YOLO i lokalnych modeli wizyjnych (ustawienia w `.env.example` są dobrane pod 16 GB VRAM; na CPU działa, ale bardzo wolno)
- [Ollama](https://ollama.com/) – opcjonalnie, do darmowych opisów punktów i raportu lokalnie
- klucz API Anthropic (raport przez Claude) i/lub Google (opisy punktów przez Gemini) – opcjonalnie

## Instalacja

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
```

### 1. PyTorch z CUDA

Zainstaluj go **przed** resztą zależności, inaczej `ultralytics` pociągnie wersję tylko na CPU. Wybierz wersję CUDA na [pytorch.org](https://pytorch.org/get-started/locally/), np.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Sprawdzenie:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. TacticLLM

```bash
pip install -e ".[all,dev]"
```

Grupy zależności: `vision` (wideo, YOLO, mapy cieplne), `annotate` (Ollama, Gemini), `llm` (Agno, Claude), `dev` (pytest). Same statystyki i testy działają bez żadnej z nich.

### 3. Ollama (opcjonalnie)

Zainstaluj Ollamę, a potem pobierz modele:

```bash
ollama pull qwen2.5vl:7b    # opisy punktów (model wizyjny)
ollama pull qwen3:14b       # raport i czat, gdy TACTIC_REPORT_PROVIDER=ollama
```

Ollama musi działać w tle (`ollama serve`, domyślnie `http://localhost:11434`).

### 4. Konfiguracja

```bash
copy .env.example .env          # Linux/macOS: cp .env.example .env
```

Najważniejsze zmienne (pełna lista w [config.py](src/tacticllm/config.py), prefiks `TACTIC_`):

| Zmienna | Domyślnie | Znaczenie |
|---|---|---|
| `TACTIC_REPORT_PROVIDER` | `anthropic` | `anthropic` (Claude) albo `ollama` |
| `ANTHROPIC_API_KEY` | – | klucz do raportu przez Claude |
| `TACTIC_ANNOTATOR` | `ollama` | `ollama` albo `gemini` |
| `GOOGLE_API_KEY` | – | klucz do Gemini |
| `TACTIC_OLLAMA_MODEL` | `qwen2.5vl:7b` | model wizyjny do opisów punktów |
| `TACTIC_OLLAMA_FRAME_WIDTH` | `896` | szerokość klatek dla modelu (na 8 GB VRAM: `672`) |
| `TACTIC_YOLO_WEIGHTS` | `yolo11l.pt` | wagi YOLO (pobierane automatycznie) |
| `TACTIC_DEVICE` | `0` | `0` = pierwsze GPU, `cpu` = bez GPU |
| `TACTIC_ANALYSIS_FPS` | `10` | ile klatek na sekundę analizuje YOLO |

## Przebieg pracy

Wszystkie pliki jednej analizy trafiają do katalogu `--run` (domyślnie `runs/mecz`).

### Dane meczu

Skopiuj [examples/match.yaml](examples/match.yaml) i uzupełnij pary i zawodników. `start_side: near` ma para, która w **pierwszym punkcie fragmentu** stoi bliżej kamery. `focus_team` decyduje, dla kogo są wskazówki w raporcie.

### Wszystko naraz

```bash
tacticllm analyze "https://www.youtube.com/watch?v=..." --start 1:02:00 --end 1:45:00 --match mecz.yaml --run runs/final
```

Kroki, które już mają wynik w katalogu (wideo, kalibracja, segmenty), są pomijane, a opisy punktów można wznowić po przerwaniu.

### Krok po kroku

```bash
tacticllm download "https://www.youtube.com/watch?v=..." --start 1:02:00 --end 1:45:00 -r runs/final
tacticllm calibrate -r runs/final --at 0:10
tacticllm track -r runs/final
tacticllm scores -r runs/final --box "85,50,510,132" --match mecz.yaml
tacticllm points -r runs/final
tacticllm annotate -r runs/final
tacticllm stats -r runs/final --match mecz.yaml
tacticllm report -r runs/final
tacticllm chat -r runs/final
```

1. **download** – pobiera film albo tylko fragment (link obsługiwany przez yt-dlp lub ścieżka do pliku).
2. **calibrate** – na klatce z czasu `--at` (musi to być widok z głównej kamery!) klikasz 4 narożniki podłogi kortu: dalszy-lewy, dalszy-prawy, bliższy-prawy, bliższy-lewy. Backspace cofa, Enter zapisuje. Gdy narożniki są poza kadrem, użyj `--kind service` i kliknij końce linii serwisowych przy ścianach. Bez okna: `--corners "x,y;x,y;x,y;x,y"`. Sprawdź `calibration_preview.jpg` – linie kortu muszą pokrywać się z obrazem.
3. **track** – wykrywa zawodników na ujęciach z głównej kamery i dzieli film na punkty (`segments.json`). Gdy realizator nie przełączy kamery między punktami, ujęcie jest dzielone na przerwie przed kolejnym serwisem.
4. **scores** – odczytuje tablicę wyników przed każdym punktem (lokalnie, `qwen3-vl:8b-instruct` w Ollamie) i ustala zwycięzcę punktu (`scores.json`). `--box` to prostokąt z grafiką wyniku w pikselach klatki (x1,y1,x2,y2 – zapamiętywany); bez niego model dostaje całą klatkę. Wiersze tablicy są dopasowywane do par po nazwiskach z pliku meczu, więc `--match` jest tu potrzebne. Ten zwycięzca ma pierwszeństwo przed oceną modelu z `annotate`.
5. **points** – lista wykrytych punktów do szybkiej weryfikacji.
6. **annotate** – model wizyjny opisuje każdy punkt: przebieg, zakończenie i taktyki (`annotations.json`). Najlepiej wypada Gemini (`--backend gemini`, darmowy plan: 20 zapytań dziennie na model; po wyczerpaniu limitu uruchom tę samą komendę następnego dnia). `--limit 10 --model qwen3-vl:8b-instruct --out ann_qwen3.json` pozwala porównać modele na kilku punktach.
7. **stats** – statystyki (`stats.json`) i mapy cieplne (`heatmaps/`).
8. **report** – raport taktyczny w Markdown (`report.md`).
9. **chat** – rozmowa z agentem o meczu, np. „dlaczego przegrali 2. seta?”.

### Pliki w katalogu analizy

| Plik | Zawartość |
|---|---|
| `video.mp4` | analizowany fragment |
| `calibration.json`, `reference.jpg`, `calibration_preview.jpg` | kalibracja kortu |
| `segments.json` | punkty i pozycje zawodników (metry) |
| `scores.json` | wynik z tablicy przed każdym punktem i zwycięzca punktu |
| `annotations.json` | opisy punktów z modelu wizyjnego |
| `stats.json`, `heatmaps/` | statystyki i mapy cieplne |
| `report.md` | raport dla trenera |

## Ograniczenia

- Brak śledzenia piłki – zwycięzca punktu pochodzi z tablicy wyników (`scores`), a przebieg i taktyki z modelu wizyjnego (`annotate`). Na próbnym fragmencie (18 punktów) tablica dała 18/18, Gemini ok. 80%, lokalne modele 7–8B ok. 50%.
- „drive” / „revés” to **pozycja** na korcie, a nie konkretny zawodnik – po zamianie stron w trakcie wymiany statystyki obu się mieszają.
- Ujęcia głównej kamery są rozpoznawane przez porównanie z klatką z kalibracji; zmiana kamery lub oświetlenia może zgubić punkty.
- Bez `scores` i `annotate` przerwy i powtórki z widokiem głównej kamery liczą się jako punkty, a raport nie wie, kto wygrywał.

## Testy

```bash
pytest
```

Testy nie wymagają GPU, wideo ani modeli.
