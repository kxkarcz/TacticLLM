from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Ustawienia z zmiennych środowiskowych (prefiks TACTIC_) lub pliku .env."""

    model_config = SettingsConfigDict(env_prefix="TACTIC_", env_file=".env", extra="ignore")

    # klucze bez prefiksu TACTIC_ – biblioteki szukają ich w zmiennych środowiskowych,
    # a nie w .env, więc przekazujemy je jawnie
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(default=None, validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"))

    # raport i czat: "anthropic" (Claude), "gemini" (darmowy plan wystarcza – raport to jedno zapytanie)
    # albo "ollama" (lokalnie, za darmo)
    report_provider: str = "anthropic"
    report_model: str = "claude-sonnet-5-5"
    gemini_report_model: str = "gemini-3.6-flash"
    ollama_report_model: str = "qwen3:14b"
    ollama_report_ctx: int = 32768
    report_language: str = "polski"

    # adnotacje punktów: "ollama" (lokalnie, za darmo) albo "gemini" (API Google)
    annotator: str = "ollama"
    video_model: str = "gemini-3.6-flash"  # darmowy plan: 20 zapytań dziennie na model
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5vl:7b"
    ollama_fps: float = 3.0  # ile klatek na sekundę punktu ogląda model (cały punkt, od serwisu do końca)
    ollama_max_frames: int = 64  # limit na bardzo długie wymiany (wtedy fps jest zmniejszany)
    ollama_frame_width: int = 896  # na kartach 8 GB: 672
    # odczyt tablicy wyników (zwycięzca punktu) – lokalnie, qwen3-vl czyta grafikę bezbłędnie
    scoreboard_model: str = "qwen3-vl:8b-instruct"

    yolo_weights: str = "yolo11l.pt"
    device: str = "0"  # "0" = pierwsze GPU, "cpu" = bez GPU
    yolo_imgsz: int = 1280  # większa rozdzielczość = lepiej widać dalszą parę
    yolo_conf: float = 0.25

    analysis_fps: float = 10.0  # ile klatek na sekundę analizujemy
    main_view_threshold: float = 0.55  # podobieństwo do klatki referencyjnej (główna kamera)
    min_point_seconds: float = 3.0  # krótsze ujęcia głównej kamery nie są punktami
    court_margin_m: float = 1.5  # tolerancja poza kortem (gra za drzwiami itp.)


settings = Settings()
