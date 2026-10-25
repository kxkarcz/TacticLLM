from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Ustawienia z zmiennych środowiskowych (prefiks TACTIC_) lub pliku .env."""

    model_config = SettingsConfigDict(env_prefix="TACTIC_", env_file=".env", extra="ignore")

    # raport i czat: "anthropic" (Claude, jedno wywołanie na mecz) albo "ollama" (lokalnie, za darmo)
    report_provider: str = "anthropic"
    report_model: str = "claude-sonnet-5-5"
    ollama_report_model: str = "qwen3:14b"
    ollama_report_ctx: int = 32768
    report_language: str = "polski"

    # adnotacje punktów: "ollama" (lokalnie, za darmo) albo "gemini" (API Google)
    annotator: str = "ollama"
    video_model: str = "gemini-2.5-flash"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5vl:7b"
    ollama_fps: float = 3.0  # ile klatek na sekundę punktu ogląda model (cały punkt, od serwisu do końca)
    ollama_max_frames: int = 64  # limit na bardzo długie wymiany (wtedy fps jest zmniejszany)
    ollama_frame_width: int = 672

    yolo_weights: str = "yolo11m.pt"
    device: str = "0"  # "0" = pierwsze GPU, "cpu" = bez GPU
    yolo_imgsz: int = 1280  # większa rozdzielczość = lepiej widać dalszą parę
    yolo_conf: float = 0.25

    analysis_fps: float = 10.0  # ile klatek na sekundę analizujemy
    main_view_threshold: float = 0.55  # podobieństwo do klatki referencyjnej (główna kamera)
    min_point_seconds: float = 3.0  # krótsze ujęcia głównej kamery nie są punktami
    court_margin_m: float = 1.5  # tolerancja poza kortem (gra za drzwiami itp.)


settings = Settings()
