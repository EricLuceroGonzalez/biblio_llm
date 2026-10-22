"""Configuración estable (``.env``). El tema y el idioma van por la CLI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Prefijo propio: variables genéricas de otros proyectos en la shell
    # (TEMPLATES_DIR, MODEL, LOG_DIR…) no pueden colarse en esta configuración.
    model_config = SettingsConfigDict(
        env_prefix="BIBLIO_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_url: str = "http://localhost:11434"
    model: str = "deepseek-r1:14b"
    request_timeout: float = Field(default=900.0, gt=0)  # 14B en CPU puede tardar
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 42
    num_ctx: int = Field(default=8192, ge=2048)
    num_predict: int = Field(default=8192, ge=512)  # incluye los tokens de razonamiento
    think: bool = True

    templates_dir: Path = Path().cwd() / "plantillas"
    runs_dir: Path = Path().cwd() / "runs"

    # Valores por defecto de la CLI (se pueden sobrescribir con --idioma, etc.)
    default_idioma: str = "español"
    default_audiencia: str = "investigadores y estudiantes de doctorado"

    log_dir: Path = Path("logs")
    log_level: str = "INFO"

    @field_validator("ollama_url")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
