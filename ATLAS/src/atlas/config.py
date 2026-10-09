"""Configuration loaded from environment variables and .env files.

Lookup order for .env (first found wins; real environment variables always take priority):
  1. $ATLAS_ENV_FILE
  2. .env in the ATLAS install folder (e.g. /data/ATLAS/.env)
  3. ~/.config/atlas/.env
A project's own .env is deliberately never read: it may hold unrelated secrets.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_LOCAL_MODEL = "qwen2.5-coder:7b"
# src/atlas/config.py -> the ATLAS folder (valid for the editable install setup.sh performs)
INSTALL_DIR = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, got {value}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def find_env_file() -> Path | None:
    candidates = []
    if os.getenv("ATLAS_ENV_FILE"):
        candidates.append(Path(os.environ["ATLAS_ENV_FILE"]).expanduser())
    candidates.append(INSTALL_DIR / ".env")
    candidates.append(Path.home() / ".config" / "atlas" / ".env")
    for path in candidates:
        if path.is_file():
            return path
    return None


def env_file_is_private(path: Path) -> bool:
    """True if the file is readable only by its owner (no group/other permissions)."""
    mode = path.stat().st_mode
    return not (mode & (stat.S_IRWXG | stat.S_IRWXO))


@dataclass(frozen=True)
class Settings:
    workspace: Path
    data_dir: Path
    ollama_host: str = "http://localhost:11434"
    local_model: str = DEFAULT_LOCAL_MODEL
    num_ctx: int = 8192
    temperature: float = 0.2
    request_timeout: int = 300
    cloud_base_url: str = ""
    cloud_model: str = ""
    cloud_api_key: str = field(default="", repr=False)  # never printed
    cloud_allowed: bool = False
    log_level: str = "INFO"
    env_file: Path | None = None

    @property
    def cloud_configured(self) -> bool:
        return bool(self.cloud_base_url and self.cloud_model and self.cloud_api_key)

    @property
    def log_file(self) -> Path:
        return self.data_dir / "atlas.log"


def load_settings(workspace: str | Path | None = None, env_file: Path | None = None) -> Settings:
    """Read settings. `workspace` overrides ATLAS_WORKSPACE (defaults to the current directory)."""
    env_path = env_file or find_env_file()
    if env_path is not None:
        load_dotenv(env_path, override=False)

    ws = Path(workspace or os.getenv("ATLAS_WORKSPACE") or Path.cwd()).expanduser().resolve()
    data_dir = Path(
        os.getenv("ATLAS_DATA_DIR") or Path.home() / ".local" / "share" / "atlas"
    ).expanduser()

    log_level = os.getenv("ATLAS_LOG_LEVEL", "INFO").upper()
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ConfigError(f"ATLAS_LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR, got {log_level!r}")

    temperature = _env_float("ATLAS_TEMPERATURE", 0.2)
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError(f"ATLAS_TEMPERATURE must be between 0 and 2, got {temperature}")

    return Settings(
        workspace=ws,
        data_dir=data_dir,
        ollama_host=os.getenv("OLLAMA_HOST_URL", "http://localhost:11434").rstrip("/"),
        local_model=os.getenv("ATLAS_LOCAL_MODEL", DEFAULT_LOCAL_MODEL),
        num_ctx=_env_int("ATLAS_NUM_CTX", 8192, minimum=512),
        temperature=temperature,
        request_timeout=_env_int("ATLAS_REQUEST_TIMEOUT", 300),
        cloud_base_url=os.getenv("ATLAS_CLOUD_BASE_URL", "").rstrip("/"),
        cloud_model=os.getenv("ATLAS_CLOUD_MODEL", ""),
        cloud_api_key=os.getenv("ATLAS_CLOUD_API_KEY", ""),
        cloud_allowed=_env_bool("ATLAS_CLOUD_ALLOWED", False),
        log_level=log_level,
        env_file=env_path,
    )
