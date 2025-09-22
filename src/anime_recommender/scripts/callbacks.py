import logging
import threading

from pathlib import Path
from logging.config import dictConfig

import yaml


class EventCB:
    """Custom Callback that can be reconfigured from any YAML file."""

    _configured = False
    _lock = threading.Lock()
    _CONFIG = Path("src") / "config" / "log-config.yaml"

    def __init__(self) -> None:
        self._ensure_configured()

    def _ensure_configured(self) -> None:
        if not self._configured:
            with self._lock:
                if not self._configured:
                    self._configure_logging()

    @classmethod
    def _configure_logging(cls, config_file: str | Path | None = None) -> None:
        """Configure logging once at startup."""

        config_path = Path(config_file or cls._CONFIG)

        try:
            if config_path.exists():
                with config_path.open("rt") as stream:
                    config = yaml.safe_load(stream)
                dictConfig(config=config)
            else:
                cls._setup_default_logging()
        except Exception:
            cls._setup_default_logging()

        cls._configured = True

    @staticmethod
    def _setup_default_logging() -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance - Configuration already done in the constructor."""

        return logging.getLogger(f"{self.__class__.__name__}")

    @classmethod
    def reconfigure_from_file(cls, config_file: str | Path) -> None:
        """Allow manual configuration with another file."""

        with cls._lock:
            cls._configure_logging(config_file=config_file)
