import os
import logging
import logging.handlers
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Telegram ──────────────────────────────────────────────────────────────
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")

# ── OpenAI chat ───────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-5-mini-2025-08-07")
MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "5"))
PROMPTS_FILE: str = "prompts.json"

# ── Sora video (separate credentials if proxy doesn't support /v1/videos) ─
SORA_API_KEY: str = os.getenv("SORA_API_KEY", OPENAI_API_KEY)
SORA_BASE_URL: str = os.getenv("SORA_BASE_URL", OPENAI_BASE_URL)
SORA_MODEL: str = os.getenv("SORA_MODEL", "sora-2")
SORA_SIZE: str = os.getenv("SORA_SIZE", "1280x720")         # 720x1280 | 1280x720 | 1024x1792 | 1792x1024
SORA_SECONDS: str = os.getenv("SORA_SECONDS", "4")          # "4" | "8" | "12"
SORA_POLL_INTERVAL_MS: int = int(os.getenv("SORA_POLL_INTERVAL_MS", "10000"))

# ── Pricing (USD per 1 M tokens / per second) ────────────────────────────
PRICE_INPUT_PER_1M: float = float(os.getenv("PRICE_INPUT_PER_1M", "0.15"))
PRICE_OUTPUT_PER_1M: float = float(os.getenv("PRICE_OUTPUT_PER_1M", "0.60"))
SORA_PRICE_PER_SEC: float = float(os.getenv("SORA_PRICE_PER_SEC", "0.10"))

# ── Logging ───────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = os.getenv("LOG_FILE", "bot.log")


def setup_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL.upper())
    root.addHandler(console)
    root.addHandler(file_handler)

    logging.getLogger("aiogram").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
