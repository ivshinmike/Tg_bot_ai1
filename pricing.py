from __future__ import annotations

import logging
import time

import httpx

import config

log = logging.getLogger(__name__)

_usd_rub_cache: tuple[float, float] = (0.0, 0.0)  # (rate, timestamp)
_CACHE_TTL = 3600  # 1 hour


async def get_usd_rub() -> float:
    """Fetch current USD/RUB rate from CBR, cached for 1 hour."""
    global _usd_rub_cache
    rate, fetched_at = _usd_rub_cache
    if rate and (time.time() - fetched_at) < _CACHE_TTL:
        return rate

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://www.cbr-xml-daily.ru/daily_json.js")
            resp.raise_for_status()
            data = resp.json()
        rate = float(data["Valute"]["USD"]["Value"])
        _usd_rub_cache = (rate, time.time())
        log.info("CBR USD/RUB updated: %.4f", rate)
        return rate
    except Exception:
        log.exception("Failed to fetch CBR rate, using fallback")
        return _usd_rub_cache[0] if _usd_rub_cache[0] else 90.0


def calc_chat_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * config.PRICE_INPUT_PER_1M / 1_000_000
        + completion_tokens * config.PRICE_OUTPUT_PER_1M / 1_000_000
    )


def calc_video_cost_usd(seconds: int) -> float:
    return seconds * config.SORA_PRICE_PER_SEC


async def format_chat_cost(
    prompt_tokens: int,
    completion_tokens: int,
) -> str:
    cost_usd = calc_chat_cost_usd(prompt_tokens, completion_tokens)
    rate = await get_usd_rub()
    cost_rub = cost_usd * rate

    return (
        f"📊 Токены: {prompt_tokens} вх / {completion_tokens} вых\n"
        f"💰 Стоимость: ${cost_usd:.6f} ≈ {cost_rub:.4f} ₽"
    )


async def format_video_cost(seconds: int) -> str:
    cost_usd = calc_video_cost_usd(seconds)
    rate = await get_usd_rub()
    cost_rub = cost_usd * rate

    return (
        f"📊 Видео: {seconds} сек, модель {config.SORA_MODEL}\n"
        f"💰 Стоимость: ${cost_usd:.2f} ≈ {cost_rub:.2f} ₽"
    )


def calc_image_cost_usd() -> float:
    return config.DALLE_PRICE_PER_IMAGE


async def format_image_cost() -> str:
    cost_usd = calc_image_cost_usd()
    rate = await get_usd_rub()
    cost_rub = cost_usd * rate

    return (
        f"📊 Изображение: {config.DALLE_MODEL}, {config.DALLE_SIZE}\n"
        f"💰 Стоимость: ${cost_usd:.4f} ≈ {cost_rub:.4f} ₽"
    )
