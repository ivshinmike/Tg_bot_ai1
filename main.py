import asyncio
import base64
import logging
import time
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from openai import AsyncOpenAI

import config
import pricing
from memory import ConversationMemory, PromptManager

config.setup_logging()
log = logging.getLogger(__name__)

router = Router()

pm = PromptManager()
memory = ConversationMemory(pm)

chat_client = AsyncOpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
)

sora_client = AsyncOpenAI(
    api_key=config.SORA_API_KEY,
    base_url=config.SORA_BASE_URL,
)

MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

_video_locks: dict[int, bool] = {}
_image_locks: dict[int, bool] = {}


# ── /start ───────────────────────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: types.Message) -> None:
    if not message.from_user:
        return
    uid = message.from_user.id
    log.info("user=%d /start (username=%s)", uid, message.from_user.username)

    mode_name = pm.prompts[memory.get_mode(uid)]["name"]
    await message.answer(
        f"Привет, {message.from_user.first_name}!\n\n"
        f"Текущий режим: <b>{mode_name}</b>\n\n"
        "Команды:\n"
        "/mode — выбрать режим\n"
        "/reset — очистить историю диалога\n"
        "/image &lt;описание&gt; — сгенерировать картинку (DALL·E)\n"
        "/video &lt;описание&gt; — сгенерировать видео (Sora-2)",
        parse_mode="HTML",
    )


# ── /reset ───────────────────────────────────────────────────────────────

@router.message(Command("reset"))
async def cmd_reset(message: types.Message) -> None:
    if not message.from_user:
        return
    log.info("user=%d /reset", message.from_user.id)
    memory.reset(message.from_user.id)
    await message.answer("🗑 История диалога очищена.")


# ── /mode ────────────────────────────────────────────────────────────────

@router.message(Command("mode"))
async def cmd_mode(message: types.Message) -> None:
    if not message.from_user:
        return
    uid = message.from_user.id
    current = memory.get_mode(uid)
    log.info("user=%d /mode (current=%s)", uid, current)

    buttons = []
    for key, name, desc in pm.list_modes():
        mark = " ✓" if key == current else ""
        buttons.append(
            [InlineKeyboardButton(
                text=f"{name}{mark}",
                callback_data=f"setmode:{key}",
            )]
        )
    await message.answer(
        "Выберите режим:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("setmode:"))
async def cb_set_mode(callback: types.CallbackQuery) -> None:
    if not callback.data or not callback.from_user or not callback.message:
        await callback.answer()
        return

    mode_key = callback.data.split(":", 1)[1]
    uid = callback.from_user.id

    if not pm.mode_exists(mode_key):
        log.warning("user=%d tried unknown mode '%s'", uid, mode_key)
        await callback.answer("Режим не найден.", show_alert=True)
        return

    memory.set_mode(uid, mode_key)
    name = pm.prompts[mode_key]["name"]
    log.info("user=%d selected mode '%s'", uid, mode_key)

    if isinstance(callback.message, types.Message):
        await callback.message.edit_text(
            f"Режим переключён на: <b>{name}</b>", parse_mode="HTML",
        )
    await callback.answer()


# ── /image ───────────────────────────────────────────────────────────────

@router.message(Command("image"))
async def cmd_image(message: types.Message) -> None:
    if not message.from_user or not message.text:
        return

    uid = message.from_user.id
    prompt = message.text.partition(" ")[2].strip()
    if not prompt:
        await message.answer(
            "Использование: /image &lt;описание картинки&gt;\n\n"
            f"Параметры: модель <b>{config.DALLE_MODEL}</b>, "
            f"размер {config.DALLE_SIZE}",
            parse_mode="HTML",
        )
        return

    if _image_locks.get(uid):
        await message.answer("⏳ У вас уже генерируется изображение. Дождитесь завершения.")
        return
    _image_locks[uid] = True

    log.info(
        "user=%d /image prompt='%s' model=%s size=%s",
        uid, prompt[:120], config.DALLE_MODEL, config.DALLE_SIZE,
    )

    placeholder = await message.answer("🎨 Генерация изображения…")

    t0 = time.monotonic()
    try:
        resp = await chat_client.images.generate(
            model=config.DALLE_MODEL,
            prompt=prompt,
            n=1,
            size=config.DALLE_SIZE,  # type: ignore[arg-type]
            quality=config.DALLE_QUALITY,  # type: ignore[arg-type]
            response_format="b64_json",
        )
        elapsed = time.monotonic() - t0

        if not resp.data:
            await placeholder.edit_text("❌ API вернул пустой ответ.")
            return

        b64 = getattr(resp.data[0], "b64_json", None)
        if not b64:
            await placeholder.edit_text("❌ API вернул пустой ответ.")
            return

        image_bytes = base64.b64decode(b64)
        log.info(
            "user=%d image ok in %.2fs size=%.0f KB",
            uid, elapsed, len(image_bytes) / 1024,
        )

        cost_text = await pricing.format_image_cost()
        log.info("user=%d image cost: %s", uid, cost_text.replace("\n", " | "))

        await message.answer_photo(
            photo=BufferedInputFile(image_bytes, filename="image.png"),
            caption=f"🎨 Готово за {elapsed:.1f} сек\n\n{cost_text}",
        )
        await placeholder.delete()

    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.exception("user=%d image EXCEPTION in %.1fs: %s", uid, elapsed, exc)
        await placeholder.edit_text(f"❌ Ошибка генерации: {exc}")
    finally:
        _image_locks.pop(uid, None)


# ── /video ───────────────────────────────────────────────────────────────

@router.message(Command("video"))
async def cmd_video(message: types.Message) -> None:
    if not message.from_user or not message.text:
        return

    uid = message.from_user.id
    prompt = message.text.partition(" ")[2].strip()
    if not prompt:
        await message.answer(
            "Использование: /video &lt;описание видео&gt;\n\n"
            f"Параметры: модель <b>{config.SORA_MODEL}</b>, "
            f"размер {config.SORA_SIZE}, {config.SORA_SECONDS} сек",
            parse_mode="HTML",
        )
        return

    if _video_locks.get(uid):
        await message.answer("⏳ У вас уже генерируется видео. Дождитесь завершения.")
        return
    _video_locks[uid] = True

    log.info(
        "user=%d /video prompt='%s' model=%s size=%s sec=%s",
        uid, prompt[:120], config.SORA_MODEL, config.SORA_SIZE, config.SORA_SECONDS,
    )

    status_msg = await message.answer(
        f"🎬 Генерация видео запущена…\n"
        f"Модель: <b>{config.SORA_MODEL}</b> | "
        f"{config.SORA_SIZE} | {config.SORA_SECONDS} сек\n\n"
        f"⏳ Это может занять несколько минут.",
        parse_mode="HTML",
    )

    t0 = time.monotonic()
    try:
        video = await sora_client.videos.create_and_poll(
            model=config.SORA_MODEL,
            prompt=prompt,
            size=config.SORA_SIZE,  # type: ignore[arg-type]
            seconds=config.SORA_SECONDS,  # type: ignore[arg-type]
            poll_interval_ms=config.SORA_POLL_INTERVAL_MS,
        )
        elapsed = time.monotonic() - t0

        if video.status != "completed":
            error_detail = ""
            if video.error:
                error_detail = getattr(video.error, "message", str(video.error))
            log.error(
                "user=%d video FAILED in %.1fs status=%s error=%s",
                uid, elapsed, video.status, error_detail,
            )
            await status_msg.edit_text(f"❌ Генерация не удалась: {error_detail or video.status}")
            return

        log.info("user=%d video COMPLETED in %.1fs id=%s", uid, elapsed, video.id)

        file_path = MEDIA_DIR / f"{uid}_{video.id}.mp4"
        content = await sora_client.videos.download_content(video.id, variant="video")
        content.write_to_file(str(file_path))
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        log.info("user=%d video saved %s (%.1f MB)", uid, file_path.name, file_size_mb)

        secs = int(config.SORA_SECONDS)
        cost_text = await pricing.format_video_cost(secs)
        log.info("user=%d video cost: %s", uid, cost_text.replace("\n", " | "))

        await message.answer_video(
            video=FSInputFile(file_path),
            caption=f"🎬 Готово за {elapsed:.0f} сек\n\n{cost_text}",
        )
        await status_msg.delete()
        file_path.unlink(missing_ok=True)

    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.exception("user=%d video EXCEPTION in %.1fs: %s", uid, elapsed, exc)
        await status_msg.edit_text(f"❌ Ошибка генерации видео: {exc}")
    finally:
        _video_locks.pop(uid, None)


# ── Обработка текстовых сообщений ────────────────────────────────────────

@router.message(F.text)
async def handle_text(message: types.Message) -> None:
    if not message.from_user or not message.text:
        return

    uid = message.from_user.id
    mode = memory.get_mode(uid)
    log.info("user=%d message (%d chars) mode=%s", uid, len(message.text), mode)

    memory.add_message(uid, "user", message.text)
    placeholder = await message.answer("⏳")

    t0 = time.monotonic()
    try:
        response = await chat_client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=memory.build_messages(uid),
        )
        elapsed = time.monotonic() - t0
        reply = response.choices[0].message.content or "(пустой ответ)"

        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        log.info(
            "user=%d openai ok in %.2fs tokens=[in=%d, out=%d] reply=%d chars",
            uid, elapsed, tokens_in, tokens_out, len(reply),
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.exception("user=%d openai FAILED in %.2fs: %s", uid, elapsed, exc)
        reply = f"Ошибка при обращении к API: {exc}"
        await placeholder.edit_text(reply)
        return

    memory.add_message(uid, "assistant", reply)

    cost_text = await pricing.format_chat_cost(tokens_in, tokens_out)
    log.info("user=%d cost: %s", uid, cost_text.replace("\n", " | "))

    full_reply = f"{reply}\n\n<i>{cost_text}</i>"
    await placeholder.edit_text(full_reply, parse_mode="HTML")


# ── Запуск ───────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("Pre-fetching USD/RUB rate…")
    rate = await pricing.get_usd_rub()
    log.info("USD/RUB = %.4f", rate)

    bot = Bot(token=config.BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    log.info(
        "Bot starting | chat=%s | sora=%s | base_url=%s | history=%d",
        config.MODEL_NAME, config.SORA_MODEL,
        config.OPENAI_BASE_URL, config.MAX_HISTORY,
    )
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
