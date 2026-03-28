from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field

from openai.types.chat import ChatCompletionMessageParam

import config

log = logging.getLogger(__name__)


@dataclass
class UserSession:
    mode: str = ""
    history: list[dict[str, str]] = field(default_factory=list)


class PromptManager:
    """Loads and provides access to system prompts from a JSON file."""

    def __init__(self, path: str = config.PROMPTS_FILE) -> None:
        raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        self.default_mode: str = raw["default_prompt"]
        self.prompts: dict[str, dict[str, str]] = raw["prompts"]
        log.info(
            "Loaded %d prompt mode(s): %s (default: %s)",
            len(self.prompts),
            ", ".join(self.prompts),
            self.default_mode,
        )

    def get_system_prompt(self, mode: str) -> str:
        return self.prompts[mode]["system_prompt"]

    def list_modes(self) -> list[tuple[str, str, str]]:
        """Return list of (key, name, description) for every available mode."""
        return [
            (key, p["name"], p["description"])
            for key, p in self.prompts.items()
        ]

    def mode_exists(self, mode: str) -> bool:
        return mode in self.prompts


class ConversationMemory:
    """Per-user conversation history and mode selection."""

    def __init__(self, prompt_manager: PromptManager) -> None:
        self.pm = prompt_manager
        self._sessions: dict[int, UserSession] = {}

    def _get(self, user_id: int) -> UserSession:
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession(mode=self.pm.default_mode)
        return self._sessions[user_id]

    def set_mode(self, user_id: int, mode: str) -> None:
        session = self._get(user_id)
        old_mode = session.mode
        session.mode = mode
        session.history.clear()
        log.info("user=%d mode changed %s → %s (history cleared)", user_id, old_mode, mode)

    def get_mode(self, user_id: int) -> str:
        return self._get(user_id).mode

    def add_message(self, user_id: int, role: str, content: str) -> None:
        session = self._get(user_id)
        session.history.append({"role": role, "content": content})
        trimmed = False
        if len(session.history) > config.MAX_HISTORY * 2:
            session.history = session.history[-(config.MAX_HISTORY * 2):]
            trimmed = True
        log.debug(
            "user=%d +%s (%d chars) history_len=%d%s",
            user_id, role, len(content), len(session.history),
            " [trimmed]" if trimmed else "",
        )

    def build_messages(self, user_id: int) -> list[ChatCompletionMessageParam]:
        """Build the full message list for the OpenAI API call."""
        session = self._get(user_id)
        system = self.pm.get_system_prompt(session.mode)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
        ]
        for msg in session.history:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            else:
                messages.append({"role": "assistant", "content": msg["content"]})
        return messages

    def reset(self, user_id: int) -> None:
        session = self._get(user_id)
        count = len(session.history)
        session.history.clear()
        log.info("user=%d history reset (%d messages removed)", user_id, count)
