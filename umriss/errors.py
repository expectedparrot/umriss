from __future__ import annotations

from typing import Any


class UmrissError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
        next_steps: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.context = context or {}
        self.next_steps = next_steps or []
