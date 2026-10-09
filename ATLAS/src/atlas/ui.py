"""Tiny terminal styling helpers (ANSI), disabled automatically when output isn't a terminal."""

from __future__ import annotations

import os
import sys
from typing import TextIO


class Style:
    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self.enabled = stream.isatty() and "NO_COLOR" not in os.environ and os.getenv("TERM") != "dumb"

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, t: str) -> str:
        return self._wrap("1", t)

    def dim(self, t: str) -> str:
        return self._wrap("2", t)

    def cyan(self, t: str) -> str:
        return self._wrap("36", t)

    def green(self, t: str) -> str:
        return self._wrap("32", t)

    def yellow(self, t: str) -> str:
        return self._wrap("33", t)

    def red(self, t: str) -> str:
        return self._wrap("31", t)
