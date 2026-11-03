"""Logging configuration with credential redaction.

Named ``logging_config`` rather than ``logging`` so it cannot be confused with
the standard library module at a glance.

The redaction filter is a belt-and-braces measure. The primary defence is that
no code path logs a credential in the first place; this filter guarantees that a
future careless log call, or a third-party library that echoes its own
configuration, still cannot emit a registered secret.
"""

from __future__ import annotations

import logging
from typing import Any

from infrastructure.utils.paths import resolve

REDACTION_PLACEHOLDER = "***REDACTED***"

# Minimum length for a value to be registered. Short values (e.g. a two-letter
# username) would match too much unrelated text and turn logs into noise.
_MIN_SECRET_LENGTH = 6

_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Register a value to be scrubbed from every log record.

    Called by credential providers the moment a secret is read, so the filter
    knows about it before any code has a chance to log it.

    Values shorter than six characters are ignored: they would match ordinary
    text and make the logs unreadable without adding real protection.
    """
    if value and len(value) >= _MIN_SECRET_LENGTH:
        _secrets.add(value)


def clear_secrets() -> None:
    """Forget all registered secrets. Intended for test isolation."""
    _secrets.clear()


def redact(text: str) -> str:
    """Replace every registered secret in ``text`` with a placeholder."""
    for secret in _secrets:
        if secret in text:
            text = text.replace(secret, REDACTION_PLACEHOLDER)
    return text


class SecretRedactingFilter(logging.Filter):
    """Scrubs registered secrets from log records before they are emitted."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite the record's message in place, then always allow it through."""
        if not _secrets:
            return True
        try:
            message = record.getMessage()
        except (TypeError, ValueError):  # pragma: no cover - malformed log call
            return True
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


def configure_logging(config: dict[str, Any] | None = None) -> None:
    """Configure root logging from the ``logging`` section of the app config.

    Idempotent: repeated calls replace the handlers this function installed
    rather than stacking duplicates.

    Args:
        config: Full application config. When omitted, sensible defaults are
            used so that logging still works before config is available.
    """
    settings: dict[str, Any] = (config or {}).get("logging", {}) or {}
    level_name = str(settings.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(
        fmt=str(settings.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")),
        datefmt=str(settings.get("date_format", "%Y-%m-%d %H:%M:%S")),
    )
    redactor = SecretRedactingFilter()

    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, "_fedsecureai", False)]:
        root.removeHandler(handler)
        handler.close()

    handlers: list[logging.Handler] = []
    if settings.get("console", True):
        handlers.append(logging.StreamHandler())

    log_file = settings.get("file")
    if log_file:
        path = resolve(str(log_file))
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(redactor)
        handler._fedsecureai = True  # type: ignore[attr-defined]  # marks ours for replacement
        root.addHandler(handler)

    root.setLevel(level)


__all__ = [
    "REDACTION_PLACEHOLDER",
    "SecretRedactingFilter",
    "clear_secrets",
    "configure_logging",
    "redact",
    "register_secret",
]
