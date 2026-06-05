"""
Structured logger for pmprov middleware errors and warnings.

Silent by default (NullHandler only). Enable logging explicitly::

    from tracker import enable_logging
    enable_logging()                          # WARNING + ERROR to stderr
    enable_logging(level="DEBUG")             # all levels
    enable_logging(file="pmprov.log")         # write to file instead
"""
from __future__ import annotations

import logging
import sys
from typing import Any

_logger = logging.getLogger("pmprov")
_logger.addHandler(logging.NullHandler())

_FMT = "[pmprov] %(asctime)s %(levelname)s %(component)s – %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger() -> logging.Logger:
    """Return the pmprov logger for external configuration."""
    return _logger


def enable_logging(level: str | int = "WARNING", file: str | None = None) -> None:
    """
    Enable pmprov log output.

    Parameters
    ----------
    level:
        Minimum log level — ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``,
        or the corresponding ``logging`` integer constants.
    file:
        Optional path to a log file. If omitted, output goes to stderr.
    """
    numeric = logging.getLevelNamesMapping().get(level.upper(), logging.WARNING) if isinstance(level, str) else level
    handler: logging.Handler = (
        logging.FileHandler(file) if file else logging.StreamHandler(sys.stderr)
    )
    handler.setFormatter(logging.Formatter(fmt=_FMT, datefmt=_DATEFMT))
    if not any(type(h) is type(handler) and getattr(h, "baseFilename", None) == getattr(handler, "baseFilename", None)
               for h in _logger.handlers):
        _logger.addHandler(handler)
    _logger.setLevel(numeric)


def log_storage_error(exc: Exception, **ctx: Any) -> None:
    component = ctx.pop("component", "storage")
    _logger.error("%s%s", exc, _fmt_ctx(ctx), exc_info=exc, extra={"component": component})


def log_trace_warning(msg: str, **ctx: Any) -> None:
    component = ctx.pop("component", "runtime.trace_step")
    _logger.warning("%s%s", msg, _fmt_ctx(ctx), extra={"component": component})


def _fmt_ctx(ctx: dict[str, Any]) -> str:
    if not ctx:
        return ""
    return " (" + ", ".join(f"{k}={v!r}" for k, v in ctx.items()) + ")"
