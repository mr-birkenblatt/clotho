import contextlib
import threading
from typing import Callable, cast, Iterator

from system.logger.backend import (
    DEFAULT_LOGGER_CONTEXT,
    LoggerBackend,
    LoggerContext,
    LoggerContextUpdate,
)


LOCAL = threading.local()
TL_LOGGER_CTX = "logger_context"


class LoggerFrontend:
    def __init__(self) -> None:
        self._backends: list[LoggerBackend] = []

    def register_backend(self, backend: LoggerBackend) -> None:
        self._backends.append(backend)

    def _get_context(self) -> LoggerContext:
        return getattr(LOCAL, TL_LOGGER_CTX, DEFAULT_LOGGER_CONTEXT)

    def _set_context(self, new_context: LoggerContext) -> None:
        old_context = self._get_context()
        setattr(LOCAL, TL_LOGGER_CTX, new_context)
        self._on_context_change(old_context)

    def _with_context(self, context: LoggerContextUpdate) -> None:
        # FIXME: type should be inferable
        new_context = cast(LoggerContext, {
            key: context.get(key, value)
            for key, value in self._get_context().items()
        })
        self._set_context(new_context)

    @contextlib.contextmanager
    def context(
            self, context: LoggerContextUpdate) -> Iterator['LoggerFrontend']:
        ctx = self._get_context()
        self._with_context(context)
        try:
            yield self
        finally:
            self._set_context(ctx)

    def _apply_call(
            self,
            call: Callable[[LoggerBackend, LoggerContext], None]) -> None:
        ctx = self._get_context()
        for backend in self._backends:
            call(backend, ctx)

    def _on_context_change(
            self,
            old_context: LoggerContext) -> None:
        self._apply_call(
            lambda backend, ctx: backend.on_context_change(old_context, ctx))

    def log_count(self, name: str) -> None:
        self._apply_call(lambda backend, ctx: backend.log_count(ctx, name))

    def log_num(self, name: str, num: float) -> None:
        self._apply_call(lambda backend, ctx: backend.log_num(ctx, name, num))


ROOT_LOGGER = LoggerFrontend()


def get_logger() -> LoggerFrontend:
    return ROOT_LOGGER


@contextlib.contextmanager
def logger_context(context: LoggerContextUpdate) -> Iterator[LoggerFrontend]:
    with get_logger().context(context) as logger:
        yield logger
