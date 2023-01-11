from typing import TypedDict


LoggerContext = TypedDict('LoggerContext', {
    "module": str | None,
})
LoggerContextUpdate = TypedDict('LoggerContextUpdate', {
    "module": str,
}, total=False)


DEFAULT_LOGGER_CONTEXT: LoggerContext = {
    "module": None,
}


class LoggerBackend:
    def log_note(self, context: LoggerContext, name: str, msg: str) -> None:
        raise NotImplementedError()

    def log_count(self, context: LoggerContext, name: str) -> None:
        raise NotImplementedError()

    def log_num(self, context: LoggerContext, name: str, num: float) -> None:
        raise NotImplementedError()

    def on_context_change(
            self,
            old_context: LoggerContext,
            new_context: LoggerContext) -> None:
        raise NotImplementedError()
