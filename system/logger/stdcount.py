from system.logger.backend import LoggerBackend, LoggerContext


class StdcountLogger(LoggerBackend):
    def log_note(self, context: LoggerContext, name: str, msg: str) -> None:
        print(msg)

    def log_count(self, context: LoggerContext, name: str) -> None:
        print(context, name, "+1")

    def log_num(self, context: LoggerContext, name: str, num: float) -> None:
        print(context, name, f"+{num}" if num >= 0 else f"{num}")

    def on_context_change(
            self,
            old_context: LoggerContext,
            new_context: LoggerContext) -> None:
        pass
