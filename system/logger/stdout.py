from system.logger.backend import LoggerBackend, LoggerContext


class StdoutLogger(LoggerBackend):
    def log_note(self, context: LoggerContext, name: str, msg: str) -> None:
        print(msg)

    def log_count(self, context: LoggerContext, name: str) -> None:
        pass

    def log_num(self, context: LoggerContext, name: str, num: float) -> None:
        pass

    def on_context_change(
            self,
            old_context: LoggerContext,
            new_context: LoggerContext) -> None:
        pass
