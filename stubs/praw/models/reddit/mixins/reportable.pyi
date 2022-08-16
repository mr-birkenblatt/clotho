from ....const import API_PATH as API_PATH

class ReportableMixin:
    def report(self, reason: str): ...
