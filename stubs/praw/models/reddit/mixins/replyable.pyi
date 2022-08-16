from ....const import API_PATH as API_PATH

class ReplyableMixin:
    def reply(self, *, body: str): ...
