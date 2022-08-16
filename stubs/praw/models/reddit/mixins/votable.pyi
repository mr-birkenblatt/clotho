from ....const import API_PATH as API_PATH

class VotableMixin:
    def clear_vote(self) -> None: ...
    def downvote(self) -> None: ...
    def upvote(self) -> None: ...
