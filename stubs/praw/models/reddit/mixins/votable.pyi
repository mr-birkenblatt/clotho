# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from ....const import API_PATH as API_PATH


class VotableMixin:
    def clear_vote(self) -> None: ...
    def downvote(self) -> None: ...
    def upvote(self) -> None: ...
