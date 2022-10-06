# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Generator, Optional

import praw


class ModNoteMixin:
    def create_note(
        self, *, label: Optional[str] = ..., note: str, **other_settings: Any,
        ) -> praw.models.ModNote: ...

    def author_notes(
        self, **generator_kwargs: Any,
        ) -> Generator['praw.models.ModNote', None, None]: ...
