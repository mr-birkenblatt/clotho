# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Generator, Optional

import praw

class ModNoteMixin:
    def create_note(
        self, *, label: Optional[str] = ..., note: str, **other_settings,
        ) -> praw.models.ModNote: ...

    def author_notes(
        self, **generator_kwargs,
        ) -> Generator['praw.models.ModNote', None, None]: ...
