# Stubs for pandas.io.html (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,redefined-outer-name,invalid-name
# pylint: disable=relative-beyond-top-level,arguments-differ
# pylint: disable=no-member,too-few-public-methods,keyword-arg-before-vararg
# pylint: disable=super-init-not-called,abstract-method,redefined-builtin
# pylint: disable=unused-import,useless-import-alias,signature-differs
# pylint: disable=blacklisted-name,c-extension-no-member


from typing import Any, Optional

class _HtmlFrameParser:
    io: Any = ...
    match: Any = ...
    attrs: Any = ...
    encoding: Any = ...
    displayed_only: Any = ...

    def __init__(
            self, io: Any, match: Any, attrs: Any, encoding: Any,
            displayed_only: Any) -> None:
        ...

    def parse_tables(self) -> Any:
        ...


class _BeautifulSoupHtml5LibFrameParser(_HtmlFrameParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...


class _LxmlFrameParser(_HtmlFrameParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...


def read_html(
        io: Any, match: str = ..., flavor: Optional[Any] = ...,
        header: Optional[Any] = ..., index_col: Optional[Any] = ...,
        skiprows: Optional[Any] = ..., attrs: Optional[Any] = ...,
        parse_dates: bool = ..., thousands: str = ...,
        encoding: Optional[Any] = ..., decimal: str = ...,
        converters: Optional[Any] = ..., na_values: Optional[Any] = ...,
        keep_default_na: bool = ..., displayed_only: bool = ...) -> Any:
    ...
