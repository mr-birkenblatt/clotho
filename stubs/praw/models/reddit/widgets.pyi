# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name,too-few-public-methods
# pylint: disable=super-init-not-called
from json import JSONEncoder
from typing import Any, Dict, List, Union

import praw
from _typeshed import Incomplete

from ...const import API_PATH as API_PATH
from ...util.cache import cachedproperty as cachedproperty
from ..base import PRAWBase as PRAWBase
from ..list.base import BaseList as BaseList


WidgetType: Incomplete


class Button(PRAWBase):
    ...


class CalendarConfiguration(PRAWBase):
    ...


class Hover(PRAWBase):
    ...


class Image(PRAWBase):
    ...


class ImageData(PRAWBase):
    ...


class MenuLink(PRAWBase):
    ...


class Styles(PRAWBase):
    ...


class Submenu(BaseList):
    CHILD_ATTRIBUTE: str


class SubredditWidgets(PRAWBase):
    def id_card(self) -> praw.models.IDCard: ...
    def items(self) -> Dict[str, 'praw.models.Widget']: ...
    def mod(self) -> praw.models.SubredditWidgetsModeration: ...
    def moderators_widget(self) -> praw.models.ModeratorsWidget: ...
    def sidebar(self) -> List['praw.models.Widget']: ...
    def topbar(self) -> List['praw.models.Menu']: ...
    def refresh(self) -> None: ...
    def __getattr__(self, attribute: str) -> Any: ...
    subreddit: Incomplete
    progressive_images: bool
    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...


class SubredditWidgetsModeration:
    def __init__(
        self, subreddit: praw.models.Subreddit,
        reddit: praw.Reddit) -> None: ...

    def add_button_widget(
        self, *, buttons: List[Dict[str, Union[
            Dict[str, str], str, int, Dict[str, Union[str, int]]]]],
        description: str, short_name: str, styles: Dict[str, str],
        **other_settings: Any) -> praw.models.ButtonWidget: ...

    def add_calendar(
        self, *, configuration: Dict[str, Union[bool, int]],
        google_calendar_id: str, requires_sync: bool, short_name: str,
        styles: Dict[str, str], **other_settings: Any,
        ) -> praw.models.Calendar: ...

    def add_community_list(
        self, *, data: List[Union[str, 'praw.models.Subreddit']],
        description: str = ..., short_name: str,
        styles: Dict[str, str], **other_settings: Any,
        ) -> praw.models.CommunityList: ...

    def add_custom_widget(
        self, *, css: str, height: int,
        image_data: List[Dict[str, Union[str, int]]], short_name: str,
        styles: Dict[str, str], text: str, **other_settings: Any,
        ) -> praw.models.CustomWidget: ...

    def add_image_widget(
        self, *, data: List[Dict[str, Union[str, int]]], short_name: str,
        styles: Dict[str, str], **other_settings: Any,
        ) -> praw.models.ImageWidget: ...

    def add_menu(
        self, *, data: List[Dict[str, Union[List[Dict[str, str]], str]]],
        **other_settings: Any) -> praw.models.Menu: ...

    def add_post_flair_widget(
        self, *, display: str, order: List[str], short_name: str,
        styles: Dict[str, str],
        **other_settings: Any) -> praw.models.PostFlairWidget: ...

    def add_text_area(
        self, *, short_name: str, styles: Dict[str, str], text: str,
        **other_settings: Any) -> praw.models.TextArea: ...

    def reorder(
        self, new_order: List[Union['WidgetType', str]],
        *, section: str = ...) -> None: ...

    def upload_image(self, file_path: str) -> str: ...


class Widget(PRAWBase):
    def mod(self) -> praw.models.WidgetModeration: ...
    def __eq__(self, other: Any) -> bool: ...
    subreddit: str
    id: str
    def __init__(self, reddit: praw.Reddit, _data: Dict[str, Any]) -> None: ...


class ButtonWidget(Widget, BaseList):
    CHILD_ATTRIBUTE: str


class Calendar(Widget):
    ...


class CommunityList(Widget, BaseList):
    CHILD_ATTRIBUTE: str


class CustomWidget(Widget):
    def __init__(self, reddit: praw.Reddit, _data: Dict[str, Any]) -> None: ...


class IDCard(Widget):
    ...


class ImageWidget(Widget, BaseList):
    CHILD_ATTRIBUTE: str


class Menu(Widget, BaseList):
    CHILD_ATTRIBUTE: str


class ModeratorsWidget(Widget, BaseList):
    CHILD_ATTRIBUTE: str
    def __init__(self, reddit: praw.Reddit, _data: Dict[str, Any]) -> None: ...


class PostFlairWidget(Widget, BaseList):
    CHILD_ATTRIBUTE: str


class RulesWidget(Widget, BaseList):
    CHILD_ATTRIBUTE: str
    def __init__(self, reddit: praw.Reddit, _data: Dict[str, Any]) -> None: ...


class TextArea(Widget):
    ...


class WidgetEncoder(JSONEncoder):
    def default(self, o: Any) -> Any: ...


class WidgetModeration:
    widget: Incomplete

    def __init__(
        self, widget: praw.models.Widget,
        subreddit: Union['praw.models.Subreddit', str],
        reddit: praw.Reddit) -> None: ...

    def delete(self) -> None: ...

    def update(self, **kwargs: Any) -> 'WidgetType': ...
