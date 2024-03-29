# Stubs for pandas.tseries.holiday (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,invalid-name,too-few-public-methods

from typing import Any, List, Optional


def next_monday(dt: Any) -> Any:
    ...


def next_monday_or_tuesday(dt: Any) -> Any:
    ...


def previous_friday(dt: Any) -> Any:
    ...


def sunday_to_monday(dt: Any) -> Any:
    ...


def weekend_to_monday(dt: Any) -> Any:
    ...


def nearest_workday(dt: Any) -> Any:
    ...


def next_workday(dt: Any) -> Any:
    ...


def previous_workday(dt: Any) -> Any:
    ...


def before_nearest_workday(dt: Any) -> Any:
    ...


def after_nearest_workday(dt: Any) -> Any:
    ...


class Holiday:
    name: Any = ...
    year: Any = ...
    month: Any = ...
    day: Any = ...
    offset: Any = ...
    start_date: Any = ...
    end_date: Any = ...
    observance: Any = ...
    days_of_week: Any = ...

    def __init__(
            self, name: Any, year: Optional[Any] = ...,
            month: Optional[Any] = ..., day: Optional[Any] = ...,
            offset: Optional[Any] = ..., observance: Optional[Any] = ...,
            start_date: Optional[Any] = ...,
            end_date: Optional[Any] = ...,
            days_of_week: Optional[Any] = ...) -> None:
        ...

    def dates(
            self, start_date: Any, end_date: Any,
            return_name: bool = ...) -> Any:
        ...


holiday_calendars: Any


def register(cls: Any) -> None:
    ...


def get_calendar(name: Any) -> Any:
    ...


class HolidayCalendarMetaClass(type):
    def __new__(cls, clsname: Any, bases: Any, attrs: Any) -> Any:
        ...


class AbstractHolidayCalendar(metaclass=HolidayCalendarMetaClass):
    rules: List[Holiday] = ...
    start_date: Any = ...
    end_date: Any = ...
    name: Any = ...

    def __init__(
            self, name: Optional[Any] = ...,
            rules: Optional[Any] = ...) -> None:
        ...

    def rule_from_name(self, name: Any) -> Any:
        ...

    def holidays(
            self, start: Optional[Any] = ..., end: Optional[Any] = ...,
            return_name: bool = ...) -> Any:
        ...

    @staticmethod
    def merge_class(base: Any, other: Any) -> Any:
        ...

    def merge(self, other: Any, inplace: bool = ...) -> Any:
        ...


USMemorialDay: Any
USLaborDay: Any
USColumbusDay: Any
USThanksgivingDay: Any
USMartinLutherKingJr: Any
USPresidentsDay: Any
GoodFriday: Any
EasterMonday: Any


class USFederalHolidayCalendar(AbstractHolidayCalendar):
    rules: Any = ...


def HolidayCalendarFactory(
        name: Any, base: Any, other: Any, base_class: Any = ...) -> Any:
    ...
