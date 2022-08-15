# Stubs for pandas.tseries.offsets (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,super-init-not-called,no-name-in-module
# pylint: disable=c-extension-no-member,invalid-name,disallowed-name
# pylint: disable=protected-access,too-few-public-methods
from typing import Any, Optional

from pandas._libs.tslibs import offsets as liboffsets
from pandas._libs.tslibs.offsets import BaseOffset

class DateOffset(BaseOffset):
    normalize: bool = ...

    def __init__(
            self, n: int = ..., normalize: bool = ...,
            **kwds: Any) -> None:
        ...

    def apply(self, other: Any) -> Any:
        ...

    def apply_index(self, i: Any) -> Any:
        ...

    def isAnchored(self) -> bool:
        ...

    @property
    def name(self) -> Any:
        ...

    def rollback(self, dt: Any) -> Any:
        ...

    def rollforward(self, dt: Any) -> Any:
        ...

    def onOffset(self, dt: Any) -> Any:
        ...

    @property
    def rule_code(self) -> Any:
        ...

    def freqstr(self) -> Any:
        ...

    @property
    def nanos(self) -> None:
        ...


class SingleConstructorOffset(DateOffset):
    ...


class _CustomMixin:
    def __init__(self, weekmask: Any, holidays: Any, calendar: Any) -> None:
        ...


class BusinessMixin:
    @property
    def offset(self) -> Any:
        ...


class BusinessDay(BusinessMixin, SingleConstructorOffset):
    def __init__(
            self, n: int = ..., normalize: bool = ...,
            offset: Any = ...) -> None:
        ...

    def apply(self, other: Any) -> Any:
        ...

    def apply_index(self, i: Any) -> Any:
        ...

    def onOffset(self, dt: Any) -> Any:
        ...


class BusinessHourMixin(BusinessMixin):
    def __init__(
            self, start: str = ..., end: str = ...,
            offset: Any = ...) -> None:
        ...

    def next_bday(self) -> Any:
        ...

    def rollback(self, dt: Any) -> Any:
        ...

    def rollforward(self, dt: Any) -> Any:
        ...

    def apply(self, other: Any) -> Any:
        ...

    def onOffset(self, dt: Any) -> Any:
        ...


class BusinessHour(BusinessHourMixin, SingleConstructorOffset):
    def __init__(
            self, n: int = ..., normalize: bool = ..., start: str = ...,
            end: str = ..., offset: Any = ...) -> None:
        ...


class CustomBusinessDay(_CustomMixin, BusinessDay):
    def __init__(
            self, n: int = ..., normalize: bool = ...,
            weekmask: str = ..., holidays: Optional[Any] = ...,
            calendar: Optional[Any] = ..., offset: Any = ...) -> None:
        ...

    def apply(self, other: Any) -> Any:
        ...

    def apply_index(self, i: Any) -> None:
        ...

    def onOffset(self, dt: Any) -> Any:
        ...


class CustomBusinessHour(
        _CustomMixin,
        BusinessHourMixin,
        SingleConstructorOffset):
    def __init__(
            self, n: int = ..., normalize: bool = ...,
            weekmask: str = ..., holidays: Optional[Any] = ...,
            calendar: Optional[Any] = ..., start: str = ...,
            end: str = ..., offset: Any = ...) -> None:
        ...


class MonthOffset(SingleConstructorOffset):
    __init__: Any = ...

    @property
    def name(self) -> Any:
        ...

    def onOffset(self, dt: Any) -> Any:
        ...

    def apply(self, other: Any) -> Any:
        ...

    def apply_index(self, i: Any) -> Any:
        ...


class MonthEnd(MonthOffset):
    ...


class MonthBegin(MonthOffset):
    ...


class BusinessMonthEnd(MonthOffset):
    ...


class BusinessMonthBegin(MonthOffset):
    ...


class _CustomBusinessMonth(_CustomMixin, BusinessMixin, MonthOffset):
    onOffset: Any = ...
    apply_index: Any = ...

    def __init__(
            self, n: int = ..., normalize: bool = ...,
            weekmask: str = ..., holidays: Optional[Any] = ...,
            calendar: Optional[Any] = ..., offset: Any = ...) -> None:
        ...

    def cbday_roll(self) -> Any:
        ...

    def m_offset(self) -> Any:
        ...

    def month_roll(self) -> Any:
        ...

    def apply(self, other: Any) -> Any:
        ...


class CustomBusinessMonthEnd(_CustomBusinessMonth):
    ...


class CustomBusinessMonthBegin(_CustomBusinessMonth):
    ...


class SemiMonthOffset(DateOffset):
    def __init__(
            self, n: int = ..., normalize: bool = ...,
            day_of_month: Optional[Any] = ...) -> None:
        ...

    @property
    def rule_code(self) -> Any:
        ...

    def apply(self, other: Any) -> Any:
        ...

    def apply_index(self, i: Any) -> Any:
        ...


class SemiMonthEnd(SemiMonthOffset):
    def onOffset(self, dt: Any) -> Any:
        ...


class SemiMonthBegin(SemiMonthOffset):
    def onOffset(self, dt: Any) -> Any:
        ...


class Week(DateOffset):
    def __init__(
            self, n: int = ..., normalize: bool = ...,
            weekday: Optional[Any] = ...) -> None:
        ...

    def isAnchored(self) -> bool:
        ...

    def apply(self, other: Any) -> Any:
        ...

    def apply_index(self, i: Any) -> Any:
        ...

    def onOffset(self, dt: Any) -> Any:
        ...

    @property
    def rule_code(self) -> Any:
        ...


class _WeekOfMonthMixin:
    def apply(self, other: Any) -> Any:
        ...

    def onOffset(self, dt: Any) -> Any:
        ...


class WeekOfMonth(_WeekOfMonthMixin, DateOffset):
    def __init__(
            self, n: int = ..., normalize: bool = ..., week: int = ...,
            weekday: int = ...) -> None:
        ...

    @property
    def rule_code(self) -> Any:
        ...


class LastWeekOfMonth(_WeekOfMonthMixin, DateOffset):
    def __init__(
            self, n: int = ..., normalize: bool = ...,
            weekday: int = ...) -> None:
        ...

    @property
    def rule_code(self) -> Any:
        ...


class QuarterOffset(DateOffset):
    def __init__(
            self, n: int = ..., normalize: bool = ...,
            startingMonth: Optional[Any] = ...) -> None:
        ...

    def isAnchored(self) -> bool:
        ...

    @property
    def rule_code(self) -> Any:
        ...

    def apply(self, other: Any) -> Any:
        ...

    def onOffset(self, dt: Any) -> Any:
        ...

    def apply_index(self, i: Any) -> Any:
        ...


class BQuarterEnd(QuarterOffset):
    ...


class BQuarterBegin(QuarterOffset):
    ...


class QuarterEnd(QuarterOffset):
    ...


class QuarterBegin(QuarterOffset):
    ...


class YearOffset(DateOffset):
    def apply(self, other: Any) -> Any:
        ...

    def apply_index(self, i: Any) -> Any:
        ...

    def onOffset(self, dt: Any) -> Any:
        ...

    def __init__(
            self, n: int = ..., normalize: bool = ...,
            month: Optional[Any] = ...) -> None:
        ...

    @property
    def rule_code(self) -> Any:
        ...


class BYearEnd(YearOffset):
    ...


class BYearBegin(YearOffset):
    ...


class YearEnd(YearOffset):
    ...


class YearBegin(YearOffset):
    ...


class FY5253(DateOffset):
    def __init__(
            self, n: int = ..., normalize: bool = ..., weekday: int = ...,
            startingMonth: int = ..., variation: str = ...) -> None:
        ...

    def isAnchored(self) -> bool:
        ...

    def onOffset(self, dt: Any) -> Any:
        ...

    def apply(self, other: Any) -> Any:
        ...

    def get_year_end(self, dt: Any) -> Any:
        ...

    @property
    def rule_code(self) -> Any:
        ...

    def get_rule_code_suffix(self) -> Any:
        ...


class FY5253Quarter(DateOffset):
    def __init__(
            self, n: int = ..., normalize: bool = ..., weekday: int = ...,
            startingMonth: int = ..., qtr_with_extra_week: int = ...,
            variation: str = ...) -> None:
        ...

    def isAnchored(self) -> bool:
        ...

    def apply(self, other: Any) -> Any:
        ...

    def get_weeks(self, dt: Any) -> Any:
        ...

    def year_has_extra_week(self, dt: Any) -> Any:
        ...

    def onOffset(self, dt: Any) -> Any:
        ...

    @property
    def rule_code(self) -> str:
        ...


class Easter(DateOffset):
    __init__: Any = ...

    def apply(self, other: Any) -> Any:
        ...

    def onOffset(self, dt: Any) -> Any:
        ...


class Tick(liboffsets._Tick, SingleConstructorOffset):
    def __init__(self, n: int = ..., normalize: bool = ...) -> None:
        ...

    __gt__: Any = ...
    __ge__: Any = ...
    __lt__: Any = ...
    __le__: Any = ...

    def __add__(self, other: Any) -> Any:
        ...

    def __eq__(self, other: Any) -> Any:
        ...

    def __hash__(self) -> int:
        ...

    def __ne__(self, other: Any) -> Any:
        ...

    @property
    def delta(self) -> Any:
        ...

    @property
    def nanos(self) -> Any:
        ...

    def apply(self, other: Any) -> Any:
        ...

    def isAnchored(self) -> bool:
        ...


class Day(Tick):
    ...


class Hour(Tick):
    ...


class Minute(Tick):
    ...


class Second(Tick):
    ...


class Milli(Tick):
    ...


class Micro(Tick):
    ...


class Nano(Tick):
    ...


BDay = BusinessDay
BMonthEnd = BusinessMonthEnd
BMonthBegin = BusinessMonthBegin
CBMonthEnd = CustomBusinessMonthEnd
CBMonthBegin = CustomBusinessMonthBegin
CDay = CustomBusinessDay
