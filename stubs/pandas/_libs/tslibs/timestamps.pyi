# pylint: disable=unused-argument,too-few-public-methods,no-self-use
# Stubs for pandas._libs.tslibs (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
from typing import Optional, Any, TYPE_CHECKING, overload, Union
import datetime
if TYPE_CHECKING:
    from pandas import Timedelta
    from pandas.tseries.offsets import BusinessMixin


TDistance = Union['Timedelta', 'BusinessMixin', datetime.timedelta]


class Timestamp:
    def __init__(
            self, ts_input: Any, freq: Any = ..., tz: Any = ...,
            unit: Optional[str] = ..., year: Optional[int] = ...,
            month: Optional[int] = ..., day: Optional[int] = ...,
            hour: Optional[int] = ..., minute: Optional[int] = ...,
            second: Optional[int] = ..., microsecond: Optional[int] = ...,
            nanosecond: Optional[int] = ...,
            tzinfo: Optional[datetime.tzinfo] = ...):
        ...

    def date(self) -> datetime.date:
        ...

    @overload
    def __add__(self, other: 'Timedelta') -> 'Timestamp':
        ...

    @overload
    def __add__(self, other: TDistance) -> 'Timestamp':
        ...

    @overload
    def __sub__(self, other: TDistance) -> 'Timestamp':
        ...

    @overload
    def __sub__(self, other: 'Timestamp') -> 'Timedelta':
        ...

    def __lt__(self, other: 'Timestamp') -> bool:
        ...

    def __le__(self, other: 'Timestamp') -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __gt__(self, other: 'Timestamp') -> bool:
        ...

    def __ge__(self, other: 'Timestamp') -> bool:
        ...

    def replace(
            self, year: Optional[int] = None,
            month: Optional[int] = None, day: Optional[int] = None,
            hour: Optional[int] = None, minute: Optional[int] = None,
            second: Optional[int] = None,
            microsecond: Optional[int] = None,
            nanosecond: Optional[int] = None,
            tzinfo: Any = ..., fold: Optional[int] = 0) -> 'Timestamp':
        ...
