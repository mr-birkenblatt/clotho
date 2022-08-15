# Stubs for pandas.core.indexes.api (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,useless-import-alias
from typing import Any

from pandas._libs.tslibs import NaT as NaT
from pandas.core.indexes.base import _new_Index as _new_Index
from pandas.core.indexes.base import ensure_index as ensure_index
from pandas.core.indexes.base import (
    ensure_index_from_sequences as ensure_index_from_sequences,
)
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.base import InvalidIndexError as InvalidIndexError
from pandas.core.indexes.category import CategoricalIndex as CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex as IntervalIndex
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.indexes.numeric import Float64Index as Float64Index
from pandas.core.indexes.numeric import Int64Index as Int64Index
from pandas.core.indexes.numeric import NumericIndex as NumericIndex
from pandas.core.indexes.numeric import UInt64Index as UInt64Index
from pandas.core.indexes.period import PeriodIndex as PeriodIndex
from pandas.core.indexes.range import RangeIndex as RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex as TimedeltaIndex

__all__ = [
    "Index",
    "MultiIndex",
    "NumericIndex",
    "Float64Index",
    "Int64Index",
    "CategoricalIndex",
    "IntervalIndex",
    "RangeIndex",
    "UInt64Index",
    "InvalidIndexError",
    "TimedeltaIndex",
    "PeriodIndex",
    "DatetimeIndex",
    "_new_Index",
    "NaT",
    "ensure_index",
    "ensure_index_from_sequences",
    "_get_combined_index",
    "_get_objs_combined_axis",
    "_union_indexes",
    "_get_consensus_names",
    "_all_indexes_same",
]


def _get_objs_combined_axis(
        objs: Any, intersect: bool = False,
        axis: int = 0, sort: bool = True) -> Any:
    ...


def _get_distinct_objs(objs: Any) -> Any:
    ...


def _get_combined_index(
        indexes: Any, intersect: bool = False,
        sort: bool = False) -> Any:
    ...


def _union_indexes(indexes: Any, sort: bool = True) -> Any:
    ...


def _sanitize_and_check(indexes: Any) -> Any:
    ...


def _get_consensus_names(indexes: Any) -> Any:
    ...


def _all_indexes_same(indexes: Any) -> bool:
    ...
