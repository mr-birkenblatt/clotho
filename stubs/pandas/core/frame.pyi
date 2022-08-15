# Stubs for pandas.core.frame (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-import,unused-argument,invalid-name,redefined-builtin
# pylint: disable=too-few-public-methods,function-redefined
# pylint: disable=redefined-outer-name,too-many-ancestors,super-init-not-called
# pylint: disable=too-many-arguments,keyword-arg-before-vararg
# pylint: disable=arguments-differ,signature-differs,disallowed-name
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    overload,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import numpy as np
from pandas.core.generic import NDFrame

if TYPE_CHECKING:
    import pandas as pd


SelectLike = Union[
    'DataFrame',
    'pd.Index',
    'pd.Series',
    List[Any],
    np.ndarray,
    slice,
    str,
]


class DataFrame(NDFrame):
    def __init__(
            self, data: Optional[Any] = ..., index: Optional[Any] = ...,
            columns: Optional[Any] = ..., dtype: Optional[Any] = ...,
            copy: bool = ...) -> None:
        ...

    @property
    def _constructor_expanddim(self) -> Any:
        ...

    @property
    def axes(self) -> Any:
        ...

    @property
    def shape(self) -> Tuple[int, int]:
        ...

    def to_string(
            self, buf: Optional[Any] = ...,
            columns: Optional[Any] = ...,
            col_space: Optional[Any] = ...,
            header: bool = ...,
            index: bool = ...,
            na_rep: str = ...,
            formatters: Optional[Any] = ...,
            float_format: Optional[Any] = ...,
            sparsify: Optional[Any] = ...,
            index_names: bool = ...,
            justify: Optional[Any] = ...,
            max_rows: Optional[Any] = ...,
            min_rows: Optional[Any] = ...,
            max_cols: Optional[Any] = ...,
            show_dimensions: bool = ...,
            decimal: str = ...,
            line_width: Optional[Any] = ...) -> Any:
        ...

    @property
    def style(self) -> Any:
        ...

    def items(self) -> Any:
        ...

    def iteritems(self) -> Any:
        ...

    def iterrows(self) -> Iterator[Tuple[Any, 'pd.Series']]:
        ...

    def itertuples(self, index: bool = ..., name: str = ...) -> Any:
        ...

    def __len__(self) -> int:
        ...

    def dot(self, other: Any) -> Any:
        ...

    def __matmul__(self, other: Any) -> Any:
        ...

    def __rmatmul__(self, other: Any) -> Any:
        ...

    @classmethod
    def from_dict(
            cls, data: Any, orient: str = ...,
            dtype: Optional[Any] = ...,
            columns: Optional[Any] = ...) -> Any:
        ...

    def to_numpy(
            self,
            dtype: Optional[Any] = ...,
            copy: bool = ...) -> np.ndarray:
        ...

    def to_dict(
            self, orient: str = ...,
            into: Any = ...) -> Dict[Any, List[Any]]:
        ...

    def to_gbq(
            self, destination_table: Any, project_id: Optional[Any] = ...,
            chunksize: Optional[Any] = ..., reauth: bool = ...,
            if_exists: str = ..., auth_local_webserver: bool = ...,
            table_schema: Optional[Any] = ...,
            location: Optional[Any] = ..., progress_bar: bool = ...,
            credentials: Optional[Any] = ..., verbose: Optional[Any] = ...,
            private_key: Optional[Any] = ...) -> None:
        ...

    @classmethod
    def from_records(
            cls, data: Any, index: Optional[Any] = ...,
            exclude: Optional[Any] = ...,
            columns: Optional[Any] = ..., coerce_float: bool = ...,
            nrows: Optional[Any] = ...) -> Any:
        ...

    def to_records(
            self, index: bool = ...,
            convert_datetime64: Optional[Any] = ...,
            column_dtypes: Optional[Any] = ...,
            index_dtypes: Optional[Any] = ...) -> Any:
        ...

    @classmethod
    def from_items(
            cls, items: Any, columns: Optional[Any] = ...,
            orient: str = ...) -> Any:
        ...

    def to_sparse(
            self, fill_value: Optional[Any] = ...,
            kind: str = ...) -> Any:
        ...

    def to_stata(
            self, fname: Any, convert_dates: Optional[Any] = ...,
            write_index: bool = ..., encoding: str = ...,
            byteorder: Optional[Any] = ...,
            time_stamp: Optional[Any] = ...,
            data_label: Optional[Any] = ...,
            variable_labels: Optional[Any] = ..., version: int = ...,
            convert_strl: Optional[Any] = ...) -> None:
        ...

    def to_feather(self, fname: Any) -> None:
        ...

    def to_parquet(
            self, fname: Any, engine: str = ..., compression: str = ...,
            index: Optional[Any] = ...,
            partition_cols: Optional[Any] = ...,
            **kwargs: Any) -> bytes:
        ...

    def to_html(
            self, buf: Optional[Any] = ..., columns: Optional[Any] = ...,
            col_space: Optional[Any] = ..., header: bool = ...,
            index: bool = ..., na_rep: str = ...,
            formatters: Optional[Any] = ...,
            float_format: Optional[Any] = ...,
            sparsify: Optional[Any] = ...,
            index_names: bool = ...,
            justify: Optional[Any] = ...,
            max_rows: Optional[Any] = ...,
            max_cols: Optional[Any] = ...,
            show_dimensions: bool = ...,
            decimal: str = ...,
            bold_rows: bool = ...,
            classes: Optional[Any] = ...,
            escape: bool = ...,
            notebook: bool = ...,
            border: Optional[Any] = ...,
            table_id: Optional[Any] = ...,
            render_links: bool = ...) -> Any:
        ...

    def info(
            self, verbose: Optional[Any] = ..., buf: Optional[Any] = ...,
            max_cols: Optional[Any] = ..., memory_usage: Optional[Any] = ...,
            null_counts: Optional[Any] = ...) -> Any:
        ...

    def memory_usage(self, index: bool = ..., deep: bool = ...) -> Any:
        ...

    def transpose(self, *args: Any, **kwargs: Any) -> Any:
        ...

    T: 'DataFrame' = ...

    def get_value(self, index: Any, col: Any, takeable: bool = ...) -> Any:
        ...

    def set_value(
            self, index: Any, col: Any, value: Any,
            takeable: bool = ...) -> Any:
        ...

    @overload
    def __getitem__(self, key: str) -> 'pd.Series':
        ...

    @overload
    def __getitem__(self, key: np.ndarray) -> 'DataFrame':
        ...

    @overload
    def __getitem__(self, key: 'pd.Index') -> 'DataFrame':
        ...

    @overload
    def __getitem__(
            self,
            key: Tuple[SelectLike, SelectLike]) -> 'DataFrame':
        ...

    @overload
    def __getitem__(
            self,
            key: List[Any]) -> 'DataFrame':
        ...

    @overload
    def __getitem__(
            self,
            key: 'pd.Series') -> 'DataFrame':
        ...

    @overload
    def __getitem__(self, key: 'DataFrame') -> 'DataFrame':
        ...

    def query(self, expr: Any, inplace: bool = ..., **kwargs: Any) -> Any:
        ...

    def eval(self, expr: Any, inplace: bool = ..., **kwargs: Any) -> Any:
        ...

    def select_dtypes(
            self, include: Optional[Any] = ...,
            exclude: Optional[Any] = ...) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any) -> Any:
        ...

    def insert(
            self, loc: Any, column: Any, value: Any,
            allow_duplicates: bool = ...) -> None:
        ...

    def assign(self, **kwargs: Any) -> Any:
        ...

    def lookup(self, row_labels: Any, col_labels: Any) -> Any:
        ...

    def align(
            self, other: Any, join: str = ..., axis: Optional[Any] = ...,
            level: Optional[Any] = ..., copy: bool = ...,
            fill_value: Optional[Any] = ..., method: Optional[Any] = ...,
            limit: Optional[Any] = ..., fill_axis: int = ...,
            broadcast_axis: Optional[Any] = ...) -> Any:
        ...

    def reindex(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def drop(
            self, labels: Optional[Any] = ..., axis: int = ...,
            index: Optional[Any] = ..., columns: Optional[Any] = ...,
            level: Optional[Any] = ..., inplace: bool = ...,
            errors: str = ...) -> Any:
        ...

    def rename(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def fillna(
            self, value: Optional[Any] = ..., method: Optional[Any] = ...,
            axis: Optional[Any] = ..., inplace: bool = ...,
            limit: Optional[Any] = ..., downcast: Optional[Any] = ...,
            **kwargs: Any) -> Any:
        ...

    def replace(
            self, to_replace: Optional[Any] = ...,
            value: Optional[Any] = ..., inplace: bool = ...,
            limit: Optional[Any] = ..., regex: bool = ...,
            method: str = ...) -> Any:
        ...

    def shift(
            self, periods: int = ..., freq: Optional[Any] = ...,
            axis: int = ...,
            fill_value: Optional[Any] = ...) -> 'DataFrame':
        ...

    def set_index(
            self, keys: Any, drop: bool = ..., append: bool = ...,
            inplace: Literal[False] = ...,
            verify_integrity: bool = ...) -> 'DataFrame':
        ...

    def reset_index(
            self, level: Optional[Any] = ..., drop: bool = ...,
            inplace: Literal[False] = ..., col_level: int = ...,
            col_fill: str = ...) -> 'DataFrame':
        ...

    def isna(self) -> 'DataFrame':
        ...

    def isnull(self) -> 'DataFrame':
        ...

    def notna(self) -> 'DataFrame':
        ...

    def notnull(self) -> 'DataFrame':
        ...

    def dropna(
            self, axis: int = ..., how: str = ...,
            thresh: Optional[Any] = ..., subset: Optional[Any] = ...,
            inplace: Literal[False] = ...) -> 'DataFrame':
        ...

    def drop_duplicates(
            self, subset: Optional[Any] = ...,
            keep: str = ..., inplace: bool = ...) -> Any:
        ...

    def duplicated(self, subset: Optional[Any] = ..., keep: str = ...) -> Any:
        ...

    def sort_index(
            self, axis: int = ..., level: Optional[Any] = ...,
            ascending: bool = ..., inplace: bool = ...,
            kind: str = ..., na_position: str = ...,
            sort_remaining: bool = ...,
            by: Optional[Any] = ...) -> Any:
        ...

    def nlargest(self, n: Any, columns: Any, keep: str = ...) -> Any:
        ...

    def nsmallest(self, n: Any, columns: Any, keep: str = ...) -> Any:
        ...

    def swaplevel(self, i: int = ..., j: int = ..., axis: int = ...) -> Any:
        ...

    def reorder_levels(self, order: Any, axis: int = ...) -> Any:
        ...

    def combine(
            self, other: Any, func: Any, fill_value: Optional[Any] = ...,
            overwrite: bool = ...) -> Any:
        ...

    def combine_first(self, other: Any) -> Any:
        ...

    def update(
            self, other: Any, join: str = ..., overwrite: bool = ...,
            filter_func: Optional[Any] = ..., errors: str = ...) -> None:
        ...

    def pivot(
            self, index: Optional[Any] = ..., columns: Optional[Any] = ...,
            values: Optional[Any] = ...) -> Any:
        ...

    def pivot_table(
            self, values: Optional[Any] = ...,
            index: Optional[Any] = ..., columns: Optional[Any] = ...,
            aggfunc: str = ..., fill_value: Optional[Any] = ...,
            margins: bool = ..., dropna: bool = ...,
            margins_name: str = ..., observed: bool = ...) -> Any:
        ...

    def stack(self, level: int = ..., dropna: bool = ...) -> Any:
        ...

    def explode(self, column: Union[str, Tuple]) -> 'DataFrame':
        ...

    def unstack(
            self, level: int = ...,
            fill_value: Optional[Any] = ...) -> Any:
        ...

    def melt(
            self, id_vars: Optional[Any] = ...,
            value_vars: Optional[Any] = ..., var_name: Optional[Any] = ...,
            value_name: str = ..., col_level: Optional[Any] = ...) -> Any:
        ...

    def diff(self, periods: int = ..., axis: int = ...) -> Any:
        ...

    def aggregate(
            self, func: Any, axis: int = ...,
            *args: Any, **kwargs: Any) -> Any:
        ...

    agg: Any = ...

    def transform(
            self, func: Any, axis: int = ...,
            *args: Any, **kwargs: Any) -> Any:
        ...

    def apply(
            self, func: Any, axis: int = ..., broadcast: Optional[Any] = ...,
            raw: bool = ..., reduce: Optional[Any] = ...,
            result_type: Optional[Any] = ..., args: Any = ...,
            **kwds: Any) -> Any:
        ...

    def applymap(self, func: Any) -> Any:
        ...

    def append(
            self, other: Any, ignore_index: bool = ...,
            verify_integrity: bool = ..., sort: Optional[Any] = ...) -> Any:
        ...

    def join(
            self, other: Any, on: Optional[Any] = ..., how: str = ...,
            lsuffix: str = ..., rsuffix: str = ..., sort: bool = ...) -> Any:
        ...

    def merge(
            self, right: Any, how: str = ..., on: Optional[Any] = ...,
            left_on: Optional[Any] = ..., right_on: Optional[Any] = ...,
            left_index: bool = ..., right_index: bool = ...,
            sort: bool = ..., suffixes: Any = ..., copy: bool = ...,
            indicator: bool = ..., validate: Optional[Any] = ...) -> Any:
        ...

    def round(self, decimals: int = ..., *args: Any, **kwargs: Any) -> Any:
        ...

    def corr(self, method: str = ..., min_periods: int = ...) -> Any:
        ...

    def cov(self, min_periods: Optional[Any] = ...) -> Any:
        ...

    def corrwith(
            self, other: Any, axis: int = ..., drop: bool = ...,
            method: str = ...) -> Any:
        ...

    def count(
            self, axis: int = ..., level: Optional[Any] = ...,
            numeric_only: bool = ...) -> Any:
        ...

    def nunique(self, axis: int = ..., dropna: bool = ...) -> Any:
        ...

    def idxmin(self, axis: int = ..., skipna: bool = ...) -> Any:
        ...

    def idxmax(self, axis: int = ..., skipna: bool = ...) -> Any:
        ...

    def mode(
            self, axis: int = ..., numeric_only: bool = ...,
            dropna: bool = ...) -> Any:
        ...

    def quantile(
            self, q: float = ..., axis: int = ...,
            numeric_only: bool = ..., interpolation: str = ...) -> Any:
        ...

    def to_timestamp(
            self, freq: Optional[Any] = ..., how: str = ...,
            axis: int = ..., copy: bool = ...) -> Any:
        ...

    def to_period(
            self, freq: Optional[Any] = ..., axis: int = ...,
            copy: bool = ...) -> Any:
        ...

    def isin(self, values: Any) -> Any:
        ...

    plot: Any = ...
    hist: Any = ...
    boxplot: Any = ...
    sparse: Any = ...

    def __lt__(self, other: Any) -> 'DataFrame':
        ...

    def __le__(self, other: Any) -> 'DataFrame':
        ...

    def __eq__(self, other: object) -> bool:  # NOTE: cannot change return type
        ...

    def __ne__(self, other: object) -> bool:  # NOTE: cannot change return type
        ...

    def __gt__(self, other: Any) -> 'DataFrame':
        ...

    def __ge__(self, other: Any) -> 'DataFrame':
        ...
