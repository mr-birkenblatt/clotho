# Stubs for pandas.conftest (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,redefined-outer-name,invalid-name
# pylint: disable=relative-beyond-top-level,arguments-differ
# pylint: disable=no-member,too-few-public-methods,keyword-arg-before-vararg
# pylint: disable=super-init-not-called,abstract-method
from typing import Any


def pytest_addoption(parser: Any) -> None:
    ...


def pytest_runtest_setup(item: Any) -> None:
    ...


def configure_tests() -> None:
    ...


def add_imports(doctest_namespace: Any) -> None:
    ...


def spmatrix(request: Any) -> Any:
    ...


def axis(request: Any) -> Any:
    ...


axis_frame = axis


def axis_series(request: Any) -> Any:
    ...


def ip() -> Any:
    ...


def observed(request: Any) -> Any:
    ...


def ordered_fixture(request: Any) -> Any:
    ...


def all_arithmetic_operators(request: Any) -> Any:
    ...


def all_arithmetic_functions(request: Any) -> Any:
    ...


def all_numeric_reductions(request: Any) -> Any:
    ...


def all_boolean_reductions(request: Any) -> Any:
    ...


def cython_table_items(request: Any) -> Any:
    ...


def all_compare_operators(request: Any) -> Any:
    ...


def compare_operators_no_eq_ne(request: Any) -> Any:
    ...


def compression(request: Any) -> Any:
    ...


def compression_only(request: Any) -> Any:
    ...


def writable(request: Any) -> Any:
    ...


def datetime_tz_utc() -> Any:
    ...


def utc_fixture(request: Any) -> Any:
    ...


def join_type(request: Any) -> Any:
    ...


def strict_data_files(pytestconfig: Any) -> Any:
    ...


def datapath(strict_data_files: Any) -> Any:
    ...


def iris(datapath: Any) -> Any:
    ...


def nselect_method(request: Any) -> Any:
    ...


def closed(request: Any) -> Any:
    ...


def other_closed(request: Any) -> Any:
    ...


def nulls_fixture(request: Any) -> Any:
    ...


nulls_fixture2 = nulls_fixture


def unique_nulls_fixture(request: Any) -> Any:
    ...


unique_nulls_fixture2 = unique_nulls_fixture
TIMEZONES: Any
TIMEZONE_IDS: Any


def tz_naive_fixture(request: Any) -> Any:
    ...


def tz_aware_fixture(request: Any) -> Any:
    ...


tz_aware_fixture2 = tz_aware_fixture
UNSIGNED_INT_DTYPES: Any
UNSIGNED_EA_INT_DTYPES: Any
SIGNED_INT_DTYPES: Any
SIGNED_EA_INT_DTYPES: Any
ALL_INT_DTYPES: Any
ALL_EA_INT_DTYPES: Any
FLOAT_DTYPES: Any
COMPLEX_DTYPES: Any
STRING_DTYPES: Any
DATETIME64_DTYPES: Any
TIMEDELTA64_DTYPES: Any
BOOL_DTYPES: Any
BYTES_DTYPES: Any
OBJECT_DTYPES: Any
ALL_REAL_DTYPES: Any
ALL_NUMPY_DTYPES: Any


def string_dtype(request: Any) -> Any:
    ...


def bytes_dtype(request: Any) -> Any:
    ...


def object_dtype(request: Any) -> Any:
    ...


def datetime64_dtype(request: Any) -> Any:
    ...


def timedelta64_dtype(request: Any) -> Any:
    ...


def float_dtype(request: Any) -> Any:
    ...


def complex_dtype(request: Any) -> Any:
    ...


def sint_dtype(request: Any) -> Any:
    ...


def uint_dtype(request: Any) -> Any:
    ...


def any_int_dtype(request: Any) -> Any:
    ...


def any_real_dtype(request: Any) -> Any:
    ...


def any_numpy_dtype(request: Any) -> Any:
    ...


ids: Any


def any_skipna_inferred_dtype(request: Any) -> Any:
    ...


def tick_classes(request: Any) -> Any:
    ...


cls: Any


def float_frame() -> Any:
    ...
