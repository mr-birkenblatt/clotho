# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import abc
from typing import Dict, NamedTuple, Optional

from _typeshed import Incomplete


class MetricData(NamedTuple):
    timestamp: Incomplete
    group_name: Incomplete
    name: Incomplete
    value: Incomplete


class MetricsConfig:
    params: Incomplete
    def __init__(self, params: Optional[Dict[str, str]] = ...) -> None: ...


class MetricHandler(abc.ABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def emit(self, metric_data: MetricData): ...


class ConsoleMetricHandler(MetricHandler):
    def emit(self, metric_data: MetricData): ...


class NullMetricHandler(MetricHandler):
    def emit(self, metric_data: MetricData): ...


class MetricStream:
    group_name: Incomplete
    handler: Incomplete
    def __init__(self, group_name: str, handler: MetricHandler) -> None: ...
    def add_value(self, metric_name: str, metric_value: int): ...


def configure(handler: MetricHandler, group: str = ...): ...


def getStream(group: str): ...


def prof(fn: Incomplete | None = ..., group: str = ...): ...


def profile(group: Incomplete | None = ...): ...


def put_metric(
    metric_name: str, metric_value: int, metric_group: str = ...): ...


def publish_metric(metric_group: str, metric_name: str, metric_value: int): ...


def get_elapsed_time_ms(start_time_in_seconds: float): ...
