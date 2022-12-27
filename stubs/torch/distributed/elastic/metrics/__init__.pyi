# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.distributed.elastic.metrics.static_init import *

from .api import ConsoleMetricHandler as ConsoleMetricHandler


        MetricData as MetricData, MetricHandler as MetricHandler,
        MetricsConfig as MetricsConfig,
        NullMetricHandler as NullMetricHandler, configure as configure,
        getStream as getStream, get_elapsed_time_ms as get_elapsed_time_ms,
        prof as prof, profile as profile, publish_metric as publish_metric,
        put_metric as put_metric
from typing import Optional


def initialize_metrics(cfg: Optional[MetricsConfig] = ...): ...
