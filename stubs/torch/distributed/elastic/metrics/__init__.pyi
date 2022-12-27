# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Optional

from torch.distributed.elastic.metrics.static_init import *

from .api import configure as configure
from .api import ConsoleMetricHandler as ConsoleMetricHandler
from .api import get_elapsed_time_ms as get_elapsed_time_ms
from .api import getStream as getStream
from .api import MetricData as MetricData
from .api import MetricHandler as MetricHandler
from .api import MetricsConfig as MetricsConfig
from .api import NullMetricHandler as NullMetricHandler
from .api import prof as prof
from .api import profile as profile
from .api import publish_metric as publish_metric
from .api import put_metric as put_metric


def initialize_metrics(cfg: Optional[MetricsConfig] = ...): ...
